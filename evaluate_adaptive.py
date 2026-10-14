"""
Evaluates intra-generation re-scheduling under changing system load.

Per-request schedulers pick one strategy before the first token. If the load
changes while a long response is being generated, that choice goes stale.
Here policies may re-decide every N tokens from live telemetry; because all
strategies share the same weights and KV cache, switching is free and the
generated tokens are unchanged.

Load scenarios (background busy workers, changes after a number of generated
tokens, so every policy meets the change at the same point of the response):
    steady_idle   0 workers throughout
    steady_heavy  H workers throughout
    spike         0 workers, H workers from token K
    relief        H workers, 0 workers from token K

Conditions:
    static, balanced, latency_optimized   fixed strategies
    rule_adaptive                         external-CPU threshold, re-decided with hysteresis
    rl_once / rl_adaptive                 offline DQN, per request / re-decided with hysteresis
    bandit_adaptive                       online bandit learning from per-segment token latency,
                                          warm-started from the DQN; one instance for the whole run

Each context (prompt x scenario) runs every condition in shuffled order, with
the load schedule restarted for each run. All runs generate exactly
max_new_tokens tokens (EOS ignored) so latencies are comparable.

Usage:
    python evaluate_adaptive.py --num-prompts 10 --prompt-offset 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.benchmark.evaluate import paired_differences, summarize_runs
from src.benchmark.plots import plot_latency_by_load, plot_paired_differences, plot_strategy_timeline
from src.data.load_prompts import load_prompts
from src.llm.partition import PartitionStrategy
from src.scheduler.adaptive import CoresHeuristicPolicy, FixedPolicy, OnlineBanditPolicy, RLPolicy, ThresholdPolicy
from src.telemetry.load_generator import TokenScheduledLoad
from src.telemetry.monitor import BackgroundLoadSampler

FIXED = ["static", "balanced", "latency_optimized"]


def _load_inference():
    from src.llm.inference import generate_partitioned_response

    return generate_partitioned_response


def _load_dqn_agent():
    from src.rl.dqn_agent import DQNAgent

    return DQNAgent


def _count_decision_switches(decision_log):
    configurations = [
        (row["strategy_id"], tuple(row.get("thread_counts", ())))
        for row in decision_log
    ]
    return sum(previous != current for previous, current in zip(configurations, configurations[1:]))


def make_scenarios(heavy: int, switch_token: int, ramp=None):
    scenarios = {
        "steady_idle": [(0, 0)],
        "steady_heavy": [(0, heavy)],
        "spike": [(0, 0), (switch_token, heavy)],
        "relief": [(0, heavy), (switch_token, 0)]
    }
    if ramp:
        scenarios["ramp"] = [(0, 0)] + list(ramp)
    return scenarios


def parse_ramp(text):
    """'60:8,140:12' -> [(60, 8), (140, 12)] (token, workers)."""
    return [tuple(int(v) for v in part.split(":")) for part in text.split(",")] if text else None


def make_policies(agent, threshold, sampler, patience=2, seed=0, bandit_edges=(35.0,), bin_margin=8.0,
                  thread_budgets=PartitionStrategy.THREAD_BUDGETS, emergency_cpu=None):
    policies = {name: (lambda sid=sid: FixedPolicy(sid))
                for sid, name in PartitionStrategy.STRATEGIES.items()}
    policies["rule_adaptive"] = lambda: ThresholdPolicy(
        threshold, sampler=sampler, patience=patience, emergency_cpu=emergency_cpu,
        emergency_action=1
    )
    policies["rule_noh"] = lambda: ThresholdPolicy(threshold, sampler=sampler, patience=0,
                                                    emergency_cpu=emergency_cpu, emergency_action=1)
    policies["cores_heuristic"] = lambda: CoresHeuristicPolicy(
        sampler=sampler, patience=patience, emergency_cpu=emergency_cpu
    )

    prior = None
    if agent is not None:
        policies["rl_once"] = lambda: RLPolicy(agent, once=True, sampler=sampler)
        policies["rl_adaptive"] = lambda: RLPolicy(agent, sampler=sampler, patience=patience,
                               emergency_cpu=emergency_cpu, emergency_action=1)
        DQNAgent = _load_dqn_agent()
        prior = DQNAgent(state_size=agent.state_size, action_size=agent.action_size)
        prior.model.load_state_dict(agent.model.state_dict())

    # The bandit keeps learning across requests: one shared instance
    bandit = OnlineBanditPolicy(prior_agent=prior, sampler=sampler, seed=seed, edges=bandit_edges,
                                bin_margin=bin_margin, emergency_cpu=emergency_cpu)
    policies["bandit_adaptive"] = lambda: bandit
    return policies, bandit


def run_adaptive_evaluation(
    model_path="data/processed/dqn_model_v3.pth",
    num_prompts=10,
    prompt_offset=20,
    max_new_tokens=200,
    decision_interval=10,
    heavy=8,
    switch_token=60,
    threshold=40.0,
    patience=2,
    seed=11,
    tag="v4",
    resume=False,
    ramp=None,
    bandit_edges=(35.0,),
    bin_margin=8.0,
    thread_budgets=PartitionStrategy.THREAD_BUDGETS,
    emergency_cpu=None,
    emergency_latency_factor=None
):
    data_dir = ROOT_DIR / "data" / "processed"
    results_dir = ROOT_DIR / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    if decision_interval <= 0:
        raise ValueError("decision_interval must be positive")
    thread_budgets = tuple(int(value) for value in thread_budgets)

    agent = None
    resolved_model_path = ROOT_DIR / model_path
    if not resolved_model_path.exists() and (ROOT_DIR / 'data' / 'processed' / 'dqn_model_v6.pth').exists():
        resolved_model_path = ROOT_DIR / 'data' / 'processed' / 'dqn_model_v6.pth'
    if resolved_model_path.exists():
        DQNAgent = _load_dqn_agent()
        agent = DQNAgent(state_size=4, action_size=3)
        agent.load(str(resolved_model_path))
    else:
        print(f"No DQN model at {model_path}; RL conditions skipped, bandit starts without a prior.", flush=True)

    rng = np.random.default_rng(seed)
    scenarios = make_scenarios(heavy, switch_token, ramp)
    prompts = load_prompts(csv_path=str(data_dir / "prompts.csv"), limit=prompt_offset + num_prompts)[prompt_offset:]

    contexts = [(prompt_offset + i + 1, p, s) for i, p in enumerate(prompts) for s in scenarios]
    rng.shuffle(contexts)

    sampler = BackgroundLoadSampler().start()
    generate_partitioned_response = _load_inference()
    policies, bandit = make_policies(agent, threshold, sampler, patience=patience, seed=seed,
                                     bandit_edges=bandit_edges, bin_margin=bin_margin,
                                     thread_budgets=thread_budgets, emergency_cpu=emergency_cpu)
    # Drawn up front so a resumed run replays exactly the same condition orders
    orders = [rng.permutation(list(policies)) for _ in contexts]

    paths = {name: data_dir / f"adaptive_{name}_{tag}.csv" for name in ["runs", "decisions", "timelines"]}
    state_path = data_dir / f"adaptive_bandit_state_{tag}.json"
    runs, decisions, timelines = [], [], []
    if resume and paths["runs"].exists():
        runs = pd.read_csv(paths["runs"]).to_dict("records")
        for name, rows in [("decisions", decisions), ("timelines", timelines)]:
            if paths[name].exists():
                rows.extend(pd.read_csv(paths[name]).to_dict("records"))
        if state_path.exists():
            bandit.load_state_dict(json.loads(state_path.read_text()))
        else:
            print("No bandit checkpoint found: the bandit restarts from its prior.", flush=True)
    done = {r["context_id"] for r in runs}
    if done:
        print(f"Resuming: {len(done)} contexts already done.", flush=True)

    print("=" * 80, flush=True)
    print(f"ADAPTIVE RE-SCHEDULING EVALUATION: {len(contexts)} contexts x {len(policies)} conditions, "
          f"{max_new_tokens} tokens, decision every {decision_interval} tokens, load change at token {switch_token}",
          flush=True)
    print("=" * 80, flush=True)

    generate_partitioned_response(prompts[0], strategy=0, max_new_tokens=8)  # warm-up

    t_begin = time.perf_counter()
    num_new = 0

    try:
        for c_idx, (prompt_id, prompt, scenario) in enumerate(contexts, start=1):
            if c_idx in done:
                continue
            print(f"[{c_idx:03d}/{len(contexts)}] prompt {prompt_id} {scenario:12s}:", end="", flush=True)
            lat = {}

            for run_order, cond in enumerate(orders[c_idx - 1]):
                policy = policies[cond]()
                log = []
                time.sleep(1.0)  # let the previous run's load disappear
                load = TokenScheduledLoad(scenarios[scenario])
                with load:
                    t0 = time.perf_counter()
                    _, stages, token_ids = generate_partitioned_response(
                        prompt, max_new_tokens=max_new_tokens, return_token_ids=True,
                        scheduler=policy, decision_interval=decision_interval, decision_log=log,
                        step_callback=load.on_step, ignore_eos=True,
                        emergency_latency_factor=None if getattr(policy, "is_fixed", False) else emergency_latency_factor
                    )
                    latency = time.perf_counter() - t0

                per_token = pd.DataFrame(stages).groupby("token_step").agg(
                    token_seconds=("stage_latency_seconds", "sum"), strategy_id=("strategy_id", "first"))
                runs.append({
                    "context_id": c_idx,
                    "prompt_id": prompt_id,
                    "scenario": scenario,
                    "condition": cond,
                    "run_order": run_order,
                    "num_generated_tokens": len(token_ids),
                    "latency_seconds": round(latency, 4),
                    "tokens_per_second": round(len(token_ids) / latency, 3),
                    "num_decisions": len(log),
                    "num_switches": _count_decision_switches(log),
                    "decision_overhead_seconds": round(sum(d["decision_overhead_seconds"] for d in log), 4),
                    "thread_config_overhead_seconds": round(
                        sum(s.get("thread_config_overhead_seconds", 0.0) for s in stages), 8
                    ),
                    "decision_interval": decision_interval,
                    "patience": patience,
                    "policy_patience": 0 if cond == "rule_noh" else patience if cond in {
                        "rule_adaptive", "cores_heuristic", "rl_adaptive"
                    } else None,
                    "emergency_cpu": emergency_cpu,
                    "thread_budgets": ",".join(map(str, thread_budgets)),
                    **{f"share_{PartitionStrategy(s).strategy_name}": round((per_token["strategy_id"] == s).mean(), 3)
                       for s in PartitionStrategy.STRATEGIES},
                    # summarize_runs expects these columns
                    "pre_cpu_percent": log[0].get("external_cpu_percent", np.nan) if log else np.nan,
                    "pre_ram_percent": log[0].get("ram_percent", np.nan) if log else np.nan,
                    "total_handoff_latency_seconds": sum(s["handoff_latency_seconds"] for s in stages),
                    "total_activation_handoff_mb": sum(s["activation_size_mb"] for s in stages),
                    "token_ids": " ".join(map(str, token_ids))
                })
                for d in log:
                    decisions.append({"context_id": c_idx, "scenario": scenario, "condition": cond, **d})
                for step, row in per_token.iterrows():
                    timelines.append({"context_id": c_idx, "scenario": scenario, "condition": cond,
                                      "token_step": step, "token_seconds": row["token_seconds"],
                                      "strategy_id": row["strategy_id"]})
                lat[cond] = round(latency, 1)

            num_new += 1
            eta = (time.perf_counter() - t_begin) / num_new * (len(contexts) - c_idx)
            print(f" {lat} | ETA {eta / 60:.0f} min", flush=True)
            # Checkpoint after every context so an interrupted run can --resume
            for name, rows in [("runs", runs), ("decisions", decisions), ("timelines", timelines)]:
                pd.DataFrame(rows).to_csv(paths[name], index=False)
            state_path.write_text(json.dumps(bandit.state_dict()))
    finally:
        sampler.stop()

    df = pd.DataFrame(runs)
    pd.DataFrame(bandit.history).to_csv(data_dir / f"adaptive_bandit_history_{tag}.csv", index=False)
    df_timeline = pd.DataFrame(timelines)

    report_adaptive_results(df, df_timeline, data_dir, results_dir, tag)
    return df


def stalled_contexts(df, ratio=2.0, min_excess_seconds=5.0):
    """
    Contexts containing a latency-only outlier under the supplied threshold.

    This is a sensitivity flag, not evidence that paging occurred. Paging-based
    exclusion requires independent page-fault or swap telemetry.
    """
    median = df.groupby(["scenario", "condition"])["latency_seconds"].transform("median")
    stalled = (df["latency_seconds"] > ratio * median) & (df["latency_seconds"] > median + min_excess_seconds)
    return sorted(int(c) for c in df.loc[stalled, "context_id"].unique())


def report_adaptive_results(df, df_timeline, data_dir, results_dir, tag, stall_ratio=2.0):
    output_tag = f"{tag}r" if tag == "v5" else tag
    stalled = stalled_contexts(df, stall_ratio) if stall_ratio else []

    same_output = df.groupby("context_id")["token_ids"].nunique().eq(1).mean()

    # PRIMARY ANALYSIS: all contexts retained
    summary = summarize_runs(df, ["scenario", "condition"])
    summary.to_csv(data_dir / f"adaptive_summary_{output_tag}.csv", index=False)

    conditions = set(df["condition"])
    primary = next(c for c in ["rl_adaptive", "rule_adaptive", "bandit_adaptive"] if c in conditions)
    diffs = paired_differences(df, "condition", baseline=sorted(conditions - {primary}),
                               conditions=[primary], group_col="scenario")
    diffs.to_csv(data_dir / f"adaptive_paired_diffs_{output_tag}.csv", index=False)

    wide = df.pivot_table(index="context_id", columns="condition", values="latency_seconds")
    scenario_of = df.groupby("context_id")["scenario"].first()
    fixed = [c for c in FIXED if c in wide]
    regret = wide.sub(wide[fixed].min(axis=1), axis=0)
    regret["scenario"] = scenario_of

    print("\n" + "=" * 80, flush=True)
    print("PRIMARY ANALYSIS: ALL CONTEXTS (seconds)", flush=True)
    print("=" * 80, flush=True)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.pivot_table(index="scenario", columns="condition", values="latency_seconds", aggfunc="mean")
              .round(2).to_string(), flush=True)
        print(f"\nPRIMARY PAIRED DIFFERENCES: {primary} VS OTHERS (negative = {primary} faster)", flush=True)
        cols_show = ["scenario", "baseline", "num_contexts", "mean_diff_seconds", "ci95_low", "ci95_high",
                     "mean_diff_percent", "p_value_t_raw", "p_value_t_adjusted", "significant"]
        valid_cols = [c for c in cols_show if c in diffs.columns]
        print(diffs[valid_cols].to_string(index=False), flush=True)
        print("\nMEAN REGRET VS BEST FIXED STRATEGY IN HINDSIGHT, PER CONTEXT (seconds)", flush=True)
        print(regret.groupby("scenario").mean().round(2).to_string(), flush=True)
        print("\nMEAN STRATEGY SWITCHES PER REQUEST", flush=True)
        print(df.pivot_table(index="scenario", columns="condition", values="num_switches").round(2).to_string(),
              flush=True)

    # SENSITIVITY ANALYSIS: clearly labeled table excluding heuristic outliers
    if stalled:
        print("\n" + "=" * 80, flush=True)
        print(f"SENSITIVITY ANALYSIS: EXCLUDING {len(stalled)} LATENCY OUTLIER CONTEXT(S): {stalled}", flush=True)
        print("(Sensitivity table: no independent paging telemetry in source data)", flush=True)
        print("=" * 80, flush=True)
        df_sens = df[~df["context_id"].isin(stalled)]
        sens_summary = summarize_runs(df_sens, ["scenario", "condition"])
        sens_summary.to_csv(data_dir / f"adaptive_summary_sensitivity_{output_tag}.csv", index=False)
        sens_diffs = paired_differences(df_sens, "condition", baseline=sorted(conditions - {primary}),
                                        conditions=[primary], group_col="scenario")
        sens_diffs.to_csv(data_dir / f"adaptive_paired_diffs_sensitivity_{output_tag}.csv", index=False)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(df_sens.pivot_table(index="scenario", columns="condition", values="latency_seconds", aggfunc="mean")
                  .round(2).to_string(), flush=True)
            print(f"\nSENSITIVITY PAIRED DIFFERENCES: {primary} VS OTHERS", flush=True)
            valid_sens_cols = [c for c in cols_show if c in sens_diffs.columns]
            print(sens_diffs[valid_sens_cols].to_string(index=False), flush=True)

    if "bandit_adaptive" in regret:
        # Learning curve: does the online bandit improve as it serves more requests?
        half = len(regret) // 2
        ordered = regret.sort_index()
        print("\nONLINE BANDIT LEARNING: mean regret, first vs second half of the run (seconds)", flush=True)
        for label, part in [("first half", ordered.iloc[:half]), ("second half", ordered.iloc[half:])]:
            cols = [c for c in ["bandit_adaptive", "rl_adaptive", "rl_once"] if c in part]
            print(f"  {label:11s} " + "  ".join(f"{c}={part[c].mean():.2f}" for c in cols), flush=True)

    print(f"\nIdentical generated tokens across all conditions: {same_output:.0%} of contexts", flush=True)
    print(f"Mean decision overhead per request: {df['decision_overhead_seconds'].mean():.4f}s", flush=True)

    plot_latency_by_load(summary, "condition", results_dir / f"adaptive_latency_by_scenario_{output_tag}.png",
                         "Latency under changing load", group_col="scenario")
    if not diffs.empty:
        plot_paired_differences(diffs, results_dir / f"adaptive_paired_diffs_{output_tag}.png",
                                f"{primary} latency difference vs other policies", group_col="scenario", label_col="baseline")
    for scenario in ["spike", "relief"]:
        ctx = df[df["scenario"] == scenario]["context_id"]
        if len(ctx):
            plot_strategy_timeline(df_timeline[df_timeline["context_id"] == ctx.iloc[-1]],
                                   results_dir / f"adaptive_{scenario}_timeline_{output_tag}.png",
                                   f"Per-token latency, load {'arrives' if scenario == 'spike' else 'leaves'} mid-generation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="data/processed/dqn_model_v3.pth")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--prompt-offset", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--interval", type=int, choices=[5, 10, 20], default=10,
                        help="Tokens between scheduling decisions.")
    parser.add_argument("--heavy", type=int, default=8, help="Busy workers during heavy load.")
    parser.add_argument("--switch-token", type=int, default=60, help="Generated tokens until the load changes.")
    parser.add_argument("--threshold", type=float, default=40.0, help="External CPU %% threshold of the rule baseline.")
    parser.add_argument("--patience", type=int, choices=[0, 1, 2], default=2,
                        help="Hysteresis decisions before switching (0 and 1 both switch immediately).")
    parser.add_argument("--thread-budgets", default="2,4,6,8,10,12",
                        help="Comma-separated core-budget candidates for cores_heuristic.")
    parser.add_argument("--emergency-cpu", type=float, default=None,
                        help="Scheduler CPU emergency: external CPU percent that bypasses policy patience "
                             "at the next decision interval (off by default).")
    parser.add_argument("--emergency-latency-factor", type=float, default=None,
                        help="Engine latency switch for adaptive conditions: next token on balanced when a token "
                             "takes more than this factor x the median of the last 10 (off by default).")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--tag", default="v4")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted run with the same tag.")
    parser.add_argument("--report-only", action="store_true", help="Re-analyse saved results without running.")
    parser.add_argument("--ramp", default="", help="Extra 'ramp' scenario from idle, e.g. '60:8,140:12' (token:workers).")
    parser.add_argument("--bandit-edges", default="35", help="External CPU %% bin edges of the online bandit.")
    parser.add_argument("--bin-margin", type=float, default=8.0, help="Bandit bin hysteresis in CPU %% points.")
    parser.add_argument("--keep-awake", action="store_true", help="Stop Windows from sleeping while this runs.")
    args = parser.parse_args()

    if args.keep_awake and sys.platform == "win32":
        import ctypes
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: released automatically when the process exits
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    if args.report_only:
        data_dir = ROOT_DIR / "data" / "processed"
        report_adaptive_results(pd.read_csv(data_dir / f"adaptive_runs_{args.tag}.csv"),
                                pd.read_csv(data_dir / f"adaptive_timelines_{args.tag}.csv"),
                                data_dir, ROOT_DIR / "results", args.tag)
        sys.exit(0)

    run_adaptive_evaluation(
        model_path=args.model,
        num_prompts=args.num_prompts,
        prompt_offset=args.prompt_offset,
        max_new_tokens=args.max_new_tokens,
        decision_interval=args.interval,
        heavy=args.heavy,
        switch_token=args.switch_token,
        threshold=args.threshold,
        patience=args.patience,
        seed=args.seed,
        tag=args.tag,
        resume=args.resume,
        ramp=parse_ramp(args.ramp),
        bandit_edges=tuple(float(e) for e in args.bandit_edges.split(",")),
        bin_margin=args.bin_margin,
        thread_budgets=tuple(int(value) for value in args.thread_budgets.split(",")),
        emergency_cpu=args.emergency_cpu,
        emergency_latency_factor=args.emergency_latency_factor
    )
