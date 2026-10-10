"""
Evaluates intra-generation re-scheduling under changing system load.

Per-request schedulers pick one strategy before the first token. If the load
changes while a long response is being generated, that choice goes stale.
Here policies may re-decide every N tokens from live telemetry; because all
strategies share the same weights and KV cache, switching is free and the
generated tokens are unchanged.

Load scenarios (background busy workers, times relative to request start):
    steady_idle   0 workers throughout
    steady_heavy  H workers throughout
    spike         0 workers, H workers from T seconds
    relief        H workers, 0 workers from T seconds

Conditions: static, latency_optimized (fixed), rule_once / rule_adaptive
(threshold baseline), rl_once / rl_adaptive (trained DQN). Each context
(prompt x scenario) runs every condition in shuffled order, with the load
schedule restarted for each run.

Usage:
    python evaluate_adaptive.py --num-prompts 4 --prompt-offset 20
"""

import argparse
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
from src.llm.inference import generate_partitioned_response
from src.llm.partition import PartitionStrategy
from src.rl.dqn_agent import DQNAgent
from src.scheduler.adaptive import FixedPolicy, RLPolicy, ThresholdPolicy
from src.telemetry.load_generator import DynamicCPULoad


def make_scenarios(heavy: int, switch_at: float):
    return {
        "steady_idle": [(0.0, 0)],
        "steady_heavy": [(0.0, heavy)],
        "spike": [(0.0, 0), (switch_at, heavy)],
        "relief": [(0.0, heavy), (switch_at, 0)]
    }


def make_policies(agent, threshold):
    policies = {
        "static": lambda: FixedPolicy(0),
        "latency_optimized": lambda: FixedPolicy(2),
        "rule_once": lambda: ThresholdPolicy(threshold, once=True),
        "rule_adaptive": lambda: ThresholdPolicy(threshold, once=False)
    }
    if agent is not None:
        policies["rl_once"] = lambda: RLPolicy(agent, once=True)
        policies["rl_adaptive"] = lambda: RLPolicy(agent, once=False)
    return policies


def run_adaptive_evaluation(
    model_path="data/processed/dqn_model_v3.pth",
    num_prompts=4,
    prompt_offset=20,
    max_new_tokens=100,
    decision_interval=10,
    heavy=8,
    switch_at=5.0,
    threshold=40.0,
    seed=11,
    tag="v3"
):
    data_dir = ROOT_DIR / "data" / "processed"
    results_dir = ROOT_DIR / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    agent = None
    if (ROOT_DIR / model_path).exists():
        agent = DQNAgent(state_size=4, action_size=3)
        agent.load(str(ROOT_DIR / model_path))
    else:
        print(f"No DQN model at {model_path}; RL conditions skipped.", flush=True)

    rng = np.random.default_rng(seed)
    scenarios = make_scenarios(heavy, switch_at)
    policies = make_policies(agent, threshold)
    prompts = load_prompts(csv_path=str(data_dir / "prompts.csv"), limit=prompt_offset + num_prompts)[prompt_offset:]

    contexts = [(prompt_offset + i + 1, p, s) for i, p in enumerate(prompts) for s in scenarios]
    rng.shuffle(contexts)

    print("=" * 80, flush=True)
    print(f"ADAPTIVE RE-SCHEDULING EVALUATION: {len(contexts)} contexts x {len(policies)} conditions, "
          f"decision every {decision_interval} tokens", flush=True)
    print("=" * 80, flush=True)

    generate_partitioned_response(prompts[0], strategy=0, max_new_tokens=8)  # warm-up

    runs, decisions, timelines = [], [], []
    runs_path = data_dir / f"adaptive_runs_{tag}.csv"
    t_begin = time.perf_counter()

    for c_idx, (prompt_id, prompt, scenario) in enumerate(contexts, start=1):
        print(f"[{c_idx:03d}/{len(contexts)}] prompt {prompt_id} {scenario:12s}:", end="", flush=True)
        lat = {}

        for run_order, cond in enumerate(rng.permutation(list(policies))):
            policy = policies[cond]()
            log = []
            time.sleep(1.0)  # let the previous run's load disappear
            with DynamicCPULoad(scenarios[scenario]):
                t0 = time.perf_counter()
                _, stages, token_ids = generate_partitioned_response(
                    prompt, max_new_tokens=max_new_tokens, return_token_ids=True,
                    scheduler=policy, decision_interval=decision_interval, decision_log=log
                )
                latency = time.perf_counter() - t0

            chosen = [d["strategy_id"] for d in log]
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
                "num_switches": int(sum(a != b for a, b in zip(chosen, chosen[1:]))),
                "decision_overhead_seconds": round(sum(d["decision_overhead_seconds"] for d in log), 4),
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

        eta = (time.perf_counter() - t_begin) / c_idx * (len(contexts) - c_idx)
        print(f" {lat} | ETA {eta / 60:.0f} min", flush=True)
        pd.DataFrame(runs).to_csv(runs_path, index=False)

    df = pd.DataFrame(runs)
    df.to_csv(runs_path, index=False)
    pd.DataFrame(decisions).to_csv(data_dir / f"adaptive_decisions_{tag}.csv", index=False)
    df_timeline = pd.DataFrame(timelines)
    df_timeline.to_csv(data_dir / f"adaptive_timelines_{tag}.csv", index=False)

    report_adaptive_results(df, df_timeline, data_dir, results_dir, tag)
    return df


def _only(diffs, condition):
    return diffs[diffs["condition"] == condition] if not diffs.empty else diffs


def report_adaptive_results(df, df_timeline, data_dir, results_dir, tag):
    same_output = df.groupby("context_id")["token_ids"].nunique().eq(1).mean()

    summary = summarize_runs(df, ["scenario", "condition"])
    summary.to_csv(data_dir / f"adaptive_summary_{tag}.csv", index=False)

    best = "rl_adaptive" if "rl_adaptive" in set(df["condition"]) else "rule_adaptive"
    diffs = pd.concat([
        _only(paired_differences(df, "condition", baseline=b, group_col="scenario"), best)
        for b in sorted(set(df["condition"]) - {best})
    ], ignore_index=True)
    diffs.to_csv(data_dir / f"adaptive_paired_diffs_{tag}.csv", index=False)

    wide = df.pivot_table(index="context_id", columns="condition", values="latency_seconds")
    scenario_of = df.groupby("context_id")["scenario"].first()
    fixed = [c for c in ["static", "latency_optimized"] if c in wide]
    regret = wide.drop(columns="scenario", errors="ignore").sub(wide[fixed].min(axis=1), axis=0)
    regret["scenario"] = scenario_of

    print("\n" + "=" * 80, flush=True)
    print("MEAN LATENCY BY SCENARIO AND CONDITION", flush=True)
    print("=" * 80, flush=True)
    print(df.pivot_table(index="scenario", columns="condition", values="latency_seconds", aggfunc="mean")
          .round(2).to_string(), flush=True)
    print(f"\nPAIRED DIFFERENCES: {best} VS OTHERS (negative = {best} faster)", flush=True)
    print(diffs[["scenario", "baseline", "num_contexts", "mean_diff_seconds", "ci95_low", "ci95_high",
                 "mean_diff_percent", "significant"]].to_string(index=False), flush=True)
    print("\nMEAN REGRET VS BEST FIXED STRATEGY IN HINDSIGHT (seconds)", flush=True)
    print(regret.groupby("scenario").mean().round(2).to_string(), flush=True)
    print("\nMEAN STRATEGY SWITCHES PER REQUEST", flush=True)
    print(df.pivot_table(index="scenario", columns="condition", values="num_switches").round(2).to_string(), flush=True)
    print(f"\nIdentical generated tokens across all conditions: {same_output:.0%} of contexts", flush=True)
    print(f"Mean decision overhead per request: {df['decision_overhead_seconds'].mean():.3f}s", flush=True)

    plot_latency_by_load(summary, "condition", results_dir / f"adaptive_latency_by_scenario_{tag}.png",
                         "Latency under changing load", group_col="scenario")
    if not diffs.empty:
        plot_paired_differences(diffs, results_dir / f"adaptive_paired_diffs_{tag}.png",
                                f"{best} latency difference vs other policies", group_col="scenario", label_col="baseline")
    spike_ctx = df[df["scenario"] == "spike"]["context_id"]
    if len(spike_ctx):
        plot_strategy_timeline(df_timeline[df_timeline["context_id"] == spike_ctx.iloc[0]],
                               results_dir / f"adaptive_spike_timeline_{tag}.png",
                               "Per-token latency when load arrives mid-generation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="data/processed/dqn_model_v3.pth")
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--prompt-offset", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--interval", type=int, default=10, help="Tokens between scheduling decisions.")
    parser.add_argument("--heavy", type=int, default=8, help="Busy workers during heavy load.")
    parser.add_argument("--switch-at", type=float, default=5.0, help="Seconds until the load changes.")
    parser.add_argument("--threshold", type=float, default=40.0, help="External CPU %% threshold of the rule baseline.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--tag", default="v3")
    args = parser.parse_args()

    run_adaptive_evaluation(
        model_path=args.model,
        num_prompts=args.num_prompts,
        prompt_offset=args.prompt_offset,
        max_new_tokens=args.max_new_tokens,
        decision_interval=args.interval,
        heavy=args.heavy,
        switch_at=args.switch_at,
        threshold=args.threshold,
        seed=args.seed,
        tag=args.tag
    )
