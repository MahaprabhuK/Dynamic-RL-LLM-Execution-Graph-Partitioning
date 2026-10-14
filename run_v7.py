"""
Plan-driven runner for the v7 experiments (smoke test, pilot, main run,
realistic-load check).

Everything that defines a run comes from a committed plan file
(experiments/<stage>_plan.yaml): prompts, scenarios, conditions, seeds, token
count. The runner never changes the plan; it refuses to start when the plan
file has uncommitted changes.

Per run it records: latency, decode s/token, switches, emergency triggers
(latency / cpu), external CPU before and during, RAM % and available MB,
paging deltas (process page faults, machine-wide pages input, swap), the git
commit hash and wall-clock times. Before every run the RAM guard waits while
RAM is above the plan's limit and logs skipped runs to skipped_runs_<tag>.csv.

Outputs (data/processed/):
    v7runs_<tag>.csv, v7timelines_<tag>.csv, v7decisions_<tag>.csv,
    v7segments_<tag>.csv, skipped_runs_<tag>.csv, v7bandit_state_<tag>.json

Usage:
    python run_v7.py --plan experiments/smoke_plan.yaml --resume --keep-awake
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.load_prompts import load_prompts
from src.llm.partition import apply_partition_strategy
from src.telemetry.load_generator import TokenScheduledLoad
from src.telemetry.monitor import BackgroundLoadSampler
from src.telemetry.run_guard import git_commit_hash, git_is_dirty, ram_guard, run_metadata, summarize_segments

DATA_DIR = ROOT_DIR / "data" / "processed"


def load_plan(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plan_is_committed(path):
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--", str(path)], cwd=ROOT_DIR, text=True)
        tracked = subprocess.call(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT_DIR,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
        return tracked and not out.strip()
    except Exception:
        return False


def count_switches(decision_log):
    configs = [(d["strategy_id"], tuple(d.get("thread_counts", ()))) for d in decision_log]
    return sum(a != b for a, b in zip(configs, configs[1:]))


def build_policy(cond, sampler, shared):
    """Policy object for one condition of the plan (see experiments/*_plan.yaml)."""
    from src.scheduler import adaptive

    kind = cond["policy"]
    patience = cond.get("patience", 2)
    emergency_cpu = cond.get("emergency_cpu")
    if kind == "fixed":
        return adaptive.FixedPolicy(cond["arm"])
    if kind == "rule":
        return adaptive.ThresholdPolicy(cond["threshold"], low_load_action=cond["low_arm"],
                                        high_load_action=cond["high_arm"], sampler=sampler, patience=patience,
                                        emergency_cpu=emergency_cpu, emergency_action=cond.get("emergency_arm"))
    if kind == "rule_lookup":
        return adaptive.LookupRulePolicy.from_file(ROOT_DIR / cond["rule_file"], sampler=sampler, patience=patience,
                                                   emergency_cpu=emergency_cpu,
                                                   emergency_action=cond.get("emergency_arm"))
    if kind == "cores_heuristic":
        kwargs = {"arms": cond["arms"]} if "arms" in cond else {}
        return adaptive.CoresHeuristicPolicy(sampler=sampler, patience=patience, emergency_cpu=emergency_cpu,
                                             **kwargs)
    if kind == "rl":
        key = ("rl", cond["model"])
        if key not in shared:
            from src.rl.dqn_agent import DQNAgent
            agent = DQNAgent(state_size=cond.get("state_size", 4), action_size=cond.get("action_size", 3))
            agent.load(str(ROOT_DIR / cond["model"]))
            shared[key] = agent
        agent = shared[key]
        if cond.get("v7_state", False):
            return adaptive.RLPolicyV7(agent, arms=cond["arms"], sampler=sampler, patience=patience,
                                       emergency_cpu=emergency_cpu)
        return adaptive.RLPolicy(agent, sampler=sampler, patience=patience, emergency_cpu=emergency_cpu,
                                 emergency_action=cond.get("emergency_arm"))
    if kind == "bandit":
        key = ("bandit", cond["name"])
        if key not in shared:
            if cond.get("v7", False):
                shared[key] = adaptive.OnlineBanditPolicyV7.from_file(ROOT_DIR / cond["prior_file"], sampler=sampler,
                                                                      seed=cond.get("seed", 0))
            else:
                prior = None
                if cond.get("prior_model"):
                    from src.rl.dqn_agent import DQNAgent
                    prior = DQNAgent(state_size=4, action_size=3)
                    prior.load(str(ROOT_DIR / cond["prior_model"]))
                shared[key] = adaptive.OnlineBanditPolicy(prior_agent=prior, sampler=sampler,
                                                          seed=cond.get("seed", 0),
                                                          edges=tuple(cond.get("edges", (35.0,))))
        return shared[key]
    raise ValueError(f"Unknown policy kind {kind!r}")


def make_load(plan, schedule):
    load_cfg = plan.get("load", {"type": "busy"})
    if load_cfg.get("type", "busy") == "busy":
        return TokenScheduledLoad(schedule)
    if load_cfg["type"] == "realistic":
        from src.telemetry.realistic_load import TokenScheduledRealisticLoad
        return TokenScheduledRealisticLoad(schedule, **{k: v for k, v in load_cfg.items() if k != "type"})
    raise ValueError(f"Unknown load type {load_cfg['type']!r}")


def run_plan(plan_path, resume=False, only_prompts=None, allow_uncommitted=False):
    plan_path = Path(plan_path)
    plan = load_plan(plan_path)
    tag = plan["tag"]
    if not allow_uncommitted and not plan_is_committed(plan_path):
        raise SystemExit(f"{plan_path} is not committed (or has local changes); commit the plan before running.")

    paths = {name: DATA_DIR / f"v7{name}_{tag}.csv" for name in ["runs", "timelines", "decisions", "segments"]}
    skipped_csv = DATA_DIR / f"skipped_runs_{tag}.csv"
    state_dir = DATA_DIR / f"v7bandit_state_{tag}"
    if not resume and paths["runs"].exists():
        raise SystemExit(f"{paths['runs']} exists; use --resume (results are never overwritten).")

    seed = int(plan["seed"])
    rng = np.random.default_rng(seed)
    prompt_ids = list(plan["prompts"])
    scenarios = {name: [tuple(step) for step in steps] for name, steps in plan["scenarios"].items()}
    conditions = {c["name"]: c for c in plan["conditions"]}
    max_new_tokens = int(plan["max_new_tokens"])
    guard = plan.get("ram_guard", {})
    flag_cfg = plan.get("paging_flag", {})
    flag_pages = flag_cfg.get("system_pages_input_per_segment")

    all_prompts = load_prompts(csv_path=str(DATA_DIR / "prompts.csv"), limit=max(prompt_ids))
    # Context order: prompt-major (so a night can cover a block of prompts), scenarios shuffled per prompt.
    # Condition order: shuffled per context. All drawn up front so --resume replays the same orders.
    contexts = []
    for pid in prompt_ids:
        for scen in rng.permutation(list(scenarios)):
            contexts.append((pid, str(scen), [str(c) for c in rng.permutation(list(conditions))]))

    frames = {name: [] for name in paths}
    if resume:
        for name, path in paths.items():
            if path.exists():
                frames[name] = pd.read_csv(path).to_dict("records")
    done = {(r["prompt_id"], r["scenario"], r["condition"]) for r in frames["runs"]}

    from src.llm.inference import generate_partitioned_response
    generate_partitioned_response(all_prompts[prompt_ids[0] - 1], strategy="balanced", max_new_tokens=8)  # warm-up

    sampler = BackgroundLoadSampler().start()
    shared = {}
    # Restore shared bandit states
    for name, cond in conditions.items():
        if cond["policy"] == "bandit":
            policy = build_policy(cond, sampler, shared)
            state_file = state_dir / f"{name}.json"
            if resume and state_file.exists():
                policy.load_state_dict(json.loads(state_file.read_text()))

    total = len(contexts) * len(conditions)
    todo = [(c_idx, ctx) for c_idx, ctx in enumerate(contexts, start=1)
            if only_prompts is None or ctx[0] in only_prompts]
    print("=" * 80, flush=True)
    print(f"v7 RUN tag={tag} plan={plan_path.name} commit={git_commit_hash()[:10]}"
          f"{' (dirty)' if git_is_dirty() else ''}", flush=True)
    print(f"{len(contexts)} contexts x {len(conditions)} conditions = {total} runs; "
          f"{len(done)} already done; {max_new_tokens} tokens", flush=True)
    print("=" * 80, flush=True)
    t_begin, new_runs = time.perf_counter(), 0

    try:
        for c_idx, (pid, scen, order) in todo:
            prompt = all_prompts[pid - 1]
            pending = [c for c in order if (pid, scen, c) not in done]
            if not pending:
                continue
            print(f"[ctx {c_idx:03d}/{len(contexts)}] prompt {pid} {scen:12s}:", end="", flush=True)
            for run_order, cond_name in enumerate(order):
                if (pid, scen, cond_name) in done:
                    continue
                cond = conditions[cond_name]
                run_info = {"tag": tag, "context_id": c_idx, "prompt_id": pid, "scenario": scen,
                            "condition": cond_name, "run_order": run_order}
                ok, _ = ram_guard(skipped_csv, run_info, max_percent=guard.get("max_percent", 85.0),
                                  wait_seconds=guard.get("wait_seconds", 30.0), retries=guard.get("retries", 10))
                if not ok:
                    continue
                policy = build_policy(cond, sampler, shared)
                interval = int(cond.get("interval", plan.get("decision_interval", 10)))
                latency_factor = cond.get("emergency_latency_factor")

                time.sleep(1.0)  # let the previous run's load disappear
                meta_before = run_metadata()
                cpu_before = sampler.latest()["external_cpu_percent"]
                cpu_during = []
                load = make_load(plan, scenarios[scen])

                def on_step(step, load=load):
                    load.on_step(step)
                    if step % 5 == 0 and step > 0:
                        cpu_during.append(sampler.latest()["external_cpu_percent"])

                log, segments = [], []
                with load:
                    t0 = time.perf_counter()
                    _, stages, token_ids = generate_partitioned_response(
                        prompt, max_new_tokens=max_new_tokens, return_token_ids=True, scheduler=policy,
                        decision_interval=interval, decision_log=log, step_callback=on_step, ignore_eos=True,
                        emergency_latency_factor=latency_factor, conservative_strategy=cond.get("conservative_arm",
                                                                                                "balanced"),
                        segment_log=segments
                    )
                    latency = time.perf_counter() - t0
                meta_after = run_metadata()

                stage_df = pd.DataFrame(stages)
                per_token = stage_df.groupby("token_step").agg(
                    token_seconds=("stage_latency_seconds", "sum"), arm=("arm", "first"),
                    emergency_trigger=("emergency_trigger", "first"), emergency_kind=("emergency_kind", "first"))
                paging = summarize_segments(segments)
                sys_pages = [s["system_pages_input_delta"] for s in segments]
                max_sys = max(sys_pages) if sys_pages else -1
                emergencies = per_token[per_token["emergency_trigger"]]
                arm_share = per_token["arm"].value_counts(normalize=True)
                row = {
                    **run_info,
                    "max_new_tokens": max_new_tokens,
                    "num_generated_tokens": len(token_ids),
                    "latency_seconds": round(latency, 4),
                    "decode_sec_per_token": round(float(per_token["token_seconds"].iloc[1:].median()), 6)
                    if len(per_token) > 1 else np.nan,
                    "sum_token_seconds": round(float(per_token["token_seconds"].sum()), 4),
                    "num_decisions": sum(1 for d in log if d.get("emergency_kind") != "latency"),
                    "num_switches": count_switches(log),
                    "emergency_triggers": int(len(emergencies)),
                    "emergency_latency_triggers": int((emergencies["emergency_kind"] == "latency").sum()),
                    "emergency_cpu_triggers": int((emergencies["emergency_kind"] == "cpu").sum()),
                    "decision_interval": interval,
                    "patience": cond.get("patience") if cond["policy"] != "fixed" else None,
                    "emergency_latency_factor": latency_factor,
                    "emergency_cpu": cond.get("emergency_cpu"),
                    "arms_used": ",".join(sorted(arm_share.index)),
                    "arm_shares": json.dumps({k: round(float(v), 3) for k, v in arm_share.items()}),
                    "external_cpu_before": cpu_before,
                    "external_cpu_during": round(float(np.mean(cpu_during)), 2) if cpu_during else np.nan,
                    "ram_percent_before": meta_before["ram_percent"],
                    "available_mb_before": meta_before["available_mb"],
                    "ram_percent_after": meta_after["ram_percent"],
                    "available_mb_after": meta_after["available_mb"],
                    **paging,
                    "system_pages_input_delta": int(sum(p for p in sys_pages if p >= 0)) if sys_pages else -1,
                    "max_segment_system_pages_input": max_sys,
                    "paging_flag": bool(flag_pages is not None and max_sys >= flag_pages),
                    "commit_hash": git_commit_hash(),
                    "tree_dirty": git_is_dirty(),
                    "wall_clock_start": meta_before["wall_clock"],
                    "wall_clock_end": meta_after["wall_clock"],
                    "token_ids": " ".join(map(str, token_ids)),
                }
                frames["runs"].append(row)
                for step, t in per_token.iterrows():
                    frames["timelines"].append({**run_info, "token_step": int(step),
                                                "token_seconds": round(float(t["token_seconds"]), 6),
                                                "arm": t["arm"], "emergency_trigger": bool(t["emergency_trigger"]),
                                                "emergency_kind": t["emergency_kind"]})
                for d in log:
                    frames["decisions"].append({**run_info, **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                                                               for k, v in d.items()}})
                for s in segments:
                    frames["segments"].append({**run_info, **s})
                done.add((pid, scen, cond_name))
                new_runs += 1
                print(f" {cond_name}={latency:.1f}", end="", flush=True)

                # Checkpoint after every run
                for name, path in paths.items():
                    pd.DataFrame(frames[name]).to_csv(path, index=False)
                for name, c in conditions.items():
                    if c["policy"] == "bandit" and ("bandit", name) in shared:
                        state_dir.mkdir(parents=True, exist_ok=True)
                        (state_dir / f"{name}.json").write_text(json.dumps(shared[("bandit", name)].state_dict()))
            remaining = sum(1 for _, (p, s, o) in todo for c in o if (p, s, c) not in done)
            eta = (time.perf_counter() - t_begin) / max(new_runs, 1) * remaining
            print(f" | {remaining} runs left, ETA {eta / 3600:.1f} h", flush=True)
    finally:
        sampler.stop()
    print(f"Finished: {len(frames['runs'])} runs in {paths['runs'].name} at {datetime.now():%H:%M:%S}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prompts", default="", help="Comma-separated subset of the plan's prompts to run now.")
    parser.add_argument("--keep-awake", action="store_true")
    args = parser.parse_args()
    if args.keep_awake and sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    only = {int(p) for p in args.prompts.split(",")} if args.prompts else None
    run_plan(args.plan, resume=args.resume, only_prompts=only)
