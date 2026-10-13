"""
Stage 6B/6C: bandit v7 prior and the main-run plan.

B. experiments/e4_bandit_prior.yaml: per external-CPU bin of experiments/e3_rule.yaml,
   the median s/token of every v7 arm over the load-sweep runs whose measured
   external CPU (during the run) falls into that bin. Bin i: edges[i-1] < cpu <= edges[i].
C. experiments/e4_plan.yaml: main run (Stages 7/8). Interval/patience/latency
   switch of rule_lookup come from experiments/e2_frozen.yaml; cores_heuristic and
   rl_adaptive use the same interval/patience. Values are copied by this script,
   never typed by hand.

    python experiments/build_e4_plan.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
SWEEP = "data/processed/load_sweep_v7.csv"


def bandit_prior(df, edges, arms):
    cpu_bin = np.searchsorted(edges, df["external_cpu_during"].to_numpy(), side="left")
    df = df.assign(bin=cpu_bin)
    prior, counts = {}, {}
    for b in range(len(edges) + 1):
        g = df[df["bin"] == b]
        med = g.groupby("arm")["decode_sec_per_token"].median()
        if set(med.index) != set(arms):
            continue
        prior[b] = [round(float(med[a]), 5) for a in arms]
        counts[b] = {a: int((g["arm"] == a).sum()) for a in arms}
    return prior, counts


def main():
    rule = yaml.safe_load((ROOT / "experiments" / "e3_rule.yaml").read_text(encoding="utf-8"))
    frozen = yaml.safe_load((ROOT / "experiments" / "e2_frozen.yaml").read_text(encoding="utf-8"))
    sweep_plan = yaml.safe_load((ROOT / "experiments" / "e1_plan.yaml").read_text(encoding="utf-8"))
    arms = list(sweep_plan["arms"])
    df = pd.read_csv(ROOT / SWEEP)

    prior, counts = bandit_prior(df, rule["edges"], arms)
    prior_path = ROOT / "experiments" / "e4_bandit_prior.yaml"
    plan_path = ROOT / "experiments" / "e4_plan.yaml"
    for p in (prior_path, plan_path):
        if p.exists():
            raise SystemExit(f"{p} exists; results are never overwritten")
    prior_path.write_text(yaml.safe_dump({
        "source": SWEEP, "built_by": "experiments/build_e4_plan.py",
        "bins_from": "experiments/e3_rule.yaml", "edges": rule["edges"], "arms": arms,
        "prior_weight": 2.0, "prior_sec_per_token": prior, "runs_per_bin_and_arm": counts,
        "note": "bins without every arm measured get no prior (start untried)",
    }, sort_keys=False), encoding="utf-8")

    interval, patience = int(frozen["decision_interval"]), int(frozen["patience"])
    rule_cond = {"name": "rule_lookup", "policy": "rule_lookup", "rule_file": "experiments/e3_rule.yaml",
                 "interval": interval, "patience": patience}
    if frozen.get("emergency_latency_factor") is not None:
        rule_cond["emergency_latency_factor"] = float(frozen["emergency_latency_factor"])
        rule_cond["conservative_arm"] = frozen.get("conservative_arm", "balanced")

    plan = {
        "stage": "e4_main_run", "tag": "v7",
        "frozen_from": "experiments/e2_frozen.yaml",
        "prompts": list(range(21, 36)),
        "max_new_tokens": 200, "ignore_eos": True,
        "seed": 11,  # v5 seed
        "decision_interval": 10,
        "scenarios": {  # as v5: token-triggered busy-loop workers
            "steady_idle": [[0, 0]],
            "steady_heavy": [[0, 10]],
            "spike": [[0, 0], [60, 10]],
            "relief": [[0, 10], [60, 0]],
            "ramp": [[0, 0], [60, 8], [140, 12]],
        },
        "ram_guard": {"max_percent": 85.0, "wait_seconds": 30, "retries": 10},
        "paging_flag": {"system_pages_input_per_segment": 1000},
        "conditions": [
            {"name": "threads_2", "policy": "fixed", "arm": "threads_2"},
            {"name": "balanced", "policy": "fixed", "arm": "balanced"},
            {"name": "latency_optimized", "policy": "fixed", "arm": "latency_optimized"},
            {"name": "static", "policy": "fixed", "arm": "static"},
            {"name": "rule_v5", "policy": "rule", "threshold": 60.0, "low_arm": "static", "high_arm": "balanced",
             "patience": 2, "interval": 10},
            rule_cond,
            {"name": "cores_heuristic", "policy": "cores_heuristic", "interval": interval, "patience": patience},
            {"name": "rl_adaptive", "policy": "rl", "v7_state": True, "model": "data/processed/dqn_model_v7.pth",
             "state_size": 3, "action_size": len(arms), "arms": arms, "interval": interval, "patience": patience},
            {"name": "bandit_adaptive", "policy": "bandit", "v7": True, "prior_file": "experiments/e4_bandit_prior.yaml",
             "interval": 10, "seed": 11},
        ],
        "condition_order": "randomised within each context (prompt x scenario), seeded; scenario order per prompt seeded",
        "total_runs": 15 * 5 * 9,
        "primary_analysis": {
            "test": "paired t-test on per-context latency (seconds), two-sided",
            "correction": "Holm over the 20 tests below, alpha 0.05",
            "alongside": ["Wilcoxon signed-rank p", "95% bootstrap CI of the mean paired difference"],
            "tests": [{"a": a, "b": b, "scenario": s}
                      for a, b in [("rule_lookup", "balanced"), ("rule_lookup", "threads_2"),
                                   ("rl_adaptive", "rule_lookup"), ("cores_heuristic", "rule_lookup")]
                      for s in ["steady_idle", "steady_heavy", "spike", "relief", "ramp"]],
            "exclusions": "none in the primary analysis",
        },
        "descriptive": "everything else; no significance labels",
        "sensitivity": "same tables with paging-flagged runs (paging_flag) removed, labelled as sensitivity",
        "night_1": "prompts 21-28", "night_2": "prompts 29-35",
    }
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    print(f"wrote {prior_path.name} (bins {sorted(prior)}) and {plan_path.name} "
          f"(rule_lookup interval {interval}, patience {patience}, latency switch "
          f"{frozen.get('emergency_latency_factor')})")


if __name__ == "__main__":
    main()
