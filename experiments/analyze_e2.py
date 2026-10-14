"""
Stage 5 analysis: reaction-delay pilot summary and the scripted freeze rule.

Reads data/processed/v7runs_pilot_v7.csv and experiments/e2_plan.yaml, writes
experiments/e2_summary.txt, experiments/e2_summary.csv and experiments/e2_frozen.yaml.

    python experiments/analyze_e2.py
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


def summarize(runs, configs):
    rows = []
    for (cond, scen), g in runs.groupby(["condition", "scenario"]):
        rows.append({"condition": cond, "scenario": scen, "n": len(g),
                     "mean_latency_s": g["latency_seconds"].mean(),
                     "switches_per_request": g["num_switches"].mean(),
                     "emergency_triggers_per_request": g["emergency_triggers"].mean()})
    table = pd.DataFrame(rows)
    spike_bal = table[(table["condition"] == "balanced") & (table["scenario"] == "spike")]["mean_latency_s"].iloc[0]
    spike = table[table["scenario"] == "spike"].set_index("condition")["mean_latency_s"]
    table["spike_gap_vs_balanced_s"] = table["condition"].map(spike - spike_bal)
    table["spike_gap_vs_balanced_percent"] = table["condition"].map(100 * (spike - spike_bal) / spike_bal)
    return table


def freeze(runs, configs, tolerance=0.02):
    cand = runs[runs["condition"].isin(configs)]
    per = cand.groupby(["condition", "scenario"])["latency_seconds"].mean().unstack("scenario")
    total = per["spike"] + per["ramp"]
    switches = cand.groupby("condition")["num_switches"].mean()
    best_total = total.min()
    close = total[total <= best_total * (1 + tolerance)].index
    order = sorted(close, key=lambda c: (switches[c], total[c]))
    winner = order[0]
    return winner, total, switches, list(close)


def main():
    plan = yaml.safe_load((ROOT / "experiments" / "e2_plan.yaml").read_text(encoding="utf-8"))
    tag = plan["tag"]
    src = f"data/processed/v7runs_{tag}.csv"
    runs = pd.read_csv(ROOT / src)
    conds = {c["name"]: c for c in plan["conditions"]}
    configs = [n for n, c in conds.items() if c["policy"] == "rule_lookup"]
    expected = plan["total_runs"]

    table = summarize(runs, configs)
    winner, total, switches, close = freeze(runs, configs)
    w = conds[winner]
    frozen = {
        "source": src, "built_by": "experiments/analyze_e2.py", "freeze_rule": plan["freeze_rule"].strip(),
        "winner": winner,
        "policy": "rule_lookup", "rule_file": w["rule_file"],
        "decision_interval": int(w["interval"]), "patience": int(w["patience"]),
        "emergency_latency_factor": w.get("emergency_latency_factor"),
        "conservative_arm": w.get("conservative_arm", "balanced"),
        "total_mean_latency_spike_plus_ramp_s": round(float(total[winner]), 3),
        "switches_per_request": round(float(switches[winner]), 3),
        "configs_within_2_percent": close,
    }
    out_yaml = ROOT / "experiments" / "e2_frozen.yaml"
    out_txt = ROOT / "experiments" / "e2_summary.txt"
    out_csv = ROOT / "experiments" / "e2_summary.csv"
    for p in (out_yaml, out_txt, out_csv):
        if p.exists():
            raise SystemExit(f"{p} exists; results are never overwritten")
    out_yaml.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    table.to_csv(out_csv, index=False)

    skipped = ROOT / "data" / "processed" / f"skipped_runs_{tag}.csv"
    n_skipped = len(pd.read_csv(skipped)) if skipped.exists() else 0
    lines = [f"Reaction-delay pilot summary (experiments/analyze_e2.py), source: {src}",
             f"runs completed: {len(runs)} of {expected}; skipped (RAM guard): {n_skipped}; "
             f"paging-flagged runs: {int(runs['paging_flag'].astype(bool).sum())}",
             f"commit hashes: {sorted(runs['commit_hash'].unique())}",
             "", "Per config and scenario:",
             table.round(3).to_string(index=False), "",
             "Freeze: total mean latency spike + ramp (s) and switches per request, rule_lookup configs:",
             pd.DataFrame({"total_s": total.round(3), "switches_per_request": switches.round(3)})
             .sort_values("total_s").to_string(), "",
             f"configs within 2% of the best total: {close}",
             f"FROZEN: {winner} -> interval {frozen['decision_interval']}, patience {frozen['patience']}, "
             f"latency switch {frozen['emergency_latency_factor']} (experiments/e2_frozen.yaml)"]
    text = "\n".join(lines) + "\n"
    out_txt.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
