"""
Stage 4: build the non-learned policies' inputs from the v7 load sweep.

A. rule_lookup: per workers level, the measured external CPU % during runs
   (mean over all runs at that level) and the arm with the lowest median
   s/token. Levels are ordered by measured CPU; consecutive levels with the
   same best arm form one bin. If that gives more than 4 bins, the adjacent
   pair whose merge adds the least regret is merged (merged bin arm = lowest
   mean of per-level medians) until 4 remain. Bin edges sit halfway between the
   highest CPU of one bin and the lowest CPU of the next. Written to
   experiments/e3_rule.yaml with the data behind each choice.

C. Offline check (descriptive): per workers level, the arm rule_lookup and
   cores_heuristic pick from the measured CPU, and the regret in s/token vs the
   best arm at that level. Written to experiments/e3_offline_policies.csv.

    python experiments/build_rule_v7.py
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SWEEP = "data/processed/load_sweep_v7.csv"
MAX_BINS = 4


def level_table(df):
    med = df.pivot_table(index="load_workers", columns="arm", values="decode_sec_per_token", aggfunc="median")
    cpu = df.groupby("load_workers")["external_cpu_during"].mean().rename("external_cpu_during")
    table = med.join(cpu).sort_values("external_cpu_during")
    arms = list(med.columns)
    table["best_arm"] = table[arms].idxmin(axis=1)
    return table, arms


def build_bins(table, arms, max_bins=MAX_BINS):
    bins = []
    for w, row in table.iterrows():
        if bins and bins[-1]["arm"] == row["best_arm"]:
            bins[-1]["levels"].append(w)
        else:
            bins.append({"arm": row["best_arm"], "levels": [w]})

    def best_for(levels):
        means = table.loc[levels, arms].mean()
        return means.idxmin(), float(means.min())

    def regret(levels, arm):
        return float((table.loc[levels, arm] - table.loc[levels, arms].min(axis=1)).sum())

    while len(bins) > max_bins:
        costs = []
        for i in range(len(bins) - 1):
            union = bins[i]["levels"] + bins[i + 1]["levels"]
            arm, _ = best_for(union)
            added = regret(union, arm) - regret(bins[i]["levels"], bins[i]["arm"]) - \
                regret(bins[i + 1]["levels"], bins[i + 1]["arm"])
            costs.append((added, i, arm))
        _, i, arm = min(costs)
        bins[i:i + 2] = [{"arm": arm, "levels": bins[i]["levels"] + bins[i + 1]["levels"]}]

    edges = []
    for lo, hi in zip(bins, bins[1:]):
        top = table.loc[lo["levels"], "external_cpu_during"].max()
        bottom = table.loc[hi["levels"], "external_cpu_during"].min()
        edges.append(round(float((top + bottom) / 2), 2))
    return bins, edges


def main():
    from src.scheduler.adaptive import CoresHeuristicPolicy, LookupRulePolicy

    df = pd.read_csv(ROOT / SWEEP)
    table, arms = level_table(df)
    bins, edges = build_bins(table, arms)

    rule = {
        "source": SWEEP,
        "built_by": "experiments/build_rule_v7.py",
        "metric": "median decode s/token per (workers level, arm) over prompts x repeats",
        "cpu_measure": "mean external CPU % during runs at each workers level (all arms)",
        "edges": edges,
        "arms": [b["arm"] for b in bins],
        "bins": [{
            "arm": b["arm"],
            "cpu_range": [None if i == 0 else edges[i - 1], None if i == len(bins) - 1 else edges[i]],
            "levels": [{
                "workers": int(w),
                "external_cpu_during": round(float(table.loc[w, "external_cpu_during"]), 2),
                "best_arm": table.loc[w, "best_arm"],
                "median_sec_per_token": {a: round(float(table.loc[w, a]), 5) for a in arms},
            } for w in b["levels"]],
        } for i, b in enumerate(bins)],
    }
    rule_path = ROOT / "experiments" / "e3_rule.yaml"
    out_csv = ROOT / "experiments" / "e3_offline_policies.csv"
    for path in (rule_path, out_csv):
        if path.exists():
            raise SystemExit(f"{path} exists; results are never overwritten")
    rule_path.write_text(yaml.safe_dump(rule, sort_keys=False), encoding="utf-8")

    lookup = LookupRulePolicy(edges, rule["arms"], patience=0)
    heuristic = CoresHeuristicPolicy(patience=0)
    rows = []
    for w, row in table.sort_index().iterrows():
        cpu = float(row["external_cpu_during"])
        best = row["best_arm"]
        rule_arm = lookup.arms[lookup.bin_of(cpu)]
        heur_arm = f"threads_{heuristic.snap(heuristic.target_threads(cpu))}"
        out = {"load_workers": int(w), "external_cpu_during": round(cpu, 2), "best_arm": best,
               "best_median_sec_per_token": round(float(row[best]), 5),
               "rule_lookup_arm": rule_arm, "rule_lookup_regret_sec_per_token": round(float(row[rule_arm] - row[best]), 5),
               "cores_heuristic_target_threads": heuristic.target_threads(cpu), "cores_heuristic_arm": heur_arm,
               "cores_heuristic_regret_sec_per_token": round(float(row[heur_arm] - row[best]), 5)
               if heur_arm in row else None}
        for a in arms:
            out[f"median_{a}"] = round(float(row[a]), 5)
        rows.append(out)
    offline = pd.DataFrame(rows)
    offline.to_csv(out_csv, index=False)

    print(f"rule_lookup (source {SWEEP}) -> experiments/e3_rule.yaml")
    for i, b in enumerate(bins):
        lo = "-inf" if i == 0 else f"{edges[i - 1]}"
        hi = "+inf" if i == len(bins) - 1 else f"{edges[i]}"
        print(f"  ({lo}, {hi}] external CPU % -> {b['arm']}  (workers levels {b['levels']})")
    print("\noffline check -> experiments/e3_offline_policies.csv")
    print(offline[["load_workers", "external_cpu_during", "best_arm", "rule_lookup_arm",
                   "rule_lookup_regret_sec_per_token", "cores_heuristic_arm",
                   "cores_heuristic_regret_sec_per_token"]].to_string(index=False))


if __name__ == "__main__":
    main()
