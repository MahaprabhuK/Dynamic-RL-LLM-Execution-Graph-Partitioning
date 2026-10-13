"""
Segment oracle from logged per-token timelines (offline hindsight bound, not a policy).

For every context (prompt x scenario) all fixed arms were run under the same
token-triggered load, so their per-token times can be compared segment by
segment. Per context:
    segment oracle   = sum over 10-token segments of the cheapest fixed arm's segment time
    best single fixed = the fixed arm with the lowest summed token time in that context
Per scenario (means over contexts): oracle, best single fixed, max adaptive gain %
(= 100 * (best fixed - oracle) / best fixed), and the mean summed token time of each
adaptive policy (and its regret vs the oracle).

Times are sums of per-token compute time (token_seconds), not wall-clock latency.

    python experiments/segment_oracle.py --timelines data/processed/adaptive_timelines_v5.csv \
        --fixed balanced,latency_optimized,static --policies rule_adaptive,rl_adaptive,bandit_adaptive \
        --out experiments/e0b_segment_oracle
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def segment_oracle_table(timelines, fixed, policies, segment_size=10, arm_col="condition"):
    t = timelines.copy()
    t["segment"] = t["token_step"] // segment_size
    seg = t.groupby(["context_id", "scenario", arm_col, "segment"], as_index=False)["token_seconds"].sum()

    fixed_seg = seg[seg[arm_col].isin(fixed)]
    complete = fixed_seg.groupby(["context_id", "segment"])[arm_col].nunique()
    if (complete != len(fixed)).any():
        bad = complete[complete != len(fixed)].index.tolist()[:5]
        raise ValueError(f"incomplete fixed-arm coverage for context/segment {bad}")

    oracle = fixed_seg.groupby(["context_id", "scenario", "segment"])["token_seconds"].min() \
        .groupby(["context_id", "scenario"]).sum().rename("segment_oracle")
    fixed_tot = fixed_seg.groupby(["context_id", "scenario", arm_col])["token_seconds"].sum().unstack(arm_col)
    per_ctx = pd.concat([oracle, fixed_tot.min(axis=1).rename("best_single_fixed"),
                         fixed_tot.idxmin(axis=1).rename("best_fixed_arm")], axis=1)
    totals = seg.groupby(["context_id", "scenario", arm_col])["token_seconds"].sum().unstack(arm_col)
    for p in policies:
        if p in totals:
            per_ctx[p] = totals[p]
    per_ctx = per_ctx.reset_index()

    rows = []
    for scen, g in per_ctx.groupby("scenario", sort=True):
        row = {"scenario": scen, "num_contexts": len(g),
               "mean_segment_oracle_s": g["segment_oracle"].mean(),
               "mean_best_single_fixed_s": g["best_single_fixed"].mean(),
               "max_adaptive_gain_percent": 100 * (g["best_single_fixed"].mean() - g["segment_oracle"].mean())
               / g["best_single_fixed"].mean()}
        for p in policies:
            if p in g:
                row[f"mean_{p}_s"] = g[p].mean()
                row[f"{p}_regret_vs_oracle_s"] = (g[p] - g["segment_oracle"]).mean()
        rows.append(row)
    return pd.DataFrame(rows), per_ctx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timelines", required=True)
    parser.add_argument("--fixed", required=True, help="comma-separated fixed conditions")
    parser.add_argument("--policies", default="", help="comma-separated adaptive conditions")
    parser.add_argument("--segment", type=int, default=10)
    parser.add_argument("--arm-col", default="condition")
    parser.add_argument("--out", required=True, help="output prefix; writes <out>.csv, <out>_per_context.csv, <out>.txt")
    args = parser.parse_args()

    src = Path(args.timelines)
    timelines = pd.read_csv(src if src.is_absolute() else ROOT / src)
    fixed = args.fixed.split(",")
    policies = [p for p in args.policies.split(",") if p]
    table, per_ctx = segment_oracle_table(timelines, fixed, policies, args.segment, args.arm_col)

    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    for path in [out.with_suffix(".csv"), out.with_name(out.name + "_per_context.csv"), out.with_suffix(".txt")]:
        if path.exists():
            raise SystemExit(f"{path} exists; results are never overwritten")
    table.to_csv(out.with_suffix(".csv"), index=False)
    per_ctx.to_csv(out.with_name(out.name + "_per_context.csv"), index=False)
    text = "\n".join([
        f"Segment oracle (experiments/segment_oracle.py), source: {args.timelines}",
        f"segment = {args.segment} tokens; fixed arms: {', '.join(fixed)}; times = summed per-token seconds",
        "",
        table.round(2).to_string(index=False),
        "",
    ])
    out.with_suffix(".txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
