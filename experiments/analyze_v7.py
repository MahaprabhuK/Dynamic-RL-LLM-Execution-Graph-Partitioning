"""
Stage 8B: analysis of the v7 main run. Reads only the v7 CSVs and experiments/e4_plan.yaml.

Writes experiments/e4_summary.txt and CSVs (experiments/e4_*.csv), plots in results/v7/:
 1. per scenario and condition: mean, median, SD, min, max latency; switches and
    emergency triggers per request                                   -> e4_latency.csv
 2. the 20 pre-registered tests: mean paired diff, 95% bootstrap CI, raw t p, Holm t p,
    Wilcoxon p, n                                                    -> e4_tests.csv
 3. segment-oracle regret per policy per scenario (experiments/segment_oracle.py,
    four fixed arms)                                                 -> e4_segment_oracle.csv
 4. sensitivity: (1) and (2) with paging-flagged runs removed        -> e4_*_sensitivity.csv
 5. skipped runs and paging events
 6. plots

    python experiments/analyze_v7.py [--plan experiments/e4_plan.yaml]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from segment_oracle import segment_oracle_table  # noqa: E402

BOOT = 10000
ALPHA = 0.05


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, (m - rank) * p[ok][i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out


def latency_table(runs):
    g = runs.groupby(["scenario", "condition"])
    t = g["latency_seconds"].agg(n="count", mean="mean", median="median", sd="std", min="min", max="max")
    t["switches_per_request"] = g["num_switches"].mean()
    t["emergency_triggers_per_request"] = g["emergency_triggers"].mean()
    return t.reset_index()


def paired_tests(runs, tests, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in tests:
        sub = runs[(runs["scenario"] == t["scenario"]) & runs["condition"].isin([t["a"], t["b"]])]
        wide = sub.pivot_table(index="prompt_id", columns="condition", values="latency_seconds").dropna()
        n = len(wide)
        row = {"a": t["a"], "b": t["b"], "scenario": t["scenario"], "n": n}
        if n >= 2 and {t["a"], t["b"]} <= set(wide.columns):
            d = (wide[t["a"]] - wide[t["b"]]).to_numpy()
            boots = rng.choice(d, size=(BOOT, n), replace=True).mean(axis=1)
            row.update({
                "mean_a_s": wide[t["a"]].mean(), "mean_b_s": wide[t["b"]].mean(),
                "mean_diff_s": d.mean(), "mean_diff_percent": 100 * d.mean() / wide[t["b"]].mean(),
                "ci95_low_s": np.percentile(boots, 2.5), "ci95_high_s": np.percentile(boots, 97.5),
                "p_t_raw": stats.ttest_rel(wide[t["a"]], wide[t["b"]]).pvalue if np.any(d != 0) else 1.0,
                "p_wilcoxon": stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0,
            })
        rows.append(row)
    df = pd.DataFrame(rows)
    df["p_t_holm"] = holm(df["p_t_raw"]) if "p_t_raw" in df else np.nan
    df["significant_holm_0.05"] = df["p_t_holm"] < ALPHA
    return df


def oracle_regret(timelines, plan):
    fixed = [c["name"] for c in plan["conditions"] if c["policy"] == "fixed"]
    policies = [c["name"] for c in plan["conditions"] if c["policy"] != "fixed"]
    t = timelines.copy()
    t["context_id"] = t["prompt_id"].astype(str) + "_" + t["scenario"]
    complete = t.groupby("context_id")["condition"].nunique()
    t = t[t["context_id"].isin(complete[complete == len(plan["conditions"])].index)]
    table, _ = segment_oracle_table(t, fixed, policies)
    return table


def plot_all(runs, timelines, plan, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    conds = [c["name"] for c in plan["conditions"]]
    scenarios = list(plan["scenarios"])
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.2 * len(scenarios), 4.8), sharey=True)
    for ax, scen in zip(np.atleast_1d(axes), scenarios):
        sub = runs[runs["scenario"] == scen]
        for i, c in enumerate(conds):
            vals = sub[sub["condition"] == c]["latency_seconds"]
            if vals.empty:
                continue
            ax.barh(i, vals.mean(), color="#9ab3d5", height=0.6)
            ax.scatter(vals, np.full(len(vals), i) + np.random.default_rng(i).uniform(-0.2, 0.2, len(vals)),
                       s=8, color="#26456e", zorder=3)
            ax.plot([vals.median()] * 2, [i - 0.32, i + 0.32], color="#c0392b", lw=2, zorder=4)
        ax.set_title(scen)
        ax.set_xlabel("latency (s), 200 tokens")
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels(conds)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("v7 main run: mean (bar), per-context (dots), median (red tick)")
    fig.tight_layout()
    fig.savefig(out_dir / "latency_by_scenario_v7.png", dpi=130)
    plt.close(fig)

    for scen in ["spike", "ramp"]:
        sub = timelines[timelines["scenario"] == scen]
        if sub.empty:
            continue
        pid = sorted(sub["prompt_id"].unique())[0]
        sub = sub[sub["prompt_id"] == pid]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for c in conds:
            line = sub[sub["condition"] == c].sort_values("token_step")
            if not line.empty:
                ax.plot(line["token_step"], line["token_seconds"], lw=1, label=c)
        for step, _ in plan["scenarios"][scen][1:]:
            ax.axvline(step, color="grey", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("token")
        ax.set_ylabel("token time (s, log)")
        ax.set_title(f"{scen}, prompt {pid}: per-token time (dashed = load change)")
        ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{scen}_timeline_prompt{pid}_v7.png", dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="experiments/e4_plan.yaml")
    args = parser.parse_args()
    plan = yaml.safe_load((ROOT / args.plan).read_text(encoding="utf-8"))
    tag = plan["tag"]
    src_runs = f"data/processed/v7runs_{tag}.csv"
    src_tl = f"data/processed/v7timelines_{tag}.csv"
    runs = pd.read_csv(ROOT / src_runs)
    timelines = pd.read_csv(ROOT / src_tl)
    tests = plan["primary_analysis"]["tests"]
    exp = ROOT / "experiments"
    outputs = {k: exp / f"e4_{k}.csv" for k in
               ["latency", "tests", "segment_oracle", "latency_sensitivity", "tests_sensitivity"]}
    summary_path = exp / "e4_summary.txt"
    for p in [*outputs.values(), summary_path]:
        if p.exists():
            raise SystemExit(f"{p} exists; results are never overwritten")

    lat = latency_table(runs)
    tst = paired_tests(runs, tests)
    orc = oracle_regret(timelines, plan)
    flagged = runs["paging_flag"].astype(bool)
    clean = runs[~flagged]
    lat_s = latency_table(clean)
    tst_s = paired_tests(clean, tests)
    for k, df in zip(outputs, [lat, tst, orc, lat_s, tst_s]):
        df.to_csv(outputs[k], index=False)
    plot_all(runs, timelines, plan, ROOT / "results" / "v7")

    skipped = ROOT / "data" / "processed" / f"skipped_runs_{tag}.csv"
    skipped_df = pd.read_csv(skipped) if skipped.exists() else pd.DataFrame()
    show = ["a", "b", "scenario", "n", "mean_diff_s", "ci95_low_s", "ci95_high_s", "mean_diff_percent",
            "p_t_raw", "p_t_holm", "p_wilcoxon", "significant_holm_0.05"]
    oracle_cols = ["scenario", "num_contexts", "mean_segment_oracle_s", "mean_best_single_fixed_s"] + \
        [c for c in orc.columns if c.endswith("_regret_vs_oracle_s")]
    mean_wide = runs.pivot_table(index="scenario", columns="condition", values="latency_seconds", aggfunc="mean")
    lines = [
        f"v7 main run analysis (experiments/analyze_v7.py), sources: {src_runs}, {src_tl}, {args.plan}",
        f"runs: {len(runs)} of {plan['total_runs']}; contexts with all conditions: "
        f"{int((runs.groupby(['prompt_id', 'scenario'])['condition'].nunique() == len(plan['conditions'])).sum())}",
        f"commit hashes: {sorted(runs['commit_hash'].unique())}",
        f"token IDs identical across conditions in "
        f"{runs.groupby(['prompt_id', 'scenario'])['token_ids'].nunique().eq(1).mean():.1%} of contexts",
        "", "1. Latency per scenario and condition (s, 200 tokens) -> experiments/e4_latency.csv",
        "Mean latency table:", mean_wide.round(2).to_string(), "",
        lat.round(3).to_string(index=False), "",
        "2. Pre-registered tests (a - b, seconds; negative = a faster), paired t, Holm over 20 -> experiments/e4_tests.csv",
        tst[show].round(4).to_string(index=False), "",
        "3. Segment-oracle regret vs oracle (summed token seconds; 4 fixed arms) -> experiments/e4_segment_oracle.csv",
        orc[oracle_cols].round(2).to_string(index=False), "",
        f"4. SENSITIVITY (paging-flagged runs removed: {int(flagged.sum())} runs) -> experiments/e4_*_sensitivity.csv",
        lat_s.round(3).to_string(index=False), "",
        tst_s[show].round(4).to_string(index=False), "",
        "5. Skipped runs and paging events",
        f"skipped runs (RAM guard): {len(skipped_df)}" + (f" -> {skipped.name}" if len(skipped_df) else ""),
        f"paging-flagged runs: {int(flagged.sum())}",
    ]
    if flagged.any():
        lines.append(runs[flagged][["prompt_id", "scenario", "condition", "max_segment_system_pages_input",
                                    "latency_seconds"]].to_string(index=False))
    if len(skipped_df):
        lines.append(skipped_df.to_string(index=False))
    lines += ["", "6. Plots: results/v7/latency_by_scenario_v7.png, results/v7/spike_timeline_*.png, "
                  "results/v7/ramp_timeline_*.png"]
    text = "\n".join(lines) + "\n"
    summary_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
