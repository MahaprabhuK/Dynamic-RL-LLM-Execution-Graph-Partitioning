"""
Descriptive supplement to experiments/analyze_v7.py for the v7 main run (no significance labels).
Reads the v7 CSVs, experiments/e4_plan.yaml, experiments/e4_segment_oracle.csv and git.

Writes experiments/e4_supplement.txt and CSVs:
 a. per scenario: rule_lookup vs the best of the four fixed arms, chosen per scenario from the
    same data (lowest mean latency)                                  -> e4_best_fixed.csv
 b. segment-oracle regret per policy per scenario (from e4_segment_oracle.csv, written by
    analyze_v7.py)
 c. 13 Oct rows vs 18-19 Oct rows: mean decode s/token of each fixed arm in steady_idle and
    steady_heavy, and mean external CPU during runs, per date      -> e4_by_date.csv
 d. commit_hash per row, files changed in each run commit, and git diff of run code, configs
    and plan files between each run commit and HEAD

    python experiments/analyze_main_v7.py [--plan experiments/e4_plan.yaml]
"""

import argparse
import hashlib
import subprocess
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
FIXED = ["threads_2", "balanced", "latency_optimized", "static"]
DATE_SCENARIOS = ["steady_idle", "steady_heavy"]
CODE_PATHS = ["run_v7.py", "src/", "configs/", "experiments/e4_plan.yaml", "experiments/e2_frozen.yaml",
              "experiments/e3_rule.yaml", "experiments/e4_bandit_prior.yaml"]
UNTRACKED_INPUTS = ["data/processed/dqn_model_v7.pth"]  # gitignored; hash shown, not verifiable against run time


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def best_fixed_table(runs, scenarios):
    mean = runs.pivot_table(index="scenario", columns="condition", values="latency_seconds", aggfunc="mean")
    rows = []
    for scen in scenarios:
        best = mean.loc[scen, FIXED].idxmin()
        sub = runs[(runs["scenario"] == scen) & runs["condition"].isin(["rule_lookup", best])]
        wide = sub.pivot_table(index="prompt_id", columns="condition", values="latency_seconds").dropna()
        d = wide["rule_lookup"] - wide[best]
        rows.append({"scenario": scen, "best_fixed": best, "n": len(wide),
                     "mean_rule_lookup_s": wide["rule_lookup"].mean(), "mean_best_fixed_s": wide[best].mean(),
                     "mean_diff_s": d.mean(), "mean_diff_percent": 100 * d.mean() / wide[best].mean(),
                     "contexts_rule_lookup_faster": int((d < 0).sum())})
    return pd.DataFrame(rows)


def by_date_table(runs):
    day = runs["wall_clock_start"].str[:10]
    runs = runs.assign(session=day.map(lambda s: "13 Oct" if s == "2026-10-13" else "18-19 Oct"))
    sub = runs[runs["condition"].isin(FIXED) & runs["scenario"].isin(DATE_SCENARIOS)]
    t = sub.groupby(["scenario", "condition", "session"]).agg(
        n=("decode_sec_per_token", "count"), mean_decode_sec_per_token=("decode_sec_per_token", "mean"),
        mean_external_cpu_during=("external_cpu_during", "mean")).reset_index()
    cpu = runs.groupby("session").agg(n=("external_cpu_during", "count"),
                                      mean_external_cpu_during=("external_cpu_during", "mean"))
    cpu_scen = runs.groupby(["scenario", "session"])["external_cpu_during"].agg(["count", "mean"]).unstack()
    return t, cpu, cpu_scen, runs


def provenance_lines(runs):
    lines = []
    head = git("rev-parse", "HEAD").strip()
    counts = runs["commit_hash"].value_counts()
    lines.append(f"commit_hash values over {len(runs)} rows: {counts.to_dict()}")
    lines.append(f"all rows share one commit_hash: {'yes' if len(counts) == 1 else 'NO'}")
    for commit, n in counts.items():
        sub = runs[runs["commit_hash"] == commit]
        lines.append("")
        lines.append(f"commit {commit[:10]}: {n} rows, wall clock {sub['wall_clock_start'].min()} .. "
                     f"{sub['wall_clock_start'].max()}, tree_dirty {sub['tree_dirty'].astype(bool).value_counts().to_dict()}")
        lines.append(f"  files changed in that commit (git show --name-status {commit[:10]}):")
        lines += ["    " + l for l in git("show", "--name-status", "--format=", commit).strip().splitlines()]
        diff = git("diff", "--stat", commit, head, "--", *CODE_PATHS).strip()
        lines.append(f"  git diff --stat {commit[:10]}..{head[:10]} -- {' '.join(CODE_PATHS)}:")
        lines += ["    " + l for l in (diff.splitlines() or ["(empty: unchanged)"])]
    lines.append("")
    lines.append("git status at run start is not recorded. tree_dirty is `git status --porcelain --untracked-files=no`")
    lines.append("at write time; it also counts the tracked output CSVs the run itself writes.")
    for p in UNTRACKED_INPUTS:
        f = ROOT / p
        if f.exists():
            lines.append(f"{p} (gitignored): sha256 {hashlib.sha256(f.read_bytes()).hexdigest()[:16]}..., "
                         f"modified {pd.Timestamp(f.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')} UTC")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="experiments/e4_plan.yaml")
    args = parser.parse_args()
    plan = yaml.safe_load((ROOT / args.plan).read_text(encoding="utf-8"))
    tag = plan["tag"]
    runs = pd.read_csv(ROOT / f"data/processed/v7runs_{tag}.csv")
    exp = ROOT / "experiments"
    out_best, out_date, summary_path = exp / "e4_best_fixed.csv", exp / "e4_by_date.csv", exp / "e4_supplement.txt"
    for p in [out_best, out_date, summary_path]:
        if p.exists():
            raise SystemExit(f"{p} exists; results are never overwritten")

    scenarios = list(plan["scenarios"])
    best = best_fixed_table(runs, scenarios)
    oracle = pd.read_csv(exp / "e4_segment_oracle.csv")
    by_date, cpu, cpu_scen, _ = by_date_table(runs)
    best.to_csv(out_best, index=False)
    by_date.to_csv(out_date, index=False)

    regret_cols = [c for c in oracle.columns if c.endswith("_regret_vs_oracle_s")]
    regret = oracle[["scenario", "num_contexts", "mean_segment_oracle_s", "mean_best_single_fixed_s"] + regret_cols]
    regret.columns = [c.replace("_regret_vs_oracle_s", "") for c in regret.columns]
    date_wide = by_date.pivot_table(index=["scenario", "condition"], columns="session",
                                    values=["n", "mean_decode_sec_per_token"])
    lines = [
        f"v7 main run, descriptive supplement (experiments/analyze_main_v7.py); no significance labels",
        f"source: data/processed/v7runs_{tag}.csv ({len(runs)} rows), {args.plan}", "",
        "a. rule_lookup vs best fixed arm per scenario (best = lowest mean latency among "
        f"{', '.join(FIXED)}, chosen from the same data; diff = rule_lookup - best, negative = rule_lookup faster)"
        " -> experiments/e4_best_fixed.csv",
        best.round(3).to_string(index=False), "",
        "b. Segment-oracle regret per policy per scenario (s, summed token seconds vs per-segment oracle over the"
        " four fixed arms; from experiments/e4_segment_oracle.csv, also section 3 of e4_summary.txt)",
        regret.round(2).to_string(index=False), "",
        "c. By session: fixed arms in steady_idle and steady_heavy, mean decode s/token (prefill excluded)"
        " -> experiments/e4_by_date.csv",
        date_wide.round(4).to_string(), "",
        "Mean external CPU during runs (%), all rows, by session:", cpu.round(2).to_string(), "",
        "Mean external CPU during runs (%), by scenario and session (count, mean):",
        cpu_scen.round(2).to_string(), "",
        "13 Oct rows are prompts 21-22 only (contexts 1-8); 18-19 Oct rows are prompts 22-30.",
        "Session differences therefore mix machine state with prompt differences.", "",
        "d. Provenance",
    ] + provenance_lines(runs)
    text = "\n".join(lines) + "\n"
    summary_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
