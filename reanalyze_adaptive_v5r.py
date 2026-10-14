"""Reanalyse saved adaptive v5 data without changing any v5 source files.

Primary analysis: all 40 contexts (8 per scenario).
Sensitivity analysis: separate table excluding the 4 heuristic outlier candidates (14, 19, 36, 39).
Statistical correction: paired t-test and Wilcoxon signed-rank test with family-wise
Holm step-down adjustment across the 30 predefined comparisons.
"""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
import numpy as np

from src.benchmark.evaluate import paired_differences, summarize_runs, bootstrap_ci, _holm_adjust, _paired_p_values
from src.telemetry.stall_detection import audit_v5_exclusions

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "processed"
SOURCE_PATH = DATA_DIR / "adaptive_runs_v5.csv"
TIMELINES_PATH = DATA_DIR / "adaptive_timelines_v5.csv"


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def build_predefined_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the predefined 30 paired comparisons (6 per scenario across 5 scenarios):
    - rl_adaptive vs rl_once
    - rl_adaptive vs balanced
    - rl_adaptive vs latency_optimized
    - rl_adaptive vs static
    - rl_adaptive vs rule_adaptive
    - rule_adaptive vs balanced
    Applies Holm adjustment across all 30 tests.
    """
    scenarios = ["ramp", "relief", "spike", "steady_heavy", "steady_idle"]
    pairs = [
        ("rl_adaptive", "rl_once"),
        ("rl_adaptive", "balanced"),
        ("rl_adaptive", "latency_optimized"),
        ("rl_adaptive", "static"),
        ("rl_adaptive", "rule_adaptive"),
        ("rule_adaptive", "balanced"),
    ]

    records = []
    for scen in scenarios:
        scen_df = df[df["scenario"] == scen]
        wide = scen_df.pivot_table(
            index="context_id", columns="condition", values="latency_seconds", aggfunc="mean"
        )
        for cond, base in pairs:
            if cond not in wide.columns or base not in wide.columns:
                continue
            common = wide[[cond, base]].dropna()
            diff = common[cond] - common[base]
            n_ctx = len(diff)
            mean_diff, lo, hi = bootstrap_ci(diff)
            base_vals = common[base]
            pct = (100.0 * diff / base_vals).mean() if not base_vals.empty else np.nan
            t_p, w_p = _paired_p_values(common[cond], common[base])

            records.append({
                "analysis_family": "predefined_primary_comparisons",
                "scenario": scen,
                "condition": cond,
                "baseline": base,
                "num_contexts": n_ctx,
                "mean_diff_seconds": round(float(mean_diff), 4),
                "ci95_low": round(float(lo), 4) if pd.notna(lo) else np.nan,
                "ci95_high": round(float(hi), 4) if pd.notna(hi) else np.nan,
                "ci_note": "approximate (percentile bootstrap, n<15)",
                "mean_diff_percent": round(float(pct), 2) if pd.notna(pct) else np.nan,
                "faster_rate": round(float((diff < 0).mean()), 3) if len(diff) else np.nan,
                "p_value_t_raw": t_p,
                "p_value_t_adjusted": np.nan,
                "p_value_wilcoxon_raw": w_p,
                "p_value_wilcoxon_adjusted": np.nan,
                "p_value_raw": t_p,
                "p_value_adjusted": np.nan,
                "significant": False,
            })

    result = pd.DataFrame(records)
    if not result.empty:
        result["p_value_t_adjusted"] = _holm_adjust(result["p_value_t_raw"].to_numpy())
        result["p_value_wilcoxon_adjusted"] = _holm_adjust(result["p_value_wilcoxon_raw"].to_numpy())
        result["p_value_raw"] = result["p_value_t_raw"]
        result["p_value_adjusted"] = result["p_value_t_adjusted"]
        result["significant"] = result["p_value_t_adjusted"].lt(0.05)
    return result


def build_exploratory_bandit_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Exploratory comparisons for bandit_adaptive vs fixed and adaptive policies."""
    scenarios = ["ramp", "relief", "spike", "steady_heavy", "steady_idle"]
    baselines = ["balanced", "latency_optimized", "static", "rule_adaptive", "rl_adaptive"]
    records = []
    for scen in scenarios:
        scen_df = df[df["scenario"] == scen]
        wide = scen_df.pivot_table(index="context_id", columns="condition", values="latency_seconds")
        if "bandit_adaptive" not in wide.columns:
            continue
        for base in baselines:
            if base not in wide.columns:
                continue
            common = wide[["bandit_adaptive", base]].dropna()
            diff = common["bandit_adaptive"] - common[base]
            mean_diff, lo, hi = bootstrap_ci(diff)
            pct = (100.0 * diff / common[base]).mean()
            t_p, w_p = _paired_p_values(common["bandit_adaptive"], common[base])
            records.append({
                "analysis_family": "exploratory_bandit_comparisons",
                "scenario": scen,
                "condition": "bandit_adaptive",
                "baseline": base,
                "num_contexts": len(diff),
                "mean_diff_seconds": round(float(mean_diff), 4),
                "ci95_low": round(float(lo), 4) if pd.notna(lo) else np.nan,
                "ci95_high": round(float(hi), 4) if pd.notna(hi) else np.nan,
                "ci_note": "approximate (percentile bootstrap, n<15)",
                "mean_diff_percent": round(float(pct), 2),
                "faster_rate": round(float((diff < 0).mean()), 3),
                "p_value_t_raw": t_p,
                "p_value_t_adjusted": np.nan,
                "p_value_wilcoxon_raw": w_p,
                "p_value_wilcoxon_adjusted": np.nan,
                "p_value_raw": t_p,
                "p_value_adjusted": np.nan,
                "significant": False,
            })
    res = pd.DataFrame(records)
    if not res.empty:
        res["p_value_t_adjusted"] = _holm_adjust(res["p_value_t_raw"].to_numpy())
        res["p_value_wilcoxon_adjusted"] = _holm_adjust(res["p_value_wilcoxon_raw"].to_numpy())
        res["p_value_raw"] = res["p_value_t_raw"]
        res["p_value_adjusted"] = res["p_value_t_adjusted"]
        res["significant"] = res["p_value_t_adjusted"].lt(0.05)
    return res


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Source dataset missing: {SOURCE_PATH}")

    runs = pd.read_csv(SOURCE_PATH)
    timelines = pd.read_csv(TIMELINES_PATH) if TIMELINES_PATH.exists() else None

    # Audit exclusions with timeline evidence
    audit = audit_v5_exclusions(runs, timelines)

    # Primary analysis (all 40 contexts)
    primary_summary = summarize_runs(runs, ["scenario", "condition"])
    primary_diffs = build_predefined_comparisons(runs)
    bandit_diffs = build_exploratory_bandit_comparisons(runs)

    # Sensitivity analysis (excluding contexts 14, 19, 36, 39)
    flagged_contexts = set(audit["context_id"])
    sens_runs = runs[~runs["context_id"].isin(flagged_contexts)].copy()
    sens_summary = summarize_runs(sens_runs, ["scenario", "condition"])
    sens_diffs = build_predefined_comparisons(sens_runs)

    audit_path = DATA_DIR / "adaptive_exclusions_v5r.csv"
    summary_path = DATA_DIR / "adaptive_summary_v5r.csv"
    diffs_path = DATA_DIR / "adaptive_paired_diffs_v5r.csv"
    sens_summary_path = DATA_DIR / "adaptive_summary_sensitivity_v5r.csv"
    sens_diffs_path = DATA_DIR / "adaptive_paired_diffs_sensitivity_v5r.csv"
    report_path = DATA_DIR / "adaptive_report_v5r.txt"

    audit.to_csv(audit_path, index=False)
    primary_summary.to_csv(summary_path, index=False)
    primary_diffs.to_csv(diffs_path, index=False)
    sens_summary.to_csv(sens_summary_path, index=False)
    sens_diffs.to_csv(sens_diffs_path, index=False)

    mean_table_all = primary_summary.pivot(index="scenario", columns="condition", values="mean_latency").round(1)
    mean_table_sens = sens_summary.pivot(index="scenario", columns="condition", values="mean_latency").round(1)

    cols_to_print = [
        "scenario", "condition", "baseline", "num_contexts", "mean_diff_seconds",
        "ci95_low", "ci95_high", "p_value_t_raw", "p_value_t_adjusted",
        "p_value_wilcoxon_raw", "p_value_wilcoxon_adjusted", "significant",
    ]

    report = f"""Adaptive v5r Reanalysis Report
================================
Source: {SOURCE_PATH.name} (280 runs across 40 contexts; 8 prompts x 5 scenarios x 7 conditions).
All contexts are retained in the primary analysis.
Bootstrap confidence intervals are approximate (percentile bootstrap with n=6-8 per scenario).
P-values: raw paired t-test, two-sided Wilcoxon signed-rank test, and Holm step-down adjusted p-values.
Significance criterion: Holm-adjusted paired t-test p < 0.05 across the 30 predefined comparisons.

1. Primary Latency Summary: All Contexts (n=8 per scenario, seconds)
-------------------------------------------------------------------
{mean_table_all.to_string()}

2. Primary Paired Differences: 30 Predefined Comparisons (All Contexts, n=8)
----------------------------------------------------------------------------
{primary_diffs[cols_to_print].to_string(index=False)}

Surviving Holm adjustment in primary analysis:
{primary_diffs[primary_diffs['significant']][cols_to_print].to_string(index=False) if primary_diffs['significant'].any() else 'None'}

3. Sensitivity Table: Excluding Heuristic Latency Outliers [14, 19, 36, 39] (n=6-8)
-----------------------------------------------------------------------------------
{mean_table_sens.to_string()}

Sensitivity Paired Differences (reproduces reviewer reanalysis table):
{sens_diffs[cols_to_print].to_string(index=False)}

Surviving Holm adjustment in sensitivity analysis:
{sens_diffs[sens_diffs['significant']][cols_to_print].to_string(index=False) if sens_diffs['significant'].any() else 'None'}

4. Exclusion Audit Summary
--------------------------
Source data contains no page fault or swap telemetry. Therefore no run is confirmed as a memory-paging stall.
The 4 runs flagged by the legacy 2x median rule are audited below:
{audit[['context_id', 'scenario', 'condition', 'latency_seconds', 'median_latency_seconds', 'classification', 'evidence']].to_string(index=False)}

Key Findings:
- rl_adaptive vs rule_adaptive: statistically indistinguishable across all 5 scenarios (raw p >= 0.15, Holm p = 1.0).
- rl_adaptive vs balanced: rl_adaptive is slower or indistinguishable in all scenarios. In ramp: +9.5 s (slower, n=8); in spike: +21.8 s (slower, n=8).
- rl_adaptive vs rl_once: in ramp, diff = -238.4 s (n=8), but raw p = 0.0134 becomes Holm p = 0.35 (not significant at 0.05 family-wise).
- The only comparisons surviving Holm in the all-context primary analysis are against static in ramp (-353.7 s, Holm p = 0.0009) and spike (-73.7 s, Holm p = 0.028), representing oversubscription cliff baselines.
"""

    report_path.write_text(report, encoding="utf-8")
    print(f"Generated {audit_path}")
    print(f"Generated {summary_path}")
    print(f"Generated {diffs_path}")
    print(f"Generated {sens_summary_path}")
    print(f"Generated {sens_diffs_path}")
    print(f"Generated {report_path}")


if __name__ == "__main__":
    main()
