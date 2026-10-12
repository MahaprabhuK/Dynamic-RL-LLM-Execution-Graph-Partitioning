import numpy as np
import pandas as pd
from scipy import stats


def _holm_adjust(p_values):
    """Holm step-down family-wise error correction for multiple comparisons."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(p_values)
    if not finite_mask.any():
        return adjusted

    finite_indices = np.flatnonzero(finite_mask)
    order = finite_indices[np.argsort(p_values[finite_mask])]
    sorted_p = p_values[order]
    sorted_adjusted = np.empty_like(sorted_p)
    running = 0.0

    for rank in range(1, sorted_p.size + 1):
        current = sorted_p[rank - 1] * (sorted_p.size - rank + 1)
        running = max(running, current)
        sorted_adjusted[rank - 1] = min(1.0, running)

    adjusted[order] = sorted_adjusted
    return adjusted


def _paired_p_values(cond: pd.Series, baseline: pd.Series):
    """Return paired t-test and Wilcoxon p-values from the same complete pairs."""
    common = pd.concat({"condition": cond, "baseline": baseline}, axis=1).dropna()
    diff = common["condition"].to_numpy(dtype=float) - common["baseline"].to_numpy(dtype=float)
    if diff.size < 2:
        return np.nan, np.nan
    if np.all(diff == 0.0):
        return 1.0, 1.0

    scale = max(1.0, float(np.max(np.abs(diff))))
    if np.ptp(diff) <= np.finfo(float).eps * scale * 8:
        t_p = 0.0
    else:
        t_p = float(stats.ttest_1samp(diff, 0.0).pvalue)

    try:
        w_p = float(stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
    except ValueError:
        w_p = 1.0 if np.allclose(diff, 0.0) else np.nan
    return t_p, w_p


def bootstrap_ci(values, n_boot: int = 10000, ci: float = 0.95, seed: int = 0):
    """Mean and percentile-bootstrap confidence interval of the mean."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        return values[0], np.nan, np.nan

    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    return values.mean(), np.quantile(means, alpha), np.quantile(means, 1 - alpha)


def summarize_runs(df: pd.DataFrame, group_cols, value_col: str = "latency_seconds"):
    """Per-group latency/throughput summary with a 95% CI on mean latency."""
    rows = []
    for keys, sub in df.groupby(list(group_cols), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        mean, lo, hi = bootstrap_ci(sub[value_col])
        row = dict(zip(group_cols, keys))
        row.update({
            "num_runs": len(sub),
            "mean_latency": round(mean, 4),
            "ci95_low": round(lo, 4),
            "ci95_high": round(hi, 4),
            "median_latency": round(sub[value_col].median(), 4),
            "std_latency": round(sub[value_col].std(), 4),
            "mean_tokens_per_sec": round(sub["tokens_per_second"].mean(), 3),
            "mean_pre_cpu_percent": round(sub["pre_cpu_percent"].mean(), 2),
            "mean_pre_ram_percent": round(sub["pre_ram_percent"].mean(), 2),
            "mean_total_handoff_latency_seconds": round(sub["total_handoff_latency_seconds"].mean(), 8),
            "mean_total_activation_handoff_mb": round(sub["total_activation_handoff_mb"].mean(), 4)
        })
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(
    df: pd.DataFrame,
    condition_col: str,
    baseline: str | list[str],
    value_col: str = "latency_seconds",
    context_col: str = "context_id",
    group_col: str | None = None,
    conditions=None,
    holm_group_cols=None
):
    """
    Paired per-context differences (condition - baseline) with bootstrap 95% CIs.
    Negative mean_diff = the condition is faster than the baseline.
    The backward-compatible p_value_* columns alias the paired t-test values.
    By default all emitted comparisons form one Holm family; holm_group_cols
    explicitly partitions that family when the analysis plan calls for it.
    """
    baselines = [baseline] if isinstance(baseline, str) else list(baseline)
    groups = [(None, df)] if group_col is None else list(df.groupby(group_col, sort=True))
    rows = []
    for g_key, g_df in groups:
        wide = g_df.pivot_table(index=context_col, columns=condition_col, values=value_col, aggfunc="mean")
        group_rows = []
        test_conditions = list(conditions) if conditions is not None else list(wide.columns)
        for base in baselines:
            if base not in wide.columns:
                continue
            for cond in test_conditions:
                if cond == base or cond not in wide.columns:
                    continue
                diff = (wide[cond] - wide[base]).dropna()
                mean, lo, hi = bootstrap_ci(diff)
                baseline_values = wide.loc[diff.index, base]
                pct = (100 * diff[baseline_values != 0] / baseline_values[baseline_values != 0]).mean()
                t_p, w_p = _paired_p_values(wide[cond], wide[base])
                row = {} if group_col is None else {group_col: g_key}
                row.update({
                    "condition": cond,
                    "baseline": base,
                    "num_contexts": len(diff),
                    "mean_diff_seconds": round(mean, 4),
                    "ci95_low": round(lo, 4),
                    "ci95_high": round(hi, 4),
                    "mean_diff_percent": round(float(pct), 2) if pd.notna(pct) else np.nan,
                    "faster_rate": round((diff < 0).mean(), 3) if len(diff) else np.nan,
                    "p_value_t_raw": t_p,
                    "p_value_t_adjusted": np.nan,
                    "p_value_wilcoxon_raw": w_p,
                    "p_value_wilcoxon_adjusted": np.nan,
                    "p_value_raw": t_p,
                    "p_value_adjusted": np.nan,
                    "significant_ci95": bool(pd.notna(lo) and (hi < 0 or lo > 0)),
                    "significant": False
                })
                group_rows.append(row)
        rows.extend(group_rows)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    family_cols = list(holm_group_cols or [])
    family_groups = [(None, result.index)] if not family_cols else result.groupby(family_cols, sort=True).groups.items()
    for _, indexes in family_groups:
        indexes = list(indexes)
        for raw_col, adjusted_col in (("p_value_t_raw", "p_value_t_adjusted"),
                                      ("p_value_wilcoxon_raw", "p_value_wilcoxon_adjusted")):
            result.loc[indexes, adjusted_col] = _holm_adjust(result.loc[indexes, raw_col].to_numpy())

    result["p_value_raw"] = result["p_value_t_raw"]
    result["p_value_adjusted"] = result["p_value_t_adjusted"]
    result["significant"] = result["p_value_t_adjusted"].lt(0.05)
    return result


def oracle_wins(df: pd.DataFrame, condition_col: str = "strategy", value_col: str = "latency_seconds",
                context_col: str = "context_id", group_col: str | None = "load_workers"):
    """How often each condition was the fastest within a context."""
    idx = df.groupby(context_col)[value_col].idxmin()
    best = df.loc[idx]
    keys = [group_col, condition_col] if group_col else [condition_col]
    return best.groupby(keys).size().rename("num_wins").reset_index()
