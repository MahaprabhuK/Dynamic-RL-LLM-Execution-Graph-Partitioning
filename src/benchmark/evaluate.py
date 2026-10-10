import numpy as np
import pandas as pd


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
    baseline: str,
    value_col: str = "latency_seconds",
    context_col: str = "context_id",
    group_col: str | None = None
):
    """
    Paired per-context differences (condition - baseline) with bootstrap 95% CIs.
    Negative mean_diff = the condition is faster than the baseline.
    A CI that excludes 0 indicates a reliable difference.
    """
    groups = [(None, df)] if group_col is None else list(df.groupby(group_col, sort=True))
    rows = []
    for g_key, g_df in groups:
        wide = g_df.pivot_table(index=context_col, columns=condition_col, values=value_col, aggfunc="mean")
        if baseline not in wide.columns:
            continue
        for cond in wide.columns:
            if cond == baseline:
                continue
            diff = (wide[cond] - wide[baseline]).dropna()
            mean, lo, hi = bootstrap_ci(diff)
            row = {} if group_col is None else {group_col: g_key}
            row.update({
                "condition": cond,
                "baseline": baseline,
                "num_contexts": len(diff),
                "mean_diff_seconds": round(mean, 4),
                "ci95_low": round(lo, 4),
                "ci95_high": round(hi, 4),
                "mean_diff_percent": round(100 * (diff / wide.loc[diff.index, baseline]).mean(), 2),
                "faster_rate": round((diff < 0).mean(), 3),
                "significant": bool(not np.isnan(lo) and (hi < 0 or lo > 0))
            })
            rows.append(row)
    return pd.DataFrame(rows)


def oracle_wins(df: pd.DataFrame, condition_col: str = "strategy", value_col: str = "latency_seconds",
                context_col: str = "context_id", group_col: str | None = "load_workers"):
    """How often each condition was the fastest within a context."""
    idx = df.groupby(context_col)[value_col].idxmin()
    best = df.loc[idx]
    keys = [group_col, condition_col] if group_col else [condition_col]
    return best.groupby(keys).size().rename("num_wins").reset_index()
