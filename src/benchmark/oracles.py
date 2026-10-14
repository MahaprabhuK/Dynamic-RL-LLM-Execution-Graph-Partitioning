"""Offline upper bounds that must never be used as live scheduling policies."""

import numpy as np
import pandas as pd


def segment_oracle(
    outcomes: pd.DataFrame,
    context_col: str = "context_id",
    segment_col: str = "segment_id",
    action_col: str = "action",
    value_col: str = "latency_seconds",
) -> pd.DataFrame:
    """Choose the fastest observed action per segment using complete hindsight outcomes.

    This is an offline upper bound, not a deployable policy. Every context/segment
    must contain an outcome for the same complete action set; incomplete tables are
    rejected rather than silently selecting from different candidate sets.
    """
    required = {context_col, segment_col, action_col, value_col}
    missing = sorted(required - set(outcomes.columns))
    if missing:
        raise ValueError(f"segment oracle input is missing columns: {missing}")
    if outcomes.empty:
        return pd.DataFrame(columns=[context_col, segment_col, action_col, value_col,
                                     "oracle_type", "uses_future_outcomes", "online_usable"])

    available_actions = set(outcomes[action_col].dropna().unique())
    grouped = outcomes.groupby([context_col, segment_col], sort=True, dropna=False)
    records = []
    for key, group in grouped:
        if set(group[action_col].dropna().unique()) != available_actions:
            raise ValueError(f"incomplete action outcomes for context/segment {key}")
        means = group.groupby(action_col, sort=True)[value_col].mean()
        chosen_action = means.idxmin()
        records.append({
            context_col: key[0],
            segment_col: key[1],
            action_col: chosen_action,
            value_col: means.loc[chosen_action],
            "oracle_type": "segment_hindsight_upper_bound",
            "uses_future_outcomes": True,
            "online_usable": False,
        })
    return pd.DataFrame(records)


def compute_timeline_segment_oracle(
    timelines_df: pd.DataFrame,
    segment_size: int = 10,
    fixed_conditions: tuple = ("static", "balanced", "latency_optimized"),
    context_col: str = "context_id",
    scenario_col: str = "scenario",
    condition_col: str = "condition",
    step_col: str = "token_step",
    latency_col: str = "token_seconds",
) -> pd.DataFrame:
    """
    Computes hindsight segment oracle and best-single-fixed-arm bounds from logged per-token timelines.
    
    For each (context_id, scenario), partitions tokens into segments of `segment_size` tokens.
    For each segment, picks the minimum latency among the fixed conditions.
    Summing across segments yields `segment_oracle_seconds`.
    Also computes `best_single_fixed_seconds` (the best single arm for the whole request in hindsight).
    """
    t_fixed = timelines_df[timelines_df[condition_col].isin(fixed_conditions)].copy()
    if t_fixed.empty:
        raise ValueError(f"No fixed conditions {fixed_conditions} found in timelines dataframe.")

    t_fixed["segment_idx"] = t_fixed[step_col] // segment_size
    
    # Cost per (context, scenario, segment, condition)
    seg_costs = t_fixed.groupby([context_col, scenario_col, "segment_idx", condition_col])[latency_col].sum().unstack(condition_col)
    
    # Hindsight segment oracle selects minimum arm per segment
    oracle_seg = seg_costs.min(axis=1)
    oracle_per_ctx = oracle_seg.groupby([context_col, scenario_col]).sum().rename("segment_oracle_seconds")
    
    # Best single fixed arm for the whole request
    fixed_totals = t_fixed.groupby([context_col, scenario_col, condition_col])[latency_col].sum().unstack(condition_col)
    best_fixed = fixed_totals.min(axis=1).rename("best_single_fixed_seconds")
    best_fixed_arm = fixed_totals.idxmin(axis=1).rename("best_single_fixed_arm")
    
    result = pd.concat([oracle_per_ctx, best_fixed, best_fixed_arm], axis=1).reset_index()
    result["oracle_gap_seconds"] = result["best_single_fixed_seconds"] - result["segment_oracle_seconds"]
    return result


def compute_policy_regrets(
    runs_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    context_col: str = "context_id",
    scenario_col: str = "scenario",
    condition_col: str = "condition",
    latency_col: str = "latency_seconds",
) -> pd.DataFrame:
    """
    Computes regret for each policy against:
    1. segment_oracle (dynamic switching upper bound in hindsight)
    2. best_single_fixed (best single fixed arm in hindsight)
    """
    merged = pd.merge(
        runs_df,
        oracle_df,
        on=[context_col, scenario_col],
        how="inner"
    )
    merged["regret_vs_segment_oracle"] = merged[latency_col] - merged["segment_oracle_seconds"]
    merged["regret_vs_best_fixed"] = merged[latency_col] - merged["best_single_fixed_seconds"]
    
    summary = merged.groupby([scenario_col, condition_col]).agg(
        num_runs=(latency_col, "count"),
        mean_latency=(latency_col, "mean"),
        mean_regret_vs_oracle=("regret_vs_segment_oracle", "mean"),
        median_regret_vs_oracle=("regret_vs_segment_oracle", "median"),
        mean_regret_vs_best_fixed=("regret_vs_best_fixed", "mean"),
        median_regret_vs_best_fixed=("regret_vs_best_fixed", "median"),
    ).reset_index()
    return summary
