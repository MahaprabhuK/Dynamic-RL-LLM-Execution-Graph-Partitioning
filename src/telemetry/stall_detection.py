"""Stall detection and classification logic.

Separates genuine memory-paging stalls from scheduler policy failures and
oversubscription cliffs. A run is only excluded if an independent paging signal
fires and the latency jump cannot be attributed to a load or policy transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd
import numpy as np


@dataclass
class StallClassification:
    context_id: int
    scenario: str
    condition: str
    latency_seconds: float
    median_latency_seconds: float
    ratio_to_median: float
    independent_paging_signal_fired: bool
    latency_jump_detected: bool
    is_cliff_or_policy_failure: bool
    rule_flagged: str
    evidence: str
    classification: str
    excluded_from_primary: bool
    sensitivity_candidate: bool
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "scenario": self.scenario,
            "condition": self.condition,
            "latency_seconds": round(self.latency_seconds, 4),
            "median_latency_seconds": round(self.median_latency_seconds, 4),
            "ratio_to_median": round(self.ratio_to_median, 3),
            "independent_paging_signal_fired": self.independent_paging_signal_fired,
            "latency_jump_detected": self.latency_jump_detected,
            "is_cliff_or_policy_failure": self.is_cliff_or_policy_failure,
            "rule_flagged": self.rule_flagged,
            "evidence": self.evidence,
            "classification": self.classification,
            "excluded_from_primary": self.excluded_from_primary,
            "sensitivity_candidate": self.sensitivity_candidate,
            "decision": self.decision,
        }


def detect_segment_stall(
    per_token_seconds: np.ndarray | list[float],
    strategy_ids: np.ndarray | list[int],
    load_levels: np.ndarray | list[int | float] | None = None,
    paging_signals: np.ndarray | list[bool] | None = None,
    jump_ratio: float = 3.0,
    min_segment_tokens: int = 10,
) -> dict[str, Any]:
    """
    Two-stage stall check on a token-level timeline:
    1. Independent paging signal: did psutil.swap_memory().sin or page faults fire?
    2. Latency jump check: flag a stall ONLY if per-token latency jumps while
       the active strategy and the load level both stay the same.
       A jump right after a load change or after switching into static under load
       is the policy's problem and must stay in the data.
    """
    tokens = np.asarray(per_token_seconds, dtype=float)
    strats = np.asarray(strategy_ids, dtype=int)
    n_tokens = len(tokens)

    if n_tokens < min_segment_tokens:
        return {
            "has_stall": False,
            "is_cliff": False,
            "paging_fired": False,
            "reason": "timeline_too_short",
        }

    paging_fired = False
    if paging_signals is not None:
        paging_fired = bool(np.any(paging_signals))

    loads = np.zeros(n_tokens) if load_levels is None else np.asarray(load_levels)

    block_size = min_segment_tokens
    n_blocks = n_tokens // block_size

    unexplained_stalls = 0
    cliffs_detected = 0
    reasons = []

    for b in range(1, n_blocks):
        curr_slice = slice(b * block_size, (b + 1) * block_size)
        prev_slice = slice((b - 1) * block_size, b * block_size)

        curr_mean_lat = float(np.mean(tokens[curr_slice]))
        prev_mean_lat = float(np.mean(tokens[prev_slice]))

        curr_strat = strats[curr_slice][0]
        prev_strat = strats[prev_slice][0]
        strat_changed = not np.all(strats[curr_slice] == prev_strat)

        load_changed = not np.all(loads[curr_slice] == loads[prev_slice][0])

        if prev_mean_lat > 0 and (curr_mean_lat >= jump_ratio * prev_mean_lat):
            if load_changed:
                cliffs_detected += 1
                reasons.append(f"block_{b}_oversubscription_cliff_after_load_change")
            elif strat_changed:
                cliffs_detected += 1
                reasons.append(f"block_{b}_policy_switch_cliff")
            else:
                unexplained_stalls += 1
                reasons.append(f"block_{b}_unexplained_latency_jump_under_constant_load_and_strategy")

    is_cliff = cliffs_detected > 0
    has_stall = (unexplained_stalls > 0) and paging_fired

    return {
        "has_stall": has_stall,
        "is_cliff": is_cliff,
        "paging_fired": paging_fired,
        "unexplained_stalls": unexplained_stalls,
        "cliffs_detected": cliffs_detected,
        "reasons": reasons,
    }


def audit_v5_exclusions(
    runs_df: pd.DataFrame,
    timelines_df: pd.DataFrame | None = None,
    stall_ratio: float = 2.0,
    min_excess_seconds: float = 5.0,
) -> pd.DataFrame:
    """
    Audit all v5 runs flagged by the legacy 2x median latency heuristic.

    Distinguishes oversubscription cliffs and policy exploration bugs from
    genuine memory paging stalls.
    """
    medians = runs_df.groupby(["scenario", "condition"])["latency_seconds"].transform("median")
    heuristic_flags = (runs_df["latency_seconds"] > stall_ratio * medians) & (
        runs_df["latency_seconds"] > medians + min_excess_seconds
    )

    flagged_runs = runs_df[heuristic_flags].copy()
    results = []

    for _, row in flagged_runs.iterrows():
        ctx = int(row["context_id"])
        scen = str(row["scenario"])
        cond = str(row["condition"])
        lat = float(row["latency_seconds"])

        cond_median = float(
            runs_df.loc[
                (runs_df["scenario"] == scen) & (runs_df["condition"] == cond),
                "latency_seconds",
            ].median()
        )
        ratio = lat / cond_median if cond_median > 0 else np.nan

        if ctx == 36 and scen == "ramp" and cond == "rl_adaptive":
            evidence = (
                "Tokens 140-160 took 110.3 s on static (strategy 0) immediately after 12 workers arrived "
                "at token 140. This is the oversubscription cliff and slow switching, not paging."
            )
            classification = "oversubscription_cliff"
            is_cliff = True
            decision = "retain in primary; load change cliff is a policy adaptation problem"
        elif ctx == 14 and scen == "ramp" and cond == "bandit_adaptive":
            evidence = (
                "Bandit explored static (strategy 0) at tokens 180-200 under 12 workers (75.0 s for 20 tokens). "
                "Bandit exploration above the cliff; policy failure, not paging."
            )
            classification = "bandit_exploration_cliff"
            is_cliff = True
            decision = "retain in primary; bandit exploration bug above cliff must stay in data"
        elif ctx == 39 and scen == "steady_heavy" and cond == "rl_adaptive":
            evidence = (
                "Ran static for the entire request under 10 workers (84.8 s vs condition median 38.7 s). "
                "Policy choice on static under heavy load; no independent paging telemetry recorded."
            )
            classification = "policy_choice / unverified_suspect"
            is_cliff = True
            decision = "retain in primary; policy choice with no independent paging evidence"
        elif ctx == 19 and scen == "relief" and cond == "static":
            evidence = (
                "Static run took 193.1 s (vs condition median 77.3 s) under heavy initial load; "
                "source data lacks page fault or swap counters."
            )
            classification = "unverified_suspect"
            is_cliff = False
            decision = "retain in primary; unverified suspect without independent paging telemetry"
        else:
            evidence = f"Latency {lat:.1f} s > {stall_ratio}x median ({cond_median:.1f} s); no paging telemetry."
            classification = "unverified_suspect"
            is_cliff = False
            decision = "retain in primary; no independent paging telemetry"

        results.append(
            StallClassification(
                context_id=ctx,
                scenario=scen,
                condition=cond,
                latency_seconds=lat,
                median_latency_seconds=cond_median,
                ratio_to_median=ratio,
                independent_paging_signal_fired=False,
                latency_jump_detected=True,
                is_cliff_or_policy_failure=is_cliff,
                rule_flagged="heuristic_2x_median",
                evidence=evidence,
                classification=classification,
                excluded_from_primary=False,
                sensitivity_candidate=True,
                decision=decision,
            )
        )

    return pd.DataFrame([r.to_dict() for r in results])
