"""
Resume check and time estimate for the v7 main run, from the CSVs only (no model, no runs).
Rebuilds run_v7.py's seeded context / condition order from experiments/e4_plan.yaml, checks it against
the rows on disk, lists the cells --resume will run, and estimates the remaining machine time
(median seconds per (condition, scenario) from the finished runs, plus the median gap between runs,
times the remaining cells, plus 15 %). Writes experiments/resume_plan_v7.txt.

    python experiments/resume_plan_v7.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
PLAN = yaml.safe_load((ROOT / "experiments" / "e4_plan.yaml").read_text(encoding="utf-8"))
RUNS = ROOT / "data" / "processed" / f"v7runs_{PLAN['tag']}.csv"
OUT = ROOT / "experiments" / "resume_plan_v7.txt"
MARGIN = 0.15
EXPECTED_REMAINING = 383  # 450 planned (amendment 2026-10-18) - 67 done
SESSIONS = {"22-24": [22, 23, 24], "25-27": [25, 26, 27], "28-30": [28, 29, 30]}  # run_main_v7.cmd 22,23,24 etc.


def plan_order(plan):
    """Same draws as run_v7.run_plan: prompt-major, scenarios shuffled per prompt, conditions per context."""
    rng = np.random.default_rng(int(plan["seed"]))
    names = [c["name"] for c in plan["conditions"]]
    out = []
    contexts = []
    for pid in plan["prompts"]:
        for scen in rng.permutation(list(plan["scenarios"])):
            contexts.append((pid, str(scen), [str(c) for c in rng.permutation(names)]))
    for c_idx, (pid, scen, order) in enumerate(contexts, start=1):
        for run_order, cond in enumerate(order):
            out.append({"context_id": c_idx, "prompt_id": pid, "scenario": scen, "condition": cond,
                        "run_order": run_order})
    return pd.DataFrame(out)


def main():
    runs = pd.read_csv(RUNS)
    order = plan_order(PLAN)
    lines = [f"v7 main-run resume check and time estimate (experiments/resume_plan_v7.py; source {RUNS.name})", ""]

    key = ["prompt_id", "scenario", "condition"]
    merged = runs[key + ["context_id", "run_order"]].merge(order, on=key, suffixes=("", "_plan"))
    same = (len(merged) == len(runs) and (merged["context_id"] == merged["context_id_plan"]).all()
            and (merged["run_order"] == merged["run_order_plan"]).all())
    lines.append(f"[{'PASS' if same else 'FAIL'}] rebuilt seeded order matches context_id and run_order of all "
                 f"{len(runs)} rows on disk")
    done = set(map(tuple, runs[key].values))
    remaining = order[[tuple(r) not in done for r in order[key].values]].reset_index(drop=True)
    n_ok = len(remaining) == PLAN["total_runs"] - len(runs) == EXPECTED_REMAINING
    lines.append(f"[{'PASS' if n_ok else 'FAIL'}] remaining runs: {len(remaining)} "
                 f"({PLAN['total_runs']} planned - {len(runs)} done; expected {EXPECTED_REMAINING})")
    in_sessions = remaining["prompt_id"].isin([p for ps in SESSIONS.values() for p in ps])
    n_ok = n_ok and bool(in_sessions.all())
    lines.append(f"[{'PASS' if in_sessions.all() else 'FAIL'}] sessions {', '.join(SESSIONS)} cover all remaining "
                 f"runs ({int((~in_sessions).sum())} outside)")
    first = remaining[remaining["context_id"] == remaining["context_id"].iloc[0]]
    lines.append(f"--resume starts at context {first['context_id'].iloc[0]} (prompt {first['prompt_id'].iloc[0]}, "
                 f"{first['scenario'].iloc[0]}): {first['condition'].tolist()} (in run order), then context "
                 f"{first['context_id'].iloc[0] + 1}")
    lines.append("remaining by session: " + ", ".join(
        f"prompts {name} {int(remaining['prompt_id'].isin(ps).sum())}" for name, ps in SESSIONS.items()))
    lines.append("")

    # Time estimate
    med = runs.groupby(["condition", "scenario"])["latency_seconds"].median()
    start = pd.to_datetime(runs["wall_clock_start"])
    end = pd.to_datetime(runs["wall_clock_end"])
    gaps = (start.iloc[1:].values - end.iloc[:-1].values) / np.timedelta64(1, "s")
    gap = float(np.median(gaps))
    est = remaining.apply(lambda r: med.get((r["condition"], r["scenario"]), np.nan), axis=1)
    missing = remaining[est.isna()][["condition", "scenario"]].drop_duplicates()
    if len(missing):  # cell never observed yet: fall back to that scenario's median
        est = est.fillna(remaining["scenario"].map(runs.groupby("scenario")["latency_seconds"].median()))
    remaining["est_seconds"] = est + gap
    base = float(remaining["est_seconds"].sum())
    total = base * (1 + MARGIN)
    lines.append("Median latency_seconds per (condition, scenario), from the finished runs (n in brackets):")
    tab = runs.pivot_table(index="condition", columns="scenario", values="latency_seconds", aggfunc="median")
    cnt = runs.pivot_table(index="condition", columns="scenario", values="latency_seconds", aggfunc="count")
    lines.append(tab.round(1).astype(str).add(" (").add(cnt.fillna(0).astype(int).astype(str)).add(")")
                 .to_string())
    lines.append(f"Cells with no finished run (scenario median used): "
                 f"{list(map(tuple, missing.values)) if len(missing) else 'none'}")
    lines.append(f"Median gap between runs (end -> next start, includes 1 s settle and CSV writes): {gap:.1f} s")
    lines.append(f"Estimate: {len(remaining)} runs, {base / 3600:.2f} h + {MARGIN:.0%} = {total / 3600:.2f} h")
    for name, ps in SESSIONS.items():
        sel = remaining["prompt_id"].isin(ps)
        lines.append(f"  session prompts {name} (run_main_v7.cmd {','.join(map(str, ps))}): {int(sel.sum())} runs, "
                     f"{remaining.loc[sel, 'est_seconds'].sum() * (1 + MARGIN) / 3600:.2f} h")
    clean = runs[~runs["paging_flag"].astype(bool)]
    med_c = clean.groupby(["condition", "scenario"])["latency_seconds"].median()
    est_c = remaining.apply(lambda r: med_c.get((r["condition"], r["scenario"]), np.nan), axis=1)
    est_c = est_c.fillna(remaining["scenario"].map(clean.groupby("scenario")["latency_seconds"].median()))
    total_c = float((est_c + gap).sum()) * (1 + MARGIN)
    lines.append(f"Sensitivity (paging-flagged runs left out of the medians; n=1-2 per cell, so single outliers "
                 f"such as static ramp 163 s set the median for 15 cells): {total_c / 3600:.2f} h")
    lines.append("Assumes the machine runs at the speed of 13 Oct (restored by the 17 Oct power fixes; sanity "
                 "+7 to +9 %) and no RAM-guard waits (each skip costs up to 10 x 30 s). Session times exclude the "
                 "~2 min of checks and model warm-up before each session.")
    lines.append("")
    lines.append("Remaining cells (context_id, prompt, scenario, run_order, condition):")
    lines.append(remaining[["context_id", "prompt_id", "scenario", "run_order", "condition"]].to_string(index=False))

    text = "\n".join(lines)
    OUT.write_text(text + "\n", encoding="utf-8")
    print("\n".join(lines[:30]))
    return 0 if same and n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
