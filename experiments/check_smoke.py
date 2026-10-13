"""
Automatic checks for the Stage 2 smoke test (reads data/processed/v7runs_smoke_v7.csv
and v7timelines/v7segments). Exit code 1 if any check fails.

    python experiments/check_smoke.py [tag]
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
TAG = sys.argv[1] if len(sys.argv) > 1 else "smoke_v7"
DATA = ROOT / "data" / "processed"
PLAN = yaml.safe_load((ROOT / "experiments" / "smoke_plan.yaml").read_text(encoding="utf-8"))

runs = pd.read_csv(DATA / f"v7runs_{TAG}.csv")
segments = pd.read_csv(DATA / f"v7segments_{TAG}.csv")
timelines = pd.read_csv(DATA / f"v7timelines_{TAG}.csv")
fixed = [c["name"] for c in PLAN["conditions"] if c["policy"] == "fixed"]
expected = [c["name"] for c in PLAN["conditions"]]

results = []


def check(name, ok, detail):
    results.append((name, bool(ok), detail))


check("all planned conditions ran", sorted(runs["condition"]) == sorted(expected),
      f"ran {sorted(runs['condition'])}")
f = runs[runs["condition"].isin(fixed)]
check("fixed conditions: 0 switches", (f["num_switches"] == 0).all(),
      dict(zip(f["condition"], f["num_switches"])))
check("fixed conditions: 0 emergency triggers", (f["emergency_triggers"] == 0).all(),
      dict(zip(f["condition"], f["emergency_triggers"])))
ft = timelines[timelines["condition"].isin(fixed)]
check("fixed conditions: one arm per run in timeline", (ft.groupby("condition")["arm"].nunique() == 1).all(),
      ft.groupby("condition")["arm"].unique().to_dict())
check("token IDs identical across all conditions", runs["token_ids"].nunique() == 1,
      f"{runs['token_ids'].nunique()} distinct token sequences")
check("all runs generated max_new_tokens", (runs["num_generated_tokens"] == PLAN["max_new_tokens"]).all(),
      runs["num_generated_tokens"].unique().tolist())
paging_run_cols = ["page_faults_delta", "system_pages_input_delta", "swap_sin_delta", "swap_sout_delta",
                   "min_available_mb", "max_segment_system_pages_input", "paging_flag"]
paging_seg_cols = ["page_faults_delta", "system_pages_input_delta", "swap_sin_delta", "swap_sout_delta",
                   "available_mb"]
check("paging columns exist (runs)", set(paging_run_cols) <= set(runs.columns),
      sorted(set(paging_run_cols) - set(runs.columns)))
check("paging columns exist (segments)", set(paging_seg_cols) <= set(segments.columns),
      sorted(set(paging_seg_cols) - set(segments.columns)))
cols_r = [c for c in paging_run_cols if c in runs]
cols_s = [c for c in paging_seg_cols if c in segments]
check("paging columns have no nulls", not runs[cols_r].isna().any().any() and not segments[cols_s].isna().any().any(),
      {**runs[cols_r].isna().sum()[lambda s: s > 0].to_dict(), **segments[cols_s].isna().sum()[lambda s: s > 0].to_dict()})
check("system pages-input counter available", (segments["system_pages_input_delta"] >= 0).all(),
      f"{int((segments['system_pages_input_delta'] < 0).sum())} segments without the counter")
check("commit hash filled", runs["commit_hash"].notna().all() and (runs["commit_hash"].str.len() == 40).all(),
      runs["commit_hash"].unique().tolist())
check("RAM and wall-clock columns filled",
      runs[["ram_percent_before", "available_mb_before", "wall_clock_start", "wall_clock_end"]].notna().all().all(), "")

lines = [f"Smoke checks for tag {TAG} (experiments/check_smoke.py; source data/processed/v7runs_{TAG}.csv)"]
for name, ok, detail in results:
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
lines.append("")
lines.append("Runs:")
lines.append(runs[["condition", "run_order", "latency_seconds", "decode_sec_per_token", "num_switches",
                   "emergency_triggers", "arms_used", "external_cpu_before", "ram_percent_before",
                   "page_faults_delta", "system_pages_input_delta", "paging_flag"]].to_string(index=False))
text = "\n".join(lines)
print(text)
(ROOT / "experiments" / f"check_smoke_{TAG}.txt").write_text(text + "\n", encoding="utf-8")
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
