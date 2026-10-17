"""
Integrity checks on the partial v7 main run (data/processed/v7*_v7.csv) before resuming it.
Writes experiments/check_main_partial.txt. Exit code 1 if any check fails.

    python experiments/check_main_partial.py [tag]
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
TAG = sys.argv[1] if len(sys.argv) > 1 else "v7"
DATA = ROOT / "data" / "processed"
PLAN = yaml.safe_load((ROOT / "experiments" / "e4_plan.yaml").read_text(encoding="utf-8"))
CODE_PATHS = ["run_v7.py", "src/", "configs/", "experiments/e4_plan.yaml", "experiments/e2_frozen.yaml",
              "experiments/e3_rule.yaml", "experiments/e4_bandit_prior.yaml"]
ALLOWED_OTHER_CHANGES = {"tests/test_adaptive.py"}

runs = pd.read_csv(DATA / f"v7runs_{TAG}.csv")
timelines = pd.read_csv(DATA / f"v7timelines_{TAG}.csv")
skipped_path = DATA / f"skipped_runs_{TAG}.csv"
skipped = pd.read_csv(skipped_path) if skipped_path.exists() else pd.DataFrame()
fixed = [c["name"] for c in PLAN["conditions"] if c["policy"] == "fixed"]
conditions = [c["name"] for c in PLAN["conditions"]]

results = []
extra = []


def check(name, ok, detail):
    results.append((name, bool(ok), detail))


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


# commit hash and tree state
check("commit_hash filled (40 chars) on every row",
      runs["commit_hash"].notna().all() and (runs["commit_hash"].str.len() == 40).all(),
      runs["commit_hash"].value_counts().to_dict())
check("tree_dirty false on every row", (~runs["tree_dirty"].astype(bool)).all(),
      runs["tree_dirty"].value_counts().to_dict())

# code unchanged since the run commit(s)
head = git("rev-parse", "HEAD").strip()
for commit in sorted(runs["commit_hash"].dropna().unique()):
    code_diff = git("diff", "--stat", commit, "HEAD", "--", *CODE_PATHS).strip()
    check(f"git diff {commit[:10]}..HEAD on run code/configs is empty", code_diff == "", code_diff or "(empty)")
    changed = [p for p in git("diff", "--name-only", commit, "HEAD").split() if p]
    other = [p for p in changed if not p.startswith("data/processed/") and not p.startswith("experiments/")
             and p not in ALLOWED_OTHER_CHANGES]
    check(f"no other code changed since {commit[:10]} (only data/, experiments/ outputs, {sorted(ALLOWED_OTHER_CHANGES)})",
          not other, other or "none")
    extra.append(f"git diff --stat {commit[:10]} {head[:10]} (whole tree):")
    extra.append(git("diff", "--stat", commit, "HEAD").rstrip() or "(empty)")
    extra.append(f"git diff --stat {commit[:10]} {head[:10]} -- tests/:")
    extra.append(git("diff", "--stat", commit, "HEAD", "--", "tests/").rstrip()
                 or "(empty: a63992a and its revert cancel out)")
    extra.append("")

# token IDs identical across conditions within each context
tok = runs.groupby("context_id")["token_ids"].nunique()
check("token IDs identical across conditions within each context", (tok == 1).all(),
      {int(k): int(v) for k, v in tok.items()})
check("all runs generated max_new_tokens", (runs["num_generated_tokens"] == PLAN["max_new_tokens"]).all(),
      runs["num_generated_tokens"].unique().tolist())

# fixed conditions
f = runs[runs["condition"].isin(fixed)]
check("fixed conditions: 0 switches", (f["num_switches"] == 0).all(),
      f.groupby("condition")["num_switches"].sum().to_dict())
check("fixed conditions: 0 emergency triggers", (f["emergency_triggers"] == 0).all(),
      f.groupby("condition")["emergency_triggers"].sum().to_dict())
ft = timelines[timelines["condition"].isin(fixed)]
check("fixed conditions: one arm per run in timeline",
      (ft.groupby(["context_id", "condition"])["arm"].nunique() == 1).all(), "")

# no duplicate cells
dup = runs.duplicated(["prompt_id", "scenario", "condition"]).sum()
check("no duplicate (prompt, scenario, condition) rows", dup == 0, f"{dup} duplicates")

lines = [f"Partial main-run checks for tag {TAG} (experiments/check_main_partial.py; HEAD {head[:10]})",
         f"Rows in v7runs_{TAG}.csv: {len(runs)} of {PLAN['total_runs']}", ""]
for name, ok, detail in results:
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
lines.append("")
lines += extra

# paging and RAM guard
pf = runs[runs["paging_flag"].astype(bool)]
lines.append(f"Paging-flagged runs (system pages input >= {PLAN['paging_flag']['system_pages_input_per_segment']} "
             f"in a segment): {len(pf)} of {len(runs)}")
if len(pf):
    lines.append(pf[["context_id", "prompt_id", "scenario", "condition", "latency_seconds",
                     "max_segment_system_pages_input", "ram_percent_before", "min_available_mb"]].to_string(index=False))
lines.append(f"Paging-flagged by condition: {pf['condition'].value_counts().to_dict()}")
lines.append(f"RAM before runs: min {runs['ram_percent_before'].min():.1f} %, median "
             f"{runs['ram_percent_before'].median():.1f} %, max {runs['ram_percent_before'].max():.1f} %")
lines.append(f"RAM-guard skips: {len(skipped)} ({skipped_path.name} "
             f"{'exists' if skipped_path.exists() else 'does not exist: no skips were logged'})")
lines.append("")

# cells done
lines.append("Cells done (rows = (prompt, scenario) context in run order; x = done):")
ctx = runs.groupby("context_id").agg(prompt_id=("prompt_id", "first"), scenario=("scenario", "first"))
grid = pd.DataFrame("", index=ctx.index, columns=conditions)
for _, r in runs.iterrows():
    grid.loc[r["context_id"], r["condition"]] = "x"
grid.insert(0, "scenario", ctx["scenario"])
grid.insert(0, "prompt", ctx["prompt_id"])
lines.append(grid.to_string())
partial = [c for c in ctx.index if (grid.loc[c, conditions] == "x").sum() < len(conditions)]
for c in partial:
    done = [k for k in conditions if grid.loc[c, k] == "x"]
    lines.append(f"Context {c} (prompt {ctx.loc[c, 'prompt_id']}, {ctx.loc[c, 'scenario']}) partial: "
                 f"done {done}, missing {[k for k in conditions if k not in done]}")
lines.append(f"Complete contexts: {len(ctx) - len(partial)} of {len(PLAN['prompts']) * len(PLAN['scenarios'])}")

text = "\n".join(lines)
print(text)
(ROOT / "experiments" / "check_main_partial.txt").write_text(text + "\n", encoding="utf-8")
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
