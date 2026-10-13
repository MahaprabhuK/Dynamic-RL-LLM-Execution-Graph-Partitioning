"""
Load sweep v7 (Stage 3): fixed single-stage arms under steady busy-loop load.

Reads experiments/e1_plan.yaml (must be committed). For each prompt, repeat and
workers level (seeded order) the background workers are started once, then
every arm runs once in a seeded random order. Writes one row per run to
data/processed/load_sweep_v7.csv (checkpointed after every run; --resume skips
finished runs) and skipped runs to data/processed/skipped_runs_load_sweep_v7.csv.

    python run_load_sweep_v7.py --resume --keep-awake
    python run_load_sweep_v7.py --summary-only
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from run_v7 import plan_is_committed
from src.data.load_prompts import load_prompts
from src.telemetry.load_generator import BackgroundCPULoad
from src.telemetry.monitor import BackgroundLoadSampler
from src.telemetry.run_guard import git_commit_hash, git_is_dirty, ram_guard, run_metadata, summarize_segments

PLAN_PATH = ROOT_DIR / "experiments" / "e1_plan.yaml"
SKIPPED_NAME = "skipped_runs_load_sweep_v7.csv"


def blocks(plan):
    """(prompt, repeat, workers, arm order) blocks, reproducible from the plan seed."""
    rng = np.random.default_rng(plan["seed"])
    out = []
    for pid in plan["prompts"]:
        for rep in range(plan["repeats"]):
            for workers in rng.permutation(plan["background_workers"]):
                out.append((pid, rep, int(workers), [str(a) for a in rng.permutation(plan["arms"])]))
    return out


def run_sweep(plan_path=PLAN_PATH, resume=False, data_dir=None, allow_uncommitted=False, settle=1.0):
    from src.llm.inference import generate_partitioned_response

    plan = yaml.safe_load(Path(plan_path).read_text(encoding="utf-8"))
    if not allow_uncommitted and not plan_is_committed(plan_path):
        raise SystemExit(f"{plan_path} is not committed; commit the plan before running.")
    data_dir = Path(data_dir) if data_dir else ROOT_DIR / "data" / "processed"
    out_csv = data_dir / Path(plan["output"]).name
    skipped_csv = data_dir / SKIPPED_NAME
    if out_csv.exists() and not resume:
        raise SystemExit(f"{out_csv} exists; use --resume (results are never overwritten).")
    rows = pd.read_csv(out_csv).to_dict("records") if resume and out_csv.exists() else []
    done = {(r["prompt_id"], r["repeat"], r["load_workers"], r["arm"]) for r in rows}

    guard = plan["ram_guard"]
    flag_pages = plan["paging_flag"]["system_pages_input_per_segment"]
    tokens = int(plan["decode_tokens"]) + 1  # + prefill token
    prompts = load_prompts(csv_path=str(ROOT_DIR / "data" / "processed" / "prompts.csv"), limit=max(plan["prompts"]))
    plan_blocks = blocks(plan)
    total = sum(len(b[3]) for b in plan_blocks)
    print(f"LOAD SWEEP v7: {total} runs, {len(done)} done, commit {git_commit_hash()[:10]}"
          f"{' (dirty)' if git_is_dirty() else ''}", flush=True)
    generate_partitioned_response(prompts[plan["prompts"][0] - 1], strategy="balanced", max_new_tokens=8)

    t_begin, new = time.perf_counter(), 0
    with BackgroundLoadSampler() as sampler:
        for b_idx, (pid, rep, workers, arms) in enumerate(plan_blocks, start=1):
            if all((pid, rep, workers, a) in done for a in arms):
                continue
            prompt = prompts[pid - 1]
            with BackgroundCPULoad(workers):
                time.sleep(settle)
                for order, arm in enumerate(arms):
                    if (pid, rep, workers, arm) in done:
                        continue
                    info = {"block": b_idx, "prompt_id": pid, "repeat": rep, "load_workers": workers, "arm": arm,
                            "arm_order": order}
                    ok, _ = ram_guard(skipped_csv, info, max_percent=guard["max_percent"],
                                      wait_seconds=guard["wait_seconds"], retries=guard["retries"])
                    if not ok:
                        continue
                    time.sleep(settle)
                    before = run_metadata()
                    cpu_before = sampler.latest()["external_cpu_percent"]
                    cpu_during, segments = [], []

                    def on_step(step):
                        if step > 0 and step % 3 == 0:
                            cpu_during.append(sampler.latest()["external_cpu_percent"])

                    t0 = time.perf_counter()
                    _, stages, ids = generate_partitioned_response(
                        prompt, strategy=arm, max_new_tokens=tokens, return_token_ids=True, ignore_eos=True,
                        step_callback=on_step, segment_log=segments)
                    latency = time.perf_counter() - t0
                    after = run_metadata()
                    per_token = pd.DataFrame(stages).groupby("token_step")["stage_latency_seconds"].sum()
                    sys_pages = [s["system_pages_input_delta"] for s in segments]
                    rows.append({
                        **info,
                        "decode_tokens": tokens - 1,
                        "decode_sec_per_token": round(float(per_token.iloc[1:].median()), 6),
                        "mean_decode_sec_per_token": round(float(per_token.iloc[1:].mean()), 6),
                        "prefill_seconds": round(float(per_token.iloc[0]), 6),
                        "latency_seconds": round(latency, 4),
                        "external_cpu_before": cpu_before,
                        "external_cpu_during": round(float(np.mean(cpu_during)), 2) if cpu_during else np.nan,
                        "ram_percent_before": before["ram_percent"], "available_mb_before": before["available_mb"],
                        "ram_percent_after": after["ram_percent"], "available_mb_after": after["available_mb"],
                        **summarize_segments(segments),
                        "system_pages_input_delta": int(sum(p for p in sys_pages if p >= 0)),
                        "max_segment_system_pages_input": max(sys_pages) if sys_pages else -1,
                        "paging_flag": bool(sys_pages and max(sys_pages) >= flag_pages),
                        "commit_hash": git_commit_hash(), "tree_dirty": git_is_dirty(),
                        "wall_clock_start": before["wall_clock"], "wall_clock_end": after["wall_clock"],
                        "num_generated_tokens": len(ids),
                    })
                    done.add((pid, rep, workers, arm))
                    new += 1
                    pd.DataFrame(rows).to_csv(out_csv, index=False)
                    left = total - len(done)
                    eta = (time.perf_counter() - t_begin) / new * left / 3600
                    print(f"[{len(done):03d}/{total}] p{pid} r{rep} w{workers:02d} {arm:10s} "
                          f"{rows[-1]['decode_sec_per_token']:.4f} s/tok cpu {rows[-1]['external_cpu_during']} "
                          f"ram {before['ram_percent']}% | ETA {eta:.1f} h", flush=True)
            time.sleep(settle)
    return pd.DataFrame(rows)


def write_summary(plan_path=PLAN_PATH, data_dir=None, out_path=None):
    plan = yaml.safe_load(Path(plan_path).read_text(encoding="utf-8"))
    data_dir = Path(data_dir) if data_dir else ROOT_DIR / "data" / "processed"
    df = pd.read_csv(data_dir / Path(plan["output"]).name)
    arms = plan["arms"]
    lines = [f"Load sweep v7 summary (run_load_sweep_v7.py --summary-only), source: {plan['output']}",
             f"runs completed: {len(df)} of {plan['total_runs']}; metric: median decode s/token (prefill excluded)",
             f"commit hashes: {sorted(df['commit_hash'].unique())}", ""]

    lines.append("1. s/token per workers level and arm: mean [min, max] over prompts x repeats; best arm = lowest mean")
    stats = df.groupby(["load_workers", "arm"])["decode_sec_per_token"].agg(["mean", "min", "max", "count"])
    for w, g in stats.groupby(level=0):
        g = g.droplevel(0)
        lines.append(f"workers {w:2d}: best {g['mean'].idxmin()}")
        for a in arms:
            if a in g.index:
                lines.append(f"    {a:10s} {g.loc[a, 'mean']:.4f} [{g.loc[a, 'min']:.4f}, {g.loc[a, 'max']:.4f}] "
                             f"n={int(g.loc[a, 'count'])}")
    lines.append("")

    lines.append("2. Paired difference threads_3 vs balanced: 100 * (threads_3 - balanced) / balanced, "
                 "per (prompt, repeat)")
    wide = df.pivot_table(index=["load_workers", "prompt_id", "repeat"], columns="arm", values="decode_sec_per_token")
    if {"threads_3", "balanced"} <= set(wide.columns):
        pct = (100 * (wide["threads_3"] - wide["balanced"]) / wide["balanced"]).dropna()
        for w, g in pct.groupby(level=0):
            lines.append(f"workers {w:2d}: mean {g.mean():+.2f} %  range [{g.min():+.2f}, {g.max():+.2f}] %  n={len(g)}")
    lines.append("")

    lines.append("3. Mean external CPU % per workers level (before run / during run)")
    cpu = df.groupby("load_workers")[["external_cpu_before", "external_cpu_during"]].mean()
    for w, r in cpu.iterrows():
        lines.append(f"workers {w:2d}: before {r['external_cpu_before']:.1f} %  during {r['external_cpu_during']:.1f} %")
    lines.append("")

    lines.append("4. Skipped runs and paging events")
    skipped = data_dir / SKIPPED_NAME
    n_skipped = len(pd.read_csv(skipped)) if skipped.exists() else 0
    lines.append(f"skipped runs (RAM guard): {n_skipped}" + (f" (see {skipped.name})" if n_skipped else ""))
    flagged = df[df["paging_flag"].astype(bool)]
    lines.append(f"paging-flagged runs (max segment system pages input >= "
                 f"{plan['paging_flag']['system_pages_input_per_segment']}): {len(flagged)}")
    if len(flagged):
        lines.append(flagged[["prompt_id", "repeat", "load_workers", "arm", "max_segment_system_pages_input",
                              "decode_sec_per_token"]].to_string(index=False))
    lines.append(f"RAM % before runs: min {df['ram_percent_before'].min()}, max {df['ram_percent_before'].max()}")

    text = "\n".join(lines) + "\n"
    out_path = Path(out_path) if out_path else ROOT_DIR / plan["summary"]
    out_path.write_text(text, encoding="utf-8")
    print(text)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-awake", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.keep_awake and sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    if not args.summary_only:
        run_sweep(resume=args.resume)
    write_summary()
