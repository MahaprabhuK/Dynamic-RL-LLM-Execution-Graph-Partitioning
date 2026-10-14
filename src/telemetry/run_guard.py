"""
Run bookkeeping shared by the v7 experiment scripts.

ram_guard (F4): before every run, wait while RAM use is above the limit
(default 85 %, 30 s apart, up to 10 retries). If RAM is still too high, the
run is skipped and written to a skipped-runs CSV; it is never skipped silently.

run_metadata: git commit hash, RAM %, available MB and wall-clock time, recorded
in every run row.
"""

import csv
import subprocess
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import psutil

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


@lru_cache(maxsize=1)
def git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True).strip()
    except Exception:
        return "unknown"


def git_is_dirty():
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT_DIR, text=True)
        return bool(out.strip())
    except Exception:
        return True


def ram_snapshot():
    vm = psutil.virtual_memory()
    return {"ram_percent": vm.percent, "available_mb": round(vm.available / 2 ** 20, 1)}


def run_metadata():
    return {"commit_hash": git_commit_hash(), **ram_snapshot(),
            "wall_clock": datetime.now().isoformat(timespec="seconds")}


def ram_guard(skipped_csv, run_info, max_percent=85.0, wait_seconds=30.0, retries=10, sleep=time.sleep,
              snapshot=ram_snapshot):
    """
    Returns (ok, snapshot). When RAM stays above `max_percent` after `retries`
    waits, appends `run_info` plus the reason to `skipped_csv` and returns False.
    """
    snap = snapshot()
    attempts = 0
    while snap["ram_percent"] > max_percent and attempts < retries:
        attempts += 1
        print(f"  RAM {snap['ram_percent']}% > {max_percent}%: waiting {wait_seconds:.0f} s "
              f"(retry {attempts}/{retries})", flush=True)
        sleep(wait_seconds)
        snap = snapshot()
    if snap["ram_percent"] <= max_percent:
        return True, snap

    row = {**run_info, **snap, "retries": attempts, "max_ram_percent": max_percent,
           "reason": f"RAM {snap['ram_percent']}% above {max_percent}% after {attempts} retries",
           "commit_hash": git_commit_hash(), "wall_clock": datetime.now().isoformat(timespec="seconds")}
    path = Path(skipped_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
    print(f"  SKIPPED: {row['reason']} -> {path}", flush=True)
    return False, snap


def summarize_segments(segment_log):
    """Per-run paging totals from the engine's per-segment log."""
    if not segment_log:
        return {"swap_sin_delta": 0, "swap_sout_delta": 0, "page_faults_delta": 0,
                "max_segment_page_faults": 0, "min_available_mb": float("nan")}
    return {
        "swap_sin_delta": sum(s["swap_sin_delta"] for s in segment_log),
        "swap_sout_delta": sum(s["swap_sout_delta"] for s in segment_log),
        "page_faults_delta": sum(s["page_faults_delta"] for s in segment_log),
        "max_segment_page_faults": max(s["page_faults_delta"] for s in segment_log),
        "min_available_mb": min(s["available_mb"] for s in segment_log),
    }
