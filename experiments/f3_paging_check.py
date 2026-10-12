"""
F3 check: does the per-segment paging telemetry fire under memory pressure?

Runs one 50-token request normally, then again while experiments/f3_ram_hog.py
holds RAM until available memory is under 1 GB, and writes the per-segment
telemetry of both runs to experiments/f3_paging_check.txt.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from src.data.load_prompts import load_prompts
from src.llm.inference import generate_partitioned_response
from src.telemetry.run_guard import run_metadata

OUT = ROOT_DIR / "experiments" / "f3_paging_check.txt"
PROMPT = load_prompts(csv_path=str(ROOT_DIR / "data" / "processed" / "prompts.csv"), limit=1)[0]


def one_run(label):
    segments = []
    t0 = time.perf_counter()
    generate_partitioned_response(PROMPT, strategy="balanced", max_new_tokens=50, ignore_eos=True,
                                  segment_log=segments)
    frame = pd.DataFrame(segments)
    frame.insert(0, "run", label)
    return frame, time.perf_counter() - t0


def main():
    generate_partitioned_response(PROMPT, strategy="balanced", max_new_tokens=8)  # load + warm-up
    meta_before = run_metadata()
    normal, normal_s = one_run("normal")

    hog = subprocess.Popen([sys.executable, str(ROOT_DIR / "experiments" / "f3_ram_hog.py"),
                            "--target-mb", "1024", "--hold-seconds", "600"],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    ready = hog.stdout.readline().strip()
    meta_pressure = run_metadata()
    try:
        pressure, pressure_s = one_run("pressure")
    finally:
        hog.stdin.close()
        hog.wait(timeout=60)

    both = pd.concat([normal, pressure], ignore_index=True)
    totals = both.groupby("run", sort=False).agg(
        total_page_faults=("page_faults_delta", "sum"), max_segment_page_faults=("page_faults_delta", "max"),
        swap_sin=("swap_sin_delta", "sum"), swap_sout=("swap_sout_delta", "sum"),
        min_available_mb=("available_mb", "min"))
    totals["latency_seconds"] = [round(normal_s, 2), round(pressure_s, 2)]
    ratio = totals.loc["pressure", "total_page_faults"] / max(totals.loc["normal", "total_page_faults"], 1)

    lines = [
        "F3 paging check (written by experiments/f3_paging_check.py)",
        f"prompt id 1, arm balanced, 50 tokens, ignore_eos; segment = 10 tokens",
        f"commit {meta_before['commit_hash']}",
        f"before normal run:   RAM {meta_before['ram_percent']}%, available {meta_before['available_mb']} MB, "
        f"{meta_before['wall_clock']}",
        f"helper: {ready}",
        f"before pressure run: RAM {meta_pressure['ram_percent']}%, available {meta_pressure['available_mb']} MB, "
        f"{meta_pressure['wall_clock']}",
        "",
        "Per-segment telemetry:",
        both.to_string(index=False),
        "",
        "Per-run totals:",
        totals.to_string(),
        "",
        f"page-fault delta ratio pressure/normal: {ratio:.2f}",
        "note: psutil reports swap sin/sout as 0 on Windows; num_page_faults counts soft and hard faults.",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
