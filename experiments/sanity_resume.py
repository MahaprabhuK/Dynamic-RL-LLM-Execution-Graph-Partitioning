"""
Pre-resume sanity check (~2 min): prompt 1, 30 decode tokens, 10 busy workers, fixed arms threads_2
and balanced, measured the same way as run_load_sweep_v7.py (median decode s/token, prefill excluded).
Passes if each arm is within 25 % of the v7 sweep at 10 workers and RAM before each run is under 85 %.
Writes experiments/sanity_resume.txt only; never touches the main-run files. Exit code 1 on failure.

    set HF_HUB_OFFLINE=1 && python experiments/sanity_resume.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_prompts import load_prompts
from src.telemetry.load_generator import BackgroundCPULoad
from src.telemetry.monitor import BackgroundLoadSampler
from src.telemetry.run_guard import git_commit_hash, run_metadata

WORKERS, TOKENS, PROMPT_ID = 10, 30, 1
REFERENCE = {"threads_2": 0.118, "balanced": 0.107}  # v7 load sweep, workers 10, mean over 3 prompts x 2 repeats
ORDER = ["threads_2", "balanced", "balanced", "threads_2"]
TOLERANCE, RAM_LIMIT = 0.25, 85.0
OUT = ROOT / "experiments" / "sanity_resume.txt"


def main():
    from src.llm.inference import generate_partitioned_response

    prompt = load_prompts(csv_path=str(ROOT / "data" / "processed" / "prompts.csv"), limit=PROMPT_ID)[PROMPT_ID - 1]
    generate_partitioned_response(prompt, strategy="balanced", max_new_tokens=8)
    rows = []
    with BackgroundLoadSampler() as sampler, BackgroundCPULoad(WORKERS):
        time.sleep(1.0)
        for order, arm in enumerate(ORDER):
            time.sleep(1.0)
            before = run_metadata()
            cpu_during = []

            def on_step(step):
                if step > 0 and step % 3 == 0:
                    cpu_during.append(sampler.latest()["external_cpu_percent"])

            t0 = time.perf_counter()
            _, stages, _ = generate_partitioned_response(prompt, strategy=arm, max_new_tokens=TOKENS + 1,
                                                         return_token_ids=True, ignore_eos=True, step_callback=on_step)
            latency = time.perf_counter() - t0
            per_token = pd.DataFrame(stages).groupby("token_step")["stage_latency_seconds"].sum()
            rows.append({"order": order, "arm": arm,
                         "decode_sec_per_token": round(float(per_token.iloc[1:].median()), 6),
                         "latency_seconds": round(latency, 3),
                         "external_cpu_during": round(float(np.mean(cpu_during)), 2) if cpu_during else np.nan,
                         "ram_percent_before": before["ram_percent"], "available_mb_before": before["available_mb"],
                         "wall_clock_start": before["wall_clock"]})
            print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    lines = [f"Pre-resume sanity check (experiments/sanity_resume.py), commit {git_commit_hash()[:10]}",
             f"prompt {PROMPT_ID}, {TOKENS} decode tokens, {WORKERS} busy workers; metric median decode s/token "
             f"(prefill excluded), as in run_load_sweep_v7.py", "", df.to_string(index=False), ""]
    ok_all = True
    for arm, ref in REFERENCE.items():
        got = float(df[df["arm"] == arm]["decode_sec_per_token"].median())
        dev = (got - ref) / ref
        ok = abs(dev) <= TOLERANCE
        ok_all &= ok
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {arm}: {got:.4f} s/tok vs sweep {ref:.3f} ({dev:+.1%}, limit "
                     f"+/-{TOLERANCE:.0%})")
    ram_ok = bool((df["ram_percent_before"] < RAM_LIMIT).all())
    ok_all &= ram_ok
    lines.append(f"[{'PASS' if ram_ok else 'FAIL'}] RAM before each run < {RAM_LIMIT:.0f} %: "
                 f"{df['ram_percent_before'].tolist()}")
    lines.append(f"OVERALL: {'PASS' if ok_all else 'FAIL'}")
    text = "\n".join(lines)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
