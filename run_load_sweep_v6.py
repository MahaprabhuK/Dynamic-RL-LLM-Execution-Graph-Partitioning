"""
Load Sweep Runner (P1.4):
Evaluates single-stage thread budgets {2, 4, 6, 8, 10, 12} across background loads
0 to 14 workers with 3 repeats at 50 tokens per request.
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Windows PyTorch DLL loading helper
torch_lib = r"C:\Users\Pavithran\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib"
if os.path.exists(torch_lib):
    try:
        os.add_dll_directory(torch_lib)
    except Exception:
        pass
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
import pandas as pd


def run_load_sweep(
    loads=(0, 2, 4, 6, 8, 10, 12, 14),
    repeats=3,
    tokens=50,
    thread_budgets=(2, 4, 6, 8, 10, 12),
    prompt_id=1,
    seed=42,
    tag="v6",
    dry_run=False
):
    total_runs = len(loads) * repeats * len(thread_budgets)

    print("=" * 80)
    print(f"LOAD SWEEP PLAN ({tag})")
    print(f"Background loads: {list(loads)} (0 to 14 workers)")
    print(f"Repeats: {repeats}")
    print(f"Thread budgets (arms): {list(thread_budgets)}")
    print(f"Prompt ID: {prompt_id} (from training set, keeping prompts 21-40 held out)")
    print(f"Tokens per run: {tokens}")
    print(f"Total inference runs: {total_runs} ({len(loads)} loads x {repeats} reps x {len(thread_budgets)} arms)")
    print("=" * 80)

    if dry_run:
        print("\n[DRY RUN] Sweep configuration details:")
        print(f"  Loads: {list(loads)}")
        print(f"  Arms: {list(thread_budgets)}")
        print(f"  Total runs: {total_runs}")
        print(f"  Estimated duration: {total_runs} runs x ~2.0s + settle = ~{total_runs * 2.2 / 60:.1f} minutes ({total_runs * 2.2 / 3600:.2f} hours)")
        print(f"  Output path: data/processed/load_sweep_{tag}.csv")
        print("Dry run complete. No models executed.")
        return

    from src.data.load_prompts import load_prompts
    from src.llm.inference import generate_partitioned_response
    from src.llm.partition import PartitionStrategy
    from src.telemetry.load_generator import BackgroundCPULoad
    from src.telemetry.monitor import BackgroundLoadSampler

    rng = np.random.default_rng(seed)
    prompt = load_prompts(csv_path=str(ROOT_DIR / "data" / "processed" / "prompts.csv"), limit=prompt_id)[prompt_id - 1]

    # Warm up model once
    generate_partitioned_response(prompt, max_new_tokens=8)

    rows = []
    out_csv = ROOT_DIR / "data" / "processed" / f"load_sweep_{tag}.csv"

    with BackgroundLoadSampler() as sampler:
        run_count = 0
        for rep in range(repeats):
            for load in rng.permutation(loads):
                with BackgroundCPULoad(int(load)):
                    time.sleep(1.0)
                    external = sampler.latest()["external_cpu_percent"]
                    for budget in rng.permutation(thread_budgets):
                        run_count += 1
                        strategy_obj = PartitionStrategy("static", thread_budget=int(budget))
                        t0 = time.perf_counter()
                        _, stages = generate_partitioned_response(
                            prompt,
                            strategy=strategy_obj,
                            max_new_tokens=tokens,
                            ignore_eos=True
                        )
                        dur = time.perf_counter() - t0
                        per_token = pd.DataFrame(stages).groupby("token_step")["stage_latency_seconds"].sum()
                        spt = float(per_token.iloc[1:].median()) if len(per_token) > 1 else float(per_token.median())
                        cpu_during = sampler.latest()["external_cpu_percent"]

                        print(f"[{run_count:03d}/{total_runs}] rep={rep} load={load:02d} budget={budget:02d}: {spt:.4f} s/tok (total {dur:.2f}s, CPU {cpu_during:.1f}%)", flush=True)

                        rows.append({
                            "repeat": rep,
                            "load_workers": int(load),
                            "thread_budget": int(budget),
                            "strategy": f"threads_{budget}",
                            "decode_sec_per_token": spt,
                            "latency_seconds": round(dur, 4),
                            "external_cpu_idle_reading": external,
                            "external_cpu_during": cpu_during,
                            "prompt_id": prompt_id
                        })
                        pd.DataFrame(rows).to_csv(out_csv, index=False)
                time.sleep(1.0)

    df = pd.DataFrame(rows)
    table = df.pivot_table(index="load_workers", columns="thread_budget", values="decode_sec_per_token", aggfunc="mean")
    table["best_budget"] = table.idxmin(axis=1)
    table["ext_cpu_before"] = df.groupby("load_workers")["external_cpu_idle_reading"].mean()
    table["ext_cpu_during"] = df.groupby("load_workers")["external_cpu_during"].mean()
    print("\nSWEEP RESULTS SUMMARY:")
    print(table.round(4).to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run P1.4 load sweep on expanded thread action set.")
    parser.add_argument("--loads", default="0,2,4,6,8,10,12,14")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-id", type=int, default=1)
    parser.add_argument("--tag", default="v6")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and estimated duration without running inference.")
    args = parser.parse_args()

    loads = [int(x) for x in args.loads.split(",")]
    run_load_sweep(
        loads=loads,
        repeats=args.repeats,
        tokens=args.tokens,
        prompt_id=args.prompt_id,
        seed=args.seed,
        tag=args.tag,
        dry_run=args.dry_run
    )
