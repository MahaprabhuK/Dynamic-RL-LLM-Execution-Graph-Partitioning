"""
Pilot Ablation Runner (P1.2):
Ablates decision interval {5, 10, 20} x patience {0, 1, 2} with emergency switch on/off,
using the spike and ramp scenarios on 3 pilot prompts.
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
import pandas as pd


def get_ablation_grid(pruned: bool = True):
    """
    Returns the ablation configurations: (interval, patience, emergency_enabled).
    If pruned: intervals {10, 20} x patience {1, 2} x emergency {True, False} (8 configs)
    plus interval 5 x patience 2 x emergency {True, False} (2 configs) = 10 configs.
    If full: {5, 10, 20} x {0, 1, 2} x {True, False} = 18 configs.
    """
    if pruned:
        intervals = [10, 20]
        patiences = [1, 2]
        emergencies = [False, True]
        grid = [(i, p, e) for i in intervals for p in patiences for e in emergencies]
        grid.extend([(5, 2, False), (5, 2, True)])
        return grid
    else:
        intervals = [5, 10, 20]
        patiences = [0, 1, 2]
        emergencies = [False, True]
        return [(i, p, e) for i in intervals for p in patiences for e in emergencies]


def run_ablation(
    num_prompts: int = 3,
    prompt_offset: int = 20,
    max_new_tokens: int = 200,
    dry_run: bool = False,
    tag: str = "v6_pilot",
    pruned: bool = True
):
    grid = get_ablation_grid(pruned=pruned)
    scenarios = {
        "spike": [(0, 0), (60, 8)],
        "ramp": [(0, 0), (60, 8), (140, 12)]
    }
    total_runs = num_prompts * len(scenarios) * len(grid)

    print("=" * 80)
    print(f"PILOT ABLATION PLAN ({tag})")
    print(f"Prompts: {num_prompts} (IDs {prompt_offset+1} to {prompt_offset+num_prompts})")
    print(f"Scenarios: {list(scenarios.keys())}")
    print(f"Grid size: {len(grid)} configurations per context")
    print(f"Total runs: {total_runs} runs of {max_new_tokens} tokens")
    print("=" * 80)

    if dry_run:
        print("\n[DRY RUN] Grid configurations:")
        for idx, (interval, patience, emergency) in enumerate(grid, 1):
            print(f"  Config {idx:02d}: interval={interval:02d}, patience={patience}, emergency_switch={emergency}")
        print(f"\nEstimated duration: {total_runs} runs x ~55s = ~{total_runs * 55 / 60:.1f} minutes ({total_runs * 55 / 3600:.2f} hours)")
        print(f"Output artifact path: data/processed/adaptive_ablation_{tag}.csv")
        print("Dry run complete. No inference executed.")
        return

    from src.data.load_prompts import load_prompts
    from src.llm.inference import generate_partitioned_response
    from src.scheduler.adaptive import ThresholdPolicy
    from src.telemetry.load_generator import TokenScheduledLoad
    from src.telemetry.monitor import BackgroundLoadSampler

    prompts = load_prompts(limit=prompt_offset + num_prompts)[prompt_offset:]
    out_csv = ROOT_DIR / "data" / "processed" / f"adaptive_ablation_{tag}.csv"
    sampler = BackgroundLoadSampler().start()
    records = []

    try:
        run_count = 0
        for p_idx, prompt in enumerate(prompts):
            for sc_name, sc_schedule in scenarios.items():
                for interval, patience, emergency in grid:
                    run_count += 1
                    print(f"[{run_count:03d}/{total_runs}] Prompt {prompt_offset+p_idx+1} {sc_name} i={interval} p={patience} em={emergency}: ", end="", flush=True)
                    policy = ThresholdPolicy(
                        threshold=60.0,
                        sampler=sampler,
                        patience=patience,
                        emergency_threshold=85.0 if emergency else None,
                        emergency_action=1
                    )
                    load = TokenScheduledLoad(sc_schedule)
                    time.sleep(1.0)
                    with load:
                        t0 = time.perf_counter()
                        _, _, token_ids = generate_partitioned_response(
                            prompt,
                            max_new_tokens=max_new_tokens,
                            return_token_ids=True,
                            scheduler=policy,
                            decision_interval=interval,
                            emergency_switch_enabled=emergency,
                            emergency_factor=3.0,
                            conservative_strategy=1,
                            step_callback=load.on_step,
                            ignore_eos=True
                        )
                        duration = time.perf_counter() - t0
                    print(f"{duration:.2f} s", flush=True)
                    records.append({
                        "prompt_id": prompt_offset + p_idx + 1,
                        "scenario": sc_name,
                        "interval": interval,
                        "patience": patience,
                        "emergency_switch": emergency,
                        "latency_seconds": round(duration, 4),
                        "num_tokens": len(token_ids)
                    })
                    pd.DataFrame(records).to_csv(out_csv, index=False)
    finally:
        sampler.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reaction-delay ablation pilot.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and estimated duration without running inference.")
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--tag", default="v6_pilot")
    parser.add_argument("--full-grid", action="store_true", help="Run full 18-config grid instead of pruned 10-config grid.")
    args = parser.parse_args()

    run_ablation(num_prompts=args.num_prompts, dry_run=args.dry_run, tag=args.tag, pruned=not args.full_grid)
