"""
Collects the v3 fixed-strategy dataset used to train and evaluate the scheduler.

Protocol (designed to remove the run-order confound of v1/v2):
- A "context" is one request: prompt x repeat x background-load level.
- Contexts are visited in random order.
- Inside each context, the pre-decision state is measured once, then all three
  strategies run back-to-back in a freshly shuffled order, so their outcomes
  are paired and comparable.
- Background load (busy worker processes) creates genuinely different system
  states, so the best strategy can depend on the context.
- Warm-up runs precede measurement; throughput uses real generated tokens.

Usage:
    python evaluate_fixed_strategies.py --num-prompts 20 --repeats 2 --loads 0,4,8
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.benchmark.evaluate import oracle_wins, paired_differences, summarize_runs
from src.benchmark.plots import plot_latency_by_load, plot_paired_differences
from src.data.load_prompts import load_prompts
from src.llm.partition import PartitionStrategy, apply_partition_strategy
from src.telemetry.load_generator import BackgroundCPULoad
from src.telemetry.monitor import collect_metrics


def _load_inference():
    """Import the LLM inference layer only when a real generation call is needed."""
    from src.llm.inference import generate_partitioned_response, get_tokenizer
    return generate_partitioned_response, get_tokenizer

STRATEGIES = [0, 1, 2]  # static, balanced, latency_optimized


def execute_strategy(prompt: str, strategy: PartitionStrategy, max_new_tokens: int):
    """Run one request and return its measured outcome."""
    generate_partitioned_response, _ = _load_inference()
    start_time = time.perf_counter()
    response, stage_records, token_ids = generate_partitioned_response(
        prompt=prompt,
        strategy=strategy,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_token_ids=True
    )
    latency = time.perf_counter() - start_time
    post_metrics = collect_metrics()

    return {
        "response": response,
        "token_ids": token_ids,
        "stage_records": stage_records,
        "latency": latency,
        "post_metrics": post_metrics
    }


def warm_up(prompt: str):
    print("Warm-up runs...", flush=True)
    generate_partitioned_response, _ = _load_inference()
    for strat_id in STRATEGIES:
        generate_partitioned_response(prompt, strategy=strat_id, max_new_tokens=8)


def run_fixed_strategy_evaluation(
    num_prompts: int = 20,
    repeats: int = 2,
    loads=(0, 4, 8),
    max_new_tokens: int = 100,
    prompt_offset: int = 0,
    seed: int = 42,
    tag: str = "v3"
):
    data_dir = ROOT_DIR / "data" / "processed"
    results_dir = ROOT_DIR / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    all_prompts = load_prompts(csv_path=str(data_dir / "prompts.csv"), limit=prompt_offset + num_prompts)
    prompts = all_prompts[prompt_offset:]
    _, get_tokenizer = _load_inference()
    tokenizer = get_tokenizer()

    contexts = [
        (prompt_offset + p_idx + 1, prompt, rep, load)
        for p_idx, prompt in enumerate(prompts)
        for rep in range(repeats)
        for load in loads
    ]
    rng.shuffle(contexts)

    print("=" * 80, flush=True)
    print(f"V3 FIXED-STRATEGY COLLECTION: {len(prompts)} prompts x {repeats} repeats x loads {list(loads)}"
          f" = {len(contexts)} contexts x {len(STRATEGIES)} strategies", flush=True)
    for s in STRATEGIES:
        strat = PartitionStrategy(s)
        print(f"  [{s}] {strat.strategy_name:18s} layers={strat.get_layer_ranges()} threads={strat.get_thread_counts()}")
    print("=" * 80, flush=True)

    warm_up(prompts[0])

    metrics_path = data_dir / f"fixed_strategy_metrics_{tag}.csv"
    stages_path = data_dir / f"fixed_strategy_stage_metrics_{tag}.csv"
    raw_results = []
    stage_results = []
    t_begin = time.perf_counter()

    for c_idx, (prompt_id, prompt, rep, load) in enumerate(contexts, start=1):
        context_id = c_idx
        input_tokens = len(tokenizer.encode(prompt))
        order = rng.permutation(STRATEGIES)

        print(f"[{c_idx:03d}/{len(contexts)}] prompt {prompt_id:3d} rep {rep} load {load}:", end="", flush=True)

        try:
            with BackgroundCPULoad(load):
                pre = collect_metrics(interval=0.5)
                outcomes = {}
                for run_order, strat_id in enumerate(order):
                    outcomes[int(strat_id)] = (run_order, execute_strategy(prompt, apply_partition_strategy(int(strat_id)), max_new_tokens))
        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            continue

        static_ids = outcomes[0][1]["token_ids"]
        for strat_id in STRATEGIES:
            run_order, out = outcomes[strat_id]
            strat = apply_partition_strategy(strat_id)
            n_tokens = len(out["token_ids"])
            post = out["post_metrics"]

            raw_results.append({
                "context_id": context_id,
                "prompt_id": prompt_id,
                "repeat": rep,
                "load_workers": load,
                "run_order": run_order,
                "strategy": strat.strategy_name,
                "strategy_id": strat_id,
                "stage_threads": "/".join(map(str, strat.get_thread_counts())),
                "prompt_length": len(prompt),
                "input_tokens": input_tokens,
                "pre_cpu_percent": pre["cpu_percent"],
                "pre_ram_percent": pre["ram_percent"],
                "pre_ram_available_gb": pre["ram_available_gb"],
                "response_length": len(out["response"]),
                "num_generated_tokens": n_tokens,
                "latency_seconds": round(out["latency"], 4),
                "tokens_per_second": round(n_tokens / out["latency"], 3) if out["latency"] > 0 else 0.0,
                "cpu_percent": post["cpu_percent"],
                "ram_percent": post["ram_percent"],
                "ram_used_gb": post["ram_used_gb"],
                "total_activation_handoff_mb": round(sum(s["activation_size_mb"] for s in out["stage_records"]), 4),
                "total_handoff_latency_seconds": round(sum(s["handoff_latency_seconds"] for s in out["stage_records"]), 8),
                "output_matches_static": out["token_ids"] == static_ids
            })

            for stg in out["stage_records"]:
                stage_results.append({
                    "context_id": context_id,
                    "prompt_id": prompt_id,
                    "strategy": strat.strategy_name,
                    "partition_id": stg["partition_id"],
                    "layer_start": stg["layer_start"],
                    "layer_end": stg["layer_end"],
                    "num_threads": stg["num_threads"],
                    "stage_latency_seconds": stg["stage_latency_seconds"],
                    "activation_size_bytes": stg["activation_size_bytes"],
                    "activation_size_mb": stg["activation_size_mb"],
                    "handoff_latency_seconds": stg["handoff_latency_seconds"]
                })

        lat = {apply_partition_strategy(s).strategy_name[:6]: round(outcomes[s][1]["latency"], 2) for s in STRATEGIES}
        elapsed = time.perf_counter() - t_begin
        eta = elapsed / c_idx * (len(contexts) - c_idx)
        print(f" pre-cpu {pre['cpu_percent']:5.1f}% | {lat} | ETA {eta / 60:.0f} min", flush=True)

        # Save incrementally so long runs can be interrupted without losing data
        pd.DataFrame(raw_results).to_csv(metrics_path, index=False)

    df_metrics = pd.DataFrame(raw_results)
    df_metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(stage_results).to_csv(stages_path, index=False)
    print(f"\nSaved raw metrics ({len(df_metrics)} rows) to: {metrics_path}", flush=True)
    print(f"Saved stage metrics ({len(stage_results)} rows) to: {stages_path}", flush=True)

    report_fixed_strategy_results(df_metrics, data_dir, results_dir, tag)
    return df_metrics


def report_fixed_strategy_results(df_metrics: pd.DataFrame, data_dir: Path, results_dir: Path, tag: str):
    summary = summarize_runs(df_metrics, ["load_workers", "strategy"])
    summary_path = data_dir / f"fixed_strategy_summary_{tag}.csv"
    summary.to_csv(summary_path, index=False)

    diffs = paired_differences(df_metrics, "strategy", baseline="static", group_col="load_workers")
    diffs_path = data_dir / f"fixed_strategy_paired_diffs_{tag}.csv"
    diffs.to_csv(diffs_path, index=False)

    wins = oracle_wins(df_metrics)

    print("\n" + "=" * 80, flush=True)
    print("SUMMARY BY LOAD LEVEL AND STRATEGY (95% bootstrap CI on mean latency)", flush=True)
    print("=" * 80, flush=True)
    print(summary[["load_workers", "strategy", "num_runs", "mean_latency", "ci95_low", "ci95_high",
                   "mean_tokens_per_sec", "mean_pre_cpu_percent"]].to_string(index=False), flush=True)
    print("\nPAIRED DIFFERENCES VS STATIC (negative = faster)", flush=True)
    print(diffs.to_string(index=False), flush=True)
    print("\nFASTEST STRATEGY COUNTS PER LOAD LEVEL", flush=True)
    print(wins.to_string(index=False), flush=True)
    print(f"\nOutputs identical to static: {df_metrics['output_matches_static'].mean():.1%} of runs", flush=True)
    print(f"Saved: {summary_path}\nSaved: {diffs_path}", flush=True)

    plot_latency_by_load(summary, "strategy", results_dir / f"fixed_strategy_latency_by_load_{tag}.png",
                         "Mean latency by strategy and background load")
    if not diffs.empty:
        plot_paired_differences(diffs, results_dir / f"fixed_strategy_paired_diffs_{tag}.png",
                                "Paired latency difference vs static")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--loads", default="0,4,8", help="Comma-separated numbers of busy background workers.")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="v3")
    args = parser.parse_args()

    run_fixed_strategy_evaluation(
        num_prompts=args.num_prompts,
        repeats=args.repeats,
        loads=[int(x) for x in args.loads.split(",")],
        max_new_tokens=args.max_new_tokens,
        prompt_offset=args.prompt_offset,
        seed=args.seed,
        tag=args.tag
    )
