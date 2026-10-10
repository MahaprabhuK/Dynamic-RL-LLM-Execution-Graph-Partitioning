"""
Evaluation script to establish the v2 operational fixed-strategy baseline.
Evaluates 20 prompts under all 3 partition strategies using the layer-sliced execution engine:
1. static (strategy_id=0)
2. balanced (strategy_id=1)
3. latency_optimized (strategy_id=2)

Measures physical activation tensor handoffs and stage execution times on CPU.
"""

import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for plot generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.load_prompts import load_prompts
from src.llm.inference import generate_partitioned_response
from src.llm.partition import apply_partition_strategy
from src.graph.execution_graph import build_execution_graph
from src.telemetry.monitor import collect_metrics


def run_fixed_strategy_v2_evaluation(num_prompts: int = 20):
    data_dir = ROOT_DIR / "data" / "processed"
    results_dir = ROOT_DIR / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print(f"STARTING V2 LAYER-SLICED BASELINE EVALUATION ({num_prompts} PROMPTS x 3 STRATEGIES)", flush=True)
    print("=" * 80, flush=True)

    prompts = load_prompts(csv_path=str(data_dir / "prompts.csv"), limit=num_prompts)
    print(f"Loaded {len(prompts)} prompts.", flush=True)

    strategies = [0, 1, 2]  # static, balanced, latency_optimized
    raw_results = []
    stage_results = []

    for strat_id in strategies:
        strat_obj = apply_partition_strategy(strat_id)
        strat_name = strat_obj.strategy_name
        print(f"\n--- Running Strategy [{strat_id}]: {strat_name} ---", flush=True)

        for idx, prompt in enumerate(prompts, start=1):
            print(f"  [Prompt {idx:02d}/{num_prompts:02d}] Executing layer-sliced '{strat_name}'...", end="", flush=True)

            try:
                # Construct execution graph metadata
                graph = build_execution_graph(prompt_id=idx, partition_strategy=strat_obj)

                start_time = time.perf_counter()
                response, stage_records = generate_partitioned_response(
                    prompt=prompt,
                    strategy=strat_obj,
                    max_new_tokens=100,
                    do_sample=False
                )
                latency = time.perf_counter() - start_time

                metrics = collect_metrics()

                response_words = len(response.split())
                tokens_per_sec = round(response_words / latency, 2) if latency > 0 else 0.0

                total_handoff_mb = sum(stage["activation_size_mb"] for stage in stage_records)
                total_handoff_latency = sum(stage["handoff_latency_seconds"] for stage in stage_records)

                record = {
                    "prompt_id": idx,
                    "strategy": strat_name,
                    "strategy_id": strat_id,
                    "prompt_length": len(prompt),
                    "response_length": len(response),
                    "latency_seconds": round(latency, 4),
                    "tokens_per_second": tokens_per_sec,
                    "cpu_percent": metrics["cpu_percent"],
                    "ram_percent": metrics["ram_percent"],
                    "ram_used_gb": metrics["ram_used_gb"],
                    "total_activation_handoff_mb": round(total_handoff_mb, 4),
                    "total_handoff_latency_seconds": round(total_handoff_latency, 8)
                }
                raw_results.append(record)

                for stg in stage_records:
                    stg_row = {
                        "prompt_id": idx,
                        "strategy": strat_name,
                        "partition_id": stg["partition_id"],
                        "layer_start": stg["layer_start"],
                        "layer_end": stg["layer_end"],
                        "stage_latency_seconds": stg["stage_latency_seconds"],
                        "activation_size_bytes": stg["activation_size_bytes"],
                        "activation_size_mb": stg["activation_size_mb"],
                        "handoff_latency_seconds": stg["handoff_latency_seconds"]
                    }
                    stage_results.append(stg_row)

                print(f" Done ({latency:.2f}s, {tokens_per_sec} tok/s, {total_handoff_mb:.2f} MB handoffs)", flush=True)

            except Exception as e:
                print(f" ERROR: Prompt {idx} failed on strategy {strat_name}: {e}", flush=True)

    df_metrics = pd.DataFrame(raw_results)
    metrics_path = data_dir / "fixed_strategy_metrics_v2.csv"
    df_metrics.to_csv(metrics_path, index=False)

    df_stages = pd.DataFrame(stage_results)
    stages_path = data_dir / "fixed_strategy_stage_metrics_v2.csv"
    df_stages.to_csv(stages_path, index=False)

    print(f"\nSaved raw metrics v2 ({len(df_metrics)} rows) to: {metrics_path}", flush=True)
    print(f"Saved stage metrics v2 ({len(df_stages)} rows) to: {stages_path}", flush=True)

    # Compute Summary Statistics
    summary_list = []
    for strat_id in strategies:
        strat_obj = apply_partition_strategy(strat_id)
        strat_name = strat_obj.strategy_name
        sub_df = df_metrics[df_metrics["strategy_id"] == strat_id]

        if sub_df.empty:
            continue

        latencies = sub_df["latency_seconds"]
        summary_record = {
            "strategy": strat_name,
            "strategy_id": strat_id,
            "num_prompts": len(sub_df),
            "mean_latency": round(latencies.mean(), 4),
            "median_latency": round(latencies.median(), 4),
            "std_latency": round(latencies.std(), 4),
            "min_latency": round(latencies.min(), 4),
            "max_latency": round(latencies.max(), 4),
            "mean_tokens_per_sec": round(sub_df["tokens_per_second"].mean(), 2),
            "mean_cpu_percent": round(sub_df["cpu_percent"].mean(), 2),
            "mean_ram_percent": round(sub_df["ram_percent"].mean(), 2),
            "mean_ram_used_gb": round(sub_df["ram_used_gb"].mean(), 2),
            "mean_total_handoff_latency_seconds": round(sub_df["total_handoff_latency_seconds"].mean(), 8),
            "mean_total_activation_handoff_mb": round(sub_df["total_activation_handoff_mb"].mean(), 4)
        }
        summary_list.append(summary_record)

    df_summary = pd.DataFrame(summary_list)
    summary_path = data_dir / "fixed_strategy_summary_v2.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved summary metrics v2 to: {summary_path}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("V2 SUMMARY STATISTICS BY STRATEGY", flush=True)
    print("=" * 80, flush=True)
    print(df_summary.to_string(index=False), flush=True)
    print("=" * 80, flush=True)

    generate_plots_v2(df_metrics, df_summary, results_dir)

    return df_metrics, df_summary, df_stages


def generate_plots_v2(df_metrics: pd.DataFrame, df_summary: pd.DataFrame, results_dir: Path):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {"static": "#2b5c8f", "balanced": "#d95f02", "latency_optimized": "#7570b3"}

    # 1. Latency plot v2
    plt.figure(figsize=(10, 6))
    for strat in df_metrics["strategy"].unique():
        sub = df_metrics[df_metrics["strategy"] == strat]
        plt.plot(sub["prompt_id"], sub["latency_seconds"], marker="o", label=strat, color=colors.get(strat, "blue"))

    plt.title("V2 Operational Latency per Prompt across Partition Strategies", fontsize=14, fontweight="bold")
    plt.xlabel("Prompt ID", fontsize=12)
    plt.ylabel("Latency (seconds)", fontsize=12)
    plt.xticks(range(1, df_metrics["prompt_id"].max() + 1))
    plt.legend(title="Strategy")
    plt.tight_layout()
    latency_plot_path = results_dir / "fixed_strategy_latency_v2.png"
    plt.savefig(latency_plot_path, dpi=300)
    plt.close()
    print(f"Generated plot: {latency_plot_path}", flush=True)

    # 2. Throughput plot v2
    plt.figure(figsize=(10, 6))
    for strat in df_metrics["strategy"].unique():
        sub = df_metrics[df_metrics["strategy"] == strat]
        plt.plot(sub["prompt_id"], sub["tokens_per_second"], marker="s", label=strat, color=colors.get(strat, "blue"))

    plt.title("V2 Operational Throughput (Tokens/sec) across Partition Strategies", fontsize=14, fontweight="bold")
    plt.xlabel("Prompt ID", fontsize=12)
    plt.ylabel("Tokens / Second", fontsize=12)
    plt.xticks(range(1, df_metrics["prompt_id"].max() + 1))
    plt.legend(title="Strategy")
    plt.tight_layout()
    throughput_plot_path = results_dir / "fixed_strategy_throughput_v2.png"
    plt.savefig(throughput_plot_path, dpi=300)
    plt.close()
    print(f"Generated plot: {throughput_plot_path}", flush=True)

    # 3. Summary Mean Latency Bar Chart v2
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df_summary["strategy"], df_summary["mean_latency"], color=[colors.get(s, "blue") for s in df_summary["strategy"]], width=0.5)
    plt.title("V2 Operational Mean Latency by Partition Strategy", fontsize=14, fontweight="bold")
    plt.xlabel("Partition Strategy", fontsize=12)
    plt.ylabel("Mean Latency (seconds)", fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f"{yval:.2f}s", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    summary_plot_path = results_dir / "fixed_strategy_summary_v2.png"
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()
    print(f"Generated plot: {summary_plot_path}", flush=True)


if __name__ == "__main__":
    run_fixed_strategy_v2_evaluation(num_prompts=20)
