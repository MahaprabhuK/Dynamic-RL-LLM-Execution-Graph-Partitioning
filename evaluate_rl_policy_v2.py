"""
Fresh Online Evaluation Script for RL Policy v2 vs Fixed Baselines.

Evaluates 20 prompts under 4 comparison conditions using the physical v2 layer-sliced engine:
1. Fixed Static (action 0)
2. Fixed Balanced (action 1)
3. Fixed Latency-Optimized (action 2)
4. RL Policy v2 (dynamically selected by trained DQN agent)

Produces per-run CSV, summary CSV, paired latency differences, and plots.
"""

import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add workspace root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.load_prompts import load_prompts
from src.llm.inference import generate_partitioned_response, tokenizer
from src.llm.partition import apply_partition_strategy
from src.graph.execution_graph import build_execution_graph
from src.telemetry.monitor import collect_metrics
from src.rl.dqn_agent import DQNAgent


def run_rl_policy_v2_evaluation(num_prompts: int = 20):
    data_dir = ROOT_DIR / "data" / "processed"
    results_dir = ROOT_DIR / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print(f"STARTING FRESH RL POLICY V2 ONLINE EVALUATION ({num_prompts} PROMPTS)", flush=True)
    print("=" * 80, flush=True)

    # 1. Load trained DQN Model v2
    agent = DQNAgent(state_size=4, action_size=3)
    model_path = data_dir / "dqn_model_v2.pth"
    agent.load(str(model_path))
    agent.epsilon = 0.0  # Pure greedy evaluation

    prompts = load_prompts(csv_path=str(data_dir / "prompts.csv"), limit=num_prompts)
    print(f"Loaded {len(prompts)} prompts.", flush=True)

    conditions = ["static", "balanced", "latency_optimized", "rl_policy"]
    eval_results = []

    for cond in conditions:
        print(f"\n--- Running Evaluation Condition: [{cond}] ---", flush=True)

        for idx, prompt in enumerate(prompts, start=1):
            print(f"  [Prompt {idx:02d}/{num_prompts:02d}] Evaluating condition '{cond}'...", end="", flush=True)

            try:
                # Pre-decision telemetry & feature extraction
                pre_metrics = collect_metrics()
                prompt_len = len(prompt)
                input_tokens = len(tokenizer.encode(prompt))

                pre_state = np.array([
                    prompt_len / 200.0,
                    input_tokens / 100.0,
                    pre_metrics["cpu_percent"] / 100.0,
                    pre_metrics["ram_percent"] / 100.0
                ], dtype=np.float32)

                # Determine action
                if cond == "static":
                    action_id = 0
                elif cond == "balanced":
                    action_id = 1
                elif cond == "latency_optimized":
                    action_id = 2
                else:  # rl_policy
                    action_id = agent.choose_action(pre_state)

                strat_obj = apply_partition_strategy(action_id)
                strat_name = strat_obj.strategy_name

                # Build graph & verify consistency
                graph = build_execution_graph(prompt_id=idx, partition_strategy=strat_obj)
                assert graph.graph["strategy_id"] == action_id, "Graph action mismatch!"

                # Execute through physical v2 layer-sliced engine
                t_start = time.perf_counter()
                response, stage_records = generate_partitioned_response(
                    prompt=prompt,
                    strategy=strat_obj,
                    max_new_tokens=100,
                    do_sample=False
                )
                latency = time.perf_counter() - t_start

                post_metrics = collect_metrics()
                words = len(response.split())
                tok_per_sec = round(words / latency, 2) if latency > 0 else 0.0

                total_handoff_mb = sum(stg["activation_size_mb"] for stg in stage_records)
                total_handoff_lat = sum(stg["handoff_latency_seconds"] for stg in stage_records)

                record = {
                    "prompt_id": idx,
                    "prompt_length": prompt_len,
                    "input_tokens": input_tokens,
                    "condition": cond,
                    "selected_strategy": strat_name,
                    "strategy_id": action_id,
                    "latency_seconds": round(latency, 4),
                    "tokens_per_second": tok_per_sec,
                    "cpu_percent": post_metrics["cpu_percent"],
                    "ram_percent": post_metrics["ram_percent"],
                    "ram_used_gb": post_metrics["ram_used_gb"],
                    "total_handoff_latency_seconds": round(total_handoff_lat, 8),
                    "total_activation_handoff_mb": round(total_handoff_mb, 4)
                }
                eval_results.append(record)
                print(f" Done ({strat_name}, {latency:.2f}s, {tok_per_sec} tok/s)", flush=True)

            except Exception as e:
                print(f" ERROR: Condition {cond} failed on prompt {idx}: {e}", flush=True)

    df_eval = pd.DataFrame(eval_results)
    eval_csv_path = data_dir / "rl_policy_evaluation_v2.csv"
    df_eval.to_csv(eval_csv_path, index=False)
    print(f"\nSaved RL evaluation results ({len(df_eval)} rows) to: {eval_csv_path}", flush=True)

    # Calculate Summary Statistics by Condition
    summary_list = []
    for cond in conditions:
        sub_df = df_eval[df_eval["condition"] == cond]
        if sub_df.empty:
            continue

        strat_dist = sub_df["selected_strategy"].value_counts().to_dict()
        strat_dist_str = ", ".join(f"{k}:{v}" for k, v in strat_dist.items())

        lats = sub_df["latency_seconds"]
        summary_record = {
            "condition": cond,
            "num_prompts": len(sub_df),
            "mean_latency": round(lats.mean(), 4),
            "median_latency": round(lats.median(), 4),
            "std_latency": round(lats.std(), 4),
            "mean_tokens_per_sec": round(sub_df["tokens_per_second"].mean(), 2),
            "mean_cpu_percent": round(sub_df["cpu_percent"].mean(), 2),
            "mean_ram_percent": round(sub_df["ram_percent"].mean(), 2),
            "mean_ram_used_gb": round(sub_df["ram_used_gb"].mean(), 2),
            "mean_handoff_latency_seconds": round(sub_df["total_handoff_latency_seconds"].mean(), 8),
            "mean_activation_handoff_mb": round(sub_df["total_activation_handoff_mb"].mean(), 4),
            "strategy_distribution": strat_dist_str
        }
        summary_list.append(summary_record)

    df_summary = pd.DataFrame(summary_list)
    summary_csv_path = data_dir / "rl_policy_summary_v2.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved summary results to: {summary_csv_path}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("V2 RL POLICY VS FIXED BASELINES SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(df_summary.to_string(index=False), flush=True)
    print("=" * 80, flush=True)

    # Paired Per-Prompt Latency Differences vs RL Policy
    compute_paired_differences(df_eval)

    # Generate Plots
    generate_rl_plots(df_eval, df_summary, results_dir)

    return df_eval, df_summary


def compute_paired_differences(df_eval: pd.DataFrame):
    print("\n" + "=" * 80, flush=True)
    print("PAIRED PER-PROMPT LATENCY DIFFERENCES (RL vs BASELINES)", flush=True)
    print("=" * 80, flush=True)

    p_rl = df_eval[df_eval["condition"] == "rl_policy"].set_index("prompt_id")["latency_seconds"]
    p_static = df_eval[df_eval["condition"] == "static"].set_index("prompt_id")["latency_seconds"]
    p_balanced = df_eval[df_eval["condition"] == "balanced"].set_index("prompt_id")["latency_seconds"]
    p_latopt = df_eval[df_eval["condition"] == "latency_optimized"].set_index("prompt_id")["latency_seconds"]

    diff_static = p_rl - p_static
    diff_balanced = p_rl - p_balanced
    diff_latopt = p_rl - p_latopt

    print(f"Mean (RL - Static) Latency Diff:    {diff_static.mean():+7.4f}s (Median: {diff_static.median():+7.4f}s)")
    print(f"Mean (RL - Balanced) Latency Diff:  {diff_balanced.mean():+7.4f}s (Median: {diff_balanced.median():+7.4f}s)")
    print(f"Mean (RL - LatOpt) Latency Diff:    {diff_latopt.mean():+7.4f}s (Median: {diff_latopt.median():+7.4f}s)")
    print("=" * 80, flush=True)


def generate_rl_plots(df_eval: pd.DataFrame, df_summary: pd.DataFrame, results_dir: Path):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. RL Action Selection Distribution Plot
    rl_df = df_eval[df_eval["condition"] == "rl_policy"]
    counts = rl_df["selected_strategy"].value_counts()
    plt.figure(figsize=(7, 5))
    plt.bar(counts.index, counts.values, color=["#2b5c8f", "#d95f02", "#7570b3"][:len(counts)], width=0.5)
    plt.title("RL Policy Strategy Selection Distribution (v2)", fontsize=14, fontweight="bold")
    plt.xlabel("Selected Strategy", fontsize=12)
    plt.ylabel("Frequency (Count)", fontsize=12)
    for i, v in enumerate(counts.values):
        plt.text(i, v + 0.1, str(v), ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    dist_plot_path = results_dir / "rl_strategy_distribution_v2.png"
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"Generated plot: {dist_plot_path}", flush=True)

    # 2. Latency Comparison Plot
    plt.figure(figsize=(10, 6))
    cond_colors = {"static": "#2b5c8f", "balanced": "#d95f02", "latency_optimized": "#7570b3", "rl_policy": "#e7298a"}
    for cond in df_eval["condition"].unique():
        sub = df_eval[df_eval["condition"] == cond]
        plt.plot(sub["prompt_id"], sub["latency_seconds"], marker="o", label=cond, color=cond_colors.get(cond, "blue"))

    plt.title("Latency per Prompt Across Conditions (RL vs Fixed Baselines)", fontsize=14, fontweight="bold")
    plt.xlabel("Prompt ID", fontsize=12)
    plt.ylabel("Latency (seconds)", fontsize=12)
    plt.xticks(range(1, df_eval["prompt_id"].max() + 1))
    plt.legend(title="Condition")
    plt.tight_layout()
    lat_plot_path = results_dir / "rl_latency_comparison_v2.png"
    plt.savefig(lat_plot_path, dpi=300)
    plt.close()
    print(f"Generated plot: {lat_plot_path}", flush=True)

    # 3. Throughput Comparison Plot
    plt.figure(figsize=(10, 6))
    for cond in df_eval["condition"].unique():
        sub = df_eval[df_eval["condition"] == cond]
        plt.plot(sub["prompt_id"], sub["tokens_per_second"], marker="s", label=cond, color=cond_colors.get(cond, "blue"))

    plt.title("Throughput (Tokens/sec) Across Conditions (RL vs Fixed Baselines)", fontsize=14, fontweight="bold")
    plt.xlabel("Prompt ID", fontsize=12)
    plt.ylabel("Tokens / Second", fontsize=12)
    plt.xticks(range(1, df_eval["prompt_id"].max() + 1))
    plt.legend(title="Condition")
    plt.tight_layout()
    tput_plot_path = results_dir / "rl_throughput_comparison_v2.png"
    plt.savefig(tput_plot_path, dpi=300)
    plt.close()
    print(f"Generated plot: {tput_plot_path}", flush=True)


if __name__ == "__main__":
    run_rl_policy_v2_evaluation(num_prompts=20)
