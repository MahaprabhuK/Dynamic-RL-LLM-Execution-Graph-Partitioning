"""
Online evaluation of the trained RL scheduler against the fixed strategies.

Uses the same protocol as evaluate_fixed_strategies.py: for each context
(prompt x background-load level, visited in random order) the pre-decision
state is measured once, the RL policy picks a strategy from that state, and
all four conditions (static, balanced, latency_optimized, rl_policy) run
back-to-back in a shuffled order. Results are compared with paired
per-context differences and bootstrap 95% CIs.

Use prompts that were NOT in the training data (see --prompt-offset).

Usage:
    python evaluate_rl_policy.py --prompt-offset 20 --num-prompts 10 --loads 0,4,8
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add workspace root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluate_fixed_strategies import execute_strategy, warm_up
from src.benchmark.evaluate import paired_differences, summarize_runs
from src.benchmark.plots import plot_latency_by_load, plot_paired_differences, plot_strategy_distribution
from src.data.load_prompts import load_prompts
from src.graph.execution_graph import build_execution_graph
from src.llm.inference import get_tokenizer
from src.llm.partition import apply_partition_strategy
from src.rl.dqn_agent import DQNAgent
from src.rl.features import observe_state
from src.telemetry.load_generator import BackgroundCPULoad
from src.telemetry.monitor import collect_metrics

CONDITIONS = ["static", "balanced", "latency_optimized", "rl_policy"]


def run_rl_policy_evaluation(
    model_path: str = "data/processed/dqn_model_v3.pth",
    num_prompts: int = 10,
    prompt_offset: int = 20,
    loads=(0, 4, 8),
    max_new_tokens: int = 100,
    seed: int = 7,
    tag: str = "v3"
):
    data_dir = ROOT_DIR / "data" / "processed"
    results_dir = ROOT_DIR / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    agent = DQNAgent(state_size=4, action_size=3)
    agent.load(str(ROOT_DIR / model_path))
    agent.epsilon = 0.0  # Pure greedy evaluation

    rng = np.random.default_rng(seed)
    prompts = load_prompts(csv_path=str(data_dir / "prompts.csv"), limit=prompt_offset + num_prompts)[prompt_offset:]
    tokenizer = get_tokenizer()

    contexts = [(prompt_offset + i + 1, p, load) for i, p in enumerate(prompts) for load in loads]
    rng.shuffle(contexts)

    print("=" * 80, flush=True)
    print(f"RL POLICY ONLINE EVALUATION: {len(contexts)} contexts x {len(CONDITIONS)} conditions", flush=True)
    print("=" * 80, flush=True)

    warm_up(prompts[0])
    eval_results = []

    for c_idx, (prompt_id, prompt, load) in enumerate(contexts, start=1):
        print(f"[{c_idx:03d}/{len(contexts)}] prompt {prompt_id:3d} load {load}:", end="", flush=True)

        try:
            with BackgroundCPULoad(load):
                pre = collect_metrics(interval=0.5)
                state = observe_state(prompt, tokenizer, pre)
                rl_action = agent.choose_action(state)
                action_for = {"static": 0, "balanced": 1, "latency_optimized": 2, "rl_policy": rl_action}

                for run_order, cond in enumerate(rng.permutation(CONDITIONS)):
                    action_id = action_for[cond]
                    strat = apply_partition_strategy(action_id)
                    graph = build_execution_graph(prompt_id=prompt_id, partition_strategy=strat)
                    assert graph.graph["strategy_id"] == action_id, "Graph action mismatch!"

                    out = execute_strategy(prompt, strat, max_new_tokens)
                    n_tokens = len(out["token_ids"])
                    eval_results.append({
                        "context_id": c_idx,
                        "prompt_id": prompt_id,
                        "load_workers": load,
                        "run_order": run_order,
                        "condition": cond,
                        "selected_strategy": strat.strategy_name,
                        "strategy_id": action_id,
                        "prompt_length": len(prompt),
                        "input_tokens": round(float(state[1]) * 100),
                        "pre_cpu_percent": pre["cpu_percent"],
                        "pre_ram_percent": pre["ram_percent"],
                        "num_generated_tokens": n_tokens,
                        "latency_seconds": round(out["latency"], 4),
                        "tokens_per_second": round(n_tokens / out["latency"], 3) if out["latency"] > 0 else 0.0,
                        "cpu_percent": out["post_metrics"]["cpu_percent"],
                        "ram_percent": out["post_metrics"]["ram_percent"],
                        "total_handoff_latency_seconds": round(sum(s["handoff_latency_seconds"] for s in out["stage_records"]), 8),
                        "total_activation_handoff_mb": round(sum(s["activation_size_mb"] for s in out["stage_records"]), 4)
                    })
        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            continue

        print(f" pre-cpu {pre['cpu_percent']:5.1f}% | RL chose {apply_partition_strategy(rl_action).strategy_name}", flush=True)

    df_eval = pd.DataFrame(eval_results)
    eval_csv_path = data_dir / f"rl_policy_evaluation_{tag}.csv"
    df_eval.to_csv(eval_csv_path, index=False)
    print(f"\nSaved RL evaluation results ({len(df_eval)} rows) to: {eval_csv_path}", flush=True)

    summary = summarize_runs(df_eval, ["load_workers", "condition"])
    summary_path = data_dir / f"rl_policy_summary_{tag}.csv"
    summary.to_csv(summary_path, index=False)

    diffs = pd.concat([
        paired_differences(df_eval, "condition", baseline=b, group_col="load_workers").query("condition == 'rl_policy'")
        for b in ["static", "balanced", "latency_optimized"]
    ], ignore_index=True)
    diffs_path = data_dir / f"rl_policy_paired_diffs_{tag}.csv"
    diffs.to_csv(diffs_path, index=False)

    # Regret: RL latency vs the fastest fixed strategy measured in the same context
    wide = df_eval.pivot_table(index="context_id", columns="condition", values="latency_seconds")
    regret = wide["rl_policy"] - wide[["static", "balanced", "latency_optimized"]].min(axis=1)

    print("\n" + "=" * 80, flush=True)
    print("SUMMARY BY LOAD LEVEL AND CONDITION", flush=True)
    print("=" * 80, flush=True)
    print(summary[["load_workers", "condition", "num_runs", "mean_latency", "ci95_low", "ci95_high",
                   "mean_tokens_per_sec"]].to_string(index=False), flush=True)
    print("\nPAIRED DIFFERENCES: RL POLICY VS EACH FIXED STRATEGY (negative = RL faster)", flush=True)
    print(diffs.to_string(index=False), flush=True)
    print(f"\nMean latency regret vs best fixed strategy per context: {regret.mean():.3f}s", flush=True)
    rl_choices = df_eval[df_eval["condition"] == "rl_policy"]
    print("RL choices by load:\n" + rl_choices.groupby(["load_workers", "selected_strategy"]).size().to_string(), flush=True)

    plot_latency_by_load(summary, "condition", results_dir / f"rl_latency_by_load_{tag}.png",
                         "Mean latency: RL policy vs fixed strategies")
    plot_paired_differences(diffs, results_dir / f"rl_paired_diffs_{tag}.png",
                            "RL policy latency difference vs fixed strategies")
    plot_strategy_distribution(rl_choices, results_dir / f"rl_strategy_distribution_{tag}.png",
                               "RL policy choices by background load")

    return df_eval, summary, diffs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="data/processed/dqn_model_v3.pth")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--prompt-offset", type=int, default=20,
                        help="Skip the first N prompts (used for training data collection).")
    parser.add_argument("--loads", default="0,4,8")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tag", default="v3")
    args = parser.parse_args()

    run_rl_policy_evaluation(
        model_path=args.model,
        num_prompts=args.num_prompts,
        prompt_offset=args.prompt_offset,
        loads=[int(x) for x in args.loads.split(",")],
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        tag=args.tag
    )
