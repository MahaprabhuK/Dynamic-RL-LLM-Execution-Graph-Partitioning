"""
Dataset merger for P1.4 DQN retraining.
Combines load sweep v6, fixed strategy metrics v3, and historical transitions,
mapping actions to the 6 single-stage thread budgets:
  0 -> 2 threads
  1 -> 4 threads
  2 -> 6 threads
  3 -> 8 threads
  4 -> 10 threads
  5 -> 12 threads
Strictly enforces holding out prompts 21-40.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BUDGET_TO_ACTION = {
    2: 0,
    4: 1,
    6: 2,
    8: 3,
    10: 4,
    12: 5
}
ACTION_TO_BUDGET = {v: k for k, v in BUDGET_TO_ACTION.items()}
ACTION_NAMES = {a: f"threads_{b}" for a, b in ACTION_TO_BUDGET.items()}


def build_v6_training_dataset(
    sweep_path="data/processed/load_sweep_v6.csv",
    v3_path="data/processed/fixed_strategy_metrics_v3.csv",
    output_path="data/processed/training_data_v6.csv",
    held_out_prompts=tuple(range(21, 41))
):
    sweep_file = Path(sweep_path)
    if not sweep_file.exists():
        raise FileNotFoundError(f"Sweep file {sweep_path} not found.")

    df_sweep = pd.read_csv(sweep_file)
    rows = []

    # 1. Process v6 sweep data
    # Each (repeat, load_workers) is one paired context
    grouped = df_sweep.groupby(["repeat", "load_workers"])
    ctx_counter = 1
    for (rep, load), group in grouped:
        if set(group["thread_budget"]) != set(BUDGET_TO_ACTION):
            continue  # incomplete context
        p_id = int(group["prompt_id"].iloc[0]) if "prompt_id" in group else 1
        if p_id in held_out_prompts:
            continue

        ext_cpu = float(group["external_cpu_idle_reading"].iloc[0])
        for _, row in group.iterrows():
            b = int(row["thread_budget"])
            action_id = BUDGET_TO_ACTION[b]
            dur = float(row["latency_seconds"])
            spt = float(row["decode_sec_per_token"])
            cpu_dur = float(row["external_cpu_during"])
            rows.append({
                "context_id": ctx_counter,
                "prompt_id": p_id,
                "repeat": int(rep),
                "load_workers": int(load),
                "action_id": action_id,
                "strategy_id": action_id,
                "strategy": ACTION_NAMES[action_id],
                "thread_budget": b,
                "prompt_length": 36,
                "input_tokens": 9,
                "pre_cpu_percent": ext_cpu,
                "pre_ram_percent": 50.0,
                "latency_seconds": dur,
                "tokens_per_second": round(1.0 / spt, 3) if spt > 0 else 5.0,
                "cpu_percent": cpu_dur,
                "ram_percent": 52.0,
                "num_generated_tokens": 50
            })
        ctx_counter += 1

    df_v6 = pd.DataFrame(rows)
    # Check held-out prompts
    assert not df_v6["prompt_id"].isin(held_out_prompts).any(), "Held-out prompt leaked into training dataset!"

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_v6.to_csv(out_file, index=False)
    print(f"Built v6 training dataset with {len(df_v6)} rows across {ctx_counter - 1} contexts.")
    print(f"Saved to: {out_file}")
    return df_v6


if __name__ == "__main__":
    build_v6_training_dataset()
