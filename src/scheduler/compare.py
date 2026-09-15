import numpy as np
import pandas as pd

from src.rl.dqn_agent import DQNAgent
from src.scheduler.scheduler import RLScheduler
from src.scheduler.static_scheduler import StaticScheduler
from src.graph.execution_graph import build_execution_graph


def normalize_state(row):
    """
    Convert one runtime-metrics row into the normalized
    state representation used by the DQN.
    """

    return np.array([
        row["prompt_length"] / 200.0,
        row["response_length"] / 600.0,
        row["latency_seconds"] / 40.0,
        row["cpu_percent"] / 100.0,
        row["ram_percent"] / 100.0
    ], dtype=np.float32)


def compare_schedulers():

    # Load actual runtime measurements
    df = pd.read_csv(
        "data/processed/runtime_metrics.csv"
    )

    # Use the first recorded inference request
    row = df.iloc[0]

    state = normalize_state(row)

    print("REAL RUNTIME STATE")
    print("------------------")
    print(
        f"Prompt length:   {row['prompt_length']}"
    )
    print(
        f"Response length: {row['response_length']}"
    )
    print(
        f"Latency:         {row['latency_seconds']:.3f}s"
    )
    print(
        f"CPU:             {row['cpu_percent']:.1f}%"
    )
    print(
        f"RAM:             {row['ram_percent']:.1f}%"
    )

    # Build graph
    graph = build_execution_graph(
        prompt_id=int(row["prompt_id"])
    )

    # -------------------------
    # Static Scheduler
    # -------------------------

    static_scheduler = StaticScheduler()

    static_plan = static_scheduler.schedule(
        graph
    )

    print("\nSTATIC SCHEDULER")
    print("----------------")
    print(static_plan)

    # -------------------------
    # RL Scheduler
    # -------------------------

    agent = DQNAgent()
    agent.load("data/processed/dqn_model.pth")
    agent.epsilon = 0.0

    rl_scheduler = RLScheduler(agent)

    action, strategy, rl_graph = rl_scheduler.schedule(
        state,
        prompt_id=int(row["prompt_id"])
    )

    print("\nRL SCHEDULER")
    print("------------")
    print("DQN action:", action)
    print("Strategy:", strategy.strategy_name)

    print("\nRL Execution Graph:")
    for node in rl_graph.nodes:

        partition = rl_graph.nodes[node].get(
            "partition",
            "not assigned"
        )

        print(
            f"{node:10s} -> Partition {partition}"
        )


if __name__ == "__main__":
    compare_schedulers()