import pandas as pd
import numpy as np

from src.rl.reward import calculate_reward


class LLMPartitionEnv:
    """
    RL environment for dynamic LLM execution scheduling.

    State:
        [prompt_length, response_length,
         latency, cpu_percent, ram_percent]

    Actions:
        0 -> static
        1 -> balanced
        2 -> latency_optimized

    Note:
        The effects of the non-static strategies are currently
        simulated. They will later be replaced by measurements
        from the actual dynamically partitioned execution graph.
    """

    def __init__(self, csv_path="data/processed/runtime_metrics.csv"):

        self.data = pd.read_csv(csv_path)

        self.actions = [
            "static",
            "balanced",
            "latency_optimized"
        ]

        self.current_step = 0
        self.max_steps = len(self.data)

    def reset(self):
        """Reset the environment."""

        self.current_step = 0

        return self._get_state()

    def _get_state(self):
        """Return the current runtime state."""

        row = self.data.iloc[self.current_step]

        return np.array([
            row["prompt_length"] / 200.0,
            row["response_length"] / 600.0,
            row["latency_seconds"] / 40.0,
            row["cpu_percent"] / 100.0,
            row["ram_percent"] / 100.0
        ], dtype=np.float32)

    def step(self, action):
        """
        Execute a scheduling action.

        The current baseline measurements are used as the
        reference. Non-static strategy effects are simulated
        for RL development.
        """

        if action < 0 or action >= len(self.actions):
            raise ValueError("Invalid action.")

        row = self.data.iloc[self.current_step]

        base_latency = float(row["latency_seconds"])
        base_cpu = float(row["cpu_percent"])
        base_ram = float(row["ram_percent"])

        strategy = self.actions[action]

        # Prototype simulation of strategy effects.
        if strategy == "static":
            latency = base_latency
            cpu = base_cpu
            ram = base_ram

        elif strategy == "balanced":
            latency = base_latency * 0.90
            cpu = base_cpu * 0.95
            ram = base_ram * 0.98

        else:  # latency_optimized
            latency = base_latency * 0.80
            cpu = base_cpu * 1.05
            ram = base_ram * 0.99

        reward = calculate_reward(
            latency,
            cpu,
            ram
        )

        info = {
            "action": strategy,
            "latency": latency,
            "cpu": cpu,
            "ram": ram
        }

        self.current_step += 1

        done = self.current_step >= self.max_steps

        if done:
            next_state = np.zeros(5, dtype=np.float32)
        else:
            next_state = self._get_state()

        return next_state, reward, done, info


if __name__ == "__main__":

    env = LLMPartitionEnv()

    state = env.reset()

    print("Initial state:")
    print(state)

    print("\nTesting all strategies:\n")

    for action in range(3):

        env.current_step = 0

        _, reward, _, info = env.step(action)

        print(
            f"{info['action']:20s} "
            f"Latency: {info['latency']:.2f}s | "
            f"CPU: {info['cpu']:.2f}% | "
            f"RAM: {info['ram']:.2f}% | "
            f"Reward: {reward:.4f}"
        )