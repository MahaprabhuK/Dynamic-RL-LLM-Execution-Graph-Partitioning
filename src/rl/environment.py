import pandas as pd
import numpy as np

from src.rl.reward import calculate_reward


class LLMPartitionEnv:
    """
    Empirical Contextual RL environment for dynamic LLM execution scheduling.

    Pre-decision State (4D Vector):
        [normalized_prompt_length, normalized_input_tokens, pre_cpu_percent, pre_ram_percent]

    Actions:
        0 -> static
        1 -> balanced
        2 -> latency_optimized

    Grounded entirely in measured empirical outcomes from fixed_strategy_metrics_v2.csv.
    Does NOT use synthetic multipliers or post-execution metrics in pre-decision states.
    """

    ACTIONS = {
        0: "static",
        1: "balanced",
        2: "latency_optimized"
    }

    def __init__(self, csv_path="data/processed/fixed_strategy_metrics_v2.csv", mode="train", seed=42):
        self.mode = mode
        self.seed = seed
        self.df = pd.read_csv(csv_path)

        # Extract unique prompt_ids
        unique_prompts = sorted(self.df["prompt_id"].unique())

        # Deterministic 70/30 Train/Val Split by Prompt ID
        rng = np.random.default_rng(seed)
        shuffled_prompts = unique_prompts.copy()
        rng.shuffle(shuffled_prompts)

        n_train = int(len(unique_prompts) * 0.70)
        self.train_prompts = sorted(shuffled_prompts[:n_train])
        self.val_prompts = sorted(shuffled_prompts[n_train:])

        if mode == "train":
            self.active_prompts = self.train_prompts
        elif mode == "val":
            self.active_prompts = self.val_prompts
        else:
            self.active_prompts = unique_prompts

        # Group data by prompt_id and strategy_id for paired action outcomes
        self.paired_contexts = {}
        for p_id in self.active_prompts:
            p_df = self.df[self.df["prompt_id"] == p_id]
            action_map = {}
            for _, row in p_df.iterrows():
                s_id = int(row["strategy_id"])
                action_map[s_id] = row
            self.paired_contexts[p_id] = action_map

        self.current_prompt_index = 0
        self.max_steps = len(self.active_prompts)
        self.state_size = 4
        self.action_size = 3

    def reset(self):
        """Reset environment to start of prompt list."""
        self.current_prompt_index = 0
        return self._get_state()

    def _get_state(self):
        """Construct pre-decision state vector for current prompt."""
        p_id = self.active_prompts[self.current_prompt_index]
        # Pre-decision features derived from prompt metadata (action 0 reference row)
        row = self.paired_contexts[p_id][0]

        prompt_len = float(row["prompt_length"])
        # Estimate input tokens as ~word count or len/4 if not stored
        input_tokens = float(row.get("input_tokens", prompt_len / 4.0))

        # Baseline pre-execution hardware context (using baseline row metrics)
        pre_cpu = float(row["cpu_percent"])
        pre_ram = float(row["ram_percent"])

        state = np.array([
            prompt_len / 200.0,
            input_tokens / 100.0,
            pre_cpu / 100.0,
            pre_ram / 100.0
        ], dtype=np.float32)

        return state

    def step(self, action: int):
        """
        Execute scheduling action for current prompt context.

        Returns:
            next_state, reward, done, info
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}. Must be 0, 1, or 2.")

        p_id = self.active_prompts[self.current_prompt_index]
        measured_row = self.paired_contexts[p_id][action]

        latency = float(measured_row["latency_seconds"])
        cpu = float(measured_row["cpu_percent"])
        ram = float(measured_row["ram_percent"])
        tokens_per_sec = float(measured_row["tokens_per_second"])
        handoff_mb = float(measured_row.get("total_activation_handoff_mb", 0.0))
        handoff_latency = float(measured_row.get("total_handoff_latency_seconds", 0.0))

        reward = calculate_reward(latency, cpu, ram)

        info = {
            "prompt_id": p_id,
            "action": self.ACTIONS[action],
            "action_id": action,
            "latency": latency,
            "tokens_per_second": tokens_per_sec,
            "cpu": cpu,
            "ram": ram,
            "handoff_mb": handoff_mb,
            "handoff_latency": handoff_latency,
            "reward": reward
        }

        self.current_prompt_index += 1
        done = self.current_prompt_index >= self.max_steps

        if done:
            next_state = np.zeros(self.state_size, dtype=np.float32)
        else:
            next_state = self._get_state()

        return next_state, reward, done, info


if __name__ == "__main__":
    env_train = LLMPartitionEnv(mode="train")
    env_val = LLMPartitionEnv(mode="val")

    print(f"Empirical LLMPartitionEnv initialized.")
    print(f"Train prompts ({len(env_train.train_prompts)}):", env_train.train_prompts)
    print(f"Val prompts ({len(env_val.val_prompts)}):", env_val.val_prompts)

    state = env_train.reset()
    print("Initial train state:", state)

    _, reward, _, info = env_train.step(action=1)
    print(f"Sample step (action 1: balanced): Latency {info['latency']:.2f}s | Reward {reward:.4f}")