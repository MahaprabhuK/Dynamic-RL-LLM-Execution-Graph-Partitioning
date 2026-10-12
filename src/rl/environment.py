import numpy as np
import pandas as pd

from src.rl.features import STATE_SIZE, build_state
from src.rl.reward import calculate_reward


class LLMPartitionEnv:
    """
    Offline contextual-bandit environment for dynamic LLM execution scheduling.

    Each context is one request (prompt x repeat x background-load level) for
    which every strategy was measured back-to-back in random order, so all
    three action outcomes are known (paired outcomes).

    Pre-decision State (4D Vector, see src/rl/features.py):
        [normalized_prompt_length, normalized_input_tokens, pre_cpu_percent, pre_ram_percent]

    Actions:
        0 -> static
        1 -> balanced
        2 -> latency_optimized

    Contexts are independent, so the agent should be trained with gamma=0.
    An "episode" is one pass over the active contexts.

    Expects the v3 measurement CSV written by evaluate_fixed_strategies.py.
    v2 CSVs (no context/pre-run columns) are still accepted, but their CPU/RAM
    features are post-run snapshots and input_tokens is approximated.
    """

    DEFAULT_ACTIONS = {
        0: "static",
        1: "balanced",
        2: "latency_optimized"
    }

    def __init__(self, csv_path="data/processed/fixed_strategy_metrics_v3.csv", mode="train", seed=42, actions=None):
        self.mode = mode
        self.seed = seed
        self.df = self._normalize_columns(pd.read_csv(csv_path))

        if actions is not None:
            self.ACTIONS = dict(actions)
        elif "action_id" in self.df.columns and self.df["action_id"].nunique() > 3:
            unique_actions = sorted(int(a) for a in self.df["action_id"].dropna().unique())
            self.ACTIONS = {a: str(self.df[self.df["action_id"] == a]["strategy"].iloc[0]) for a in unique_actions}
        elif "strategy_id" in self.df.columns and self.df["strategy_id"].nunique() > 3:
            unique_actions = sorted(int(a) for a in self.df["strategy_id"].dropna().unique())
            self.ACTIONS = {a: str(self.df[self.df["strategy_id"] == a]["strategy"].iloc[0]) for a in unique_actions}
        else:
            self.ACTIONS = dict(self.DEFAULT_ACTIONS)

        # Train/Val Split
        unique_prompts = sorted(self.df["prompt_id"].unique())
        if len(unique_prompts) > 1:
            rng = np.random.default_rng(seed)
            shuffled_prompts = list(unique_prompts)
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
            active_df = self.df[self.df["prompt_id"].isin(self.active_prompts)]
        else:
            # Deterministic split by context_id if all rows share 1 prompt
            unique_contexts = sorted(self.df["context_id"].unique())
            n_train = max(1, int(len(unique_contexts) * 0.70))
            train_ctx = set(unique_contexts[:n_train])
            val_ctx = set(unique_contexts[n_train:])
            self.train_prompts = unique_prompts
            self.val_prompts = unique_prompts
            if mode == "train":
                active_df = self.df[self.df["context_id"].isin(train_ctx)]
            elif mode == "val":
                active_df = self.df[self.df["context_id"].isin(val_ctx)]
            else:
                active_df = self.df

        # Group data by context and strategy for paired action outcomes
        self.paired_contexts = {}
        for c_id, c_df in active_df.groupby("context_id", sort=True):
            col = "action_id" if "action_id" in c_df.columns else "strategy_id"
            action_map = {int(row[col]): row for _, row in c_df.iterrows()}
            if set(action_map) == set(self.ACTIONS):
                self.paired_contexts[c_id] = action_map

        self.context_ids = list(self.paired_contexts)
        self.current_index = 0
        self.max_steps = len(self.context_ids)
        self.state_size = STATE_SIZE
        self.action_size = len(self.ACTIONS)

    @staticmethod
    def _normalize_columns(df):
        df = df.copy()
        if "context_id" not in df.columns:
            df["context_id"] = df["prompt_id"]
        if "input_tokens" not in df.columns:
            df["input_tokens"] = df["prompt_length"] / 4.0
        if "pre_cpu_percent" not in df.columns:
            # v2 data: only post-run snapshots exist; use the static row's values per prompt
            ref = df[df["strategy_id"] == 0].set_index("context_id")
            df["pre_cpu_percent"] = df["context_id"].map(ref["cpu_percent"])
            df["pre_ram_percent"] = df["context_id"].map(ref["ram_percent"])
        if "num_generated_tokens" not in df.columns:
            df["num_generated_tokens"] = np.nan
        return df

    @property
    def current_context_id(self):
        return self.context_ids[self.current_index]

    def reset(self, shuffle=False):
        """Start a new pass over the contexts (optionally in random order)."""
        if shuffle:
            np.random.default_rng().shuffle(self.context_ids)
        self.current_index = 0
        return self._get_state()

    def _get_state(self):
        """Construct pre-decision state vector for the current context."""
        row = self.paired_contexts[self.current_context_id][0]

        return build_state(
            prompt_length=row["prompt_length"],
            input_tokens=row["input_tokens"],
            cpu_percent=row["pre_cpu_percent"],
            ram_percent=row["pre_ram_percent"]
        )

    @staticmethod
    def reward_for_row(row):
        num_tokens = row["num_generated_tokens"]
        return calculate_reward(
            latency=float(row["latency_seconds"]),
            cpu_percent=float(row["cpu_percent"]),
            ram_percent=float(row["ram_percent"]),
            num_tokens=None if pd.isna(num_tokens) else int(num_tokens)
        )

    def action_rewards(self):
        """Rewards of every action in the current context (full-feedback data)."""
        outcomes = self.paired_contexts[self.current_context_id]
        return {a: self.reward_for_row(outcomes[a]) for a in self.ACTIONS}

    def step(self, action: int):
        """
        Execute scheduling action for current context.

        Returns:
            next_state, reward, done, info
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}. Must be one of {list(self.ACTIONS)}.")

        c_id = self.current_context_id
        measured_row = self.paired_contexts[c_id][action]
        reward = self.reward_for_row(measured_row)

        info = {
            "context_id": c_id,
            "prompt_id": int(measured_row["prompt_id"]),
            "action": self.ACTIONS[action],
            "action_id": action,
            "latency": float(measured_row["latency_seconds"]),
            "tokens_per_second": float(measured_row["tokens_per_second"]),
            "cpu": float(measured_row["cpu_percent"]),
            "ram": float(measured_row["ram_percent"]),
            "handoff_mb": float(measured_row.get("total_activation_handoff_mb", 0.0)),
            "handoff_latency": float(measured_row.get("total_handoff_latency_seconds", 0.0)),
            "reward": reward
        }

        self.current_index += 1
        done = self.current_index >= self.max_steps

        if done:
            next_state = np.zeros(self.state_size, dtype=np.float32)
        else:
            next_state = self._get_state()

        return next_state, reward, done, info


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/fixed_strategy_metrics_v3.csv"
    env_train = LLMPartitionEnv(csv_path, mode="train")
    env_val = LLMPartitionEnv(csv_path, mode="val")

    print("Empirical LLMPartitionEnv initialized.")
    print(f"Train prompts ({len(env_train.train_prompts)}):", env_train.train_prompts)
    print(f"Val prompts ({len(env_val.val_prompts)}):", env_val.val_prompts)
    print(f"Train contexts: {env_train.max_steps} | Val contexts: {env_val.max_steps}")

    state = env_train.reset()
    print("Initial train state:", state)

    _, reward, _, info = env_train.step(action=1)
    print(f"Sample step (action 1: balanced): Latency {info['latency']:.2f}s | Reward {reward:.4f}")
