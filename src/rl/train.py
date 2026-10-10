from pathlib import Path
import numpy as np
import pandas as pd

from src.rl.environment import LLMPartitionEnv
from src.rl.dqn_agent import DQNAgent
from src.rl.reward import calculate_reward


def train_dqn(episodes: int = 100, seed: int = 42):
    """
    Trains the DQN Agent on empirical v2 training prompt contexts and evaluates offline
    on held-out prompt contexts.
    """
    env_train = LLMPartitionEnv(mode="train", seed=seed)
    env_val = LLMPartitionEnv(mode="val", seed=seed)

    agent = DQNAgent(
        state_size=env_train.state_size,
        action_size=env_train.action_size
    )

    print("=" * 80, flush=True)
    print("STARTING EMPIRICAL DQN TRAINING (V2)", flush=True)
    print(f"Training Prompts ({len(env_train.train_prompts)}): {env_train.train_prompts}", flush=True)
    print(f"Held-out Validation Prompts ({len(env_val.val_prompts)}): {env_val.val_prompts}", flush=True)
    print("=" * 80, flush=True)

    history = []

    for episode in range(1, episodes + 1):
        state = env_train.reset()
        total_reward = 0.0
        losses = []
        actions_taken = []
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env_train.step(action)

            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step(batch_size=16)

            if loss is not None:
                losses.append(loss)

            actions_taken.append(info["action_id"])
            total_reward += reward
            state = next_state

        agent.update_target_network()

        avg_loss = float(np.mean(losses)) if losses else 0.0
        action_counts = {0: actions_taken.count(0), 1: actions_taken.count(1), 2: actions_taken.count(2)}

        history.append({
            "episode": episode,
            "total_reward": round(total_reward, 4),
            "avg_loss": round(avg_loss, 6),
            "epsilon": round(agent.epsilon, 4),
            "static_count": action_counts[0],
            "balanced_count": action_counts[1],
            "latency_opt_count": action_counts[2]
        })

        if episode % 10 == 0 or episode == episodes:
            print(
                f"Episode {episode:03d}/{episodes:03d} | "
                f"Reward: {total_reward:7.3f} | "
                f"Loss: {avg_loss:.6f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Actions (0/1/2): {action_counts[0]}/{action_counts[1]}/{action_counts[2]}",
                flush=True
            )

    output_model_path = Path("data/processed/dqn_model_v2.pth")
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(output_model_path))

    df_hist = pd.DataFrame(history)
    hist_path = Path("data/processed/rl_training_history_v2.csv")
    df_hist.to_csv(hist_path, index=False)
    print(f"Saved training history to: {hist_path}", flush=True)

    # OFFLINE POLICY EVALUATION ON HELD-OUT PROMPT CONTEXTS
    print("\n" + "=" * 80, flush=True)
    print("OFFLINE POLICY EVALUATION ON HELD-OUT PROMPTS", flush=True)
    print("=" * 80, flush=True)

    agent.epsilon = 0.0  # Pure greedy evaluation
    val_state = env_val.reset()
    done = False
    val_records = []

    while not done:
        curr_prompt_id = env_val.active_prompts[env_val.current_prompt_index]
        p_outcomes = env_val.paired_contexts[curr_prompt_id]

        rl_action = agent.choose_action(val_state)
        next_state, rl_reward, done, rl_info = env_val.step(rl_action)

        # Evaluate fixed policies & Oracle reference on the same prompt
        oracle_action = max(p_outcomes.keys(), key=lambda a: float(p_outcomes[a]["latency_seconds"]))  # best reward
        # Compare rewards across actions
        oracle_action = max(p_outcomes.keys(), key=lambda a: float(calculate_reward(float(p_outcomes[a]["latency_seconds"]), float(p_outcomes[a]["cpu_percent"]), float(p_outcomes[a]["ram_percent"]))))

        val_records.append({
            "prompt_id": curr_prompt_id,
            "rl_action_id": rl_action,
            "rl_strategy": env_val.ACTIONS[rl_action],
            "rl_latency": rl_info["latency"],
            "rl_reward": rl_info["reward"],
            "static_latency": float(p_outcomes[0]["latency_seconds"]),
            "balanced_latency": float(p_outcomes[1]["latency_seconds"]),
            "latency_opt_latency": float(p_outcomes[2]["latency_seconds"]),
            "oracle_strategy": env_val.ACTIONS[oracle_action],
            "oracle_latency": float(p_outcomes[oracle_action]["latency_seconds"])
        })

        val_state = next_state

    df_val = pd.DataFrame(val_records)
    val_eval_path = Path("data/processed/rl_offline_evaluation_v2.csv")
    df_val.to_csv(val_eval_path, index=False)

    print(df_val.to_string(index=False), flush=True)
    print(f"\nSaved offline evaluation to: {val_eval_path}", flush=True)

    return agent, df_hist, df_val


if __name__ == "__main__":
    train_dqn(episodes=100, seed=42)

