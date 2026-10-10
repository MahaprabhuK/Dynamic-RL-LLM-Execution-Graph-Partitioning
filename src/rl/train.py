import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.rl.environment import LLMPartitionEnv
from src.rl.dqn_agent import DQNAgent


def train_dqn(
    episodes: int = 100,
    seed: int = 42,
    csv_path: str = "data/processed/fixed_strategy_metrics_v3.csv",
    tag: str = "v3",
    full_feedback: bool = True
):
    """
    Trains the DQN Agent as a contextual bandit on the training prompt contexts
    and evaluates offline on held-out prompt contexts.

    gamma=0 because contexts are independent requests: the target for
    Q(s, a) is the measured reward of that request, not the value of the next
    unrelated request.

    full_feedback: the logged data contains the outcome of every action for
    each context, so all three (state, action, reward) samples are stored per
    step instead of only the one for the chosen action.
    """
    env_train = LLMPartitionEnv(csv_path, mode="train", seed=seed)
    env_val = LLMPartitionEnv(csv_path, mode="val", seed=seed)

    agent = DQNAgent(
        state_size=env_train.state_size,
        action_size=env_train.action_size,
        gamma=0.0
    )

    print("=" * 80, flush=True)
    print(f"STARTING CONTEXTUAL-BANDIT DQN TRAINING ({tag})", flush=True)
    print(f"Training Prompts ({len(env_train.train_prompts)}): {env_train.train_prompts}", flush=True)
    print(f"Held-out Validation Prompts ({len(env_val.val_prompts)}): {env_val.val_prompts}", flush=True)
    print(f"Train contexts: {env_train.max_steps} | Val contexts: {env_val.max_steps}", flush=True)
    print("=" * 80, flush=True)

    history = []

    for episode in range(1, episodes + 1):
        state = env_train.reset(shuffle=True)
        total_reward = 0.0
        losses = []
        actions_taken = []
        done = False

        while not done:
            action = agent.choose_action(state)

            if full_feedback:
                for a, r in env_train.action_rewards().items():
                    agent.remember(state, a, r, state, True)

            next_state, reward, done, info = env_train.step(action)

            if not full_feedback:
                agent.remember(state, action, reward, next_state, True)

            loss = agent.train_step(batch_size=16)

            if loss is not None:
                losses.append(loss)

            actions_taken.append(info["action_id"])
            total_reward += reward
            state = next_state

        agent.update_target_network()

        avg_loss = float(np.mean(losses)) if losses else 0.0
        action_counts = {a: actions_taken.count(a) for a in env_train.ACTIONS}

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

    output_model_path = Path(f"data/processed/dqn_model_{tag}.pth")
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(output_model_path))

    df_hist = pd.DataFrame(history)
    hist_path = Path(f"data/processed/rl_training_history_{tag}.csv")
    df_hist.to_csv(hist_path, index=False)
    print(f"Saved training history to: {hist_path}", flush=True)

    df_val = evaluate_offline(agent, env_val)
    val_eval_path = Path(f"data/processed/rl_offline_evaluation_{tag}.csv")
    df_val.to_csv(val_eval_path, index=False)
    print(f"\nSaved offline evaluation to: {val_eval_path}", flush=True)

    return agent, df_hist, df_val


def evaluate_offline(agent, env_val):
    """
    Greedy policy evaluation on held-out contexts against every fixed policy
    and the per-context oracle (best measured reward).
    """
    print("\n" + "=" * 80, flush=True)
    print("OFFLINE POLICY EVALUATION ON HELD-OUT PROMPTS", flush=True)
    print("=" * 80, flush=True)

    agent.epsilon = 0.0  # Pure greedy evaluation
    val_state = env_val.reset()
    done = False
    val_records = []

    while not done:
        c_id = env_val.current_context_id
        p_outcomes = env_val.paired_contexts[c_id]
        rewards = env_val.action_rewards()
        oracle_action = max(rewards, key=rewards.get)

        rl_action = agent.choose_action(val_state)
        next_state, _, done, rl_info = env_val.step(rl_action)

        record = {
            "context_id": c_id,
            "prompt_id": rl_info["prompt_id"],
            "load_workers": p_outcomes[0].get("load_workers", 0),
            "rl_action_id": rl_action,
            "rl_strategy": env_val.ACTIONS[rl_action],
            "rl_latency": rl_info["latency"],
            "rl_reward": rl_info["reward"],
            "oracle_strategy": env_val.ACTIONS[oracle_action],
            "oracle_latency": float(p_outcomes[oracle_action]["latency_seconds"]),
            "regret": rewards[oracle_action] - rewards[rl_action]
        }
        for a, name in env_val.ACTIONS.items():
            record[f"{name}_latency"] = float(p_outcomes[a]["latency_seconds"])
        val_records.append(record)

        val_state = next_state

    df_val = pd.DataFrame(val_records)

    print(f"Contexts evaluated: {len(df_val)}")
    print(f"Oracle-match rate:  {(df_val['rl_strategy'] == df_val['oracle_strategy']).mean():.1%}")
    print(f"Mean regret:        {df_val['regret'].mean():.4f}")
    print(f"Mean latency  RL: {df_val['rl_latency'].mean():.3f}s | oracle: {df_val['oracle_latency'].mean():.3f}s")
    for name in env_val.ACTIONS.values():
        print(f"              {name}: {df_val[f'{name}_latency'].mean():.3f}s")
    print("RL action distribution:", df_val["rl_strategy"].value_counts().to_dict(), flush=True)

    return df_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the contextual-bandit DQN scheduler.")
    parser.add_argument("--csv", default="data/processed/fixed_strategy_metrics_v3.csv")
    parser.add_argument("--tag", default="v3")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chosen-action-only", action="store_true",
                        help="Learn only from the chosen action's reward (bandit feedback).")
    args = parser.parse_args()

    train_dqn(
        episodes=args.episodes,
        seed=args.seed,
        csv_path=args.csv,
        tag=args.tag,
        full_feedback=not args.chosen_action_only
    )
