import numpy as np

from src.rl.environment import LLMPartitionEnv
from src.rl.dqn_agent import DQNAgent


def train_dqn(episodes=50):

    env = LLMPartitionEnv()

    agent = DQNAgent(
        state_size=5,
        action_size=3
    )

    print("Starting DQN training...")
    print("Episodes:", episodes)
    print("Environment steps:", env.max_steps)
    print()

    for episode in range(1, episodes + 1):

        state = env.reset()

        total_reward = 0.0
        losses = []
        actions_taken = []

        done = False

        while not done:

            action = agent.choose_action(state)

            next_state, reward, done, info = env.step(action)

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            loss = agent.train_step(batch_size=16)

            if loss is not None:
                losses.append(loss)

            actions_taken.append(info["action"])

            total_reward += reward

            state = next_state

        # Update target network after each episode
        agent.update_target_network()

        average_loss = (
            np.mean(losses)
            if losses
            else 0.0
        )

        print(
            f"Episode {episode:02d} | "
            f"Reward: {total_reward:.3f} | "
            f"Loss: {average_loss:.4f} | "
            f"Epsilon: {agent.epsilon:.4f}"
        )

    print()
    print("DQN training completed!")
    agent.save("data/processed/dqn_model.pth")
    agent.epsilon = 0.0

    print("\nFinal strategy selection:")

    state = env.reset()

    done = False

    while not done:

        action = agent.choose_action(state)

        next_state, reward, done, info = env.step(action)

        print(
            f"Prompt {env.current_step:02d} -> "
            f"{info['action']} | "
            f"Reward: {reward:.4f}"
        )

        state = next_state


if __name__ == "__main__":
    train_dqn(episodes=50)
