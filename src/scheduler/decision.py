from src.llm.partition import apply_partition_strategy


def make_scheduling_decision(agent, state):
    """
    Use the trained DQN agent to select an execution strategy.

    Args:
        agent: DQN agent.
        state: Current runtime state.

    Returns:
        Integer action and selected partition strategy.
    """

    action = agent.choose_action(state)

    strategy = apply_partition_strategy(action)

    return action, strategy


if __name__ == "__main__":

    import numpy as np
    from src.rl.dqn_agent import DQNAgent

    # Example runtime state:
    # [prompt_length, response_length, latency, CPU, RAM]
    state = np.array(
        [0.18, 0.60, 0.50, 0.70, 0.95],
        dtype=np.float32
    )

    agent = DQNAgent()

    action, strategy = make_scheduling_decision(
        agent,
        state
    )

    print("Scheduling decision:")
    print("Action:", action)
    print("Strategy:", strategy.strategy_name)

    strategy.describe()