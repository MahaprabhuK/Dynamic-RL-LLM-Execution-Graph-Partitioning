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

    from src.rl.dqn_agent import DQNAgent

    from src.rl.features import build_state

    # Example pre-decision state: 36-char prompt, 9 input tokens, CPU 70%, RAM 95%
    state = build_state(prompt_length=36, input_tokens=9, cpu_percent=70.0, ram_percent=95.0)

    agent = DQNAgent()

    action, strategy = make_scheduling_decision(
        agent,
        state
    )

    print("Scheduling decision:")
    print("Action:", action)
    print("Strategy:", strategy.strategy_name)

    strategy.describe()