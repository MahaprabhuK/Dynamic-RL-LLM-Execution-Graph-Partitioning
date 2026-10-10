from src.scheduler.decision import make_scheduling_decision
from src.graph.execution_graph import build_execution_graph


class RLScheduler:
    """
    RL-based scheduler that selects a partition strategy
    and applies it to the LLM execution graph.
    """

    def __init__(self, agent):
        self.agent = agent

    def schedule(self, state, prompt_id):
        """
        Select a strategy using the DQN agent and build
        the corresponding execution graph.

        Args:
            state: Current runtime state.
            prompt_id: ID of the inference request.

        Returns:
            action, strategy, execution_graph
        """

        action, strategy = make_scheduling_decision(
            self.agent,
            state
        )

        graph = build_execution_graph(
            prompt_id,
            partition_strategy=strategy
        )

        return action, strategy, graph


if __name__ == "__main__":

    from src.rl.dqn_agent import DQNAgent

    from src.rl.features import build_state

    # Example pre-decision state: 36-char prompt, 9 input tokens, CPU 70%, RAM 95%
    state = build_state(prompt_length=36, input_tokens=9, cpu_percent=70.0, ram_percent=95.0)

    agent = DQNAgent()

    scheduler = RLScheduler(agent)

    action, strategy, graph = scheduler.schedule(
        state,
        prompt_id=1
    )

    print("RL Scheduler")
    print("=============")
    print("Selected action:", action)
    print("Selected strategy:", strategy.strategy_name)

    print("\nExecution Graph:")
    for node in graph.nodes:
        partition = graph.nodes[node].get(
            "partition",
            "not assigned"
        )

        print(
            f"{node:10s} -> Partition {partition}"
        )