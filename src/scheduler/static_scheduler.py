from src.llm.partition import apply_partition_strategy
from src.graph.execution_graph import build_execution_graph


class StaticScheduler:
    """
    Baseline scheduler that always selects the same partition strategy,
    regardless of the runtime state. Mirrors RLScheduler.schedule().
    """

    def __init__(self, strategy_id=0):
        self.strategy_id = strategy_id

    def schedule(self, state, prompt_id):

        strategy = apply_partition_strategy(self.strategy_id)

        graph = build_execution_graph(
            prompt_id,
            partition_strategy=strategy
        )

        return self.strategy_id, strategy, graph
