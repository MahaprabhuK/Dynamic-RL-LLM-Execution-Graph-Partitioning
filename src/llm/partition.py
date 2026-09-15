class PartitionStrategy:
    """
    Defines execution strategies for the LLM execution graph.
    """

    STRATEGIES = {
        0: "static",
        1: "balanced",
        2: "latency_optimized"
    }

    def __init__(self, strategy_id=0):
        if strategy_id not in self.STRATEGIES:
            raise ValueError("Invalid partition strategy.")

        self.strategy_id = strategy_id
        self.strategy_name = self.STRATEGIES[strategy_id]

    def get_partition(self):
        """
        Return the execution graph partition associated
        with the selected strategy.
        """

        partitions = {
            "static": [
                ["Input", "Tokenizer"],
                ["LLM"],
                ["Decoder", "Output"]
            ],

            "balanced": [
                ["Input", "Tokenizer", "LLM"],
                ["Decoder", "Output"]
            ],

            "latency_optimized": [
                ["Input", "Tokenizer"],
                ["LLM", "Decoder"],
                ["Output"]
            ]
        }

        return partitions[self.strategy_name]

    def describe(self):
        """Return a human-readable description."""

        partition = self.get_partition()

        print(f"Strategy: {self.strategy_name}")
        print("Execution partitions:")

        for i, group in enumerate(partition, 1):
            print(f"  Partition {i}: {' -> '.join(group)}")


def apply_partition_strategy(action):
    """
    Apply a partition strategy selected by the RL agent.

    Args:
        action: Integer action selected by DQN.

    Returns:
        PartitionStrategy object
    """

    strategy = PartitionStrategy(action)

    return strategy


if __name__ == "__main__":

    print("Testing partition strategies...\n")

    for action in range(3):

        strategy = apply_partition_strategy(action)

        strategy.describe()

        print()