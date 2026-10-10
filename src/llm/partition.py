import psutil

from src.config import load_config


def physical_core_count():
    return psutil.cpu_count(logical=False) or psutil.cpu_count() or 1


class PartitionStrategy:
    """
    Defines execution strategies for the LLM execution graph.

    Each strategy is a concrete execution configuration: the transformer
    layers are split into stages, and each stage runs with its own intra-op
    thread budget (see configs/config.yaml). Strategies therefore differ in
    how the work is actually executed, not only in their labels.
    """

    STRATEGIES = {
        0: "static",
        1: "balanced",
        2: "latency_optimized"
    }

    def __init__(self, strategy_id=0):
        if isinstance(strategy_id, str):
            inv_map = {v: k for k, v in self.STRATEGIES.items()}
            if strategy_id in inv_map:
                strategy_id = inv_map[strategy_id]
            else:
                raise ValueError(f"Invalid strategy name: {strategy_id}")

        if strategy_id not in self.STRATEGIES:
            raise ValueError("Invalid partition strategy.")

        self.strategy_id = strategy_id
        self.strategy_name = self.STRATEGIES[strategy_id]
        self.config = load_config()["strategies"][self.strategy_name]

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

    def get_layer_ranges(self):
        """
        Return the explicit transformer layer ranges (start_layer, end_layer)
        for each partition stage in TinyLlama (22 total layers, 0-21).
        """
        return [tuple(r) for r in self.config["layer_ranges"]]

    def get_thread_counts(self):
        """
        Return the intra-op thread count used for each layer stage.
        """
        cores = physical_core_count()
        return [
            max(1, round(fraction * cores))
            for fraction in self.config["thread_fractions"]
        ]

    def get_stage_definitions(self):
        """
        Returns structured metadata for each stage in the strategy.
        """
        layer_ranges = self.get_layer_ranges()
        threads = self.get_thread_counts()
        if self.strategy_name == "static":
            return [
                {"partition_id": 1, "stage_type": "input_tokenizer", "layer_start": None, "layer_end": None, "num_threads": None, "nodes": ["Input", "Tokenizer"]},
                {"partition_id": 2, "stage_type": "transformer_slice", "layer_start": layer_ranges[0][0], "layer_end": layer_ranges[0][1], "num_threads": threads[0], "nodes": ["LLM"]},
                {"partition_id": 3, "stage_type": "decoder_output", "layer_start": None, "layer_end": None, "num_threads": None, "nodes": ["Decoder", "Output"]}
            ]
        elif self.strategy_name == "balanced":
            return [
                {"partition_id": 1, "stage_type": "transformer_slice", "layer_start": layer_ranges[0][0], "layer_end": layer_ranges[0][1], "num_threads": threads[0], "nodes": ["Input", "Tokenizer", "LLM_part1"]},
                {"partition_id": 2, "stage_type": "transformer_slice_decoder", "layer_start": layer_ranges[1][0], "layer_end": layer_ranges[1][1], "num_threads": threads[1], "nodes": ["LLM_part2", "Decoder", "Output"]}
            ]
        else:  # latency_optimized
            return [
                {"partition_id": 1, "stage_type": "input_tokenizer", "layer_start": None, "layer_end": None, "num_threads": None, "nodes": ["Input", "Tokenizer"]},
                {"partition_id": 2, "stage_type": "transformer_decoder", "layer_start": layer_ranges[0][0], "layer_end": layer_ranges[0][1], "num_threads": threads[0], "nodes": ["LLM", "Decoder"]},
                {"partition_id": 3, "stage_type": "output", "layer_start": None, "layer_end": None, "num_threads": None, "nodes": ["Output"]}
            ]

    def describe(self):
        """Return a human-readable description."""

        partition = self.get_partition()

        print(f"Strategy: {self.strategy_name} (ID: {self.strategy_id})")
        print("Execution partitions:")

        for i, group in enumerate(partition, 1):
            print(f"  Partition {i}: {' -> '.join(group)}")
        print(f"Layer ranges: {self.get_layer_ranges()}")
        print(f"Threads per layer stage: {self.get_thread_counts()}")


def apply_partition_strategy(action):
    """
    Apply a partition strategy selected by the RL agent or user.

    Args:
        action: Integer or string strategy identifier.

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
