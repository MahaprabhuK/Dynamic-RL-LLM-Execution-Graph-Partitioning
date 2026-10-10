import networkx as nx


def build_execution_graph(prompt_id, partition_strategy=None):
    """
    Build the LLM execution graph and optionally assign
    nodes to partitions selected by the scheduler.
    """

    G = nx.DiGraph()

    # Create execution nodes
    nodes = [
        "Input",
        "Tokenizer",
        "LLM",
        "Decoder",
        "Output"
    ]

    G.add_nodes_from(nodes)

    # Create execution flow
    G.add_edges_from([
        ("Input", "Tokenizer"),
        ("Tokenizer", "LLM"),
        ("LLM", "Decoder"),
        ("Decoder", "Output")
    ])

    # Add partition and layer metadata
    if partition_strategy is not None:
        G.graph["strategy_name"] = partition_strategy.strategy_name
        G.graph["strategy_id"] = partition_strategy.strategy_id
        G.graph["layer_ranges"] = partition_strategy.get_layer_ranges()
        G.graph["stage_threads"] = partition_strategy.get_thread_counts()

        partitions = partition_strategy.get_partition()
        stage_defs = partition_strategy.get_stage_definitions()

        for partition_id, partition in enumerate(partitions, 1):
            stage_meta = stage_defs[partition_id - 1] if partition_id <= len(stage_defs) else {}
            for node in partition:
                if node in G.nodes:
                    G.nodes[node]["partition"] = partition_id
                    G.nodes[node]["strategy_name"] = partition_strategy.strategy_name
                    G.nodes[node]["strategy_id"] = partition_strategy.strategy_id
                    G.nodes[node]["layer_start"] = stage_meta.get("layer_start")
                    G.nodes[node]["layer_end"] = stage_meta.get("layer_end")
                    G.nodes[node]["num_threads"] = stage_meta.get("num_threads")
                    G.nodes[node]["stage_type"] = stage_meta.get("stage_type")

    return G


if __name__ == "__main__":

    from src.llm.partition import PartitionStrategy

    strategy = PartitionStrategy(2)

    graph = build_execution_graph(
        prompt_id=1,
        partition_strategy=strategy
    )

    print("Execution Graph")
    print("----------------")

    for node in graph.nodes:

        partition = graph.nodes[node].get(
            "partition",
            "not assigned"
        )

        print(
            f"{node:10s} -> Partition {partition}"
        )

    print("\nEdges:")
    for edge in graph.edges:
        print(f"{edge[0]} -> {edge[1]}")