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

    # Add partition information
    if partition_strategy is not None:

        partitions = partition_strategy.get_partition()

        for partition_id, partition in enumerate(partitions, 1):

            for node in partition:

                if node in G.nodes:
                    G.nodes[node]["partition"] = partition_id

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