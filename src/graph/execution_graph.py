import networkx as nx


def build_execution_graph(prompt_id):
    """
    Creates a simple execution graph for one inference request.
    This is a placeholder that will later be dynamically partitioned
    by the RL scheduler.
    """

    G = nx.DiGraph()

    G.add_node("Input")
    G.add_node("Tokenizer")
    G.add_node("LLM")
    G.add_node("Decoder")
    G.add_node("Output")

    G.add_edges_from([
        ("Input", "Tokenizer"),
        ("Tokenizer", "LLM"),
        ("LLM", "Decoder"),
        ("Decoder", "Output")
    ])

    return G


if __name__ == "__main__":
    graph = build_execution_graph(1)

    print("Nodes:", graph.nodes())
    print("Edges:", graph.edges())