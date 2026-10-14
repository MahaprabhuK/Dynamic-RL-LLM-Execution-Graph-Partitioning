from pathlib import Path

from src.data.load_prompts import load_prompts
from src.llm.inference import get_tokenizer
from src.rl.dqn_agent import DQNAgent
from src.rl.features import observe_state
from src.scheduler.scheduler import RLScheduler
from src.scheduler.static_scheduler import StaticScheduler
from src.telemetry.monitor import collect_metrics

MODEL_CANDIDATES = [
    "data/processed/dqn_model_v3.pth",
    "data/processed/dqn_model_v2.pth"
]


def print_graph(graph):
    for node in graph.nodes:
        attrs = graph.nodes[node]
        print(
            f"{node:10s} -> Partition {attrs.get('partition', 'not assigned')}"
            f" (threads: {attrs.get('num_threads')})"
        )


def compare_schedulers(prompt_index=0):
    """
    Show which strategy the static and RL schedulers pick for one request,
    using a live pre-decision state (current CPU/RAM + prompt features).
    """

    prompt = load_prompts(limit=prompt_index + 1)[prompt_index]
    pre_metrics = collect_metrics()
    state = observe_state(prompt, get_tokenizer(), pre_metrics)

    print("PRE-DECISION STATE")
    print("------------------")
    print(f"Prompt:          {prompt[:60]!r}")
    print(f"Prompt length:   {len(prompt)}")
    print(f"CPU:             {pre_metrics['cpu_percent']:.1f}%")
    print(f"RAM:             {pre_metrics['ram_percent']:.1f}%")
    print(f"State vector:    {state}")

    # -------------------------
    # Static Scheduler
    # -------------------------

    _, static_strategy, static_graph = StaticScheduler().schedule(
        state,
        prompt_id=prompt_index + 1
    )

    print("\nSTATIC SCHEDULER")
    print("----------------")
    print("Strategy:", static_strategy.strategy_name)
    print_graph(static_graph)

    # -------------------------
    # RL Scheduler
    # -------------------------

    model_path = next((p for p in MODEL_CANDIDATES if Path(p).exists()), None)
    if model_path is None:
        print("\nNo trained DQN model found. Run: python -m src.rl.train")
        return

    agent = DQNAgent()
    agent.load(model_path)
    agent.epsilon = 0.0

    action, strategy, rl_graph = RLScheduler(agent).schedule(
        state,
        prompt_id=prompt_index + 1
    )

    print("\nRL SCHEDULER")
    print("------------")
    print("DQN action:", action)
    print("Strategy:", strategy.strategy_name)
    print_graph(rl_graph)


if __name__ == "__main__":
    compare_schedulers()
