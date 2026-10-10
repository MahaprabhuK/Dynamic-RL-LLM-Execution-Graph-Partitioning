"""
Scheduling policies that can be consulted during generation.

Every policy exposes decide(step, prompt, input_tokens) -> strategy_id and is
passed to generate_partitioned_response(scheduler=...). With once=True a
policy decides only before the first token (classic per-request scheduling);
otherwise it re-decides every `decision_interval` tokens from live telemetry
(intra-generation re-scheduling).
"""

from src.rl.features import build_state
from src.telemetry.monitor import ExternalLoadMonitor, collect_metrics


class AdaptivePolicy:
    """Base class: handles telemetry and the once/adaptive behaviour."""

    name = "policy"

    def __init__(self, once: bool = False, initial_interval: float = 0.3):
        self.once = once
        self.initial_interval = initial_interval
        self.monitor = ExternalLoadMonitor()
        self.current = None
        self.last_telemetry = {}

    def observe(self, step):
        if step == 0:
            # Before the first token our own process is idle, so a short blocking
            # sample of system CPU is the external load (same as the training feature).
            m = collect_metrics(interval=self.initial_interval)
            self.monitor.reset()
            return {"cpu_percent": m["cpu_percent"], "own_cpu_percent": 0.0,
                    "external_cpu_percent": m["cpu_percent"], "ram_percent": m["ram_percent"],
                    "window_seconds": self.initial_interval}
        return self.monitor.sample()

    def decide(self, step, prompt, input_tokens):
        if self.once and self.current is not None:
            return self.current
        self.last_telemetry = self.observe(step)
        self.current = self.choose(self.last_telemetry, prompt, input_tokens)
        return self.current

    def choose(self, telemetry, prompt, input_tokens):
        raise NotImplementedError


class FixedPolicy(AdaptivePolicy):
    def __init__(self, strategy_id):
        super().__init__(once=True)
        self.strategy_id = strategy_id

    def observe(self, step):
        return {}

    def choose(self, telemetry, prompt, input_tokens):
        return self.strategy_id


class ThresholdPolicy(AdaptivePolicy):
    """
    Simple rule baseline: fewer threads when external CPU load is high.
    Defaults follow the v3 measurements: static wins when idle, balanced
    (3+3 threads) wins from ~4 busy workers (external CPU ~45%+).
    """

    def __init__(self, threshold: float = 40.0, low_load_action: int = 0, high_load_action: int = 1, once=False):
        super().__init__(once=once)
        self.threshold = threshold
        self.low_load_action = low_load_action
        self.high_load_action = high_load_action

    def choose(self, telemetry, prompt, input_tokens):
        return self.high_load_action if telemetry["external_cpu_percent"] > self.threshold else self.low_load_action


class RLPolicy(AdaptivePolicy):
    """Trained DQN agent, fed the same 4D state it was trained on."""

    def __init__(self, agent, once=False):
        super().__init__(once=once)
        self.agent = agent
        self.agent.epsilon = 0.0

    def choose(self, telemetry, prompt, input_tokens):
        state = build_state(
            prompt_length=len(prompt),
            input_tokens=input_tokens,
            cpu_percent=telemetry["external_cpu_percent"],
            ram_percent=telemetry["ram_percent"]
        )
        return self.agent.choose_action(state)
