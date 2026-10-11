"""
Scheduling policies that can be consulted during generation.

Every policy exposes decide(step, prompt, input_tokens) -> strategy_id and is
passed to generate_partitioned_response(scheduler=...). With once=True a
policy decides only before the first token (classic per-request scheduling);
otherwise it re-decides every `decision_interval` tokens from live telemetry
(intra-generation re-scheduling). Step 0 starts a new request.

Telemetry comes from a shared BackgroundLoadSampler when one is given
(non-blocking); otherwise from a short blocking sample before the first token.
"""

import random
import time

import numpy as np

from src.llm.partition import PartitionStrategy
from src.rl.features import build_state
from src.telemetry.monitor import ExternalLoadMonitor, collect_metrics


class AdaptivePolicy:
    """
    Base class: handles telemetry, the once/adaptive behaviour and hysteresis.

    patience: a different strategy must be proposed this many decisions in a
    row before the policy switches to it (1 = switch immediately).
    """

    name = "policy"

    def __init__(self, once: bool = False, initial_interval: float = 0.3, sampler=None, patience: int = 1):
        self.once = once
        self.initial_interval = initial_interval
        self.sampler = sampler
        self.patience = max(1, patience)
        self.monitor = ExternalLoadMonitor() if sampler is None else None
        self.current = None
        self.last_telemetry = {}
        self._proposal, self._streak = None, 0

    def observe(self, step):
        if self.sampler is not None:
            return self.sampler.latest()
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
        if step == 0:
            self.current, self._proposal, self._streak = None, None, 0
        if self.once and self.current is not None:
            return self.current
        self.last_telemetry = self.observe(step)
        proposal = self.choose(self.last_telemetry, prompt, input_tokens)

        if self.current is None or proposal == self.current:
            self.current, self._proposal, self._streak = proposal, None, 0
        else:
            self._streak = self._streak + 1 if proposal == self._proposal else 1
            self._proposal = proposal
            if self._streak >= self.patience:
                self.current, self._proposal, self._streak = proposal, None, 0
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

    def __init__(self, threshold: float = 40.0, low_load_action: int = 0, high_load_action: int = 1, once=False,
                 sampler=None, patience=1):
        super().__init__(once=once, sampler=sampler, patience=patience)
        self.threshold = threshold
        self.low_load_action = low_load_action
        self.high_load_action = high_load_action

    def choose(self, telemetry, prompt, input_tokens):
        return self.high_load_action if telemetry["external_cpu_percent"] > self.threshold else self.low_load_action


class RLPolicy(AdaptivePolicy):
    """Trained DQN agent, fed the same 4D state it was trained on."""

    def __init__(self, agent, once=False, sampler=None, patience=1):
        super().__init__(once=once, sampler=sampler, patience=patience)
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


class OnlineBanditPolicy(AdaptivePolicy):
    """
    Contextual bandit that keeps learning while it serves, at token-segment
    granularity.

    Every decision interval is one bandit round: the context is the external
    CPU load bin, the arm is the execution strategy, and the cost is the
    measured seconds per token of the segment that arm just generated. The
    policy measures this itself from the time between consecutive decide()
    calls, so the engine needs no extra feedback channel.

    - Warm start: per-bin estimates are initialised from the offline DQN's
      Q-values (Q = -sec_per_token / sec_per_token_ref) with `prior_weight`
      pseudo-observations, so it starts as good as the offline policy.
    - Non-stationarity: discounted estimates (factor `discount` per round in
      that bin) forget stale data, so the policy re-calibrates when the
      machine's behaviour drifts away from what the offline data showed.
    - Exploration: discounted UCB-style optimism. Arms that have not been
      tried recently get a growing bonus and are eventually re-checked, for
      the price of one segment.

    The first segment of a request is not used as feedback because it contains
    the prompt prefill. Create one instance and reuse it across requests.
    """

    def __init__(self, prior_agent=None, edges=(35.0, 55.0), discount=0.95, prior_weight=2.0, bonus=0.1,
                 sec_per_token_ref=0.5, sampler=None, seed=0):
        super().__init__(once=False, sampler=sampler)
        self.prior_agent = prior_agent
        if prior_agent is not None:
            prior_agent.epsilon = 0.0
        self.edges = list(edges)
        self.discount = discount
        self.prior_weight = prior_weight
        self.bonus = bonus
        self.sec_per_token_ref = sec_per_token_ref
        self.actions = sorted(PartitionStrategy.STRATEGIES)
        self.sums = {}     # bin -> np.array of discounted cost sums
        self.weights = {}  # bin -> np.array of discounted counts
        self.rng = random.Random(seed)
        self._pending = None
        self.history = []

    def context_bin(self, telemetry):
        return int(np.searchsorted(self.edges, telemetry["external_cpu_percent"]))

    def _ensure_bin(self, b, prompt, input_tokens, telemetry):
        if b in self.sums:
            return
        n = len(self.actions)
        self.sums[b], self.weights[b] = np.zeros(n), np.zeros(n)
        if self.prior_agent is not None:
            state = build_state(prompt_length=len(prompt), input_tokens=input_tokens,
                                cpu_percent=telemetry["external_cpu_percent"], ram_percent=telemetry["ram_percent"])
            prior = np.maximum(-self.prior_agent.q_values(state) * self.sec_per_token_ref, 1e-3)
            self.sums[b] = prior * self.prior_weight
            self.weights[b] = np.full(n, float(self.prior_weight))

    def estimates(self, b):
        w = self.weights[b]
        return np.where(w > 0, self.sums[b] / np.maximum(w, 1e-12), np.nan)

    def update(self, b, action, sec_per_token):
        i = self.actions.index(action)
        self.sums[b] *= self.discount
        self.weights[b] *= self.discount
        self.sums[b][i] += sec_per_token
        self.weights[b][i] += 1.0

    def select(self, b):
        w = self.weights[b]
        untried = [a for a, wi in zip(self.actions, w) if wi <= 1e-9]
        if untried:
            return self.rng.choice(untried)
        # Lower cost is better: optimism shrinks the estimate of rarely tried arms
        score = self.estimates(b) * (1.0 - self.bonus / np.sqrt(w))
        best = np.flatnonzero(score <= score.min() + 1e-12)
        return self.actions[int(self.rng.choice(list(best)))]

    def decide(self, step, prompt, input_tokens):
        now = time.perf_counter()
        if step == 0:
            self._pending = None
        elif self._pending is not None:
            b, action, t0, s0 = self._pending
            if s0 > 0 and step > s0:
                spt = (now - t0) / (step - s0)
                self.update(b, action, spt)
                self.history.append({"bin": b, "action": action, "sec_per_token": spt})

        telemetry = self.observe(step)
        b = self.context_bin(telemetry)
        self._ensure_bin(b, prompt, input_tokens, telemetry)
        action = self.select(b)

        est = self.estimates(b)
        self.last_telemetry = {**telemetry, "context_bin": b,
                               **{f"est_spt_{a}": round(float(e), 4) for a, e in zip(self.actions, est)}}
        self.current = action
        self._pending = (b, action, time.perf_counter(), step)
        return action

    def choose(self, telemetry, prompt, input_tokens):
        raise NotImplementedError("OnlineBanditPolicy overrides decide()")

    def state_dict(self):
        return {"sums": {str(b): v.tolist() for b, v in self.sums.items()},
                "weights": {str(b): v.tolist() for b, v in self.weights.items()},
                "history": self.history}

    def load_state_dict(self, state):
        self.sums = {int(b): np.array(v) for b, v in state["sums"].items()}
        self.weights = {int(b): np.array(v) for b, v in state["weights"].items()}
        self.history = list(state.get("history", []))
