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
import math

import numpy as np

from src.llm.partition import PartitionStrategy, apply_partition_strategy, physical_core_count
from src.rl.features import build_state
from src.telemetry.monitor import ExternalLoadMonitor, collect_metrics


class AdaptivePolicy:
    """
    Base class: handles telemetry, the once/adaptive behaviour and hysteresis.

    patience: a different strategy must be proposed this many decisions in a
    row before the policy switches to it (1 = switch immediately).
    """

    name = "policy"
    requires_telemetry = True

    def __init__(self, once: bool = False, initial_interval: float = 0.3, sampler=None, patience: int = 1,
                 emergency_cpu: float | None = None, emergency_action=None):
        self.once = once
        self.initial_interval = initial_interval
        self.sampler = sampler
        self.patience = max(0, int(patience))
        self.emergency_cpu = emergency_cpu
        self.emergency_action = emergency_action
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
        telemetry = self.observe(step)
        if not self.requires_telemetry:
            self.last_telemetry = {"telemetry_valid": True}
            self.current = self.choose(telemetry, prompt, input_tokens)
            return self.current

        cpu = telemetry.get("external_cpu_percent") if isinstance(telemetry, dict) else None
        try:
            cpu = float(cpu)
            telemetry_valid = math.isfinite(cpu) and 0.0 <= cpu <= 100.0
        except (TypeError, ValueError):
            telemetry_valid = False
        if not telemetry_valid:
            self.last_telemetry = {**(telemetry if isinstance(telemetry, dict) else {}), "telemetry_valid": False}
            self._proposal, self._streak = None, 0
            if self.current is None:
                self.current = self.safe_action()
            return self.current

        self.last_telemetry = {**telemetry, "telemetry_valid": True}
        proposal = self.choose(telemetry, prompt, input_tokens)
        emergency = self.emergency_cpu is not None and cpu >= self.emergency_cpu
        if emergency and self.emergency_action is not None:
            proposal = self.emergency_action
            if isinstance(proposal, PartitionStrategy) and proposal.thread_budget is not None:
                self.last_telemetry["selected_thread_budget"] = proposal.thread_budget
        self.last_telemetry["emergency_switch"] = bool(emergency)

        if self.current is None or self._action_key(proposal) == self._action_key(self.current):
            self.current, self._proposal, self._streak = proposal, None, 0
        elif emergency or self.patience <= 1:
            self.current, self._proposal, self._streak = proposal, None, 0
        else:
            self._streak = self._streak + 1 if self._action_key(proposal) == self._action_key(self._proposal) else 1
            self._proposal = proposal
            if self._streak >= self.patience:
                self.current, self._proposal, self._streak = proposal, None, 0
        return self.current

    @staticmethod
    def _action_key(action):
        if action is None:
            return None
        # ids, names ("threads_4", "balanced") and objects compare by what actually runs
        action = action if isinstance(action, PartitionStrategy) else apply_partition_strategy(action)
        return action.strategy_id, tuple(action.get_thread_counts())

    def on_emergency_switch(self, conservative_action=1):
        """Called when an emergency switch bypasses normal interval and hysteresis."""
        self.current = conservative_action
        self._streak = 0
        self._candidate = None

    def on_token(self, step, token_seconds):
        """Optional per-token callback for policies tracking token timing."""
        pass

    def safe_action(self):
        return 0

    def choose(self, telemetry, prompt, input_tokens):
        raise NotImplementedError


class FixedPolicy(AdaptivePolicy):
    requires_telemetry = False
    is_fixed = True  # the engine refuses the latency emergency switch for fixed arms

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
                 sampler=None, patience=1, emergency_cpu: float | None = None,
                 emergency_action: int | None = None):
        super().__init__(once=once, sampler=sampler, patience=patience,
                         emergency_cpu=emergency_cpu, emergency_action=emergency_action)
        self.threshold = threshold
        self.low_load_action = low_load_action
        self.high_load_action = high_load_action

    def choose(self, telemetry, prompt, input_tokens):
        return self.high_load_action if telemetry["external_cpu_percent"] > self.threshold else self.low_load_action


class CoresHeuristicPolicy(AdaptivePolicy):
    """
    Selects thread budget using estimated spare CPU cores.
    threads = clamp(logical_cores - estimated_busy_cores, 1, 10), where
    estimated_busy_cores = external_cpu% * 16 / 100.
    Re-decides every 10 tokens.
    """

    def __init__(self, sampler=None, logical_cores: int = 16, thread_budgets=PartitionStrategy.THREAD_BUDGETS, patience=1,
                 emergency_cpu: float | None = None, emergency_action: int = 1):
        budgets = tuple(sorted(set(int(value) for value in thread_budgets)))
        self.thread_budgets = budgets
        self.logical_cores = logical_cores
        super().__init__(sampler=sampler, patience=patience, emergency_cpu=emergency_cpu,
                         emergency_action=emergency_action)

    def choose(self, telemetry, prompt, input_tokens):
        ext_cpu = telemetry.get("external_cpu_percent", 0.0)
        estimated_busy = ext_cpu * self.logical_cores / 100.0
        target_threads = int(np.clip(round(self.logical_cores - estimated_busy), 1, 10))
        
        # Map target_threads to strategies
        # 10 threads -> static (0)
        # 6 threads -> latency_optimized (2)
        # <=4 threads -> balanced (1)
        if set(self.thread_budgets) == {3, 6, 10} or set(self.thread_budgets) == {2, 4, 6, 8, 10, 12}:
            if target_threads >= 9:
                action = 0  # static (10 threads)
            elif target_threads >= 5:
                action = 2  # latency_optimized (6 threads)
            else:
                action = 1  # balanced (3 threads)
        else:
            eligible = [b for b in self.thread_budgets if b <= target_threads]
            chosen_budget = max(eligible) if eligible else min(self.thread_budgets)
            if chosen_budget >= 10:
                action = 0
            elif chosen_budget >= 6:
                action = 2
            else:
                action = 1

        eligible = [b for b in self.thread_budgets if b <= target_threads]
        chosen_budget = max(eligible) if eligible else min(self.thread_budgets)
        self.last_telemetry.update({
            "target_threads": target_threads,
            "estimated_busy_cores": round(estimated_busy, 2),
            "cores_heuristic_action": action,
            "selected_thread_budget": chosen_budget
        })
        return PartitionStrategy(2, thread_budget=chosen_budget)

    def on_emergency_switch(self, conservative_action=1):
        """Called when an emergency switch bypasses normal interval and hysteresis."""
        self.current = conservative_action
        self._streak = 0
        self._candidate = None

    def on_token(self, step, token_seconds):
        """Optional per-token callback for policies tracking token timing."""
        pass

    def safe_action(self):
        return 1


class RLPolicy(AdaptivePolicy):
    """Trained DQN agent, fed the same 4D state it was trained on."""

    def __init__(self, agent, once=False, sampler=None, patience=1, emergency_cpu: float | None = None,
                 emergency_action: int | None = None):
        super().__init__(once=once, sampler=sampler, patience=patience,
                         emergency_cpu=emergency_cpu, emergency_action=emergency_action)
        self.agent = agent
        self.agent.epsilon = 0.0

    def choose(self, telemetry, prompt, input_tokens):
        state = build_state(
            prompt_length=len(prompt),
            input_tokens=input_tokens,
            cpu_percent=telemetry["external_cpu_percent"],
            ram_percent=telemetry["ram_percent"]
        )
        action = self.agent.choose_action(state)
        if getattr(self.agent, "action_size", 3) == 6:
            budget = {0: 2, 1: 4, 2: 6, 3: 8, 4: 10, 5: 12}.get(action, 10)
            return PartitionStrategy("static", thread_budget=budget)
        return action


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

    - Robust feedback: a segment more than `outlier_ratio` x slower than the
      best established estimate in its bin is treated as a stall (e.g. memory
      paging) and ignored, unless `regime_after` such segments arrive in a
      row, which is taken as a genuine change: the bin is reset and relearned.
    - Stable contexts: the load bin only changes within a request when the
      reading moves more than `bin_margin` points past the bin edge, so noise
      around an edge does not flip between bins.

    The first segment of a request is not used as feedback because it contains
    the prompt prefill. Create one instance and reuse it across requests.
    """

    def __init__(self, prior_agent=None, edges=(35.0,), discount=0.98, prior_weight=2.0, bonus=0.08,
                 sec_per_token_ref=0.5, sampler=None, seed=0, outlier_ratio=2.0, regime_after=5,
                 bin_margin=8.0, emergency_cpu: float | None = None, emergency_action: int = 1):
        super().__init__(once=False, sampler=sampler, emergency_cpu=emergency_cpu,
                         emergency_action=emergency_action)
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
        self.outlier_ratio = outlier_ratio
        self.regime_after = regime_after
        self.bin_margin = bin_margin
        self.counts = {}   # bin -> np.array of real (undiscounted) observation counts
        self.rng = random.Random(seed)
        self._pending = None
        self._bin = None
        self.emergency_action = emergency_action
        self._rejected = {}
        self.history = []

    def context_bin(self, telemetry):
        cpu = telemetry["external_cpu_percent"]
        b = int(np.searchsorted(self.edges, cpu))
        prev = self._bin
        if prev is not None and b != prev:
            edge = self.edges[min(b, prev)]
            if abs(cpu - edge) < self.bin_margin:
                b = prev
        return b

    def is_outlier(self, b, sec_per_token):
        established = self.counts[b] >= 3
        if not established.any():
            return False
        reference = np.nanmin(self.estimates(b)[established])
        return bool(sec_per_token > self.outlier_ratio * reference)

    def _ensure_bin(self, b, prompt, input_tokens, telemetry):
        if b in self.sums:
            return
        n = len(self.actions)
        self.sums[b], self.weights[b], self.counts[b] = np.zeros(n), np.zeros(n), np.zeros(n)
        ext_cpu = telemetry.get("external_cpu_percent", 0.0)
        # P1.3: Limit DQN warm start to training range (<=8 workers, ~75% CPU). Flat prior elsewhere.
        if self.prior_agent is not None and ext_cpu <= 75.0:
            state = build_state(prompt_length=len(prompt), input_tokens=input_tokens,
                                cpu_percent=ext_cpu, ram_percent=telemetry["ram_percent"])
            q_vals = self.prior_agent.q_values(state)
            if len(q_vals) == n:
                prior = np.maximum(-q_vals * self.sec_per_token_ref, 1e-3)
            elif len(q_vals) == 6 and n == 3:
                # Map 6 single-stage thread budgets to the 3 named aliases:
                # 0: static (10 threads -> arm 4)
                # 1: balanced (3+3 threads -> arm 2)
                # 2: latency_optimized (6 threads -> arm 2)
                q_mapped = np.array([q_vals[4], q_vals[2], q_vals[2]], dtype=np.float32)
                prior = np.maximum(-q_mapped * self.sec_per_token_ref, 1e-3)
            else:
                prior = np.full(n, float(self.sec_per_token_ref))
            self.sums[b] = prior * self.prior_weight
            self.weights[b] = np.full(n, float(self.prior_weight))
        elif self.prior_agent is not None:
            # Flat prior outside training range
            prior = np.full(n, float(self.sec_per_token_ref))
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
        self.counts[b][i] += 1

    def select(self, b):
        w = self.weights[b]
        est = self.estimates(b)
        
        # P1.3 Exploration pruning:
        # 1. Never explore static (action 0, 10 threads) in the top bin (high load).
        is_top_bin = (b >= len(self.edges) and (len(self.edges) >= 2 or self.edges[-1] >= 60.0))
        candidates = [a for a in self.actions if not (is_top_bin and a == 0)]
        if not candidates:
            candidates = list(self.actions)

        # 2. Never explore an arm whose current estimate is > 2x the best arm in this bin.
        valid_ests = [est[self.actions.index(a)] for a in candidates if not np.isnan(est[self.actions.index(a)])]
        if valid_ests:
            min_est = min(valid_ests)
            pruned = [a for a in candidates if np.isnan(est[self.actions.index(a)]) or est[self.actions.index(a)] <= 2.0 * min_est]
            if pruned:
                candidates = pruned

        untried = [a for a in candidates if w[self.actions.index(a)] <= 1e-9]
        if untried:
            return self.rng.choice(untried)

        # Lower cost is better: optimism shrinks the estimate of rarely tried arms
        scores = {a: est[self.actions.index(a)] * (1.0 - self.bonus / np.sqrt(w[self.actions.index(a)])) for a in candidates}
        min_score = min(scores.values())
        best = [a for a, s in scores.items() if s <= min_score + 1e-12]
        return self.rng.choice(best)

    def on_token(self, step, token_seconds):
        if not hasattr(self, "_segment_token_latencies"):
            self._segment_token_latencies = []
        self._segment_token_latencies.append(token_seconds)

    def decide(self, step, prompt, input_tokens):
        now = time.perf_counter()
        if step == 0:
            self._pending, self._bin = None, None
            self._segment_token_latencies = []
            self._switched_at_start = False
        elif self._pending is not None:
            b, action, t0, s0, was_switched = self._pending
            if s0 > 0 and step > s0:
                # P1.3 Cost attribution: drop the first 2 tokens after a switch from the segment cost
                latencies = getattr(self, "_segment_token_latencies", [])
                if was_switched and len(latencies) > 2:
                    spt = float(np.mean(latencies[2:]))
                elif latencies:
                    spt = float(np.mean(latencies))
                else:
                    spt = (now - t0) / (step - s0)
                self._segment_token_latencies = []

                outlier = self.is_outlier(b, spt)
                self._rejected[b] = self._rejected.get(b, 0) + 1 if outlier else 0
                accepted = not outlier or self._rejected[b] >= self.regime_after
                if outlier and accepted:
                    # Persistent, not a stall: the bin's old estimates are obsolete
                    self.sums[b], self.weights[b], self.counts[b] = (np.zeros(len(self.actions)) for _ in range(3))
                if accepted:
                    self.update(b, action, spt)
                    self._rejected[b] = 0
                self.history.append({"bin": b, "action": action, "sec_per_token": spt, "accepted": accepted})

        telemetry = self.observe(step)
        b = self._bin = self.context_bin(telemetry)
        self._ensure_bin(b, prompt, input_tokens, telemetry)
        emergency = (self.emergency_cpu is not None and
                 telemetry["external_cpu_percent"] >= self.emergency_cpu)
        action = self.emergency_action if emergency else self.select(b)

        est = self.estimates(b)
        self.last_telemetry = {**telemetry, "context_bin": b,
                       "telemetry_valid": True, "emergency_switch": bool(emergency),
                               **{f"est_spt_{a}": round(float(e), 4) for a, e in zip(self.actions, est)}}
        self.current = action
        was_switched = bool(self.current is not None and action != self.current)
        self._pending = (b, action, time.perf_counter(), step, was_switched)
        return action

    def choose(self, telemetry, prompt, input_tokens):
        raise NotImplementedError("OnlineBanditPolicy overrides decide()")

    def state_dict(self):
        return {"sums": {str(b): v.tolist() for b, v in self.sums.items()},
                "weights": {str(b): v.tolist() for b, v in self.weights.items()},
                "counts": {str(b): v.tolist() for b, v in self.counts.items()},
                "history": self.history}

    def load_state_dict(self, state):
        self.sums = {int(b): np.array(v) for b, v in state["sums"].items()}
        self.weights = {int(b): np.array(v) for b, v in state["weights"].items()}
        self.counts = {int(b): np.array(v) for b, v in state.get("counts", {}).items()} or \
            {b: np.zeros(len(self.actions)) for b in self.sums}
        self.history = list(state.get("history", []))
