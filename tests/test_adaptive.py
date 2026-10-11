"""
Fast tests for intra-generation re-scheduling (no LLM needed).

    python -m pytest tests/test_adaptive.py -v
"""

import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.llm.partition import PartitionStrategy
from src.rl.dqn_agent import DQNAgent
from src.scheduler.adaptive import FixedPolicy, OnlineBanditPolicy, RLPolicy, ThresholdPolicy
from src.telemetry.load_generator import DynamicCPULoad, TokenScheduledLoad
from src.telemetry.monitor import BackgroundLoadSampler, ExternalLoadMonitor


class ScriptedThreshold(ThresholdPolicy):
    """Threshold policy fed a scripted sequence of external-CPU readings."""

    def __init__(self, readings, **kwargs):
        super().__init__(**kwargs)
        self.readings = iter(readings)

    def observe(self, step):
        return {"external_cpu_percent": next(self.readings), "ram_percent": 50.0}


def test_threshold_adaptive_switches_when_load_changes():
    policy = ScriptedThreshold([10, 10, 90, 90], threshold=40)
    assert [policy.decide(step=s, prompt="p", input_tokens=3) for s in (0, 10, 20, 30)] == [0, 0, 1, 1]


def test_once_policy_keeps_first_decision():
    policy = ScriptedThreshold([10, 90, 90], threshold=40, once=True)
    assert [policy.decide(step=s, prompt="p", input_tokens=3) for s in (0, 10, 20)] == [0, 0, 0]


def test_fixed_policy():
    assert FixedPolicy(2).decide(step=0, prompt="p", input_tokens=1) == 2


def test_rl_policy_returns_valid_action():
    policy = RLPolicy(DQNAgent())
    assert policy.decide(step=0, prompt="hello", input_tokens=2) in PartitionStrategy.STRATEGIES
    assert "external_cpu_percent" in policy.last_telemetry


def test_external_load_monitor_sample():
    monitor = ExternalLoadMonitor()
    time.sleep(0.2)
    s = monitor.sample()
    assert 0 <= s["external_cpu_percent"] <= s["cpu_percent"] + 1e-6
    assert s["window_seconds"] > 0


def test_dynamic_load_changes_worker_count():
    me = psutil.Process()
    with DynamicCPULoad([(0.0, 0), (0.3, 2)]):
        assert len(me.children()) == 0
        time.sleep(1.0)
        assert len(me.children()) == 2
    time.sleep(0.2)
    assert len([c for c in me.children() if c.is_running()]) == 0


def test_token_scheduled_load_changes_at_token():
    me = psutil.Process()
    load = TokenScheduledLoad([(0, 0), (5, 2)])
    with load:
        for step in range(5):
            load.on_step(step)
        assert len(me.children()) == 0
        load.on_step(5)
        load._thread.join(timeout=5)
        assert len(me.children()) == 2
    time.sleep(0.2)
    assert len([c for c in me.children() if c.is_running()]) == 0


def test_background_sampler_reads_without_blocking():
    with BackgroundLoadSampler(interval=0.1) as sampler:
        t0 = time.perf_counter()
        s = sampler.latest()
        assert time.perf_counter() - t0 < 0.05
        assert 0 <= s["external_cpu_percent"] <= 100


def test_hysteresis_requires_consecutive_proposals():
    policy = ScriptedThreshold([10, 90, 10, 90, 90, 90], threshold=40, patience=2)
    chosen = [policy.decide(step=s, prompt="p", input_tokens=3) for s in range(0, 60, 10)]
    assert chosen == [0, 0, 0, 0, 1, 1]


def test_step_zero_starts_new_request():
    policy = ScriptedThreshold([10, 90], threshold=40, once=True)
    assert policy.decide(step=0, prompt="p", input_tokens=3) == 0
    assert policy.decide(step=0, prompt="p", input_tokens=3) == 1


class ScriptedSampler:
    def __init__(self, cpu):
        self.cpu = cpu

    def latest(self):
        return {"external_cpu_percent": self.cpu, "ram_percent": 50.0}


def run_bandit_request(bandit, cost_of, tokens=50, interval=10, clock=None):
    """Drive the bandit like the engine would, with a fake clock: cost_of[action] sec/token."""
    chosen = []
    for step in range(0, tokens, interval):
        a = bandit.decide(step=step, prompt="p", input_tokens=3)
        chosen.append(a)
        clock[0] += cost_of[a] * interval
    return chosen


def test_online_bandit_learns_best_arm_and_adapts(monkeypatch):
    import src.scheduler.adaptive as adaptive

    clock = [0.0]
    monkeypatch.setattr(adaptive.time, "perf_counter", lambda: clock[0])
    sampler = ScriptedSampler(80.0)
    bandit = OnlineBanditPolicy(prior_agent=None, sampler=sampler, seed=0)

    for _ in range(20):
        run_bandit_request(bandit, {0: 0.3, 1: 0.1, 2: 0.2}, clock=clock)
    assert bandit.decide(step=0, prompt="p", input_tokens=3) == 1

    # The machine drifts: strategy 2 becomes the best one in the same load bin
    for _ in range(40):
        run_bandit_request(bandit, {0: 0.3, 1: 0.2, 2: 0.05}, clock=clock)
    picks = [a for _ in range(5) for a in run_bandit_request(bandit, {0: 0.3, 1: 0.2, 2: 0.05}, clock=clock)]
    assert picks.count(2) / len(picks) > 0.7


def test_online_bandit_ignores_stalls_but_accepts_regime_change(monkeypatch):
    import src.scheduler.adaptive as adaptive

    clock = [0.0]
    monkeypatch.setattr(adaptive.time, "perf_counter", lambda: clock[0])
    bandit = OnlineBanditPolicy(prior_agent=None, sampler=ScriptedSampler(80.0), seed=0, regime_after=5)
    for _ in range(10):
        run_bandit_request(bandit, {0: 0.1, 1: 0.1, 2: 0.1}, clock=clock)
    before = bandit.estimates(1).copy()

    run_bandit_request(bandit, {0: 2.0, 1: 2.0, 2: 2.0}, tokens=30, clock=clock)  # one 2-segment stall
    assert np.allclose(bandit.estimates(1), before)
    assert not bandit.history[-1]["accepted"]

    run_bandit_request(bandit, {0: 2.0, 1: 2.0, 2: 2.0}, tokens=100, clock=clock)  # persistent slowdown
    assert bandit.history[-1]["accepted"]


def test_online_bandit_bin_hysteresis():
    sampler = ScriptedSampler(20.0)
    bandit = OnlineBanditPolicy(prior_agent=None, sampler=sampler, edges=(35.0,), bin_margin=8.0)
    bins = []
    for step, cpu in zip(range(0, 50, 10), [20, 38, 41, 30, 45]):
        sampler.cpu = cpu
        bandit.decide(step=step, prompt="p", input_tokens=3)
        bins.append(bandit.last_telemetry["context_bin"])
    assert bins == [0, 0, 0, 0, 1]


def test_online_bandit_uses_dqn_prior():
    agent = DQNAgent()
    bandit = OnlineBanditPolicy(prior_agent=agent, sampler=ScriptedSampler(10.0))
    bandit.decide(step=0, prompt="hello", input_tokens=2)
    assert (bandit.weights[0] == bandit.prior_weight).all()
    assert "est_spt_0" in bandit.last_telemetry


# ---------------------------------------------------------------------------
# End-to-end adaptive evaluation with a stubbed LLM
# ---------------------------------------------------------------------------

def fake_generate(prompt, strategy=0, max_new_tokens=100, do_sample=False, return_token_ids=False,
                  scheduler=None, decision_interval=10, decision_log=None, step_callback=None, ignore_eos=False):
    stages, current = [], PartitionStrategy(strategy)
    for step in range(max_new_tokens):
        if step_callback is not None:
            step_callback(step)
        if scheduler is not None and step % decision_interval == 0:
            current = PartitionStrategy(scheduler.decide(step=step, prompt=prompt, input_tokens=3))
            if decision_log is not None:
                decision_log.append({"step": step, "strategy_id": current.strategy_id,
                                     "decision_overhead_seconds": 0.0, **scheduler.last_telemetry})
        stages.append({"token_step": step, "strategy_id": current.strategy_id, "stage_latency_seconds": 0.001,
                       "handoff_latency_seconds": 0.0, "activation_size_mb": 0.0})
    ids = list(range(max_new_tokens))
    return ("fake", stages, ids) if return_token_ids else ("fake", stages)


class NoLoad:
    def __init__(self, schedule):
        pass

    def on_step(self, step):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_adaptive_evaluation_pipeline(tmp_path, monkeypatch):
    import evaluate_adaptive as ea

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    pd.DataFrame({"prompt_id": [1, 2], "prompt": ["first prompt", "second prompt"]}).to_csv(
        data_dir / "prompts.csv", index=False)

    agent = DQNAgent()
    agent.save(str(data_dir / "dqn.pth"))

    monkeypatch.setattr(ea, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(ea, "generate_partitioned_response", fake_generate)
    monkeypatch.setattr(ea, "TokenScheduledLoad", NoLoad)
    monkeypatch.setattr(ea, "time", types.SimpleNamespace(sleep=lambda s: None, perf_counter=time.perf_counter))

    df = ea.run_adaptive_evaluation(model_path="data/processed/dqn.pth", num_prompts=1, prompt_offset=1,
                                    max_new_tokens=30, decision_interval=10, tag="t")

    assert set(df["condition"]) == {"static", "balanced", "latency_optimized", "rule_adaptive",
                                    "rl_once", "rl_adaptive", "bandit_adaptive"}
    assert set(df["scenario"]) == {"steady_idle", "steady_heavy", "spike", "relief"}
    assert (df[df["condition"] == "rl_once"]["num_switches"] == 0).all()
    assert (df[df["condition"].str.endswith("_adaptive")]["num_decisions"] == 3).all()
    assert (data_dir / "adaptive_paired_diffs_t.csv").exists()
    assert (data_dir / "adaptive_bandit_history_t.csv").exists()

    # Resume: drop the last context, rerun, and only that context is redone
    last = df["context_id"].max()
    df[df["context_id"] != last].to_csv(data_dir / "adaptive_runs_t.csv", index=False)
    resumed = ea.run_adaptive_evaluation(model_path="data/processed/dqn.pth", num_prompts=1, prompt_offset=1,
                                         max_new_tokens=30, decision_interval=10, tag="t", resume=True)
    assert len(resumed) == len(df)
    assert list(resumed[resumed["context_id"] == last]["condition"]) == list(df[df["context_id"] == last]["condition"])


