"""
Fast tests for intra-generation re-scheduling (no LLM needed).

    python -m pytest tests/test_adaptive.py -v
"""

import contextlib
import sys
import time
import types
from pathlib import Path

import pandas as pd
import psutil
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.llm.partition import PartitionStrategy
from src.rl.dqn_agent import DQNAgent
from src.scheduler.adaptive import FixedPolicy, RLPolicy, ThresholdPolicy
from src.telemetry.load_generator import DynamicCPULoad
from src.telemetry.monitor import ExternalLoadMonitor


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


# ---------------------------------------------------------------------------
# End-to-end adaptive evaluation with a stubbed LLM
# ---------------------------------------------------------------------------

def fake_generate(prompt, strategy=0, max_new_tokens=100, do_sample=False, return_token_ids=False,
                  scheduler=None, decision_interval=10, decision_log=None):
    stages, current = [], PartitionStrategy(strategy)
    for step in range(max_new_tokens):
        if scheduler is not None and step % decision_interval == 0:
            current = PartitionStrategy(scheduler.decide(step=step, prompt=prompt, input_tokens=3))
            if decision_log is not None:
                decision_log.append({"step": step, "strategy_id": current.strategy_id,
                                     "decision_overhead_seconds": 0.0, **scheduler.last_telemetry})
        stages.append({"token_step": step, "strategy_id": current.strategy_id, "stage_latency_seconds": 0.001,
                       "handoff_latency_seconds": 0.0, "activation_size_mb": 0.0})
    ids = list(range(max_new_tokens))
    return ("fake", stages, ids) if return_token_ids else ("fake", stages)


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
    monkeypatch.setattr(ea, "DynamicCPULoad", lambda schedule: contextlib.nullcontext())
    monkeypatch.setattr(ea, "time", types.SimpleNamespace(sleep=lambda s: None, perf_counter=time.perf_counter))

    df = ea.run_adaptive_evaluation(model_path="data/processed/dqn.pth", num_prompts=1, prompt_offset=1,
                                    max_new_tokens=30, decision_interval=10, tag="t")

    assert set(df["condition"]) == {"static", "latency_optimized", "rule_once", "rule_adaptive",
                                    "rl_once", "rl_adaptive"}
    assert set(df["scenario"]) == {"steady_idle", "steady_heavy", "spike", "relief"}
    assert (df[df["condition"].str.endswith("_once")]["num_switches"] == 0).all()
    assert (df[df["condition"].str.endswith("_adaptive")]["num_decisions"] == 3).all()
    assert (data_dir / "adaptive_paired_diffs_t.csv").exists()
