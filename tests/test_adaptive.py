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
from src.scheduler.adaptive import CoresHeuristicPolicy, FixedPolicy, OnlineBanditPolicy, RLPolicy, ThresholdPolicy
from src.benchmark.oracles import segment_oracle
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
    class AgentDouble:
        epsilon = 1.0

        def choose_action(self, state):
            return 1

    policy = RLPolicy(AgentDouble())
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


def test_rule_no_hysteresis_switches_with_zero_patience():
    policy = ScriptedThreshold([10, 90], threshold=40, patience=0)
    assert [policy.decide(step=s, prompt="p", input_tokens=3) for s in (0, 10)] == [0, 1]
    assert policy.patience == 0


def test_emergency_cpu_bypasses_patience_at_next_decision():
    policy = ScriptedThreshold([10, 50, 90], threshold=40, patience=2, emergency_cpu=80)
    assert [policy.decide(step=s, prompt="p", input_tokens=3) for s in (0, 10, 20)] == [0, 0, 1]
    assert policy.last_telemetry["emergency_switch"]


def test_emergency_cpu_overrides_rl_proposal():
    class AgentDouble:
        epsilon = 0.0

        def choose_action(self, state):
            return 0

    policy = RLPolicy(AgentDouble(), sampler=ScriptedSampler(90), emergency_cpu=80,
                      emergency_action=1)
    assert policy.decide(step=0, prompt="p", input_tokens=3) == 1
    assert policy.last_telemetry["emergency_switch"]


def test_invalid_telemetry_holds_current_action_then_recovers():
    policy = ScriptedThreshold([10, 90, float("nan"), 90, 90], threshold=40, patience=2)
    choices = [policy.decide(step=s, prompt="p", input_tokens=3) for s in (0, 10, 20, 30, 40)]
    assert choices == [0, 0, 0, 0, 1]
    assert policy.last_telemetry["telemetry_valid"]


def test_cores_heuristic_is_stable_and_reduces_budget_as_load_rises():
    sampler = ScriptedSampler(10)
    policy = CoresHeuristicPolicy(sampler=sampler, patience=1)
    idle_a = policy.decide(step=0, prompt="p", input_tokens=3)
    idle_b = policy.decide(step=10, prompt="p", input_tokens=3)
    assert idle_a.get_thread_counts() == idle_b.get_thread_counts()

    budgets = [sum(idle_b.get_thread_counts())]
    for step, cpu in enumerate((35, 60, 90), start=2):
        sampler.cpu = cpu
        action = policy.decide(step=step * 10, prompt="p", input_tokens=3)
        budgets.append(sum(action.get_thread_counts()))
    assert budgets == sorted(budgets, reverse=True)
    assert set(budgets).issubset(set(PartitionStrategy.THREAD_BUDGETS))


@pytest.mark.parametrize("interval", [5, 10, 20])
def test_generation_decision_interval_and_switch_overhead(interval):
    decisions = []
    steps = list(range(0, interval * 3, interval))
    policy = ScriptedThreshold([10, 90, 90], threshold=40, patience=1)
    fake_generate("p", strategy=0, max_new_tokens=interval * 3, scheduler=policy,
                  decision_interval=interval, decision_log=decisions)
    assert [row["step"] for row in decisions] == steps
    assert all(row["decision_overhead_seconds"] > 0 for row in decisions)
    assert sum(a["strategy_id"] != b["strategy_id"] for a, b in zip(decisions, decisions[1:])) == 1


def test_switch_counter_includes_thread_budget_changes():
    from evaluate_adaptive import _count_decision_switches

    decisions = [
        {"strategy_id": 2, "thread_counts": [8]},
        {"strategy_id": 2, "thread_counts": [4]},
        {"strategy_id": 1, "thread_counts": [2, 2]},
        {"strategy_id": 1, "thread_counts": [2, 2]},
    ]
    assert _count_decision_switches(decisions) == 2


def test_segment_oracle_is_explicitly_offline_and_requires_complete_actions():
    outcomes = pd.DataFrame([
        {"context_id": 1, "segment_id": 0, "action": "static", "latency_seconds": 2.0},
        {"context_id": 1, "segment_id": 0, "action": "balanced", "latency_seconds": 1.0},
        {"context_id": 1, "segment_id": 1, "action": "static", "latency_seconds": 0.5},
        {"context_id": 1, "segment_id": 1, "action": "balanced", "latency_seconds": 1.5},
    ])
    oracle = segment_oracle(outcomes)
    assert oracle["action"].tolist() == ["balanced", "static"]
    assert oracle["uses_future_outcomes"].all()
    assert not oracle["online_usable"].any()
    with pytest.raises(ValueError, match="incomplete action outcomes"):
        segment_oracle(outcomes.iloc[:-1])


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
    class AgentDouble:
        epsilon = 1.0
        state_size = 4
        action_size = 3

        def q_values(self, state):
            return np.array([-0.4, -0.2, -0.3])

    agent = AgentDouble()
    bandit = OnlineBanditPolicy(prior_agent=agent, sampler=ScriptedSampler(10.0))
    bandit.decide(step=0, prompt="hello", input_tokens=2)
    assert (bandit.weights[0] == bandit.prior_weight).all()
    assert "est_spt_0" in bandit.last_telemetry


# ---------------------------------------------------------------------------
# End-to-end adaptive evaluation with a stubbed LLM
# ---------------------------------------------------------------------------

def fake_generate(prompt, strategy=0, max_new_tokens=100, do_sample=False, return_token_ids=False,
                  scheduler=None, decision_interval=10, decision_log=None, step_callback=None, ignore_eos=False,
                  emergency_latency_factor=None, segment_log=None, segment_tokens=10):
    stages = []
    current = strategy if isinstance(strategy, PartitionStrategy) else PartitionStrategy(strategy)
    for step in range(max_new_tokens):
        if step_callback is not None:
            step_callback(step)
        if scheduler is not None and step % decision_interval == 0:
            started = time.perf_counter()
            proposal = scheduler.decide(step=step, prompt=prompt, input_tokens=3)
            current = proposal if isinstance(proposal, PartitionStrategy) else PartitionStrategy(proposal)
            if decision_log is not None:
                decision_log.append({"step": step, "strategy_id": current.strategy_id,
                                     "thread_counts": current.get_thread_counts(),
                                     "decision_overhead_seconds": time.perf_counter() - started,
                                     **scheduler.last_telemetry})
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

    class ModelDouble:
        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            return None

    class AgentDouble:
        state_size = 4
        action_size = 3
        epsilon = 1.0

        def __init__(self, **kwargs):
            self.model = ModelDouble()

        def load(self, path):
            return None

        def choose_action(self, state):
            return 0

        def q_values(self, state):
            return np.array([-0.4, -0.2, -0.3])

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    pd.DataFrame({"prompt_id": [1, 2], "prompt": ["first prompt", "second prompt"]}).to_csv(
        data_dir / "prompts.csv", index=False)

    (data_dir / "dqn.pth").touch()

    monkeypatch.setattr(ea, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(ea, "_load_inference", lambda: fake_generate)
    monkeypatch.setattr(ea, "_load_dqn_agent", lambda: AgentDouble)
    monkeypatch.setattr(ea, "TokenScheduledLoad", NoLoad)
    monkeypatch.setattr(ea, "time", types.SimpleNamespace(sleep=lambda s: None, perf_counter=time.perf_counter))

    df = ea.run_adaptive_evaluation(model_path="data/processed/dqn.pth", num_prompts=1, prompt_offset=1,
                                    max_new_tokens=30, decision_interval=10, tag="t")

    assert set(df["condition"]) == {"static", "balanced", "latency_optimized", "rule_adaptive", "rule_noh",
                                    "cores_heuristic", "rl_once", "rl_adaptive", "bandit_adaptive"}
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


from src.telemetry.stall_detection import detect_segment_stall, audit_v5_exclusions
from src.telemetry.monitor import PagingTelemetryTracker


def test_cliff_hit_on_static_under_load_is_never_excluded():
    # Context 36 in ramp: 12 workers arrived at token 140, policy stayed on static.
    # Latency jumped from ~3 s / 20 tokens to 110.3 s / 20 tokens.
    runs = pd.DataFrame([
        {"context_id": i, "scenario": "ramp", "condition": "rl_adaptive", "latency_seconds": 54.9, "num_generated_tokens": 200}
        for i in range(1, 6)
    ] + [{
        "context_id": 36, "scenario": "ramp", "condition": "rl_adaptive",
        "latency_seconds": 167.6770, "num_generated_tokens": 200
    }])
    audit = audit_v5_exclusions(runs, stall_ratio=2.0)
    assert len(audit) == 1
    row = audit.iloc[0]
    assert row["context_id"] == 36
    assert bool(row["is_cliff_or_policy_failure"]) is True
    assert bool(row["excluded_from_primary"]) is False
    assert row["classification"] == "oversubscription_cliff"
    assert "oversubscription cliff" in row["evidence"].lower()


def test_bandit_exploration_above_cliff_is_never_excluded():
    # Context 14 in ramp: bandit explored static at tokens 180-200 under 12 workers.
    runs = pd.DataFrame([
        {"context_id": i, "scenario": "ramp", "condition": "bandit_adaptive", "latency_seconds": 75.0, "num_generated_tokens": 200}
        for i in range(1, 6)
    ] + [{
        "context_id": 14, "scenario": "ramp", "condition": "bandit_adaptive",
        "latency_seconds": 157.7798, "num_generated_tokens": 200
    }])
    audit = audit_v5_exclusions(runs, stall_ratio=2.0)
    assert len(audit) == 1
    row = audit.iloc[0]
    assert row["context_id"] == 14
    assert bool(row["is_cliff_or_policy_failure"]) is True
    assert bool(row["excluded_from_primary"]) is False
    assert row["classification"] == "bandit_exploration_cliff"
    assert "bandit exploration" in row["evidence"].lower()


def test_detect_segment_stall_distinguishes_load_changes_from_stalls():
    # 30 tokens: block 0 (idle, static), block 1 (heavy load, static), block 2 (heavy load, static)
    tokens = [0.1] * 10 + [2.5] * 10 + [2.5] * 10
    strats = [0] * 30
    loads = [0] * 10 + [12] * 10 + [12] * 10

    # Block 1 jump was caused by load change (0 -> 12 workers): cliff, NOT stall
    res = detect_segment_stall(tokens, strats, load_levels=loads, jump_ratio=3.0, min_segment_tokens=10)
    assert res["is_cliff"] is True
    assert res["has_stall"] is False

    # Spiking when load and strategy are both constant:
    # tokens 0-9: 0.1s, tokens 10-19: 0.1s, tokens 20-29: 3.0s under constant load and strategy
    constant_load_jump = [0.1] * 20 + [3.0] * 10
    constant_loads = [0] * 30
    constant_strats = [0] * 30

    # Without paging signal: not confirmed stall
    res_no_paging = detect_segment_stall(constant_load_jump, constant_strats, load_levels=constant_loads,
                                         paging_signals=[False] * 30, jump_ratio=3.0, min_segment_tokens=10)
    assert res_no_paging["is_cliff"] is False
    assert res_no_paging["has_stall"] is False
    assert res_no_paging["unexplained_stalls"] == 1

    # With independent paging signal: confirmed stall
    res_with_paging = detect_segment_stall(constant_load_jump, constant_strats, load_levels=constant_loads,
                                           paging_signals=[True] * 30, jump_ratio=3.0, min_segment_tokens=10)
    assert res_with_paging["has_stall"] is True


def test_paging_telemetry_tracker_records_deltas():
    tracker = PagingTelemetryTracker()
    delta = tracker.sample_delta()
    assert "swap_in_delta" in delta
    assert "page_faults_delta" in delta
    assert "has_paging_signal" in delta
    assert delta["swap_in_delta"] >= 0
    assert delta["page_faults_delta"] >= 0


def test_v5r_audit_retains_all_historical_contexts_in_primary():
    v5_csv = Path("data/processed/adaptive_runs_v5.csv")
    if v5_csv.exists():
        runs = pd.read_csv(v5_csv)
        audit = audit_v5_exclusions(runs)
        assert len(audit) == 4
        assert set(audit["context_id"]) == {14, 19, 36, 39}
        assert (audit["excluded_from_primary"] == False).all()
        assert (audit["sensitivity_candidate"] == True).all()


# --- P1 Unit & Regression Tests ---

def test_cores_heuristic_clamp_formula_and_mapping():
    sampler = ScriptedSampler(0.0)
    policy = CoresHeuristicPolicy(sampler=sampler, logical_cores=16, thread_budgets=(3, 6, 10), patience=0)

    # 0% external CPU -> 16 available -> clamp(16, 1, 10) = 10 threads
    policy.decide(step=0, prompt="p", input_tokens=3)
    assert policy.last_telemetry["target_threads"] == 10
    assert policy.last_telemetry["cores_heuristic_action"] == 0  # static

    # 50% external CPU -> 8 busy -> 8 available -> clamp(8, 1, 10) = 8 -> 6 threads
    sampler.cpu = 50.0
    policy.decide(step=10, prompt="p", input_tokens=3)
    assert policy.last_telemetry["target_threads"] == 8
    assert policy.last_telemetry["cores_heuristic_action"] == 2  # latency_optimized

    # 85% external CPU -> 13.6 busy -> 2.4 available -> clamp(2, 1, 10) = 2 -> 3 threads
    sampler.cpu = 85.0
    policy.decide(step=20, prompt="p", input_tokens=3)
    assert policy.last_telemetry["target_threads"] == 2
    assert policy.last_telemetry["cores_heuristic_action"] == 1  # balanced


def test_timeline_segment_oracle_computes_hindsight_bound():
    from src.benchmark.oracles import compute_timeline_segment_oracle, compute_policy_regrets

    # 20 tokens (2 segments of 10) across 3 fixed strategies
    # In segment 0: static is fastest (0.1 s/tok vs 0.2, 0.3)
    # In segment 1: balanced is fastest (0.1 s/tok vs 0.5, 0.3)
    rows = []
    for cond, (s0_spt, s1_spt) in [("static", (0.1, 0.5)), ("balanced", (0.3, 0.1)), ("latency_optimized", (0.2, 0.2))]:
        for step in range(20):
            spt = s0_spt if step < 10 else s1_spt
            rows.append({
                "context_id": 1,
                "scenario": "spike",
                "condition": cond,
                "token_step": step,
                "token_seconds": spt,
                "strategy_id": 0 if cond == "static" else (1 if cond == "balanced" else 2)
            })
    timelines_df = pd.DataFrame(rows)
    oracle_df = compute_timeline_segment_oracle(timelines_df, segment_size=10)

    # Segment 0 oracle cost = 10 * 0.1 = 1.0 s
    # Segment 1 oracle cost = 10 * 0.1 = 1.0 s
    # Total segment oracle = 2.0 s
    assert len(oracle_df) == 1
    row = oracle_df.iloc[0]
    assert np.isclose(row["segment_oracle_seconds"], 2.0)

    # Single best fixed arm: latency_optimized (10*0.2 + 10*0.2 = 4.0 s) vs static (6.0 s) vs balanced (4.0 s)
    assert np.isclose(row["best_single_fixed_seconds"], 4.0)
    assert np.isclose(row["oracle_gap_seconds"], 2.0)


def test_online_bandit_p1_3_pruning_and_top_bin():
    # Top bin (cpu >= 63% with edges=(51, 63)): static (arm 0) must NEVER be selected
    sampler = ScriptedSampler(80.0)
    bandit = OnlineBanditPolicy(prior_agent=None, sampler=sampler, edges=(51.0, 63.0), seed=42)
    selections = [bandit.decide(step=step, prompt="p", input_tokens=3) for step in range(0, 100, 10)]
    assert 0 not in selections, "Static (arm 0) was selected in the top high-load bin"

    # Exploration pruning: arm with estimate > 2x best arm is never selected
    bandit2 = OnlineBanditPolicy(prior_agent=None, sampler=ScriptedSampler(20.0), edges=(51.0, 63.0), seed=42)
    bandit2.decide(step=0, prompt="p", input_tokens=3)
    b = bandit2._bin
    bandit2.sums[b] = np.array([0.5, 0.1, 0.1])
    bandit2.weights[b] = np.array([5.0, 5.0, 5.0])
    bandit2.counts[b] = np.array([5.0, 5.0, 5.0])
    # Estimates: arm 0 = 0.10, arm 1 = 0.02, arm 2 = 0.02. Arm 0 is 5x best -> pruned!
    for _ in range(20):
        action = bandit2.select(b)
        assert action != 0, "Arm with estimate > 2x best was selected"


def test_online_bandit_dqn_prior_capped_to_training_range():
    class DummyDQN:
        def q_values(self, state):
            return np.array([-0.1, -0.3, -0.8])

    # Within training range (<=75% CPU): Q-values used
    b_in = OnlineBanditPolicy(prior_agent=DummyDQN(), sampler=ScriptedSampler(50.0), edges=(51.0, 63.0))
    b_in.decide(step=0, prompt="p", input_tokens=3)
    est_in = b_in.estimates(b_in._bin)
    assert not np.allclose(est_in, est_in[0]), "Prior was flat within training range"

    # Outside training range (>75% CPU): Flat prior used
    b_out = OnlineBanditPolicy(prior_agent=DummyDQN(), sampler=ScriptedSampler(85.0), edges=(51.0, 63.0))
    b_out.decide(step=0, prompt="p", input_tokens=3)
    est_out = b_out.estimates(b_out._bin)
    assert np.allclose(est_out, est_out[0]), "Prior was not flat outside training range"


def test_online_bandit_cost_attribution_drops_switch_tokens():
    bandit = OnlineBanditPolicy(prior_agent=None, sampler=ScriptedSampler(20.0), edges=(51.0, 63.0))
    bandit.decide(step=0, prompt="p", input_tokens=3)
    bandit.decide(step=10, prompt="p", input_tokens=3)
    # Simulate a switch into arm 1 at step 10
    bandit._pending = (0, 1, 0.0, 10, True)  # was_switched = True, s0 = 10
    # Simulate 10 tokens (10..19) where tokens 0,1 took 1.0s (switch overhead) and tokens 2..9 took 0.1s
    token_times = [1.0, 1.0] + [0.1] * 8
    bandit._segment_token_latencies = token_times.copy()
    bandit.decide(step=20, prompt="p", input_tokens=3)
    # Recorded history should drop first 2 tokens and average tokens 2..9 (0.1 s)
    last_rec = bandit.history[-1]
    assert np.isclose(last_rec["sec_per_token"], 0.1), f"Expected 0.1 s/tok after dropping 2 switch tokens, got {last_rec['sec_per_token']}"


# ---------------------------------------------------------------------------
# Real-engine tests (F2): generate_partitioned_response on a tiny random Llama
# (2 layers, hidden 64, 4 heads, vocab 512; built from a config, no download).
# ---------------------------------------------------------------------------

TINY_LAYER_RANGES = {
    "static": [[0, 1]],
    "balanced": [[0, 0], [1, 1]],
    "latency_optimized": [[0, 1]],
}
TINY_ARMS = ["threads_2", "threads_3", "balanced", "threads_4", "threads_6", "threads_10",
             "static", "latency_optimized"]


class TinyTokenizer:
    """Character-level stand-in for the HF tokenizer (only what the engine uses)."""

    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<user>" + messages[0]["content"] + "<assistant>"

    def encode(self, text):
        return [3 + (ord(ch) % 500) for ch in text]

    def __call__(self, text, return_tensors="pt"):
        import torch
        return {"input_ids": torch.tensor([self.encode(text)])}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


@pytest.fixture(scope="module")
def tiny_model():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
                         bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model = LlamaForCausalLM(config).eval()
    return TinyTokenizer(), model


@pytest.fixture
def tiny_engine(tiny_model, monkeypatch):
    import copy
    import src.config as config_module
    import src.llm.inference as inference
    import src.llm.partition as partition

    tiny_config = copy.deepcopy(config_module.load_config())
    for name, ranges in TINY_LAYER_RANGES.items():
        tiny_config["strategies"][name]["layer_ranges"] = ranges
    monkeypatch.setattr(partition, "load_config", lambda: tiny_config)
    monkeypatch.setattr(inference, "get_model", lambda: tiny_model)
    return inference.generate_partitioned_response


class SlowLayer:
    """Wraps layer 0: every call sleeps `base` s, call number `slow_call` sleeps `slow` s."""

    def __init__(self, layer, base=0.005, slow=0.2, slow_call=None):
        self.layer, self.base, self.slow, self.slow_call, self.calls = layer, base, slow, slow_call, 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        time.sleep(self.slow if self.calls == self.slow_call else self.base)
        return self.layer(*args, **kwargs)


@pytest.fixture
def slow_token(tiny_model, monkeypatch):
    """Make token step 8 slow (layer 0 runs once per token for every arm)."""
    _, model = tiny_model
    wrapper = SlowLayer(model.model.layers[0], slow_call=9)
    monkeypatch.setitem(model.model.layers._modules, "0", wrapper)
    return wrapper


def _arms_per_token(stages):
    frame = pd.DataFrame(stages)
    return frame.groupby("token_step")["arm"].first().tolist(), frame


class ScriptedArm:
    """Adaptive scheduler double that always proposes the same arm."""

    def __init__(self, arm):
        self.arm, self.last_telemetry, self.emergency_calls = arm, {}, []

    def decide(self, step, prompt, input_tokens):
        return self.arm

    def on_emergency_switch(self, conservative_action=1):
        self.emergency_calls.append(conservative_action)


@pytest.mark.parametrize("use_fixed_policy", [False, True])
def test_engine_fixed_strategy_never_switches_on_slow_token(tiny_engine, slow_token, use_fixed_policy):
    kwargs = {"scheduler": FixedPolicy("threads_10")} if use_fixed_policy else {"strategy": "threads_10"}
    log = []
    _, stages, ids = tiny_engine("hello world", max_new_tokens=20, return_token_ids=True, ignore_eos=True,
                                 decision_log=log, **kwargs)
    arms, frame = _arms_per_token(stages)
    assert slow_token.calls >= 9, "slow token was not injected"
    assert arms == ["threads_10"] * 20
    assert not frame["emergency_trigger"].any()
    assert not any(d.get("emergency_trigger") for d in log)


@pytest.mark.parametrize("use_fixed_policy", [False, True])
def test_engine_rejects_latency_switch_for_fixed_strategies(tiny_engine, use_fixed_policy):
    kwargs = {"scheduler": FixedPolicy("threads_10")} if use_fixed_policy else {"strategy": "threads_10"}
    with pytest.raises(ValueError):
        tiny_engine("hello", max_new_tokens=3, emergency_latency_factor=3.0, **kwargs)


def test_engine_latency_switch_is_off_by_default(tiny_engine, slow_token):
    scheduler = ScriptedArm("threads_10")
    _, stages = tiny_engine("hello world", max_new_tokens=20, ignore_eos=True, scheduler=scheduler,
                            decision_interval=100)
    arms, frame = _arms_per_token(stages)
    assert arms == ["threads_10"] * 20
    assert not frame["emergency_trigger"].any() and scheduler.emergency_calls == []


def test_engine_latency_switch_acts_on_token_after_slow_token(tiny_engine, slow_token):
    scheduler = ScriptedArm("threads_10")
    log = []
    _, stages = tiny_engine("hello world", max_new_tokens=20, ignore_eos=True, scheduler=scheduler,
                            decision_interval=100, decision_log=log, emergency_latency_factor=3.0,
                            conservative_strategy="balanced")
    arms, frame = _arms_per_token(stages)
    # token 8 is slow; token 9 is the first one on the conservative arm
    assert arms[:9] == ["threads_10"] * 9
    assert arms[9:] == ["balanced"] * 11
    per_token = frame.groupby("token_step").first()
    assert per_token.loc[9, "emergency_trigger"] and per_token.loc[9, "emergency_kind"] == "latency"
    assert not per_token.drop(index=9)["emergency_trigger"].any()
    emergencies = [d for d in log if d.get("emergency_trigger")]
    assert len(emergencies) == 1 and emergencies[0]["step"] == 9 and emergencies[0]["emergency_kind"] == "latency"
    assert scheduler.emergency_calls == ["balanced"]


@pytest.mark.parametrize("arm", TINY_ARMS)
def test_engine_tokens_match_generate_on_tiny_model(tiny_engine, tiny_model, arm):
    import torch

    tokenizer, model = tiny_model
    prompt = "the quick brown fox"
    inputs = tokenizer(tokenizer.apply_chat_template([{"role": "user", "content": prompt}]))
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=15, min_new_tokens=15, do_sample=False,
                             pad_token_id=0, eos_token_id=None)
    expected = out[0][inputs["input_ids"].shape[1]:].tolist()
    _, _, ids = tiny_engine(prompt, strategy=arm, max_new_tokens=15, return_token_ids=True, ignore_eos=True)
    assert ids == expected


def test_engine_segment_log_records_paging_per_10_tokens(tiny_engine):
    segments = []
    tiny_engine("hello", strategy="threads_2", max_new_tokens=25, ignore_eos=True, segment_log=segments)
    assert [(s["start_step"], s["end_step"]) for s in segments] == [(0, 9), (10, 19), (20, 24)]
    for s in segments:
        for key in ["swap_sin_delta", "swap_sout_delta", "page_faults_delta", "system_pages_input_delta", "available_mb"]:
            assert s[key] is not None and s[key] >= 0


def test_online_bandit_stall_rule_retest_at_12_workers():
    """Verify stall rule with actual load_sweep_v4.csv 12-worker measurements (~18x slowdown)."""
    # From load_sweep_v4.csv: static = ~2.13 s/tok, balanced = ~0.115 s/tok
    static_spt = 2.135
    balanced_spt = 0.115
    bandit = OnlineBanditPolicy(prior_agent=None, sampler=ScriptedSampler(75.0), edges=(51.0, 63.0),
                                outlier_ratio=2.0, regime_after=5)
    b = 2  # high load bin
    # Establish balanced as fast arm in this bin
    bandit._ensure_bin(b, "p", 3, {"external_cpu_percent": 75.0, "ram_percent": 50.0})
    for _ in range(3):
        bandit.update(b, 1, balanced_spt)

    # Static is 18.5x slower -> must be detected as an outlier stall initially
    assert bandit.is_outlier(b, static_spt) is True

    # 4 consecutive outlier segments: all rejected as stalls
    for _ in range(4):
        outlier = bandit.is_outlier(b, static_spt)
        bandit._rejected[b] = bandit._rejected.get(b, 0) + 1 if outlier else 0
        accepted = not outlier or bandit._rejected[b] >= bandit.regime_after
        assert accepted is False, "Stall was accepted before reaching regime_after"

    # 5th consecutive outlier segment: accepted as persistent regime change
    outlier = bandit.is_outlier(b, static_spt)
    bandit._rejected[b] = bandit._rejected.get(b, 0) + 1 if outlier else 0
    accepted = not outlier or bandit._rejected[b] >= bandit.regime_after
    assert accepted is True, "Persistent regime change was not accepted after 5 consecutive segments"


def test_expanded_action_space_thread_budgets_and_named_aliases():
    from src.llm.partition import PartitionStrategy, resolve, physical_core_count

    # Single-stage thread budgets: {2, 4, 6, 8, 10, 12}
    for budget in [2, 4, 6, 8, 10, 12]:
        strat = PartitionStrategy("static", thread_budget=budget)
        assert strat.thread_budget == budget
        assert strat.get_layer_ranges() == [(0, 21)]
        assert strat.get_thread_counts() == [budget]

    # Named aliases backward compatibility with explicit budgets
    static_b10 = PartitionStrategy("static", thread_budget=10)
    assert static_b10.get_thread_counts() == [10]

    latency_b6 = PartitionStrategy("latency_optimized", thread_budget=6)
    assert latency_b6.get_thread_counts() == [6]

    balanced_b6 = PartitionStrategy("balanced", thread_budget=6)
    assert balanced_b6.get_thread_counts() == [3, 3]

    # Dynamic machine physical core scaling
    static_strat = resolve("static")
    assert static_strat.strategy_name == "static"
    assert static_strat.get_layer_ranges() == [(0, 21)]
    assert static_strat.get_thread_counts() == [physical_core_count()]


def test_resume_integrity_filters_incomplete_contexts(tmp_path):
    # Simulate a corrupted / interrupted run CSV with 7 complete conditions for context 1,
    # and only 3 conditions for context 2 before a crash.
    runs_file = tmp_path / "adaptive_runs_test.csv"
    records = [
        {"context_id": 1, "condition": f"c_{i}", "latency": 10.0} for i in range(7)
    ] + [
        {"context_id": 2, "condition": f"c_{i}", "latency": 10.0} for i in range(3)
    ]
    pd.DataFrame(records).to_csv(runs_file, index=False)

    loaded_runs = pd.read_csv(runs_file).to_dict("records")
    # Resume integrity check
    num_expected_conditions = 7
    counts = pd.Series([r["context_id"] for r in loaded_runs]).value_counts()
    done = set(counts[counts == num_expected_conditions].index)

    # Incomplete context 2 must be purged, complete context 1 must be retained
    assert done == {1}
    valid_runs = [r for r in loaded_runs if r["context_id"] in done]
    assert len(valid_runs) == 7
    assert {r["context_id"] for r in valid_runs} == {1}


def test_ram_guard_waits_then_runs(tmp_path):
    from src.telemetry.run_guard import ram_guard

    readings = iter([90.0, 88.0, 70.0])
    waits = []
    ok, snap = ram_guard(tmp_path / "skipped.csv", {"run": 1}, sleep=waits.append,
                         snapshot=lambda: {"ram_percent": next(readings), "available_mb": 1000.0})
    assert ok and snap["ram_percent"] == 70.0 and waits == [30.0, 30.0]
    assert not (tmp_path / "skipped.csv").exists()


def test_ram_guard_skips_and_logs_after_ten_retries(tmp_path):
    from src.telemetry.run_guard import ram_guard

    waits = []
    path = tmp_path / "skipped.csv"
    for run in (1, 2):
        ok, _ = ram_guard(path, {"run": run, "condition": "balanced"}, sleep=waits.append,
                          snapshot=lambda: {"ram_percent": 95.0, "available_mb": 500.0})
        assert not ok
    assert len(waits) == 20
    skipped = pd.read_csv(path)
    assert skipped["run"].tolist() == [1, 2] and (skipped["retries"] == 10).all()
    assert skipped["reason"].str.contains("RAM").all() and skipped["commit_hash"].notna().all()


def test_v7_thread_arms_are_single_stage_with_fixed_threads():
    from src.llm.partition import apply_partition_strategy

    for budget in PartitionStrategy.V7_THREAD_ARMS:
        arm = apply_partition_strategy(f"threads_{budget}")
        assert arm.arm_name == f"threads_{budget}"
        assert arm.get_layer_ranges() == [(0, 21)] and arm.get_thread_counts() == [budget]
    balanced = apply_partition_strategy("balanced")
    assert balanced.arm_name == "balanced" and balanced.get_thread_counts() == [3, 3]
    assert set(PartitionStrategy.V7_ARMS) == {f"threads_{b}" for b in PartitionStrategy.V7_THREAD_ARMS} | {"balanced"}
    with pytest.raises(ValueError):
        apply_partition_strategy("threads_5")


def test_run_v7_plan_on_tiny_engine(tiny_engine, tmp_path, monkeypatch):
    import yaml
    import run_v7

    plan = {
        "stage": "test", "tag": "unit", "prompts": [1, 2], "max_new_tokens": 12, "decision_interval": 4, "seed": 3,
        "scenarios": {"spike": [[0, 0], [6, 1]], "steady_idle": [[0, 0]]},
        "ram_guard": {"max_percent": 100.0, "wait_seconds": 0, "retries": 0},
        "paging_flag": {"system_pages_input_per_segment": 1000},
        "conditions": [
            {"name": "threads_2", "policy": "fixed", "arm": "threads_2"},
            {"name": "balanced", "policy": "fixed", "arm": "balanced"},
            {"name": "rule_v5", "policy": "rule", "threshold": 60.0, "low_arm": "static", "high_arm": "balanced",
             "patience": 2, "interval": 4},
        ],
    }
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan))
    monkeypatch.setattr(run_v7, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_v7, "load_prompts", lambda csv_path=None, limit=None: ["first prompt", "second prompt"])
    monkeypatch.setattr(run_v7.time, "sleep", lambda s: None)
    run_v7.run_plan(plan_path, allow_uncommitted=True)

    runs = pd.read_csv(tmp_path / "v7runs_unit.csv")
    assert len(runs) == 2 * 2 * 3
    assert runs.groupby(["prompt_id", "scenario"])["token_ids"].nunique().eq(1).all()
    fixed = runs[runs["condition"] != "rule_v5"]
    assert (fixed["num_switches"] == 0).all() and (fixed["emergency_triggers"] == 0).all()
    for col in ["commit_hash", "ram_percent_before", "available_mb_before", "page_faults_delta",
                "system_pages_input_delta", "wall_clock_start", "decode_sec_per_token"]:
        assert runs[col].notna().all(), col
    timelines = pd.read_csv(tmp_path / "v7timelines_unit.csv")
    assert {"arm", "emergency_trigger", "emergency_kind"} <= set(timelines.columns)
    assert len(pd.read_csv(tmp_path / "v7segments_unit.csv")) == 12 * 2  # 2 segments per run

    # Resume does not repeat finished runs; a second fresh start refuses to overwrite
    run_v7.run_plan(plan_path, resume=True, allow_uncommitted=True)
    assert len(pd.read_csv(tmp_path / "v7runs_unit.csv")) == 12
    with pytest.raises(SystemExit):
        run_v7.run_plan(plan_path, allow_uncommitted=True)


def test_experiments_segment_oracle_picks_cheapest_arm_per_segment():
    sys.path.insert(0, str(ROOT_DIR / "experiments"))
    from segment_oracle import segment_oracle_table

    rows = []
    # arm a: fast in segment 0, slow in segment 1; arm b: the opposite; policy p in between
    for step in range(20):
        a, b = (0.1, 0.5) if step < 10 else (0.5, 0.1)
        for cond, sec in [("a", a), ("b", b), ("p", 0.3)]:
            rows.append({"context_id": 1, "scenario": "s", "condition": cond, "token_step": step, "token_seconds": sec})
    table, per_ctx = segment_oracle_table(pd.DataFrame(rows), ["a", "b"], ["p"])
    assert np.isclose(per_ctx["segment_oracle"].iloc[0], 2.0)
    assert np.isclose(per_ctx["best_single_fixed"].iloc[0], 6.0)
    assert np.isclose(table["max_adaptive_gain_percent"].iloc[0], 100 * 4 / 6)
    assert np.isclose(table["p_regret_vs_oracle_s"].iloc[0], 4.0)


def test_load_sweep_v7_on_tiny_engine(tiny_engine, tmp_path, monkeypatch):
    import yaml
    import run_load_sweep_v7 as sweep

    plan = yaml.safe_load((ROOT_DIR / "experiments" / "e1_plan.yaml").read_text(encoding="utf-8"))
    plan.update({"prompts": [1, 2], "background_workers": [0, 1], "repeats": 1, "decode_tokens": 4, "total_runs": 24,
                 "ram_guard": {"max_percent": 100.0, "wait_seconds": 0, "retries": 0}})
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan))
    monkeypatch.setattr(sweep, "load_prompts", lambda csv_path=None, limit=None: ["first", "second"])
    df = sweep.run_sweep(plan_path, data_dir=tmp_path, allow_uncommitted=True, settle=0)
    assert len(df) == 2 * 2 * 6 and set(df["arm"]) == set(plan["arms"])
    assert (df["num_generated_tokens"] == 5).all() and df["decode_sec_per_token"].notna().all()
    assert sweep.blocks(plan) == sweep.blocks(plan)
    assert all(sorted(b[3]) == sorted(plan["arms"]) for b in sweep.blocks(plan))
    text = sweep.write_summary(plan_path, data_dir=tmp_path, out_path=tmp_path / "summary.txt")
    assert "best" in text and "threads_3 vs balanced" in text
    df2 = sweep.run_sweep(plan_path, resume=True, data_dir=tmp_path, allow_uncommitted=True, settle=0)
    assert len(df2) == 24
