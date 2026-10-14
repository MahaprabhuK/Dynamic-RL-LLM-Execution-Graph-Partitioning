"""
Fast tests that do not load the LLM. Model calls are replaced by a stub, so
the full collection -> training -> online-evaluation pipeline can be checked
in seconds.

    python -m pytest tests/test_pipeline.py -v
"""

import importlib
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.benchmark.evaluate import _holm_adjust, _paired_p_values, bootstrap_ci, paired_differences
from src.graph.execution_graph import build_execution_graph
from src.llm.partition import PartitionStrategy, physical_core_count
from src.rl.environment import LLMPartitionEnv
from src.rl.features import STATE_SIZE, build_state
from src.rl.reward import calculate_reward


def test_strategies_differ_in_execution():
    configs = {
        (tuple(PartitionStrategy(s).get_layer_ranges()), tuple(PartitionStrategy(s).get_thread_counts()))
        for s in PartitionStrategy.STRATEGIES
    }
    assert len(configs) == len(PartitionStrategy.STRATEGIES)


@pytest.mark.parametrize("strategy_id", list(PartitionStrategy.STRATEGIES))
def test_strategy_covers_all_layers_with_valid_threads(strategy_id):
    strat = PartitionStrategy(strategy_id)
    layers = [l for start, end in strat.get_layer_ranges() for l in range(start, end + 1)]
    assert layers == list(range(22))
    threads = strat.get_thread_counts()
    assert len(threads) == len(strat.get_layer_ranges())
    assert all(1 <= t <= physical_core_count() for t in threads)


@pytest.mark.parametrize("thread_budget", PartitionStrategy.THREAD_BUDGETS)
def test_explicit_thread_budget_is_distributed_across_strategy_stages(thread_budget):
    strategy = PartitionStrategy(1, thread_budget=thread_budget)
    threads = strategy.get_thread_counts()
    assert sum(threads) == thread_budget
    assert max(threads) - min(threads) <= 1


def test_graph_carries_strategy_metadata():
    graph = build_execution_graph(1, PartitionStrategy(1))
    assert graph.graph["strategy_id"] == 1
    assert graph.graph["stage_threads"] == PartitionStrategy(1).get_thread_counts()
    assert all("partition" in graph.nodes[n] for n in graph.nodes)


def test_reward_prefers_lower_latency():
    assert calculate_reward(10.0) > calculate_reward(20.0)
    assert calculate_reward(10.0, num_tokens=100) > calculate_reward(10.0, num_tokens=50)
    assert calculate_reward(1000.0) == -1.5


def test_build_state_shape():
    state = build_state(36, 9, 50.0, 90.0)
    assert state.shape == (STATE_SIZE,)
    assert np.allclose(state, [0.18, 0.09, 0.5, 0.9])


def test_paired_differences_detects_consistent_gap():
    rows = []
    for c in range(30):
        base = 10 + c % 5
        rows += [{"context_id": c, "strategy": "static", "latency_seconds": base},
                 {"context_id": c, "strategy": "fast", "latency_seconds": base - 2}]
    diffs = paired_differences(pd.DataFrame(rows), "strategy", baseline="static")
    assert diffs.loc[0, "mean_diff_seconds"] == pytest.approx(-2)
    assert "p_value_raw" in diffs.columns
    assert bool(diffs.loc[0, "significant"])
    mean, lo, hi = bootstrap_ci([1.0, 1.0, 1.0])
    assert mean == lo == hi == 1.0


def test_paired_differences_handles_missing_pairs_and_ties():
    rows = [
        {"context_id": 0, "strategy": "baseline", "latency_seconds": 5.0},
        {"context_id": 0, "strategy": "candidate", "latency_seconds": 3.0},
        {"context_id": 1, "strategy": "baseline", "latency_seconds": 5.0},
        {"context_id": 1, "strategy": "candidate", "latency_seconds": 5.0},
        {"context_id": 2, "strategy": "baseline", "latency_seconds": 9.0},
        {"context_id": 3, "strategy": "baseline", "latency_seconds": 8.0},
        {"context_id": 3, "strategy": "candidate", "latency_seconds": 7.0},
    ]
    diffs = paired_differences(pd.DataFrame(rows), "strategy", baseline="baseline")
    assert diffs.loc[0, "num_contexts"] == 3
    assert diffs.loc[0, "mean_diff_seconds"] == pytest.approx(-1.0, abs=1e-3)
    assert "p_value_adjusted" in diffs.columns
    assert diffs.loc[0, "significant"] in {True, False}


def test_paired_differences_no_signal_keeps_non_significant_result():
    rows = []
    for c in range(20):
        base = 10.0 + (c % 3)
        direction = 1 if c % 2 == 0 else -1
        rows.append({"context_id": c, "strategy": "baseline", "latency_seconds": base})
        rows.append({"context_id": c, "strategy": "candidate", "latency_seconds": base + 0.15 * direction})
    diffs = paired_differences(pd.DataFrame(rows), "strategy", baseline="baseline")
    assert not bool(diffs.loc[0, "significant"])
    assert diffs.loc[0, "p_value_adjusted"] > 0.05


def test_holm_correction_on_small_family():
    diffs = paired_differences(pd.DataFrame([
        {"context_id": 0, "strategy": "baseline", "latency_seconds": 1.0},
        {"context_id": 0, "strategy": "a", "latency_seconds": 0.5},
        {"context_id": 1, "strategy": "baseline", "latency_seconds": 1.0},
        {"context_id": 1, "strategy": "a", "latency_seconds": 0.7},
        {"context_id": 2, "strategy": "baseline", "latency_seconds": 1.0},
        {"context_id": 2, "strategy": "a", "latency_seconds": 0.3},
        {"context_id": 0, "strategy": "baseline", "latency_seconds": 1.2},
        {"context_id": 0, "strategy": "b", "latency_seconds": 0.4},
        {"context_id": 1, "strategy": "baseline", "latency_seconds": 1.2},
        {"context_id": 1, "strategy": "b", "latency_seconds": 0.5},
        {"context_id": 2, "strategy": "baseline", "latency_seconds": 1.2},
        {"context_id": 2, "strategy": "b", "latency_seconds": 0.2},
    ]), "strategy", baseline="baseline")
    assert set(diffs["condition"].unique()) == {"a", "b"}
    assert diffs["p_value_adjusted"].between(0, 1).all()
    assert diffs["p_value_adjusted"].notna().all()


def test_paired_tests_match_known_tied_and_zero_difference_examples():
    tied_diff = pd.Series([0.0, 0.0, -1.0, -1.0, 1.0, 1.0])
    t_value, wilcoxon_value = _paired_p_values(tied_diff, pd.Series(np.zeros(len(tied_diff))))
    assert t_value == pytest.approx(1.0)
    assert wilcoxon_value == pytest.approx(1.0)

    zero_diff = pd.Series([0.0, 0.0, 0.0])
    assert _paired_p_values(zero_diff, pd.Series(np.zeros(3))) == (1.0, 1.0)


def test_paired_t_and_wilcoxon_match_independent_reference_values():
    differences = pd.Series([1.0, 2.0, 3.0, 4.0])
    t_value, wilcoxon_value = _paired_p_values(differences, pd.Series(np.zeros(4)))
    assert t_value == pytest.approx(0.030466291662170977)
    assert wilcoxon_value == pytest.approx(0.125)


def test_paired_tests_return_nan_for_one_complete_pair():
    t_value, wilcoxon_value = _paired_p_values(pd.Series([2.0]), pd.Series([1.0]))
    assert np.isnan(t_value)
    assert np.isnan(wilcoxon_value)


def test_holm_values_and_family_scope_are_explicit():
    assert np.allclose(_holm_adjust([0.01, 0.03, 0.2]), [0.03, 0.06, 0.2])
    assert np.allclose(_holm_adjust([0.01, np.nan, 0.03]), [0.02, np.nan, 0.03], equal_nan=True)

    rows = []
    for context, baseline, a, b in [
        (0, 10.0, 8.0, 9.0),
        (1, 12.0, 10.0, 11.0),
        (2, 14.0, 12.0, 13.0),
        (3, 16.0, 14.0, 15.0),
    ]:
        rows.extend([
            {"context_id": context, "strategy": "baseline", "latency_seconds": baseline},
            {"context_id": context, "strategy": "a", "latency_seconds": a},
            {"context_id": context, "strategy": "b", "latency_seconds": b},
        ])
    diffs = paired_differences(pd.DataFrame(rows), "strategy", baseline="baseline")
    expected = _holm_adjust(diffs["p_value_t_raw"].to_numpy())
    assert np.allclose(diffs["p_value_t_adjusted"], expected)
    assert np.allclose(diffs["p_value_adjusted"], diffs["p_value_t_adjusted"])


def test_environment_reads_v2_csv():
    v2 = ROOT_DIR / "data" / "processed" / "fixed_strategy_metrics_v2.csv"
    if not v2.exists():
        pytest.skip("v2 data not present")
    env = LLMPartitionEnv(str(v2), mode="train")
    state = env.reset()
    assert state.shape == (STATE_SIZE,)
    assert set(env.action_rewards()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# End-to-end pipeline with a stubbed LLM
# ---------------------------------------------------------------------------

class FakeTokenizer:
    def encode(self, text):
        return text.split()


def fake_generate(prompt, strategy=0, max_new_tokens=100, do_sample=False, return_token_ids=False):
    strat = strategy if isinstance(strategy, PartitionStrategy) else PartitionStrategy(strategy)
    time.sleep(0.001 * (1 + strat.strategy_id))
    ids = list(range(min(max_new_tokens, 5)))
    stages = [{"partition_id": i + 1, "layer_start": s, "layer_end": e, "num_threads": t,
               "stage_latency_seconds": 0.001, "activation_size_bytes": 0, "activation_size_mb": 0.0,
               "handoff_latency_seconds": 0.0}
              for i, ((s, e), t) in enumerate(zip(strat.get_layer_ranges(), strat.get_thread_counts()))]
    if return_token_ids:
        return "fake response", stages, ids
    return "fake response", stages


@pytest.fixture
def stubbed_workspace(tmp_path, monkeypatch):
    fake_inference = types.ModuleType("src.llm.inference")
    fake_inference.generate_partitioned_response = fake_generate
    fake_inference.get_tokenizer = lambda: FakeTokenizer()
    monkeypatch.setitem(sys.modules, "src.llm.inference", fake_inference)
    sys.modules.pop("evaluate_fixed_strategies", None)
    sys.modules.pop("evaluate_rl_policy", None)
    efs = importlib.import_module("evaluate_fixed_strategies")
    erp = importlib.import_module("evaluate_rl_policy")

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    pd.DataFrame({"prompt_id": range(1, 13), "prompt": [f"prompt number {i} " * (i % 4 + 1) for i in range(12)]}) \
        .to_csv(data_dir / "prompts.csv", index=False)

    for module in (efs, erp):
        monkeypatch.setattr(module, "ROOT_DIR", tmp_path)
        monkeypatch.setattr(module, "collect_metrics", lambda interval=0.1: {
            "timestamp": "", "cpu_percent": 20.0, "ram_percent": 80.0, "ram_used_gb": 1.0, "ram_available_gb": 1.0})
    monkeypatch.setattr(efs, "_load_inference", lambda: (fake_generate, lambda: FakeTokenizer()))
    monkeypatch.setattr(erp, "_load_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(erp, "_load_dqn_agent", lambda model_path: type("StubAgent", (), {"epsilon": 0.0, "choose_action": lambda self, state: 0, "load": lambda self, *args, **kwargs: None})())
    monkeypatch.chdir(tmp_path)
    return tmp_path, efs, erp


def test_end_to_end_pipeline(stubbed_workspace, monkeypatch):
    tmp_path, efs, erp = stubbed_workspace

    fake_train = types.ModuleType("src.rl.train")

    def fake_train_dqn(episodes=3, seed=42, csv_path="data/processed/fixed_strategy_metrics_v3.csv", tag="v3", full_feedback=True):
        model_path = Path(csv_path).parent / f"dqn_model_{tag}.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"stub-model")
        hist = pd.DataFrame({"episode": list(range(1, episodes + 1)), "total_reward": [0.0] * episodes})
        val = pd.DataFrame({"context_id": [1], "rl_strategy": ["static"], "regret": [0.0]})
        return object(), hist, val

    fake_train.train_dqn = fake_train_dqn
    monkeypatch.setitem(sys.modules, "src.rl.train", fake_train)
    from src.rl.train import train_dqn

    df = efs.run_fixed_strategy_evaluation(num_prompts=10, repeats=1, loads=[0], max_new_tokens=5, tag="t")
    assert len(df) == 10 * 3
    assert df.groupby("context_id")["strategy_id"].nunique().eq(3).all()
    assert df["output_matches_static"].all()

    csv_path = tmp_path / "data" / "processed" / "fixed_strategy_metrics_t.csv"
    agent, hist, val = train_dqn(episodes=3, csv_path=str(csv_path), tag="t")
    assert len(hist) == 3 and len(val) > 0
    assert (tmp_path / "data" / "processed" / "dqn_model_t.pth").exists()

    df_eval, summary, diffs = erp.run_rl_policy_evaluation(
        model_path="data/processed/dqn_model_t.pth", num_prompts=2, prompt_offset=10,
        loads=[0], max_new_tokens=5, tag="t")
    assert set(df_eval["condition"]) == set(erp.CONDITIONS)
    assert len(diffs) == 3
