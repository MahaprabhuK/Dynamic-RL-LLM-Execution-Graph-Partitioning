"""
Fast tests that do not load the LLM. Model calls are replaced by a stub, so
the full collection -> training -> online-evaluation pipeline can be checked
in seconds.

    python -m pytest tests/test_pipeline.py -v
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.benchmark.evaluate import bootstrap_ci, paired_differences
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
    assert diffs.loc[0, "significant"]
    mean, lo, hi = bootstrap_ci([1.0, 1.0, 1.0])
    assert mean == lo == hi == 1.0


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
    import evaluate_fixed_strategies as efs
    import evaluate_rl_policy as erp

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    pd.DataFrame({"prompt_id": range(1, 13), "prompt": [f"prompt number {i} " * (i % 4 + 1) for i in range(12)]}) \
        .to_csv(data_dir / "prompts.csv", index=False)

    for module in (efs, erp):
        monkeypatch.setattr(module, "ROOT_DIR", tmp_path)
        monkeypatch.setattr(module, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(efs, "generate_partitioned_response", fake_generate)
    monkeypatch.setattr(efs, "collect_metrics", lambda interval=0.1: {
        "timestamp": "", "cpu_percent": 20.0, "ram_percent": 80.0, "ram_used_gb": 1.0, "ram_available_gb": 1.0})
    monkeypatch.setattr(erp, "collect_metrics", efs.collect_metrics)
    monkeypatch.chdir(tmp_path)
    return tmp_path, efs, erp


def test_end_to_end_pipeline(stubbed_workspace):
    from src.rl.train import train_dqn

    tmp_path, efs, erp = stubbed_workspace

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
