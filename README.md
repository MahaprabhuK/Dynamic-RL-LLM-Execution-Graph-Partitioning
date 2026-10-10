# Dynamic RL-Based LLM Execution Graph Partitioning

A reinforcement-learning scheduler that picks how to execute an LLM request on a
single machine. TinyLlama-1.1B is run layer by layer through a hand-written
decode loop, and the 22 decoder layers are split into stages. A contextual-bandit
DQN observes the request and the current system load and chooses the execution
strategy that it expects to be fastest.

## Execution strategies

Each strategy is a real execution configuration (`configs/config.yaml`): a layer
split plus an intra-op thread budget per stage, as a fraction of physical cores.

| ID | Strategy | Layer stages | Threads per stage (10-core CPU) |
|----|----------|--------------|---------------------------------|
| 0 | `static` | 0-21 | 10 |
| 1 | `balanced` | 0-10, 11-21 | 3, 3 |
| 2 | `latency_optimized` | 0-21 | 6 |

All strategies run in float32 and are checked to produce token-identical output
to `model.generate()` (`tests/test_functional_equivalence.py`). On an idle
machine more threads tend to be faster; under background CPU contention fewer
threads avoid oversubscription. That trade-off is what the scheduler learns.

## Project layout

```
configs/config.yaml          model + strategy definitions
src/data/                    Alpaca download, preprocessing, prompt loading
src/llm/                     model loading, partition strategies, layer-sliced inference
src/graph/                   execution graph (networkx) annotated with partitions
src/telemetry/               CPU/RAM metrics, activation handoffs, background load generator
src/rl/                      features, reward, offline bandit environment, DQN agent, training
src/scheduler/               RL and static schedulers
src/benchmark/               bootstrap statistics and plots
evaluate_fixed_strategies.py collects the paired v3 dataset
evaluate_rl_policy.py        online evaluation of the trained policy
evaluate_adaptive.py         re-scheduling during generation under changing load
tests/                       fast pipeline tests + model equivalence test
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python -m src.data.download_dataset
python -m src.data.preprocess_dataset
```

## Workflow

```bash
# 1. Verify that every strategy reproduces model.generate() exactly (loads the model)
python -m pytest tests/test_functional_equivalence.py -v

# 2. Collect paired measurements: prompts x repeats x background-load levels.
#    Each context runs all strategies back-to-back in random order.
#    20 prompts x 2 repeats x 3 loads x 3 strategies = 360 runs (~2-3 h on CPU).
python evaluate_fixed_strategies.py --num-prompts 20 --repeats 2 --loads 0,4,8

# 3. Train the contextual-bandit DQN on that data (seconds)
python -m src.rl.train --csv data/processed/fixed_strategy_metrics_v3.csv --tag v3

# 4. Evaluate the policy online on unseen prompts (prompts 21-30)
python evaluate_rl_policy.py --prompt-offset 20 --num-prompts 10 --loads 0,4,8

# 5. Intra-generation re-scheduling under changing load (see below)
python evaluate_adaptive.py --num-prompts 3 --prompt-offset 20 --max-new-tokens 60

# Fast tests (no model needed)
python -m pytest tests/test_pipeline.py tests/test_adaptive.py
```

## Intra-generation re-scheduling

Per-request schedulers choose a strategy once, before the first token. If the
system load changes while a long response is being generated, that choice goes
stale. Because every strategy here shares the same weights and KV cache,
the engine can switch strategy *between tokens* at no cost and with identical
output. `generate_partitioned_response(scheduler=..., decision_interval=N)`
asks a policy (`src/scheduler/adaptive.py`) for a strategy every N tokens.

- Live telemetry (`ExternalLoadMonitor`) subtracts this process's own CPU usage
  from system CPU, so the policy reacts to *external* contention rather than to
  its own inference threads, matching the pre-run feature it was trained on.
- `evaluate_adaptive.py` replays load scenarios that change mid-request
  (`steady_idle`, `steady_heavy`, `spike`, `relief`; `DynamicCPULoad`) and
  compares fixed strategies, a threshold rule and the DQN, each run once per
  request (`*_once`) and re-deciding during generation (`*_adaptive`).
- Outputs: `data/processed/adaptive_*_v3.csv` and `results/adaptive_*_v3.png`,
  including a per-token timeline of a load spike.

Outputs go to `data/processed/*_v3.csv` and `results/*_v3.png`.

## Methodology notes

- **Paired, randomized measurements.** Strategies are compared within the same
  context, in shuffled order, after warm-up. This avoids the run-order and
  system-drift confound of v1/v2, where all runs of one strategy were done
  before the next.
- **Pre-decision state.** `[prompt_length, input_tokens, cpu%, ram%]` is measured
  *before* the request runs, identically in training and serving
  (`src/rl/features.py`).
- **Contextual bandit.** Requests are independent, so the DQN is trained with
  `gamma=0`. The logged data contains every action's outcome, so all three
  rewards are learned from per context (`--chosen-action-only` disables this).
- **Reward.** Negative seconds per generated token. CPU/RAM penalties are
  available in `src/rl/reward.py` but disabled by default (post-run snapshots
  are noisy).
- **Statistics.** Results report bootstrap 95% CIs and paired per-context
  differences. A difference is only meaningful if its CI excludes zero.
- **Handoff metric.** Activation handoffs are in-process `clone()` timings,
  a lower bound on a real inter-device transfer.

## Data versions

- v1/v2 (`*_v2.csv`): earlier runs in which the strategies were computationally
  equivalent and measured sequentially; differences there are noise. Kept for
  reference.
- v3: the current engine and protocol.

## Notes

- RAM: the fp32 model needs about 4.4 GB. Close other applications, or set
  `model.dtype: bfloat16` in `configs/config.yaml` to halve memory (outputs then
  differ slightly from fp32).
- `torch` in `.venv` is a CPU build; the strategies are designed for CPU-only execution.
