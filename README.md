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
#    10 prompts x 4 scenarios x 7 conditions, 200 tokens each (~2.5 h on CPU)
python evaluate_adaptive.py --num-prompts 10 --prompt-offset 20 --max-new-tokens 200

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

- Live telemetry (`BackgroundLoadSampler`) runs in a daemon thread and
  subtracts this process's own CPU usage from system CPU, so the policy reacts
  to *external* contention rather than to its own inference threads. Reading
  it never blocks generation.
- Hysteresis (`patience`): a policy only switches after proposing the new
  strategy in consecutive decisions, which prevents flapping on noisy readings.
- **Online bandit (`OnlineBanditPolicy`).** Each decision interval is one
  bandit round: context = external-load bin, arm = strategy, cost = the
  measured seconds per token of the segment just generated. It is warm-started
  from the offline DQN's Q-values, uses discounted estimates so stale data is
  forgotten when the machine's behaviour drifts, and re-checks other
  strategies with a UCB-style bonus, at the cost of one segment. One instance
  keeps learning across all requests.
- `evaluate_adaptive.py` replays load scenarios whose load changes after a
  fixed number of generated tokens (`steady_idle`, `steady_heavy`, `spike`,
  `relief`; `TokenScheduledLoad`), so every policy faces the change at the
  same point. All runs generate exactly `--max-new-tokens` tokens. It compares
  the three fixed strategies, a threshold rule, the DQN once per request
  (`rl_once`) and re-deciding (`rl_adaptive`), and the online bandit.
- Outputs: `data/processed/adaptive_*_v4.csv` and `results/adaptive_*_v4.png`,
  including per-token timelines of a load spike and a load relief.

### v4 results (10 prompts x 4 scenarios, 200 tokens, paired)

Two contexts with memory-paging stalls (a run > 2x the scenario median) are
excluded by `stalled_contexts()`; raw data is kept. Generated tokens were
identical across all conditions in 100% of contexts.

- The online bandit is significantly faster than the per-request DQN
  (`rl_once`) when load changes mid-request: -7.8% in `spike`, -7.0% in
  `relief`. It is also significantly faster than `rl_adaptive` in `spike` and
  `steady_heavy`, and than the threshold rule in three scenarios.
- It corrected a stale offline model. The v3 data said balanced (3+3 threads)
  wins under load. On the machine at evaluation time, the 6-thread
  `latency_optimized` strategy won everywhere; the DQN kept choosing balanced,
  while the bandit moved to it. Its regret against the best fixed strategy in
  hindsight is 0.3-1.2 s per request, against 0.9-2.1 s for the DQN, and fell
  from 0.94 s to 0.66 s between the first and second half of the run.
- Limitation: because one fixed strategy happened to be near-optimal in every
  scenario, the bandit only ties it in `spike` and `steady_idle` and is 4-6%
  slower in `relief` and `steady_heavy`, the cost of its exploration
  (about 3 switches per request). The bandit's value here is not knowing in
  advance which fixed strategy is best, which changed between v3 and v4.
- Decision overhead fell from 0.21 s (v3) to 0.002 s per request.

Re-analyse saved results with `python evaluate_adaptive.py --tag v4 --report-only`.

### Load sweep and v5 protocol

`pilot_load_sweep.py` measures decode seconds per token for every strategy at
0-14 busy workers (`data/processed/load_sweep_v4.csv`). On this machine there
are three regimes:

| Busy workers | External CPU reading | Best strategy |
|---|---|---|
| 0-6 | 7-49% | `static` (10 threads) |
| 8 | 54-60% | `latency_optimized` (6 threads) |
| 10-14 | 66-89% | `balanced` (3+3 threads); at 12 workers `static` oversubscribes the CPU and slows down about 20x |

The v5 run adds a `ramp` scenario (idle, then 8 workers from token 60, then 12
from token 140), in which no single fixed strategy is best for the whole
response. It uses 10 workers for the other heavy phases, puts the bandit's
bin edges at 51% and 63%, and sets the rule threshold at 60%:

```bash
python evaluate_adaptive.py --tag v5 --num-prompts 8 --heavy 10 --switch-token 60 \
    --ramp "60:8,140:12" --threshold 60 --bandit-edges "51,63" --bin-margin 4
```

The v4 diagnostics also changed the bandit:

- Memory-paging stalls had poisoned the bandit's estimates. Segments more
  than 2x slower than the best established estimate are now ignored, unless
  they persist, in which case the bin is reset.
- Readings near a bin edge had caused bin flapping, which accounted for 53 of
  114 switches. Bin hysteresis (`bin_margin`) now prevents it.
- Slower forgetting (`discount=0.98`) and a smaller exploration bonus
  (`bonus=0.08`) mean confident arms are re-checked less often.
- The stall exclusion compares each run with the same condition in the same
  scenario, so a policy that is slow every time (oversubscription) is kept.

### v5 results (8 prompts x 5 scenarios, 200 tokens, paired)

Four contexts with paging stalls were excluded (36 left). Generated tokens were
identical across all conditions in 100% of contexts; decision overhead 0.005 s
per request. Full write-up: `docs/report/Project_Report.pdf`.

- Re-deciding pays off when load crosses the oversubscription cliff. In `ramp`,
  `rl_adaptive` took 54.9 s against 389.5 s for `rl_once` (-85.7%, paired CI
  -430 to -266 s), and was 7.7% faster than the best fixed strategy
  (`balanced`). In `relief`, `rl_adaptive` and `rule_adaptive` beat every fixed
  strategy by about 9-10%.
- In `spike` all adaptive policies were about 58% slower than fixed `balanced`:
  they run `static` while idle and need 10-15 tokens at ~2.5 s/token to switch
  after 10 workers arrive (decision interval plus hysteresis).
- The online bandit was the weakest adaptive policy in v5 (+57% vs `balanced`
  in `spike`, +34% vs `rl_adaptive` in `ramp`). In the high-load bin it chose
  `latency_optimized` (median 0.234 s/token) 267 times and `balanced` (0.132
  s/token) 104 times. Its regret fell from 12.9 s to 7.9 s between the two
  halves of the run, but stayed above `rl_adaptive`.

Re-analyse with `python evaluate_adaptive.py --tag v5 --report-only`.
`run_overnight.cmd` resumes the v5 run unattended (`--resume --keep-awake`).

Results from the earlier v3 adaptive run (60-token requests, time-triggered
load, blocking telemetry) are kept as `adaptive_*_v3.*`.

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
