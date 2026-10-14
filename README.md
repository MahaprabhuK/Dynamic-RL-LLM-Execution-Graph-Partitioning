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

The historical v4 summary excluded two latency-outlier candidates using
`stalled_contexts()`; its CSV has no page-fault or swap counters, so the paging
label is unverified. Generated tokens were identical across all conditions in
100% of contexts.

- The historical report recorded descriptive online-bandit mean differences of
  -7.8% vs `rl_once` in `spike` and -7.0% in `relief`. Its significance labels
  were not Holm-adjusted and have not been recomputed.
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

- Latency outliers had been assumed to be paging stalls, but no independent
  page-fault or swap telemetry is in the saved dataset. The bandit retains a
  latency-based outlier rule; that rule does not identify paging events.
- Readings near a bin edge had caused bin flapping, which accounted for 53 of
  114 switches. Bin hysteresis (`bin_margin`) now prevents it.
- Slower forgetting (`discount=0.98`) and a smaller exploration bonus
  (`bonus=0.08`) mean confident arms are re-checked less often.
- The stall exclusion compares each run with the same condition in the same
  scenario, so a policy that is slow every time (oversubscription) is kept.

### v5r reanalysis (all 40 contexts)

The original v5 report and CSV files are preserved. Their four “paging stall”
exclusions used only a latency-outlier heuristic; the run file has no page-fault,
swap, or paging telemetry to substantiate that label. The reproducible v5r
analysis retains all 40 contexts as primary and treats contexts 14, 19, 36, and
39 only as flagged candidates in a separately labeled sensitivity analysis.
All seven conditions are paired in every context, and token IDs match across
conditions in all 40 contexts. See
[`data/processed/adaptive_report_v5r.txt`](data/processed/adaptive_report_v5r.txt)
and rerun with `python reanalyze_adaptive_v5r.py`.

For the DQN `rl_adaptive` method, all-context mean latency was slower than the
descriptively best fixed strategy in `ramp`, `spike`, and `steady_heavy`; it
was similar in `steady_idle` and slightly faster in `relief`. No comparison to
the best fixed strategy or `rule_adaptive` was significant after the declared
Holm-adjusted paired-t family. The reanalysis does not support a claim that the
DQN is superior. Raw and adjusted paired-t and Wilcoxon p-values are retained
in `adaptive_paired_diffs_v5r.csv`; the exact baseline, confidence interval,
and small-sample limitations are discussed in the report.

Older v3/v4/v5 prose and significance labels are historical analyses, not
re-verified by the v5r statistical procedure. In particular, a latency outlier
alone is not evidence of paging.

### v6 policy framework, load sweep, and DQN retraining

The framework implements all P0 and P1 algorithmic and baseline requirements:
- **Baseline policies**: `cores_heuristic` (`threads = clamp(round(16 - external_cpu% * 16 / 100), 1, 10)`),
  `rule_noh` (threshold policy with patience 0), and `segment_oracle` (hindsight upper-bound utility in `src/benchmark/oracles.py`).
- **Emergency switch**: Bypasses decision interval on token latency spikes (>3.0x recent 10-token median)
  to switch immediately to conservative execution (`configs/config.yaml`).
- **Online bandit fixes**: Drops the first 2 switch tokens from cost attribution, prunes exploration for arms
  estimated >2x the current best, disables static exploration in the top load bin, caps DQN warm-start Q-values to <=75% CPU,
  and resets bins upon detecting persistent stalls (5 consecutive slow segments).
- **Expanded action space**: Supports single-stage thread budgets `{2, 4, 6, 8, 10, 12}` alongside named backward aliases.
- **Load sweep v6 complete**: Evaluated 8 worker contention levels (0-14 workers) x 3 repeats x 6 thread budgets = 144 runs
  at 50 tokens each using Prompt 1 (`data/processed/load_sweep_v6.csv`). Confirmed optimal thread budget scales monotonically down
  from 10 threads at load 0 to 2 threads at load 14.
- **DQN v6 retraining complete**: Merged into `data/processed/training_data_v6.csv` (144 rows across 24 paired load contexts;
  Prompts 21-40 strictly held out). Contextual-bandit DQN trained for 100 episodes (`data/processed/dqn_model_v6.pth`),
  with loss converging from 0.734 to 0.056. Offline evaluation on 8 held-out load contexts (repeat 2, `data/processed/rl_offline_evaluation_v6.csv`)
  achieved mean latency of 13.91s versus oracle 13.63s (mean regret 0.0112).
- **Scope note**: Generalization across unseen prompts is not claimed; the v6 sweep and retraining use Prompt 1 evaluated under
  different background-load conditions.
- **Deferred experiments**: The P1.2 pilot ablation (`run_pilot_ablation.py`, ~55 min) and full v6 live evaluation benchmark
  (`evaluate_adaptive.py --tag v6`, 675 runs, ~14 hours) are gated awaiting explicit approval.

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
- **Statistics.** The v5r reanalysis reports paired bootstrap 95% CIs, paired
  t-tests, Wilcoxon signed-rank tests, and Holm-adjusted p-values over the
  declared comparison families. Its `significant` field means Holm-adjusted
  paired-t $p < 0.05$; the separate CI-exclusion flag is descriptive. Older
  version reports used a different, unadjusted criterion.
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
