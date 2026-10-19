# Dynamic RL-Based LLM Execution Graph Partitioning

Runs TinyLlama-1.1B (fp32) on one CPU and decides, during generation, how many threads it should use.
The model runs layer by layer through a hand-written decode loop with one shared KV cache, so the thread budget can change between tokens without changing the output.
Every 10 tokens a policy reads the CPU load from other processes and picks an arm: 2, 3, 4, 6 or 10 threads, or `balanced` (two sequential 3-thread stages).
Policies compared: a lookup table measured on this machine (`rule_lookup`), a spare-cores heuristic, an offline DQN, an online bandit, and fixed arms.
Everything is measured on one laptop (10 cores / 16 threads, 16 GB) under synthetic background load, with paired, randomized runs.

**Final report: [docs/report/final_report.html](docs/report/final_report.html)** (PDF: `docs/report/Final_Report.pdf`).

## Main result (v7, pre-registered, 450 runs of 200 tokens)

- Against the best fixed arm in each scenario (chosen afterwards from the same data), `rule_lookup` was 0.1% to 4.8% faster, and faster in 10 of 10 contexts when load arrived mid-request (spike).
- Pre-registered paired t-tests (Holm over 20): faster than fixed `threads_2` in all 5 scenarios (-8% to -20%) and than fixed `balanced` in 3 of 5 (-5% to -13%); 15 of 20 tests significant.
- An online bandit warm-started from the same sweep nearly tied it; the offline DQN was 10% to 60% slower than `rule_lookup` in 4 of 5 scenarios.

Tables: [experiments/RESULTS.md](experiments/RESULTS.md). Every number in the report is checked against its source by
`experiments/check_report_numbers.py` (output in `experiments/check_report_numbers.txt`).

## Setup

Windows, Python 3.13, CPU-only torch (see [experiments/ENVIRONMENT.md](experiments/ENVIRONMENT.md) for exact versions).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.data.download_dataset        # Alpaca prompts
python -m src.data.preprocess_dataset      # -> data/processed/prompts.csv
```

The first model load downloads TinyLlama/TinyLlama-1.1B-Chat-v1.0 (about 2.2 GB). After that, run with
`HF_HUB_OFFLINE=1`. The fp32 model needs about 4.4 GB of RAM; the v7 runs skip a run when RAM is above 85%, so close
other applications (including the Claude desktop app and browsers) before long runs.

## Reproducing the v7 stages

Times are from the runs on 13-19 Oct 2026 on the machine above. Machine runs load the model and take the CPU; run
them one at a time.

| Stage | Command | Output | Time |
|---|---|---|---|
| Tests | `set HF_HUB_OFFLINE=1` then `python -m pytest tests` | - | about 1-2 min |
| 2. Smoke test | `python run_v7.py --plan experiments/smoke_plan.yaml --resume --keep-awake` then `python experiments/check_smoke.py smoke_v7` | `v7*_smoke_v7.csv`, `experiments/check_smoke_smoke_v7.txt` | about 1 min |
| 3. Load sweep (288 runs) | `python run_load_sweep_v7.py --resume --keep-awake`, then `--summary-only` | `data/processed/load_sweep_v7.csv`, `experiments/e1_summary.txt` | about 3 h |
| 4. Lookup rule | `python experiments/build_rule_v7.py` | `experiments/e3_rule.yaml`, `e3_offline_policies.csv` | seconds |
| 5. Reaction-delay pilot (84 runs) | `python run_v7.py --plan experiments/e2_plan.yaml --resume --keep-awake` then `python experiments/analyze_e2.py` | `v7*_pilot_v7.csv`, `experiments/e2_summary.*`, `e2_frozen.yaml` | about 1 h |
| 6. DQN v7, bandit prior, plan | `python experiments/train_dqn_v7.py` then `python experiments/build_e4_plan.py` | `data/processed/dqn_model_v7.pth`, `experiments/e3_dqn_v7.*`, `e4_bandit_prior.yaml`, `e4_plan.yaml` | about 1 min |
| 7. Main run (450 runs) | `run_main_v7.cmd` (optionally `run_main_v7.cmd 22,23,24` for a subset of prompts) | `data/processed/v7*_v7.csv`, `logs/main_v7_<date>.log` | about 3.5 h of runs, plus about 2 min of checks per start |
| 8. Analysis | `python experiments/analyze_v7.py` then `python experiments/analyze_main_v7.py` | `experiments/e4_*.csv`, `e4_summary.txt`, `e4_supplement.txt`, `results/v7/*.png` | under 1 min |
| Report tables and checks | `python experiments/build_report_tables.py`, `python experiments/build_results_md.py`, `python experiments/check_report_numbers.py` | `docs/report/final_report.html` tables, `experiments/RESULTS.md`, `experiments/check_report_numbers.txt` | seconds |

`run_main_v7.cmd` checks the machine fingerprint (`experiments/check_machine.py`), CPU capacity
(`cpu_capacity_check.py`) and a short sanity run against the sweep (`sanity_resume.py`) before every start, and
resumes from the rows already written. The analysis scripts refuse to overwrite existing outputs.

Earlier versions (v3-v6) are reproduced with `evaluate_fixed_strategies.py`, `pilot_load_sweep.py`,
`evaluate_adaptive.py`, `reanalyze_adaptive_v5r.py` and `run_load_sweep_v6.py`; commands are in section 19.2 of the
report.

## Data index

| Path | Contents |
|---|---|
| `data/processed/load_sweep_v7.csv` | v7 load sweep: decode s/token per arm at 0-14 busy workers |
| `data/processed/v7runs_v7.csv` | main run, one row per run (latency, switches, paging, RAM, commit) |
| `data/processed/v7timelines_v7.csv`, `v7segments_v7.csv`, `v7decisions_v7.csv` | per token, per 10-token segment, per policy decision |
| `data/processed/skipped_runs_v7.csv` | runs skipped by the RAM guard (all re-run later) |
| `data/processed/v7*_pilot_v7.csv`, `v7*_smoke_v7.csv` | stage 5 pilot and stage 2 smoke test |
| `data/processed/dqn_model_v7.pth` | DQN loaded by `rl_adaptive` (sha256 in [experiments/MODELS.md](experiments/MODELS.md)) |
| `experiments/e*_plan.yaml`, `e2_frozen.yaml`, `e3_rule.yaml`, `e4_bandit_prior.yaml` | plans and frozen policy files |
| `experiments/e4_*.csv`, `e4_summary.txt`, `e4_supplement.txt` | main-run analysis |
| `experiments/check_main_final.txt` | final data integrity checks |
| `experiments/reference_check.md` | how each reference in the report was checked |
| `experiments/agent_progress.md` | dated log of every stage, including problems and deviations |
| `experiments/untracked_files.md` | files on disk that are not in git |
| `logs/` | main-run logs (gates, run output, retry loop) |
| `data/processed/*_v2..v6*` | earlier versions; see section 19.3 of the report |

## Layout

```
configs/config.yaml   model and strategy definitions
src/llm/              model loading, arms, layer-sliced inference
src/scheduler/        fixed, rule, lookup, cores-heuristic, DQN and bandit policies
src/telemetry/        external-CPU sampler, load generators, paging counter, RAM guard
src/rl/               features, reward, DQN agent and training
src/benchmark/        statistics, plots, oracles
run_v7.py             v7 runner for a plan file (smoke, pilot, main run)
run_load_sweep_v7.py  v7 load sweep
experiments/          plans, analysis scripts, checks, summaries
tests/                unit tests and the real-model equivalence test
docs/report/          final_report.html (current), project_report.html (superseded)
```
