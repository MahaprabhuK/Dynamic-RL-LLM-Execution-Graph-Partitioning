# Agent progress log (unfinished-work plan, v7 round)

## Setup (2026-10-13 03:00)
- CPU check: idle (12 samples over 1 min, see session); other project not running.
- Local branch fast-forwarded 90593dc -> 6fc0259 (v6 commit the plan assumes). Untracked logs and backups untouched.
- Power plan: only "Balanced" exists on this machine; High performance could not be set.
- Browsers and other apps (Chrome, Edge WebView, WhatsApp, Claude desktop) were open; not closed by the agent.
- scipy was in requirements.txt but missing from .venv; installed it.
- data/processed/dqn_model_v6.pth is not on this machine (*.pth is gitignored); Stage 2 needs it for rl_adaptive.

## Stage 1: engine, logging, tests
- F1: latency switch off by default (config `emergency_latency_switch.enabled: false`; engine arg
  `emergency_latency_factor=None`). ValueError if set with a fixed strategy or FixedPolicy.
  Renamed: engine `emergency_latency_factor` / `--emergency-latency-factor`; scheduler
  `emergency_cpu` / `--emergency-cpu` (old `emergency_threshold`, `emergency_switch_enabled`,
  `emergency_factor` removed). Stage records and decision log carry `emergency_trigger`, `emergency_kind`.
- F2: copied-logic test replaced by real-engine tests on a tiny random LlamaForCausalLM.
- F3: per-10-token segment log (swap sin/sout deltas, page-fault delta, available MB).
  **Check FAILED**: see experiments/f3_paging_check.txt. With the model loaded RAM is already ~94 %
  (~1 GB available) in the normal run, so the helper could only add 256 MB and the page-fault delta
  did not rise (pressure/normal ratio in the file). Plan says stop here.
- F4: src/telemetry/run_guard.py (85 %, 30 s, 10 retries, skipped_runs CSV) with tests. Not yet
  wired into run scripts.
- F5: arms threads_2/3/4/6/10 + balanced (PartitionStrategy.V7_ARMS), fp32.
- Tests: tests/test_pipeline.py + tests/test_adaptive.py 80 passed; with test_functional_equivalence.py 83 passed.

## STOPPED after Stage 1 (blocked)
1. F3 paging check did not show a rise (rule: stop and tell the user).
2. RAM is ~94 % once the model is loaded (v5 runs: median 96 %). The F4 guard (85 %) would skip every
   run, and each skip waits 10 x 30 s, so Stages 2-9 cannot produce data on this machine state.

## Resume 2026-10-13 07:57 (browsers closed by user)
- RAM with model loaded: 82.0 % (under the 85 % guard).
- F3 rerun -> experiments/f3_paging_check_run2.txt (first run kept unchanged). Helper reached
  818 MB available, but the process page-fault delta still did not rise (ratio in the file).
  Windows trims other processes, not the model's actively used pages. F3 still FAILED; stopped again.
- User decision (08:05): add a system-wide paging counter. Added src/telemetry/system_paging.py
  (PDH raw "\Memory\Pages Input/sec", cumulative hard-fault page reads, machine-wide) as
  `system_pages_input_delta` in every segment record; per-process page faults kept.
- F3 run 3 -> experiments/f3_paging_check_run3.txt: system pages-input rises clearly under
  pressure (totals and ratio in the file). Per-process count still does not. **F3 PASSED on the
  system counter.** Stage 1 complete.
