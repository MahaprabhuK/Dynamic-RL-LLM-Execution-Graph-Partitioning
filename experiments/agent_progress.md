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

## Stage 2: smoke test + segment oracle (08:20)
- v6 DQN retrained locally as data/processed/dqn_model_v6_local.pth (training_data_v6.csv, seed 42,
  100 episodes); used for rl_adaptive and the bandit prior in the smoke test.
- Smoke (tag smoke_v7, run_v7.py + experiments/smoke_plan.yaml): all checks PASS,
  experiments/check_smoke_smoke_v7.txt. Note: plan says 50 tokens with the spike at token 60, so the
  load never arrives in the smoke test (plumbing test only).
- Segment oracle on v5: experiments/e0b_segment_oracle.{csv,txt}; all 5 scenarios within 0.2 s of
  the expected values in the plan.

## Stage 3: load sweep v7 (finished 2026-10-13 ~11:30)
- Sweep stopped at 284/288 around 09:12 (process gone, cause unknown); resumed with --resume, 288/288.
- RAM before runs 77.8-83.5 %; 30 paging-flagged runs (system pages input >= 1000 in a segment).
- Best arm by workers: 0 threads_6, 4 threads_10, 8/9 threads_6, 10/11 threads_4, 12 balanced,
  14 threads_2. Summary at the end of data/processed/load_sweep_v7.log.
- CoresHeuristicPolicy rewritten for v7 arms (snap to nearest arm, ties to fewer threads); 91 tests pass.

## Stage 4: rule_lookup (2026-10-13 ~11:35)
- experiments/e3_rule.yaml: 4 bins (edges 63.8 / 75.09 / 83.77 % ext CPU) -> threads_6 / threads_4 /
  balanced / threads_2. Offline: zero regret except workers 4 (0.009 s/tok); cores_heuristic 0.01-0.02.

## Stage 5: reaction-delay pilot (2026-10-13 12:31)
- 78/84 done when Claude Code killed the background job for low system memory; resumed with --resume.
  Last 6 runs carry commit 60beb1a (only removed test_bfloat16.py / run_overnight.cmd; no code change).
- 0 skipped, 25 paging-flagged runs. Frozen: rule_i10_p1_off (interval 10, patience 1, no latency
  switch), experiments/e2_frozen.yaml. n=3 per cell; 4 configs within 2 % of the best total.
- All rule configs beat fixed balanced in spike by 3.7-6.9 %.

## Stage 6: DQN v7, bandit prior, main-run plan (2026-10-13 ~12:40)
- DQN v7 (leave-one-prompt-out): mean regret 0.00534 s/tok, best arm 35 %; rule_lookup fold 0.00141, 79 %;
  cores_heuristic 0.01211, 19 %. DQN does NOT beat rule_lookup offline. Final model
  data/processed/dqn_model_v7.pth (gitignored).
- experiments/e4_bandit_prior.yaml and experiments/e4_plan.yaml (9 conditions, 675 runs, prompts 21-35,
  two nights) written by build_e4_plan.py.
- Next: Stage 7 main run (needs user go-ahead, free RAM, ideally
  CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1 or run outside Claude Code).

## Stage 7 resume prep (2026-10-17)
- Main run stopped at 67/675 on 13 Oct (inside context 8, prompt 22 ramp). Committed unchanged (957fccd).
- a63992a (hardware-aware balanced test) reverted (a91ee8a); 94 tests pass with HF_HUB_OFFLINE=1.
- experiments/check_main_partial.txt: all checks PASS (commit ecec5690ce, clean tree, code diff empty,
  token IDs identical per context, fixed arms 0 switches / 0 emergencies); 18/67 paging-flagged, 0 skips.
- experiments/machine_v7.json + check_machine.py (machine fingerprint gate).
- Sanity gate FAILED: threads_2 1.04 s/tok, balanced 1.05 s/tok at 10 workers (sweep 0.118 / 0.107).
- experiments/cpu_capacity_check.txt: 16 busy workers reach only ~37 % total CPU; logical CPUs 12-15 at
  100 %, 0-11 at 7-37 %. AC core parking min cores = 4 % (Balanced scheme). Machine not usable until fixed.
- 123.9 s cores_heuristic ramp run (ctx 8): memory pressure / paging, not a policy or code bug. Arms follow
  the measured external CPU; tokens 70-159 run at 0.3-2.9 s/tok while RAM is 86-89 % after a 74k
  pages-input burst, and the whole machine stalls (total CPU 27-40 % with 8 workers); recovers at RAM 84 %.
- experiments/resume_plan_v7.txt: seeded order rebuilt and matches all 67 rows; 608 runs remain, starting
  with ctx 8 (static, rl_adaptive, bandit_adaptive, latency_optimized, balanced). Estimate 5.39 h
  (4.48 h without paging-flagged runs in the medians).
- run_main_v7.cmd: check_machine -> cpu_capacity_check -> sanity_resume -> run_v7.py --resume, log in
  logs/main_v7_<date>.log. Abort path tested (stops at the CPU check); run_v7.py not started.

## Power settings changed for the main run (2026-10-17 ~17:49) - REVERT AFTER THE MAIN RUN
- Scheme: Balanced (381b4222-f694-41f0-9685-ff5bb260df2e), subgroup SUB_PROCESSOR, AC only. DC untouched.
- CPMINCORES (0cc5b647-c1df-4637-891a-dec35c318583, core parking min cores): AC 4 -> 100.
- HETEROPOLICY (7f2f5cfa-f10c-4823-b5e1-e93ae85f46b5): AC set to 0, but it was already 0 (no change).
- Both aliases were recognised; no GUIDs needed. Full before/after: experiments/powercfg_before.txt,
  experiments/powercfg_after.txt (plain /query hides these settings, so both files also hold /qh output).
- Revert:
    powercfg /setacvalueindex scheme_current sub_processor CPMINCORES 4
    powercfg /setactive scheme_current
- Effect: none. After the change cpu_capacity_check still 36.1 % (cpu12-15 at 100 %, cpu0-11 5-30 %);
  sanity_resume threads_2 1.19 / balanced 1.07 s/tok, RAM 82.8-88.0 %. Both FAIL; main run not started.
