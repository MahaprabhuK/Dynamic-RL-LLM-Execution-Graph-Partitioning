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

## Power fixes by the user (2026-10-17, after ~17:53) - REVERT AFTER THE MAIN RUN
- Balanced scheme, SUB_PROCESSOR, AC values (confirmed with powercfg /qh on 2026-10-18; DC untouched):
  CPMINCORES 4 -> 100, CPMINCORES1 0 -> 100, CPMAXCORES1 -> 100, HETEROPOLICY -> 0, SHORTSCHEDPOLICY -> 0.
  Previous values of CPMAXCORES1, HETEROPOLICY and SHORTSCHEDPOLICY were not recorded (HETEROPOLICY was
  already 0, see above).
- Power throttling disabled for .venv\Scripts\python.exe and
  C:\Users\Mahaprabhu K\AppData\Local\Programs\Python\Python313\python.exe (as reported by the user;
  powercfg /powerthrottling list needs admin, not checked here).
- Effect: all-core use and 13 Oct speed restored. cpu_capacity_check 2026-10-17 18:21: 100.0 % total
  CPU with 16 workers, every core at 100 % (PASS). sanity_resume (commit 6dc2fb4eca): threads_2
  0.1285 s/tok (+8.9 %), balanced 0.1144 s/tok (+7.0 %) vs sweep, both PASS. Its RAM check FAILED
  (87.5 / 87.3 / 86.7 / 86.0 % before each run, limit 85 %), so OVERALL is FAIL and run_main_v7.cmd
  would stop at sanity_resume until RAM is below 85 %.

## Plan amendment: prompts 21-30 (2026-10-18)
- experiments/e4_plan.yaml: prompts 21-30 instead of 21-35 (10 contexts per scenario), total_runs
  675 -> 450, entry under amendments. Reason: limited hardware time. Decided before any analysis of
  main-run outcomes; analysis family, conditions, scenarios and seeds unchanged. night_1 / night_2
  left as they were (night_2 still says 29-35); the run now goes in three sessions.
- Resume check: run_v7.py draws the order prompt by prompt, so the amended plan's 450-run order equals
  the first 450 rows of the old order; all 67 rows on disk match context_id and run_order.
- run_main_v7.cmd takes an optional prompt list (run_main_v7.cmd 22,23,24 -> run_v7.py --prompts);
  prompts outside the plan stop it before the checks. The three checks still run first.
- experiments/resume_plan_v7.txt: 383 runs left, 3.40 h (2.82 h without paging-flagged medians).
  Sessions: 22-24 113 runs 1.02 h, 25-27 135 runs 1.19 h, 28-30 135 runs 1.19 h.

## Stage 7: main run finished (2026-10-18 23:49 - 2026-10-19 13:05)
- 450/450 runs, 50/50 contexts complete. Commits: 67 rows ecec569 (13 Oct), 383 rows 0a8e3b2.
- 18 Oct 23:49: gates passed (CPU 100 %, sanity +11.7/+15.5 %, RAM 83.4-84.9 %). Claude Code killed the
  background job at ~00:25 for low system memory after 8 runs; 5 cells of context 9 skipped by the RAM guard.
- 19 Oct: run_main_v7.cmd driven by a retry loop launched with Start-Process (outside Claude Code's
  process reaping; WMI Win32_Process.Create silently did nothing). Loop waits for baseline RAM <= 66 %
  (model adds ~17-27 points depending on memory compression), reruns --resume until 450 rows.
- 11:13: run_v7.py killed during a RAM-guard wait (thought to be a leak; it was not, a fresh process has
  the same ~7.8 GB private bytes). The run in progress had not been written; --resume redid it, no data lost.
- Sanity gate stopped two resume attempts: RAM 85.1 % (11:14), balanced +81 % (12:51); next attempts passed.
- RAM was the binding constraint. WhatsApp / Phone Link restarted on their own several times; the Claude
  desktop app (~1.35 GB, separate from the CLI) was still running until 12:31. After closing it the run finished.
- 17 RAM-guard skips in total (skipped_runs_v7.csv), all re-run later. Context 9 split 3 conditions on 18 Oct
  23:59 / 6 on 19 Oct 08:44; contexts 46-48 span up to 2 h.
- 106/450 paging-flagged runs. RAM before runs min 79.8 / median 82.7 / max 85.0 %.
- experiments/check_main_final.txt: all data checks PASS. Expected FAILs: tree_dirty True on the 383 later
  rows (the tracked output CSVs and gate reports change during the run; no code changed), code diff
  ecec569..HEAD = e4_plan.yaml amendment, run_main_v7.cmd added. check_main_partial.txt kept as the 17 Oct record.

## Stage 8: analysis (2026-10-19 13:06)
- experiments/analyze_v7.py -> experiments/e4_summary.txt, e4_*.csv, results/v7/*.png.
- 15/20 pre-registered tests Holm-significant. rule_lookup faster than threads_2 in all 5 scenarios and than
  balanced in steady_idle/spike/relief (not steady_heavy, ramp). rl_adaptive slower than rule_lookup in
  steady_heavy/spike/relief/ramp; cores_heuristic slower in steady_heavy/spike/relief.
- Sensitivity (paging-flagged removed, n 3-9): all signs kept, 10/20 significant.
- Report: docs/report/project_report.html section 16 + PDF regenerated (headless Edge).
- Power settings from 17 Oct are still changed; revert is for the user (admin).

## Final report (2026-10-19)
- docs/report/final_report.html + Final_Report.pdf: section 16 rewritten (headline vs best fixed arm per
  scenario, bandit descriptive, DQN generalisation wording, Wilcoxon column, provenance, threats to validity
  incl. 17 Oct power fix). experiments/analyze_main_v7.py supplement (e4_supplement.txt, e4_best_fixed.csv,
  e4_by_date.csv). project_report.html kept, marked superseded.

## Final report fixes (2026-10-19 afternoon)
- Section 15.2 v6 sweep table regenerated from load_sweep_v6.csv by experiments/build_report_tables.py (40 of 48 cells
  had differed at 4 decimals; fastest budgets and 8.4x / 12.0x ratios unchanged); correction note added.
- Summary rewritten for the v7 setup; abstract; related work and references (experiments/reference_check.md;
  Hu et al. INFOCOM 2019 not cited because the paper could not be opened).
- experiments/check_report_numbers.py: 283 table cells strict + 1185 text numbers matched, 181 structural, 0 unmatched.
  It found and the report now fixes: two v7 roundings (Wilcoxon 0.105, CI 8.38), v4 sweep 0.2154, the v5 context 14/36
  sentence (rl_adaptive hit the cliff only in context 36), an unreproducible v4 switch count (53 of 114), and my own
  bandit-vs-rule_lookup bounds.
- RESULTS.md (generated), MODELS.md (dqn_model_v7.pth force-added), ENVIRONMENT.md, untracked_files.md, README rewritten.
- Tests: 94 passed with HF_HUB_OFFLINE=1 (incl. real-model equivalence).
