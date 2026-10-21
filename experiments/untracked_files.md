# Files in the working tree that are not in git (19 October 2026)

Listed with `git status --porcelain --untracked-files=all` and `git status --porcelain --ignored`. Nothing was
deleted. Sizes in bytes; dates are last modification (local time).

## Untracked (not ignored)

| File | Bytes | Modified | What it is |
|---|---|---|---|
| data/processed/adaptive_v4.log | 3465 | 2026-10-11 06:47 | v4 run console log |
| data/processed/adaptive_v4_rerun.log | 8389 | 2026-10-11 18:28 | v4 rerun console log |
| data/processed/adaptive_v4_resume.log | 17482 | 2026-10-11 18:16 | v4 resume console log |
| data/processed/adaptive_v5.log | 2625 | 2026-10-11 20:32 | v5 run console log |
| data/processed/adaptive_v5_overnight.log | 18466 | 2026-10-12 04:56 | v5 overnight console log |
| data/processed/adaptive_v5_resume.log | 2036 | 2026-10-11 21:03 | v5 resume console log |
| data/processed/v4_backup/adaptive_bandit_history_v4.csv | 14768 | 2026-10-11 18:16 | copy of v4 outputs; timestamps match the v4 resume log, so probably a backup before the rerun |
| data/processed/v4_backup/adaptive_bandit_state_v4.json | 37756 | 2026-10-11 18:16 | " |
| data/processed/v4_backup/adaptive_decisions_v4.csv | 340579 | 2026-10-11 18:16 | " |
| data/processed/v4_backup/adaptive_paired_diffs_v4.csv | 2029 | 2026-10-11 18:16 | " |
| data/processed/v4_backup/adaptive_runs_v4.csv | 300697 | 2026-10-11 18:16 | " |
| data/processed/v4_backup/adaptive_summary_v4.csv | 2805 | 2026-10-11 18:16 | " |
| data/processed/v4_backup/adaptive_timelines_v4.csv | 1974569 | 2026-10-11 18:16 | " |
| data/processed/v5_backup/adaptive_bandit_state_v5.json | 9235 | 2026-10-11 21:03 | copy of partial v5 outputs; timestamps match the v5 resume log, so probably a backup before the resume |
| data/processed/v5_backup/adaptive_decisions_v5.csv | 61727 | 2026-10-11 21:03 | " |
| data/processed/v5_backup/adaptive_runs_v5.csv | 44433 | 2026-10-11 21:03 | " |
| data/processed/v5_backup/adaptive_timelines_v5.csv | 354262 | 2026-10-11 21:03 | " |

None of these are inputs to the v7 analysis or the final report.

## Ignored by .gitignore

| Path | Bytes | Files | Note |
|---|---|---|---|
| .venv/ | 1213548750 | 39878 | Python environment (see ENVIRONMENT.md); a few deep torch license folders could not be listed (path too long) |
| data/processed/dqn_model_v2.pth | 21841 | 1 | see MODELS.md |
| data/processed/dqn_model_v3.pth | 21841 | 1 | see MODELS.md |
| data/processed/dqn_model_v6_local.pth | 22681 | 1 | see MODELS.md |
| data/processed/dqn_model_v7.pth | 22353 | 1 | force-added to git in this commit (MODELS.md); still matches the ignore pattern |
| .pytest_cache/ | 8266 | 5 | test cache |
| __pycache__/ and */__pycache__/ (11 folders) | 538765 | 34 | Python bytecode |

The TinyLlama weights live in the Hugging Face cache outside the repository (MODELS.md).
