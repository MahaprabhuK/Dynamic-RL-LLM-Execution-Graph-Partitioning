# Models used by the v7 runs

## Loaded by the v7 main run (experiments/e4_plan.yaml)

| File | Used by | Size (bytes) | sha256 | In git |
|---|---|---|---|---|
| `data/processed/dqn_model_v7.pth` | `rl_adaptive` (`model:` in e4_plan.yaml) | 22353 | `b8045e99e1b2632af64a14ccedc73753701ca007d71f705f7e71b1b3a55bdcd5` | yes, force-added 19 Oct 2026 (`*.pth` is gitignored) |

- Trained by `experiments/train_dqn_v7.py` on `data/processed/load_sweep_v7.csv` (seed 42, 300 episodes, gamma 0,
  full feedback); offline evaluation in `experiments/e3_dqn_v7.txt` / `e3_dqn_v7.csv`.
- State: [external CPU %, available RAM MB, prompt tokens] (scaled); 6 actions: threads_2, threads_3, balanced,
  threads_4, threads_6, threads_10.
- File last modified 2026-10-13 12:33 IST (07:03 UTC), before the first main-run row (2026-10-13 13:04 IST).
  Git cannot prove it was not replaced between 13 and 18 Oct, since it was untracked until now; the hash above is
  the file on disk after the run.
- No other learned model is loaded by the main run. `bandit_adaptive` starts from `experiments/e4_bandit_prior.yaml`
  (built from the sweep); its learned state is saved to `data/processed/v7bandit_state_v7/bandit_adaptive.json`.
  `rule_lookup` reads `experiments/e3_rule.yaml`.

## Language model

| Model | Revision | Weights | sha256 |
|---|---|---|---|
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 (Hugging Face), float32, CPU | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` (local cache `refs/main`) | `model.safetensors`, 2200119864 bytes | `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933` |

Runs use `HF_HUB_OFFLINE=1`, so the cached snapshot above is what was loaded.

## Other DQN files on disk (not loaded by the main run, not added to git)

| File | Size (bytes) | sha256 | Used for |
|---|---|---|---|
| `data/processed/dqn_model_v2.pth` | 21841 | `88dc5c7112e320ce396f9a0146cb354ae3673e4c01d7e0e31071b9f14ed5209a` | v2 (historical) |
| `data/processed/dqn_model_v3.pth` | 21841 | `e10f4142f0a8b9f1b9033a21fe186709260c1087c18edc7a9061fe1ee7070b96` | v3-v5 `rl_adaptive` / `rl_once` |
| `data/processed/dqn_model_v6_local.pth` | 22681 | `e5eaa7544afe7edbc948a5d0e839dd7a20c2273dfc9b6104f520b3d958c7c6b0` | v7 smoke test only (retrained locally; `dqn_model_v6.pth` is not on this machine) |
