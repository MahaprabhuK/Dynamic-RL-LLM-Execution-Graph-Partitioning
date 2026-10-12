@echo off
rem Resumes the v5 adaptive evaluation (scheduled task "RL-LLM v5 overnight").
cd /d "%~dp0"
set HF_HUB_OFFLINE=1
set PYTHONUNBUFFERED=1
".venv\Scripts\python.exe" evaluate_adaptive.py --tag v5 --num-prompts 8 --prompt-offset 20 --heavy 10 --switch-token 60 --ramp "60:8,140:12" --threshold 60 --bandit-edges "51,63" --bin-margin 4 --resume --keep-awake >> "data\processed\adaptive_v5_overnight.log" 2>&1
