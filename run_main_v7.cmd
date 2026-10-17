@echo off
rem Resume the v7 main run (experiments/e4_plan.yaml). Aborts if the machine check, the CPU capacity
rem check or the sanity check fails. All output is appended to logs\main_v7_<date>.log.
rem Watch it from another window:  Get-Content logs\main_v7_<date>.log -Wait
setlocal
cd /d "%~dp0"
call .venv\Scripts\activate.bat || (echo venv activation failed & exit /b 1)
set HF_HUB_OFFLINE=1
set PYTHONUNBUFFERED=1
for /f %%d in ('python -c "import datetime; print(datetime.date.today().isoformat())"') do set RUNDATE=%%d
if not exist logs mkdir logs
set LOG=logs\main_v7_%RUNDATE%.log
echo Logging to %LOG%

echo ==== %DATE% %TIME% run_main_v7.cmd start >> "%LOG%"
python experiments\check_machine.py >> "%LOG%" 2>&1 || (echo check_machine.py FAILED, see %LOG% & echo ABORT: check_machine.py failed >> "%LOG%" & exit /b 1)
python experiments\cpu_capacity_check.py >> "%LOG%" 2>&1 || (echo cpu_capacity_check.py FAILED, see %LOG% & echo ABORT: cpu_capacity_check.py failed >> "%LOG%" & exit /b 1)
python experiments\sanity_resume.py >> "%LOG%" 2>&1 || (echo sanity_resume.py FAILED, see %LOG% & echo ABORT: sanity_resume.py failed >> "%LOG%" & exit /b 1)
python -c "import psutil; m = psutil.virtual_memory(); print(f'free RAM before main run: {m.available / 2**20:.0f} MB ({100 - m.percent:.1f} %% free, {m.percent} %% used)')" >> "%LOG%" 2>&1
echo checks passed, starting run_v7.py

python run_v7.py --plan experiments\e4_plan.yaml --resume --keep-awake >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo MAIN RUN FINISHED (run_v7.py exit code %RC%) >> "%LOG%"
python -c "import pandas as pd; print('rows in v7runs_v7.csv:', len(pd.read_csv('data/processed/v7runs_v7.csv')))" >> "%LOG%" 2>&1
echo MAIN RUN FINISHED (run_v7.py exit code %RC%)
python -c "import pandas as pd; print('rows in v7runs_v7.csv:', len(pd.read_csv('data/processed/v7runs_v7.csv')))"
exit /b %RC%
