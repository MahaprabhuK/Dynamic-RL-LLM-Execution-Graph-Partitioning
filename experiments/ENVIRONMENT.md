# Environment of the v7 runs

Recorded 19 October 2026. Machine fingerprint checked before every resume by `experiments/check_machine.py`
against `experiments/machine_v7.json` (all fields matched on 18 and 19 Oct).

## Hardware and OS

| Item | Value |
|---|---|
| CPU | 13th Gen Intel Core i5-13450HX, 10 physical cores (6 performance + 4 efficiency), 16 logical CPUs |
| RAM | 15.69 GB visible |
| GPU | RTX 3050 present, not used (torch is a CPU build) |
| OS | Windows 11 Home Single Language, build 10.0.26300 |
| Power | AC, battery saver off, Balanced scheme `381b4222-f694-41f0-9685-ff5bb260df2e` |

## Software (`.venv`)

| Package | Version |
|---|---|
| Python | 3.13.6 (MSC v.1944, 64-bit) |
| torch | 2.13.0+cpu (default intra-op threads 10) |
| transformers | 5.14.1 |
| numpy | 2.5.1 |
| pandas | 3.0.5 |
| scipy | 1.18.1 |
| psutil | 7.2.2 |
| PyYAML | 6.0.3 |

Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0, float32, revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`,
loaded with `HF_HUB_OFFLINE=1` (see `experiments/MODELS.md`).

## Power settings and the 17 October change

On 17 Oct the laptop gave only about 37% total CPU to 16 busy workers (logical CPUs 12-15 at 100%, 0-11 at 7-37%);
decode was about 10x slower than in the 13 Oct sweep. Changing CPMINCORES alone did not help
(`experiments/agent_progress.md`); full speed came back after the user changed several AC processor settings and
disabled power throttling for Python, so which change mattered is not isolated. Before the 18-19 Oct session the
AC processor settings were:

| Setting (AC, sub_processor) | Before (17 Oct, `experiments/powercfg_before.txt`) | During and after the 18-19 Oct runs (read 19 Oct) |
|---|---|---|
| CPMINCORES (core parking min cores) | 4 % | 100 % |
| CPMINCORES1 (min cores, efficiency class 1) | 0 % | 100 % |
| CPMAXCORES1 (max cores, efficiency class 1) | 100 % | 100 % |
| HETEROPOLICY (heterogeneous thread scheduling) | not in the saved query output | 0 |
| SHORTSCHEDPOLICY (short-running thread scheduling) | not in the saved query output | 0 |
| PROCTHROTTLEMIN / PROCTHROTTLEMAX | 5 % / 100 % | 5 % / 100 % (unchanged) |

DC values were not changed (CPMINCORES DC is still 4 %). Power throttling was also disabled for `.venv\Scripts\python.exe`
and the base Python 3.13 `python.exe` (reported by the user; `powercfg /powerthrottling list` needs admin and was not checked).

The power settings in force on 13 Oct, when the first 67 main-run rows and the load sweep were made, were not recorded.
After the change: `experiments/cpu_capacity_check.txt` (100% total CPU with 16 workers) and
`experiments/sanity_resume.txt` (threads_2 and balanced within 25% of the sweep). Fixed-arm speed in the 13 Oct and
18-19 Oct rows agrees within about 1% for most arms (`experiments/e4_by_date.csv`).

**Not yet reverted.** To restore the recorded original of the one setting with a known previous value (admin shell):

```
powercfg /setacvalueindex scheme_current sub_processor CPMINCORES 4
powercfg /setacvalueindex scheme_current sub_processor CPMINCORES1 0
powercfg /setactive scheme_current
```

HETEROPOLICY and SHORTSCHEDPOLICY previous AC values were not recorded.

## Run conditions

- RAM before runs: 79.8-85.0 % (median 82.7 %) with the fp32 model loaded; RAM guard 85 %, 30 s waits, 10 retries.
- Background apps closed during the 19 Oct session, including the Claude desktop app; this CLI session (about 0.3 GB) stayed open.
- Busy-loop background load from `src/telemetry` load generators (synthetic CPU load only).
