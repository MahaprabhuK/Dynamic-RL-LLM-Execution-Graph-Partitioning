"""
CPU capacity check, no model: can this machine actually deliver its cores right now?
Starts 16 busy-loop workers (src.telemetry.load_generator) for 20 s and records total and per-core
CPU % (psutil, time-based over the window), psutil.cpu_freq() before and during, power source,
battery saver state, and the active powercfg scheme with its AC processor settings.
Passes if total CPU >= 90 %. Writes experiments/cpu_capacity_check.txt. Exit code 1 on failure.

    python experiments/cpu_capacity_check.py
"""

import ctypes
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.telemetry.load_generator import BackgroundCPULoad

WORKERS, SECONDS, MIN_TOTAL = 16, 20.0, 90.0
OUT = ROOT / "experiments" / "cpu_capacity_check.txt"
# powercfg aliases under SUB_PROCESSOR
SETTINGS = {"Maximum processor state": "PROCTHROTTLEMAX",
            "Minimum processor state": "PROCTHROTTLEMIN",
            "Processor performance core parking min cores": "CPMINCORES",
            "Processor performance core parking min cores, efficiency class 1": "CPMINCORES1",
            "Heterogeneous thread scheduling policy (5 = automatic)": "SCHEDPOLICY"}


class SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]


def power_status():
    s = SYSTEM_POWER_STATUS()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
    source = {0: "battery", 1: "AC"}.get(s.ACLineStatus, "unknown")
    return source, bool(s.SystemStatusFlag & 1), s.BatteryLifePercent


def powercfg(*args):
    return subprocess.run(["powercfg", *args], capture_output=True, text=True).stdout


def ac_setting(alias):
    out = powercfg("/qh", "SCHEME_CURRENT", "SUB_PROCESSOR", alias)
    m = re.search(r"Current AC Power Setting Index:\s*0x([0-9a-fA-F]+)", out)
    return int(m.group(1), 16) if m else None


def freq():
    f = psutil.cpu_freq()
    return f"current {f.current:.0f} MHz, min {f.min:.0f}, max {f.max:.0f}" if f else "unavailable"


def main():
    lines = [f"CPU capacity check (experiments/cpu_capacity_check.py), {datetime.now().isoformat(timespec='seconds')}"]
    source, saver, pct = power_status()
    lines.append(f"power source: {source} (battery {pct} %), battery saver: {'ON' if saver else 'off'}")
    lines.append(f"active scheme: {powercfg('/getactivescheme').strip()}")
    for name, alias in SETTINGS.items():
        val = ac_setting(alias)
        unit = "" if alias == "SCHEDPOLICY" else " %"
        lines.append(f"AC {name}: {val if val is not None else 'not found'}{unit if val is not None else ''}")
    lines.append(f"cpu_freq before: {freq()} (psutil on Windows often reports the base clock only)")
    lines.append(f"logical CPUs: {psutil.cpu_count()}, physical: {psutil.cpu_count(logical=False)}")

    freqs = []
    with BackgroundCPULoad(WORKERS, settle_seconds=1.0):
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
        t_end = time.perf_counter() + SECONDS
        while time.perf_counter() < t_end:
            time.sleep(2.0)
            f = psutil.cpu_freq()
            if f:
                freqs.append(f.current)
        total = psutil.cpu_percent(interval=None)
        per_core = psutil.cpu_percent(interval=None, percpu=True)
    lines.append(f"cpu_freq during ({WORKERS} workers, every 2 s): "
                 f"{', '.join(f'{x:.0f}' for x in freqs)} MHz (max {psutil.cpu_freq().max:.0f})")
    lines.append(f"total CPU over {SECONDS:.0f} s with {WORKERS} workers: {total:.1f} %")
    lines.append("per-core CPU %: " + ", ".join(f"cpu{i} {v:.0f}" for i, v in enumerate(per_core)))
    ok = total >= MIN_TOTAL
    lines.append(f"[{'PASS' if ok else 'FAIL'}] total CPU >= {MIN_TOTAL:.0f} %")
    text = "\n".join(lines)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
