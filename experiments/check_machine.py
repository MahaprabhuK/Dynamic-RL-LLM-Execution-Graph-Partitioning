"""
Check that this machine matches the one the v7 main run started on (experiments/machine_v7.json).
Exit code 1 on any mismatch, so a resumed run cannot silently mix hardware or library versions.

    python experiments/check_machine.py            # compare against machine_v7.json
    python experiments/check_machine.py --write    # (re)write machine_v7.json from this machine
"""

import json
import platform
import sys
import winreg
from pathlib import Path

import psutil
import torch
import transformers

ROOT = Path(__file__).resolve().parent.parent
MACHINE_FILE = ROOT / "experiments" / "machine_v7.json"
RAM_TOLERANCE_GB = 0.5


def cpu_model():
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
    return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()


def current_machine():
    return {
        "cpu_model": cpu_model(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / 2**30, 2),
        "windows_build": platform.version(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def main():
    now = current_machine()
    if "--write" in sys.argv:
        MACHINE_FILE.write_text(json.dumps(now, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {MACHINE_FILE}")
        return 0
    expected = json.loads(MACHINE_FILE.read_text(encoding="utf-8"))
    bad = []
    for key, want in expected.items():
        got = now.get(key)
        ok = abs(got - want) <= RAM_TOLERANCE_GB if key == "total_ram_gb" else got == want
        print(f"[{'OK' if ok else 'MISMATCH'}] {key}: expected {want!r}, found {got!r}")
        if not ok:
            bad.append(key)
    if bad:
        print(f"Machine does not match {MACHINE_FILE.name}: {', '.join(bad)}", file=sys.stderr)
        return 1
    print("Machine matches machine_v7.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
