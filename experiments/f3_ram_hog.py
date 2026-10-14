"""
Helper for the F3 paging check: allocates RAM in 256 MB chunks (touching every
page) until available memory is under the target, prints READY, then holds the
memory until stdin closes or `hold_seconds` pass.

    python experiments/f3_ram_hog.py --target-mb 1024 --hold-seconds 300
"""

import argparse
import sys
import threading
import time

import numpy as np
import psutil

parser = argparse.ArgumentParser()
parser.add_argument("--target-mb", type=float, default=1024)
parser.add_argument("--hold-seconds", type=float, default=300)
parser.add_argument("--max-gb", type=float, default=14)
args = parser.parse_args()

chunks = []
chunk = 256 * 2 ** 20
while psutil.virtual_memory().available / 2 ** 20 > args.target_mb and len(chunks) * chunk < args.max_gb * 2 ** 30:
    chunks.append(np.ones(chunk // 8, dtype=np.float64))
print(f"READY allocated_mb={len(chunks) * 256} available_mb={psutil.virtual_memory().available / 2 ** 20:.0f}",
      flush=True)

done = threading.Event()
threading.Thread(target=lambda: (sys.stdin.read(), done.set()), daemon=True).start()
t0 = time.time()
while not done.is_set() and time.time() - t0 < args.hold_seconds:
    # Keep re-touching the memory so it stays resident and competes with the model
    for c in chunks:
        c[::512] += 1.0
        if done.is_set():
            break
    if psutil.virtual_memory().available / 2 ** 20 > args.target_mb * 1.5:
        chunks.append(np.ones(chunk // 8, dtype=np.float64))
