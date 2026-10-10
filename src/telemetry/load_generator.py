import subprocess
import sys
import time

# Plain interpreter busy loop. Run via `python -c` (not multiprocessing) so the
# workers never re-import the calling script and its torch/transformers stack.
_BUSY_LOOP = "x = 0\nwhile True:\n    x = (x * 1103515245 + 12345) & 0x7FFFFFFF\n"


class BackgroundCPULoad:
    """
    Context manager that keeps `num_workers` CPU cores busy in separate
    processes. Used to create controlled contention levels so that the best
    execution strategy genuinely depends on the system state.

        with BackgroundCPULoad(4):
            ...  # run inference under contention
    """

    def __init__(self, num_workers: int = 0, settle_seconds: float = 0.5):
        self.num_workers = num_workers
        self.settle_seconds = settle_seconds
        self._procs = []

    def __enter__(self):
        self._procs = [
            subprocess.Popen([sys.executable, "-c", _BUSY_LOOP])
            for _ in range(self.num_workers)
        ]
        if self._procs:
            time.sleep(self.settle_seconds)  # let the load settle before measuring
        return self

    def __exit__(self, *exc):
        for p in self._procs:
            p.terminate()
        for p in self._procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        self._procs = []
        return False


if __name__ == "__main__":
    import psutil

    for workers in [0, 4]:
        with BackgroundCPULoad(workers):
            print(f"workers={workers}: cpu={psutil.cpu_percent(interval=1.0)}%")
