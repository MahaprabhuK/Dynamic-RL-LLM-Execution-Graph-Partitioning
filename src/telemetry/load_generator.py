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


class DynamicCPULoad:
    """
    Background load that changes over time, to simulate contention that starts
    or stops while a request is being generated.

        schedule = [(0.0, 0), (5.0, 8)]   # idle, then 8 busy workers after 5 s
        with DynamicCPULoad(schedule):
            ...

    Times are seconds relative to entering the context.
    """

    def __init__(self, schedule):
        self.schedule = sorted(schedule)
        self._procs = []
        self._stop = None
        self._thread = None

    def _set_workers(self, n):
        while len(self._procs) < n:
            self._procs.append(subprocess.Popen([sys.executable, "-c", _BUSY_LOOP]))
        while len(self._procs) > n:
            p = self._procs.pop()
            p.terminate()
            p.wait(timeout=5)

    def _run(self, start):
        for at, workers in self.schedule:
            if self._stop.wait(timeout=max(0.0, start + at - time.perf_counter())):
                return
            self._set_workers(workers)

    def __enter__(self):
        import threading

        self._stop = threading.Event()
        start = time.perf_counter()
        # Apply the initial level synchronously so it is in place before work starts
        if self.schedule and self.schedule[0][0] <= 0:
            self._set_workers(self.schedule[0][1])
            if self.schedule[0][1] > 0:
                time.sleep(0.5)
                start = time.perf_counter()
        self._thread = threading.Thread(target=self._run, args=(start,), daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)
        self._set_workers(0)
        return False


if __name__ == "__main__":
    import psutil

    for workers in [0, 4]:
        with BackgroundCPULoad(workers):
            print(f"workers={workers}: cpu={psutil.cpu_percent(interval=1.0)}%")
