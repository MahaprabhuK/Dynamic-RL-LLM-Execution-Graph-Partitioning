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


class TokenScheduledLoad(DynamicCPULoad):
    """
    Background load that changes after a given number of generated TOKENS
    rather than seconds, so every policy faces the change at the same point of
    the response regardless of how fast it is. Pass `on_step` as the
    generator's step_callback.

        load = TokenScheduledLoad([(0, 0), (60, 8)])   # 8 workers from token 60
        with load:
            generate_partitioned_response(..., step_callback=load.on_step)

    Workers are started in a background thread so the generator never blocks.
    """

    def __init__(self, schedule):
        super().__init__(schedule)
        self._next = 0

    def _apply_async(self, workers):
        import threading

        self._thread = threading.Thread(target=self._set_workers, args=(workers,), daemon=True)
        self._thread.start()

    def on_step(self, step):
        while self._next < len(self.schedule) and step >= self.schedule[self._next][0]:
            self._apply_async(self.schedule[self._next][1])
            self._next += 1

    def __enter__(self):
        self._next = 0
        if self.schedule and self.schedule[0][0] <= 0:
            self._set_workers(self.schedule[0][1])
            self._next = 1
            if self.schedule[0][1] > 0:
                time.sleep(0.5)
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        self._set_workers(0)
        return False


if __name__ == "__main__":
    import psutil

    for workers in [0, 4]:
        with BackgroundCPULoad(workers):
            print(f"workers={workers}: cpu={psutil.cpu_percent(interval=1.0)}%")
