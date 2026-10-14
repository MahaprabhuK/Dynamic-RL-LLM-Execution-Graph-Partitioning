import os
import sys
import time
import psutil
from datetime import datetime
from typing import Any, TYPE_CHECKING

_torch_lib = r"C:\Users\Pavithran\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib"
if os.path.exists(_torch_lib):
    try:
        os.add_dll_directory(_torch_lib)
    except Exception:
        pass
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

try:
    import torch
except (ImportError, OSError) as e:
    if isinstance(e, OSError) and getattr(e, "winerror", None) != 1114 and "c10" not in str(e) and "DLL" not in str(e):
        raise
    torch = None


def collect_metrics(interval: float = 0.1):
    """
    Snapshot of system load. `cpu_percent` is averaged over `interval` seconds.
    Includes swap and page fault metrics for independent paging telemetry.
    """
    cpu = psutil.cpu_percent(interval=interval)
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    proc = psutil.Process()
    mem_info = proc.memory_info()

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_percent": cpu,
        "ram_percent": memory.percent,
        "ram_used_gb": round(memory.used / (1024 ** 3), 2),
        "ram_available_gb": round(memory.available / (1024 ** 3), 2),
        "swap_sin": getattr(swap, "sin", 0) or 0,
        "num_page_faults": getattr(mem_info, "num_page_faults", 0) or 0,
    }


class PagingTelemetryTracker:
    """
    Independent paging telemetry tracker.
    Logs:
    - psutil.swap_memory().sin delta (pages/bytes swapped into RAM from disk)
    - process page faults delta: psutil.Process().memory_info().num_page_faults (on Windows)
      or major page faults where the OS exposes them.
    """

    def __init__(self, process: psutil.Process | None = None):
        self.process = process or psutil.Process()
        self.reset()

    def reset(self):
        sm = psutil.swap_memory()
        self._last_sin = getattr(sm, "sin", 0) or 0
        mi = self.process.memory_info()
        self._last_faults = getattr(mi, "num_page_faults", 0) or 0

    def sample_delta(self) -> dict[str, Any]:
        sm = psutil.swap_memory()
        curr_sin = getattr(sm, "sin", 0) or 0
        mi = self.process.memory_info()
        curr_faults = getattr(mi, "num_page_faults", 0) or 0

        sin_delta = max(0, curr_sin - self._last_sin)
        faults_delta = max(0, curr_faults - self._last_faults)
        self._last_sin = curr_sin
        self._last_faults = curr_faults

        return {
            "swap_in_delta": sin_delta,
            "page_faults_delta": faults_delta,
            "has_paging_signal": sin_delta > 0,
        }


class SegmentPagingTracker:
    """
    Paging and RAM telemetry for one token segment (F3).

    Each sample() returns the change since the previous sample of
    psutil.swap_memory().sin / .sout (always 0 on Windows, where psutil does not
    expose them), this process's page-fault count (num_page_faults on Windows;
    includes soft faults), the machine-wide hard paging count (Windows
    "Memory\Pages Input/sec" raw counter; -1 when unavailable) and the current
    available RAM in MB.
    """

    _system = None

    def __init__(self, process: psutil.Process | None = None):
        self.process = process or psutil.Process()
        if SegmentPagingTracker._system is None:
            from src.telemetry.system_paging import SystemPagesInput
            SegmentPagingTracker._system = SystemPagesInput()  # one PDH query per process
        self._last = self._read()
        self._last_system = self._system.read()

    def _read(self):
        sm = psutil.swap_memory()
        mi = self.process.memory_info()
        faults = getattr(mi, "num_page_faults", None)
        if faults is None:  # non-Windows: major faults where available
            try:
                faults = self.process.memory_full_info().pfaults
            except Exception:
                faults = 0
        return getattr(sm, "sin", 0) or 0, getattr(sm, "sout", 0) or 0, int(faults or 0)

    def sample(self):
        sin, sout, faults = self._read()
        last_sin, last_sout, last_faults = self._last
        self._last = (sin, sout, faults)
        system, last_system = self._system.read(), self._last_system
        self._last_system = system
        vm = psutil.virtual_memory()
        return {
            "swap_sin_delta": max(0, sin - last_sin),
            "swap_sout_delta": max(0, sout - last_sout),
            "page_faults_delta": max(0, faults - last_faults),
            "system_pages_input_delta": -1 if system is None or last_system is None else max(0, system - last_system),
            "available_mb": round(vm.available / 2 ** 20, 1),
            "ram_percent": vm.percent,
        }


class ExternalLoadMonitor:
    """
    Non-blocking load telemetry for decisions made DURING generation.

    System CPU while generating includes our own inference threads, which would
    make the policy react to itself. This monitor subtracts this process's CPU
    usage, giving the load caused by everything else on the machine, which is
    comparable to the pre-run CPU measured before a request starts.

    Each sample() averages over the time since the previous sample.
    """

    def __init__(self):
        self.process = psutil.Process()
        self.num_cpus = psutil.cpu_count() or 1
        self.paging_tracker = PagingTelemetryTracker(self.process)
        self.reset()

    def reset(self):
        psutil.cpu_percent(interval=None)
        self.process.cpu_percent(interval=None)
        self.paging_tracker.reset()
        self._last_time = time.perf_counter()

    def sample(self):
        system_cpu = psutil.cpu_percent(interval=None)
        own_cpu = self.process.cpu_percent(interval=None) / self.num_cpus
        now = time.perf_counter()
        window = now - self._last_time
        self._last_time = now
        paging = self.paging_tracker.sample_delta()

        return {
            "cpu_percent": system_cpu,
            "own_cpu_percent": round(own_cpu, 2),
            "external_cpu_percent": round(max(0.0, system_cpu - own_cpu), 2),
            "ram_percent": psutil.virtual_memory().percent,
            "window_seconds": round(window, 3),
            "swap_in_delta": paging["swap_in_delta"],
            "page_faults_delta": paging["page_faults_delta"],
            "has_paging_signal": paging["has_paging_signal"],
        }


class BackgroundLoadSampler:
    """
    Continuously samples external CPU load in a daemon thread, so a scheduler
    can read the current load at any time without blocking generation.

    Uses raw cpu_times() deltas (not psutil.cpu_percent), so it does not
    interfere with other psutil users, and smooths readings with an EMA.

        with BackgroundLoadSampler() as sampler:
            sampler.latest()["external_cpu_percent"]
    """

    def __init__(self, interval: float = 0.25, smoothing: float = 0.5):
        self.interval = interval
        self.smoothing = smoothing
        self.process = psutil.Process()
        self.num_cpus = psutil.cpu_count() or 1
        self.paging_tracker = PagingTelemetryTracker(self.process)
        self._latest = None
        self._lock = None
        self._stop = None
        self._thread = None

    @staticmethod
    def _busy_and_total(t):
        total = sum(t)
        return total - t.idle, total

    def _run(self):
        last_sys = self._busy_and_total(psutil.cpu_times())
        last_own = sum(self.process.cpu_times()[:2])
        last_time = time.perf_counter()

        while not self._stop.wait(self.interval):
            busy, total = self._busy_and_total(psutil.cpu_times())
            own = sum(self.process.cpu_times()[:2])
            now = time.perf_counter()

            system_cpu = 100.0 * (busy - last_sys[0]) / max(total - last_sys[1], 1e-9)
            own_cpu = 100.0 * (own - last_own) / max((now - last_time) * self.num_cpus, 1e-9)
            external = min(100.0, max(0.0, system_cpu - own_cpu))
            last_sys, last_own, last_time = (busy, total), own, now
            paging = self.paging_tracker.sample_delta()

            with self._lock:
                prev = self._latest
                a = self.smoothing if prev is not None else 1.0
                ema = lambda key, value: value if prev is None else a * value + (1 - a) * prev[key]
                self._latest = {
                    "cpu_percent": round(ema("cpu_percent", system_cpu), 2),
                    "own_cpu_percent": round(ema("own_cpu_percent", own_cpu), 2),
                    "external_cpu_percent": round(ema("external_cpu_percent", external), 2),
                    "ram_percent": psutil.virtual_memory().percent,
                    "window_seconds": self.interval,
                    "swap_in_delta": paging["swap_in_delta"],
                    "page_faults_delta": paging["page_faults_delta"],
                    "has_paging_signal": paging["has_paging_signal"],
                }

    def start(self):
        import threading

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        time.sleep(self.interval * 2.5)  # first readings
        return self

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            self._thread = None

    def latest(self):
        with self._lock:
            if self._latest is None:
                return {"cpu_percent": 0.0, "own_cpu_percent": 0.0, "external_cpu_percent": 0.0,
                        "ram_percent": psutil.virtual_memory().percent, "window_seconds": 0.0,
                        "swap_in_delta": 0, "page_faults_delta": 0, "has_paging_signal": False}
            return dict(self._latest)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


def measure_activation_handoff(
    tensor: Any | None,
    stage_id: int,
    layer_start: int | None,
    layer_end: int | None,
    stage_latency_seconds: float,
    num_threads: int | None = None
):
    """
    Measures tensor activation copy/handoff overhead and size at a partition boundary.
    Pass tensor=None for a stage that is not followed by a boundary (no handoff).

    Note: the copy is an in-process memory clone, so it is a lower bound on the
    cost of a real inter-device or inter-process transfer.

    Returns:
        dict with activation metrics and handoff timing.
    """
    if tensor is None:
        return {
            "partition_id": stage_id,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "num_threads": num_threads,
            "stage_latency_seconds": round(stage_latency_seconds, 6),
            "activation_shape": [],
            "activation_dtype": "none",
            "activation_nelement": 0,
            "activation_size_bytes": 0,
            "activation_size_mb": 0.0,
            "handoff_latency_seconds": 0.0
        }

    if torch is None:
        raise RuntimeError("PyTorch is required to measure activation handoffs.")

    # Measure exact physical tensor memory copy
    t0 = time.perf_counter()
    _copied_tensor = tensor.clone()
    handoff_latency = time.perf_counter() - t0

    num_elements = tensor.nelement()
    size_bytes = num_elements * tensor.element_size()
    size_mb = size_bytes / (1024 ** 2)

    return {
        "partition_id": stage_id,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "num_threads": num_threads,
        "stage_latency_seconds": round(stage_latency_seconds, 6),
        "activation_shape": list(tensor.shape),
        "activation_dtype": str(tensor.dtype),
        "activation_nelement": num_elements,
        "activation_size_bytes": size_bytes,
        "activation_size_mb": round(size_mb, 6),
        "handoff_latency_seconds": round(handoff_latency, 8)
    }


if __name__ == "__main__":
    import torch

    metrics = collect_metrics()
    print(metrics)

    dummy_tensor = torch.randn(1, 100, 2048)
    handoff_info = measure_activation_handoff(dummy_tensor, stage_id=1, layer_start=0, layer_end=10, stage_latency_seconds=0.05)
    print("Sample Handoff Info:", handoff_info)
