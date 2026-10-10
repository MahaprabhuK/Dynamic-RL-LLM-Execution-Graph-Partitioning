import time
import psutil
from datetime import datetime
import torch


def collect_metrics(interval: float = 0.1):
    """
    Snapshot of system load. `cpu_percent` is averaged over `interval` seconds.
    """
    cpu = psutil.cpu_percent(interval=interval)

    memory = psutil.virtual_memory()

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_percent": cpu,
        "ram_percent": memory.percent,
        "ram_used_gb": round(memory.used / (1024 ** 3), 2),
        "ram_available_gb": round(memory.available / (1024 ** 3), 2)
    }


def measure_activation_handoff(
    tensor: torch.Tensor,
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
    metrics = collect_metrics()
    print(metrics)

    dummy_tensor = torch.randn(1, 100, 2048)
    handoff_info = measure_activation_handoff(dummy_tensor, stage_id=1, layer_start=0, layer_end=10, stage_latency_seconds=0.05)
    print("Sample Handoff Info:", handoff_info)
