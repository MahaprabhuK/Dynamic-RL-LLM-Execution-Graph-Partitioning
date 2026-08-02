import psutil
from datetime import datetime


def collect_metrics():
    cpu = psutil.cpu_percent(interval=0.1)

    memory = psutil.virtual_memory()

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_percent": cpu,
        "ram_percent": memory.percent,
        "ram_used_gb": round(memory.used / (1024 ** 3), 2)
    }


if __name__ == "__main__":
    metrics = collect_metrics()
    print(metrics)