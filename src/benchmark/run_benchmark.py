import time
import pandas as pd

from src.data.load_prompts import load_prompts
from src.llm.inference import generate_response
from src.telemetry.monitor import collect_metrics


def run_benchmark(num_prompts=10):

    prompts = load_prompts(limit=num_prompts)

    results = []

    print(f"\nRunning benchmark on {num_prompts} prompts...\n")

    for idx, prompt in enumerate(prompts, start=1):

        print(f"[{idx}/{num_prompts}] Processing...")

        start = time.perf_counter()

        response = generate_response(prompt)

        latency = time.perf_counter() - start

        metrics = collect_metrics()

        results.append({
            "prompt_id": idx,
            "timestamp": metrics["timestamp"],
            "prompt_length": len(prompt),
            "response_length": len(response),
            "latency_seconds": round(latency, 3),
            "tokens_per_second": round(len(response.split()) / latency, 2),
            "cpu_percent": metrics["cpu_percent"],
            "ram_percent": metrics["ram_percent"],
            "ram_used_gb": metrics["ram_used_gb"],
            "partition_strategy": "static"
        })

    df = pd.DataFrame(results)

    output_path = "data/processed/runtime_metrics.csv"

    df.to_csv(output_path, index=False)

    print("\nRuntime metrics saved successfully!")
    print(output_path)


if __name__ == "__main__":
    run_benchmark(num_prompts=20)