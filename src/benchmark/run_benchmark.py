import time
import pandas as pd

from src.data.load_prompts import load_prompts
from src.llm.inference import generate_response


def run_benchmark(num_prompts=10):

    prompts = load_prompts(limit=num_prompts)

    results = []

    print(f"\nRunning benchmark on {num_prompts} prompts...\n")

    for idx, prompt in enumerate(prompts, start=1):

        print(f"[{idx}/{num_prompts}] Processing...")

        start = time.perf_counter()

        response = generate_response(prompt)

        end = time.perf_counter()

        latency = end - start

        results.append({
            "prompt_id": idx,
            "prompt": prompt,
            "response": response,
            "latency_seconds": round(latency, 3)
        })

    df = pd.DataFrame(results)

    output_path = "data/processed/inference_results.csv"

    df.to_csv(output_path, index=False)

    print("\nBenchmark completed.")

    print(f"Results saved to:\n{output_path}")


if __name__ == "__main__":
    run_benchmark(num_prompts=10)