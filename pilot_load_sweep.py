"""
Load sweep: decode seconds per token of each fixed strategy at several
background-load levels, plus the external CPU reading the schedulers see.
Used to pick load levels at which the best strategy actually differs.

    python pilot_load_sweep.py --loads 0,4,8,12 --repeats 2
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.load_prompts import load_prompts
from src.llm.inference import generate_partitioned_response
from src.llm.partition import PartitionStrategy
from src.telemetry.load_generator import BackgroundCPULoad
from src.telemetry.monitor import BackgroundLoadSampler


def main(loads, repeats, tokens, seed, tag):
    rng = np.random.default_rng(seed)
    prompt = load_prompts(csv_path=str(ROOT_DIR / "data" / "processed" / "prompts.csv"), limit=21)[20]
    generate_partitioned_response(prompt, max_new_tokens=8)
    rows = []
    with BackgroundLoadSampler() as sampler:
        for rep in range(repeats):
            for load in rng.permutation(loads):
                with BackgroundCPULoad(int(load)):
                    time.sleep(0.5)
                    external = sampler.latest()["external_cpu_percent"]
                    for sid in rng.permutation(list(PartitionStrategy.STRATEGIES)):
                        _, stages = generate_partitioned_response(prompt, strategy=int(sid), max_new_tokens=tokens,
                                                                  ignore_eos=True)
                        per_token = pd.DataFrame(stages).groupby("token_step")["stage_latency_seconds"].sum()
                        rows.append({"repeat": rep, "load_workers": int(load), "strategy_id": int(sid),
                                     "strategy": PartitionStrategy(int(sid)).strategy_name,
                                     "decode_sec_per_token": per_token.iloc[1:].median(),
                                     "external_cpu_idle_reading": external,
                                     "external_cpu_during": sampler.latest()["external_cpu_percent"]})
                print(f"rep {rep} load {load} done", flush=True)
                time.sleep(1.0)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT_DIR / "data" / "processed" / f"load_sweep_{tag}.csv", index=False)
    table = df.pivot_table(index="load_workers", columns="strategy", values="decode_sec_per_token", aggfunc="mean")
    table["best"] = table.idxmin(axis=1)
    table["ext_cpu_before"] = df.groupby("load_workers")["external_cpu_idle_reading"].mean()
    table["ext_cpu_during"] = df.groupby("load_workers")["external_cpu_during"].mean()
    print(table.round(4).to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--loads", default="0,2,4,6,8,10,12,14")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--tag", default="v4")
    args = parser.parse_args()
    main([int(x) for x in args.loads.split(",")], args.repeats, args.tokens, args.seed, args.tag)
