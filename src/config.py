import copy
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"

DEFAULTS = {
    "model": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "dtype": "float32"
    },
    "strategies": {
        "static": {"layer_ranges": [[0, 21]], "thread_fractions": [1.0]},
        "balanced": {"layer_ranges": [[0, 10], [11, 21]], "thread_fractions": [0.3, 0.3]},
        "latency_optimized": {"layer_ranges": [[0, 21]], "thread_fractions": [0.6]}
    }
}


def _merge(base, override):
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


@lru_cache(maxsize=1)
def load_config():
    """Load configs/config.yaml merged over DEFAULTS."""

    config = copy.deepcopy(DEFAULTS)

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _merge(config, yaml.safe_load(f))

    return config


if __name__ == "__main__":
    import pprint
    pprint.pprint(load_config())
