from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for plot generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "static": "#2b5c8f",
    "balanced": "#d95f02",
    "latency_optimized": "#7570b3",
    "rl_policy": "#e7298a"
}


def _style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_latency_by_load(summary: pd.DataFrame, condition_col: str, path: Path, title: str):
    """Grouped bars: mean latency per condition at each background-load level, with 95% CIs."""
    _style()
    loads = sorted(summary["load_workers"].unique())
    conditions = list(dict.fromkeys(summary[condition_col]))
    width = 0.8 / len(conditions)
    x = np.arange(len(loads))

    plt.figure(figsize=(10, 6))
    for i, cond in enumerate(conditions):
        sub = summary[summary[condition_col] == cond].set_index("load_workers").reindex(loads)
        err = np.vstack([
            (sub["mean_latency"] - sub["ci95_low"]).clip(lower=0).fillna(0),
            (sub["ci95_high"] - sub["mean_latency"]).clip(lower=0).fillna(0)
        ])
        plt.bar(x + i * width - 0.4 + width / 2, sub["mean_latency"], width,
                yerr=err, capsize=3, label=cond, color=COLORS.get(cond))

    plt.xticks(x, [f"{l} busy workers" for l in loads])
    plt.ylabel("Mean latency (seconds)")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(title="Condition")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Generated plot: {path}", flush=True)


def plot_paired_differences(diffs: pd.DataFrame, path: Path, title: str):
    """Mean paired latency difference vs the baseline, with 95% CIs, per load level."""
    _style()
    labels = [f"{r.condition}\n@{r.load_workers} workers" if "load_workers" in diffs else r.condition
              for r in diffs.itertuples()]
    err = np.vstack([
        (diffs["mean_diff_seconds"] - diffs["ci95_low"]).clip(lower=0).fillna(0),
        (diffs["ci95_high"] - diffs["mean_diff_seconds"]).clip(lower=0).fillna(0)
    ])

    plt.figure(figsize=(max(8, len(diffs) * 1.2), 5))
    plt.bar(range(len(diffs)), diffs["mean_diff_seconds"], yerr=err, capsize=4,
            color=[COLORS.get(c) for c in diffs["condition"]])
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(range(len(diffs)), labels, fontsize=9)
    plt.ylabel(f"Latency difference vs {diffs['baseline'].iloc[0]} (s)\n(negative = faster)")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Generated plot: {path}", flush=True)


def plot_strategy_distribution(choices: pd.DataFrame, path: Path, title: str):
    """Stacked bars: which strategy a policy chose at each load level."""
    _style()
    table = choices.groupby(["load_workers", "selected_strategy"]).size().unstack(fill_value=0)

    plt.figure(figsize=(8, 5))
    bottom = np.zeros(len(table))
    for strat in table.columns:
        plt.bar([str(l) for l in table.index], table[strat], bottom=bottom, label=strat, color=COLORS.get(strat))
        bottom += table[strat].values

    plt.xlabel("Background busy workers")
    plt.ylabel("Count")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(title="Strategy")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Generated plot: {path}", flush=True)
