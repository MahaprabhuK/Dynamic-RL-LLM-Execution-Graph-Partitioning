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
    "rl_policy": "#e7298a",
    "rule_once": "#a6761d",
    "rule_adaptive": "#e6ab02",
    "rule_noh": "#66c2a5",
    "rl_once": "#66a61e",
    "rl_adaptive": "#e7298a",
    "bandit_adaptive": "#1b9e77",
    "cores_heuristic": "#7570b3"
}


def _style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_latency_by_load(summary: pd.DataFrame, condition_col: str, path: Path, title: str,
                         group_col: str = "load_workers"):
    """Grouped bars: mean latency per condition for each group (load level or scenario), with 95% CIs."""
    _style()
    loads = list(dict.fromkeys(summary[group_col])) if group_col != "load_workers" else sorted(summary[group_col].unique())
    conditions = list(dict.fromkeys(summary[condition_col]))
    width = 0.8 / len(conditions)
    x = np.arange(len(loads))

    plt.figure(figsize=(10, 6))
    for i, cond in enumerate(conditions):
        sub = summary[summary[condition_col] == cond].set_index(group_col).reindex(loads)
        err = np.vstack([
            (sub["mean_latency"] - sub["ci95_low"]).clip(lower=0).fillna(0),
            (sub["ci95_high"] - sub["mean_latency"]).clip(lower=0).fillna(0)
        ])
        plt.bar(x + i * width - 0.4 + width / 2, sub["mean_latency"], width,
                yerr=err, capsize=3, label=cond, color=COLORS.get(cond))

    plt.xticks(x, [f"{l} busy workers" if group_col == "load_workers" else str(l) for l in loads])
    plt.ylabel("Mean latency (seconds)")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(title="Condition")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Generated plot: {path}", flush=True)


def plot_paired_differences(diffs: pd.DataFrame, path: Path, title: str,
                            group_col: str = "load_workers", label_col: str = "condition"):
    """Mean paired latency differences with 95% CIs, one bar per (group, label) row."""
    _style()
    labels = [f"{getattr(r, label_col)}\n@{getattr(r, group_col)}" if group_col in diffs else getattr(r, label_col)
              for r in diffs.itertuples()]
    err = np.vstack([
        (diffs["mean_diff_seconds"] - diffs["ci95_low"]).clip(lower=0).fillna(0),
        (diffs["ci95_high"] - diffs["mean_diff_seconds"]).clip(lower=0).fillna(0)
    ])

    plt.figure(figsize=(max(8, len(diffs) * 1.2), 5))
    plt.bar(range(len(diffs)), diffs["mean_diff_seconds"], yerr=err, capsize=4,
            color=[COLORS.get(c) for c in diffs[label_col]])
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(range(len(diffs)), labels, fontsize=9)
    reference = diffs["baseline"].iloc[0] if label_col == "condition" else diffs["condition"].iloc[0]
    plt.ylabel(f"Latency difference (s)\n({'negative = faster than ' + reference if label_col == 'condition' else reference + ' minus baseline'})")
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


def plot_strategy_timeline(timeline: pd.DataFrame, path: Path, title: str):
    """
    Per-token latency over the course of one request for each condition,
    with markers showing which strategy was active at each token.
    """
    _style()
    markers = {0: "o", 1: "s", 2: "^"}
    names = {0: "static", 1: "balanced", 2: "latency_optimized"}

    fig, ax = plt.subplots(figsize=(11, 6))
    for cond, sub in timeline.groupby("condition"):
        sub = sub.sort_values("token_step")
        ax.plot(sub["token_step"], sub["token_seconds"], label=cond, color=COLORS.get(cond), linewidth=1.2)
        for sid, pts in sub.groupby("strategy_id"):
            ax.scatter(pts["token_step"][::5], pts["token_seconds"][::5], marker=markers.get(int(sid), "x"),
                       color=COLORS.get(cond), s=18)

    for sid, m in markers.items():
        ax.scatter([], [], marker=m, color="gray", label=f"active: {names[sid]}")
    ax.set_xlabel("Generated token")
    ax.set_ylabel("Seconds per token")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Generated plot: {path}", flush=True)