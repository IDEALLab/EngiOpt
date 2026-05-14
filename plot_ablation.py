"""
Reads all eval_n*_s*.json files from results/evaluation/ and produces
the ablation study plot: one panel per metric, x = training data size,
y = mean across seeds, shaded band = std across seeds.
"""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = "results/evaluation"
METRICS = {
    "mmd":       "MMD (Wing Shape)",
    "aoa_mse":   "MSE (Angle of Attack)",
    "vendi":     "Vendi Score",
    "shape_mse": "Shape MSE",
}


def load_results(eval_dir):
    """Returns dict: n_samples -> list of per-seed metric dicts."""
    by_n = defaultdict(list)
    for fname in sorted(os.listdir(eval_dir)):
        if not (fname.startswith("eval_n") and fname.endswith(".json")):
            continue
        path = os.path.join(eval_dir, fname)
        with open(path) as f:
            data = json.load(f)
        n = data["n_samples"]
        if n is None:
            continue
        # Store the mean value for each metric across the 10 forward passes
        by_n[n].append({k: v["mean"] for k, v in data["results"].items()})
    return by_n


def main():
    by_n = load_results(EVAL_DIR)
    if not by_n:
        print(f"No ablation JSON files found in {EVAL_DIR}")
        return

    sizes = sorted(by_n.keys())
    print(f"Found data for n_samples: {sizes}")
    for n in sizes:
        print(f"  n={n}: {len(by_n[n])} seed(s)")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (metric_key, metric_label) in zip(axes, METRICS.items()):
        means, stds = [], []
        for n in sizes:
            vals = [run[metric_key] for run in by_n[n] if metric_key in run]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(sizes, means, marker="o", linewidth=1.5, color="steelblue")
        ax.fill_between(sizes, means - stds, means + stds,
                        alpha=0.25, color="steelblue")
        ax.margins(y=0.3)
        ax.set_xlabel("Training dataset size")
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.set_xticks(sizes)
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    out_path = os.path.join("results", "ablation_plot.png")
    os.makedirs("results", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
