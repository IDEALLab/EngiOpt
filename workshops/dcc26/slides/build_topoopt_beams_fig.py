"""Build slide figures explaining topology optimization on the beams2d problem.

Produces two figures for the DCC26 intro deck:

1. ``beams2d_dataset_grid.png`` -- a diversity-sampled grid of training designs,
   each rendered like ``problem.render`` (coolwarm density field) and annotated
   with its conditions (volume fraction, rmin, force position).

2. ``beams2d_warmstart.png`` -- the constant-volume-fraction "warm start" that
   seeds the optimizer, shown next to the optimized design it converges to.
   This is the initial guess used to generate every sample in the dataset
   (``xPhys = volfrac * ones(...)`` in EngiBench's beams2d backend).

Run inside the EngiBench312 conda env:

    python workshops/dcc26/slides/build_topoopt_beams_fig.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from engibench.utils.all_problems import BUILTIN_PROBLEMS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROBLEM_ID = "beams2d"
SEED = 0
N_GRID = 12          # number of diverse designs in the dataset grid
GRID_ROWS, GRID_COLS = 3, 4
SPLIT = "train"
CMAP = "viridis"           # grid colormap
CMAP_WARMSTART = "viridis"  # warm-start figure colormap

OUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "assets")
OUT_DIR = os.path.abspath(OUT_DIR)


def farthest_point_sample(points: np.ndarray, n: int, rng: np.random.Generator) -> list[int]:
    """Greedy farthest-point sampling for maximal spread in condition space."""
    n_total = len(points)
    # Normalize each dimension to [0, 1] so no condition dominates the distance.
    mins = points.min(axis=0)
    spans = np.where(points.max(axis=0) - mins > 0, points.max(axis=0) - mins, 1.0)
    norm = (points - mins) / spans

    start = int(rng.integers(n_total))
    selected = [start]
    dists = np.linalg.norm(norm - norm[start], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(dists))
        selected.append(nxt)
        dists = np.minimum(dists, np.linalg.norm(norm - norm[nxt], axis=1))
    return selected


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    problem = BUILTIN_PROBLEMS[PROBLEM_ID]()
    problem.reset(seed=SEED)
    nely, nelx = problem.design_space.shape
    print(f"design shape (nely, nelx) = ({nely}, {nelx})")

    ds = problem.dataset[SPLIT]
    designs = np.array(ds["optimal_design"])
    volfrac = np.array(ds["volfrac"])
    rmin = np.array(ds["rmin"])
    forcedist = np.array(ds["forcedist"])
    compliance = np.array(ds["c"])
    print(f"loaded {len(designs)} {SPLIT} designs")

    # ------------------------------------------------------------------
    # Figure 1: diverse dataset grid
    # ------------------------------------------------------------------
    # Drop the extreme force position (force at the support) -- those collapse
    # to a compact blob rather than a recognizable truss, which reads poorly on
    # a slide. Sample for diversity only among the remaining designs.
    pool = np.flatnonzero(forcedist < 1.0)
    cond_pts = np.stack([volfrac[pool], rmin[pool], forcedist[pool]], axis=1)
    local_idxs = farthest_point_sample(cond_pts, N_GRID, rng)
    idxs = [int(pool[j]) for j in local_idxs]
    # Sort the chosen designs by volfrac then forcedist for a tidy reading order.
    idxs = sorted(idxs, key=lambda i: (volfrac[i], forcedist[i]))

    fig, axes = plt.subplots(
        GRID_ROWS, GRID_COLS, figsize=(GRID_COLS * 2.7, GRID_ROWS * 1.9)
    )
    fig.suptitle(
        "beams2d training data: optimized cantilevers across conditions",
        fontsize=14, fontweight="bold", y=0.99,
    )
    for ax, i in zip(axes.ravel(), idxs):
        ax.imshow(designs[i], cmap=CMAP, vmin=0, vmax=1, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"vf={volfrac[i]:.2f}  rmin={rmin[i]:.2f}\n"
            f"force@{forcedist[i]:.2f}  C={compliance[i]:.1f}",
            fontsize=8.5,
        )
    # Hide any unused axes.
    for ax in axes.ravel()[len(idxs):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    grid_path = os.path.join(OUT_DIR, "beams2d_dataset_grid.png")
    fig.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {grid_path}")

    # ------------------------------------------------------------------
    # Figure 2: warm start -> optimized
    # ------------------------------------------------------------------
    # Pick a representative mid-range sample to illustrate the pipeline.
    target_vf = 0.35
    rep = int(np.argmin(np.abs(volfrac - target_vf) + np.abs(forcedist - 0.0)))
    vf = float(volfrac[rep])
    warmstart = vf * np.ones((nely, nelx))

    fig = plt.figure(figsize=(10.5, 3.2))
    gs = GridSpec(1, 3, width_ratios=[1, 0.22, 1], wspace=0.05)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(warmstart, cmap=CMAP_WARMSTART, vmin=0, vmax=1, aspect="equal")
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title(
        f"Warm start\nuniform density = volfrac = {vf:.2f}",
        fontsize=12, fontweight="bold",
    )

    axA = fig.add_subplot(gs[1])
    axA.axis("off")
    axA.annotate(
        "", xy=(0.95, 0.5), xytext=(0.05, 0.5),
        arrowprops=dict(arrowstyle="-|>", lw=2.5, color="black"),
    )
    axA.text(
        0.5, 0.62, "topology\noptimization", ha="center", va="bottom",
        fontsize=10, transform=axA.transAxes,
    )

    ax1 = fig.add_subplot(gs[2])
    ax1.imshow(designs[rep], cmap=CMAP_WARMSTART, vmin=0, vmax=1, aspect="equal")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(
        f"Optimized design\nC={compliance[rep]:.1f}, force@{forcedist[rep]:.2f}",
        fontsize=12, fontweight="bold",
    )

    fig.suptitle(
        "Every dataset sample starts from the same uniform-density block",
        fontsize=13, y=1.10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    warm_path = os.path.join(OUT_DIR, "beams2d_warmstart.png")
    fig.savefig(warm_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {warm_path}")


if __name__ == "__main__":
    main()
