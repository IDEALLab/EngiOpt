"""Two text-light visuals for the GLUE / SuperGLUE benchmark-lifecycle slide.

1. ``glue_scoring.png``    -- how a GLUE score is computed: N task tiles, each with
                             its own metric, macro-averaged into one number.
2. ``glue_saturation.png`` -- "raising the bar": model SOTA climbs past GLUE's
                             human baseline in ~1 year, the community responds with
                             the harder SuperGLUE, and the climb repeats.

Milestones are real, well-known leaderboard values (rounded; illustrative dates).

    python workshops/dcc26/slides/build_glue_figs.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "assets"))

BG = "#fbfaf7"
INK = "#17212b"
BLUE = "#225e9b"
ORANGE = "#c47b20"
GREEN = "#2f7d62"
GREY = "#5a6675"
GREY_SOFT = "#7a8493"
BLUE_TINT = "#ecf3fa"
ORANGE_TINT = "#fbf1e2"

# GLUE's 9 tasks + the metric each is scored with (the point: many tasks, mixed metrics)
GLUE_TASKS = [
    ("CoLA", "Matthews"), ("SST-2", "acc"), ("MRPC", "F1"),
    ("STS-B", "Spearman"), ("QQP", "F1"), ("MNLI", "acc"),
    ("QNLI", "acc"), ("RTE", "acc"), ("WNLI", "acc"),
]


def fig_scoring() -> None:
    fig = plt.figure(figsize=(13, 2.0))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.012, 0.86, "How a GLUE score is computed", fontsize=15, fontweight="bold", color=INK)

    n = len(GLUE_TASKS)
    x0, x1 = 0.02, 0.66
    tile_w = (x1 - x0) / n * 0.86
    gap = (x1 - x0) / n
    for i, (task, metric) in enumerate(GLUE_TASKS):
        cx = x0 + i * gap
        ax.add_patch(FancyBboxPatch((cx, 0.34), tile_w, 0.30,
                     boxstyle="round,pad=0.004,rounding_size=0.02",
                     linewidth=1.3, edgecolor=BLUE, facecolor=BLUE_TINT))
        ax.text(cx + tile_w / 2, 0.50, task, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=INK)
        ax.text(cx + tile_w / 2, 0.405, metric, ha="center", va="center",
                fontsize=7.2, style="italic", color=GREY)
    ax.text((x0 + x1) / 2 - gap / 2, 0.20, "9 language tasks  ·  each scored by its own metric",
            ha="center", va="center", fontsize=9.5, color=GREY)

    # average arrow -> single score
    ax.add_patch(FancyArrowPatch((0.665, 0.49), (0.74, 0.49), arrowstyle="-|>",
                 mutation_scale=20, lw=2.4, color=INK))
    ax.text(0.703, 0.60, "mean", ha="center", va="bottom", fontsize=10, style="italic", color=INK)

    ax.add_patch(FancyBboxPatch((0.75, 0.30), 0.235, 0.40,
                 boxstyle="round,pad=0.006,rounding_size=0.03",
                 linewidth=0, facecolor=INK))
    ax.text(0.8675, 0.55, "GLUE score", ha="center", va="center", fontsize=11, color="#cfd6dd")
    ax.text(0.8675, 0.42, "80.5", ha="center", va="center", fontsize=22, fontweight="bold", color="white")

    ax.text(0.8675, 0.18, "unweighted average", ha="center", va="center",
            fontsize=8.5, style="italic", color=GREY_SOFT)

    fig.savefig(os.path.join(OUT_DIR, "glue_scoring.png"), dpi=220, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", os.path.join(OUT_DIR, "glue_scoring.png"))


def fig_saturation() -> None:
    # (decimal year, score) SOTA milestones
    glue = [(2018.5, 72.8, "GPT"), (2018.8, 80.5, "BERT"),
            (2019.2, 83.6, "BERT-ens"), (2019.5, 88.5, "RoBERTa"), (2019.8, 90.3, "T5")]
    sglue = [(2019.4, 69.0, "BERT"), (2019.6, 84.6, "RoBERTa"),
             (2019.9, 89.3, "T5"), (2021.0, 90.3, "DeBERTa")]
    HUMAN_GLUE, HUMAN_SGLUE = 87.1, 89.8

    fig, ax = plt.subplots(figsize=(11, 5.4))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    # human baselines
    ax.axhline(HUMAN_GLUE, color=BLUE, ls=(0, (5, 4)), lw=1.6, alpha=0.8)
    ax.axhline(HUMAN_SGLUE, color=ORANGE, ls=(0, (5, 4)), lw=1.6, alpha=0.8)
    ax.text(2018.42, HUMAN_GLUE + 0.3, "human baseline — GLUE (87.1)", fontsize=10,
            color=BLUE, va="bottom")
    ax.text(2021.05, HUMAN_SGLUE - 0.3, "human — SuperGLUE (89.8)", fontsize=10,
            color=ORANGE, va="top", ha="right")

    # GLUE climb
    gx = [p[0] for p in glue]; gy = [p[1] for p in glue]
    ax.plot(gx, gy, "-o", color=BLUE, lw=2.6, ms=8, mfc="white", mew=2, zorder=5)
    for x, y, name in glue:
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(-2, -14),
                    fontsize=8.5, color=BLUE, ha="center")
    # SuperGLUE climb
    sx = [p[0] for p in sglue]; sy = [p[1] for p in sglue]
    ax.plot(sx, sy, "-o", color=ORANGE, lw=2.6, ms=8, mfc="white", mew=2, zorder=5)
    for x, y, name in sglue:
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(2, 8),
                    fontsize=8.5, color=ORANGE, ha="center")

    # "GLUE saturated -> SuperGLUE launched" marker
    ax.axvline(2019.35, color=GREY, ls=":", lw=1.4, alpha=0.7)
    ax.annotate("GLUE saturated\n→ SuperGLUE launched", xy=(2019.35, 78),
                xytext=(2019.5, 73.5), fontsize=10.5, color=INK, fontweight="bold",
                ha="left", va="center")

    # series labels
    ax.text(2018.55, 70.0, "GLUE", color=BLUE, fontsize=14, fontweight="bold")
    ax.text(2020.55, 86.0, "SuperGLUE", color=ORANGE, fontsize=14, fontweight="bold")

    ax.set_xlim(2018.3, 2021.3); ax.set_ylim(66, 93)
    ax.set_xticks([2018, 2019, 2020, 2021])
    ax.set_xticklabels(["2018", "2019", "2020", "2021"], fontsize=11)
    ax.set_ylabel("benchmark score", fontsize=11, color=INK)
    ax.set_title("Each time models clear the bar, the community raises it",
                 fontsize=15, fontweight="bold", color=INK, loc="left", pad=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GREY_SOFT)
    ax.tick_params(colors=GREY)
    ax.grid(axis="y", color="#e6e3dd", lw=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "glue_saturation.png"), dpi=220, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", os.path.join(OUT_DIR, "glue_saturation.png"))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    fig_scoring()
    fig_saturation()


if __name__ == "__main__":
    main()
