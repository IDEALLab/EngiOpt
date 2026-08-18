"""Render the simplest-possible self-attention diagram for the DCC26 deck.

    X  ->  Q, K, V  ->  softmax(Q K^T / sqrt(d)) -> (.)V  ->  output

Produces ``self_attention.png`` in the deck palette (matplotlib mathtext, no
LaTeX install needed).

    python workshops/dcc26/slides/build_self_attention.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["mathtext.fontset"] = "cm"

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "assets"))

# deck palette
BG = "#fbfaf7"
INK = "#17212b"
BLUE = "#225e9b"
BLUE_TINT = "#ecf3fa"
GREY = "#5a6675"
QKV = {"Q": "#2c5f8a", "K": "#8a4b2c", "V": "#2f7d62"}


def box(ax, cx, cy, w, h, *, face, edge, lw=2.0, rounding=0.025):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={rounding}",
        linewidth=lw, edgecolor=edge, facecolor=face,
    ))


def arrow(ax, x0, y0, x1, y1, color=INK, lw=2.4):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=20,
        lw=lw, color=color, shrinkA=2, shrinkB=2,
    ))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(8.5, 7.5))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.955, "Self-attention", ha="center", va="center",
            fontsize=24, fontweight="bold", color=INK)

    # ---- input X ----
    box(ax, 0.5, 0.86, 0.22, 0.075, face=BLUE_TINT, edge=BLUE)
    ax.text(0.5, 0.86, r"$X$", ha="center", va="center", fontsize=20, color=INK)
    ax.text(0.5, 0.805, "input tokens", ha="center", va="center", fontsize=11,
            style="italic", color=GREY)

    # ---- Q, K, V projections ----
    qx, kx, vx, qkv_y = 0.20, 0.50, 0.80, 0.66
    for name, x in (("Q", qx), ("K", kx), ("V", vx)):
        arrow(ax, 0.5, 0.81, x, qkv_y + 0.05, color=GREY)
        box(ax, x, qkv_y, 0.18, 0.085, face="white", edge=QKV[name])
        ax.text(x, qkv_y + 0.012, f"${name}$", ha="center", va="center",
                fontsize=19, color=QKV[name])
        ax.text(x, qkv_y - 0.028, rf"$= X\,W_{name}$", ha="center", va="center",
                fontsize=11, color=GREY)

    # ---- scores: softmax(Q K^T / sqrt d) ----
    sc_y = 0.42
    box(ax, 0.35, sc_y, 0.46, 0.11, face=BLUE_TINT, edge=BLUE)
    ax.text(0.35, sc_y + 0.012,
            r"$\mathrm{softmax}\!\left(\dfrac{Q\,K^{\top}}{\sqrt{d_k}}\right)$",
            ha="center", va="center", fontsize=18, color=INK)
    ax.text(0.35, sc_y - 0.04, "attention weights", ha="center", va="center",
            fontsize=10.5, style="italic", color=GREY)
    # Q and K feed the scores box
    arrow(ax, qx, qkv_y - 0.05, 0.27, sc_y + 0.06, color=QKV["Q"])
    arrow(ax, kx, qkv_y - 0.05, 0.40, sc_y + 0.06, color=QKV["K"])

    # ---- multiply by V ----
    mul_y = 0.24
    ax.text(0.5, mul_y, r"$\times$", ha="center", va="center", fontsize=24, color=INK)
    arrow(ax, 0.35, sc_y - 0.06, 0.47, mul_y + 0.02, color=BLUE)   # scores -> x
    arrow(ax, vx, qkv_y - 0.05, 0.535, mul_y + 0.02, color=QKV["V"])  # V -> x

    # ---- output ----
    out_y = 0.095
    arrow(ax, 0.5, mul_y - 0.03, 0.5, out_y + 0.045, color=INK)
    box(ax, 0.5, out_y, 0.30, 0.08, face=INK, edge=INK)
    ax.text(0.5, out_y, "output", ha="center", va="center", fontsize=15,
            fontweight="bold", color="white")

    # ---- one-line formula footer ----
    ax.text(0.5, 0.025,
            r"$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(QK^{\top}/\sqrt{d_k}\right)V$",
            ha="center", va="center", fontsize=13, color=GREY)

    fig.savefig(os.path.join(OUT_DIR, "self_attention.png"),
                dpi=220, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", os.path.join(OUT_DIR, "self_attention.png"))


if __name__ == "__main__":
    main()
