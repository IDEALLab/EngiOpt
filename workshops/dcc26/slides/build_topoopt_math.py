"""Render a compact 'topology optimization math' panel for the DCC26 deck.

Produces ``beams2d_topoopt_math.png`` -- the SIMP compliance-minimization
formulation that EngiBench's beams2d actually solves, laid out for a slide:
the constrained problem (boxed), the two ingredients that explain the pictures
(SIMP penalization + density filter), and the Optimality-Criteria update as a
footnote.

Uses matplotlib mathtext (Computer Modern) so it needs no LaTeX install.

Run inside any env with matplotlib:

    python workshops/dcc26/slides/build_topoopt_math.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["mathtext.fontset"] = "cm"  # Computer Modern -> LaTeX look

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "assets"))

# Color accents
INK = "#1a1a1a"
ACCENT = "#2c5f8a"   # SIMP
ACCENT2 = "#8a4b2c"  # filter
BOX_FACE = "#f4f7fb"
BOX_EDGE = "#2c5f8a"


def main(transparent: bool = False) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(11, 7))
    fig.patch.set_alpha(0.0 if transparent else 1.0)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")

    # ---- Title -------------------------------------------------------
    ax.text(
        0.5, 0.945, "Topology optimization: find the stiffest layout",
        ha="center", va="center", fontsize=20, fontweight="bold", color=INK,
    )

    # ---- Tier 1: the constrained problem (boxed) ---------------------
    box = FancyBboxPatch(
        (0.06, 0.58), 0.88, 0.28,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=2, edgecolor=BOX_EDGE, facecolor=BOX_FACE, transform=ax.transAxes,
    )
    ax.add_patch(box)

    ax.text(
        0.10, 0.795,
        r"$\min_{\mathbf{x}\,\in\,[0,1]^N}\ \ "
        r"c(\mathbf{x}) \;=\; \mathbf{U}^{\top}\,\mathbf{K}(\tilde{\mathbf{x}})\,\mathbf{U}$",
        ha="left", va="center", fontsize=23, color=INK,
    )
    ax.text(
        0.10, 0.695,
        r"$\mathrm{s.t.}\ \ \ \mathbf{K}(\tilde{\mathbf{x}})\,\mathbf{U} = \mathbf{F}$",
        ha="left", va="center", fontsize=19, color=INK,
    )
    ax.text(0.345, 0.695, r"(equilibrium, FEM)",
            ha="left", va="center", fontsize=11, style="italic", color="#555")
    ax.text(
        0.585, 0.695,
        r"$\dfrac{1}{N}\sum_e \tilde{x}_e \ \leq\ v_{\mathrm{frac}}$",
        ha="left", va="center", fontsize=19, color=INK,
    )
    # annotations under the equations, still inside the box
    ax.text(0.10, 0.615, "minimize compliance (= maximize stiffness)",
            ha="left", va="center", fontsize=11, style="italic", color="#555")
    ax.text(0.585, 0.615, "stay within the volume budget",
            ha="left", va="center", fontsize=11, style="italic", color="#555")

    # ---- Tier 2: the two ingredients ---------------------------------
    ax.text(0.06, 0.49, "Two ingredients behind the pictures:",
            ha="left", va="center", fontsize=13, fontweight="bold", color=INK)

    # SIMP
    ax.text(
        0.07, 0.40,
        r"$E_e(\tilde{x}_e) = E_{\min} + \tilde{x}_e^{\,p}\,(E_0 - E_{\min})$",
        ha="left", va="center", fontsize=18, color=ACCENT,
    )
    ax.text(0.07, 0.335, r"SIMP penalization ($p=3$): intermediate 'gray' is stiffness-inefficient",
            ha="left", va="center", fontsize=11.5, color=ACCENT)

    # Filter
    ax.text(
        0.07, 0.185,
        r"$\tilde{x}_e = \dfrac{\sum_i H_{ei}\,x_i}{\sum_i H_{ei}}, \quad "
        r"H_{ei} = \max\!\left(0,\ r_{\min} - \|i-e\|\right)$",
        ha="left", va="center", fontsize=18, color=ACCENT2,
    )
    ax.text(0.07, 0.055, r"density filter: enforces the min. length scale $r_{\min}$ "
                         r"(and creates the gray boundary band)",
            ha="left", va="center", fontsize=11.5, color=ACCENT2)

    fig.savefig(os.path.join(OUT_DIR, "beams2d_topoopt_math.png"),
                dpi=220, bbox_inches="tight", transparent=transparent)
    plt.close(fig)
    print("wrote", os.path.join(OUT_DIR, "beams2d_topoopt_math.png"))

    # ---- Optional standalone: the OC update (Tier 3) -----------------
    fig2 = plt.figure(figsize=(11, 2.2))
    fig2.patch.set_alpha(0.0 if transparent else 1.0)
    ax2 = fig2.add_axes((0, 0, 1, 1))
    ax2.axis("off")
    ax2.text(0.5, 0.8, "How it's solved: Optimality Criteria update",
             ha="center", va="center", fontsize=15, fontweight="bold", color=INK)
    ax2.text(
        0.5, 0.42,
        r"$x_e^{\mathrm{new}} = \mathrm{clip}\!\left("
        r"x_e\sqrt{\dfrac{-\,\partial c/\partial x_e}{\lambda\,\partial V/\partial x_e}}"
        r"\;,\ \ x_e \pm \mathrm{move}\;,\ \ [0,1]\right)$",
        ha="center", va="center", fontsize=20, color=INK,
    )
    ax2.text(0.5, 0.08,
             r"scale each element by its compliance-per-volume sensitivity; "
             r"bisection tunes $\lambda$ to meet the volume budget",
             ha="center", va="center", fontsize=11.5, style="italic", color="#555")
    fig2.savefig(os.path.join(OUT_DIR, "beams2d_topoopt_oc_update.png"),
                 dpi=220, bbox_inches="tight", transparent=transparent)
    plt.close(fig2)
    print("wrote", os.path.join(OUT_DIR, "beams2d_topoopt_oc_update.png"))


if __name__ == "__main__":
    import sys
    main(transparent="--transparent" in sys.argv)
