"""Clean schematic of the cantilever-beam problem from simple notebook 03.

Matches the presentation / model-diagram clipart style: flat fills, the deck's
blue / green / orange / red palette, sans-serif labels, math italic for
variables. Colour encodes the benchmark role of each quantity:

    design (h, b)        -> blue
    conditions (P, L)    -> green
    objective (mass)     -> orange
    constraints          -> red

Run:
    python3 workshops/dcc26/assets/clipart/build_cantilever.py

Output:
    workshops/dcc26/assets/clipart/cantilever_problem.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent

# ---- deck palette ----
BLUE = "#225E9B"      # design
GREEN = "#2F7D62"     # conditions
ORANGE = "#C47B20"    # objective
RED = "#C84837"       # constraints
NEAR_BLACK = "#17212B"
GREY = "#5A6675"
GREY_LIGHT = "#C8CED6"
BEAM_FILL = "#9CC0E4"
BG = "#FBFAF7"


def build(out_path: Path):
    fig = plt.figure(figsize=(11.0, 6.0), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    W, H = 11.0, 6.0
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(patches.Rectangle((0, 0), W, H, facecolor=BG, edgecolor="none"))

    # ===== side view geometry =====
    wall_x = 1.05
    y0 = 3.85                 # neutral axis of the undeformed beam
    beam_th = 0.26            # half-thickness for drawing the beam bar
    x_tip = 6.55
    L_fig = x_tip - wall_x
    delta_vis = 0.95          # visual tip deflection

    # ----- fixed wall (hatched) -----
    wall = patches.Rectangle((wall_x - 0.45, y0 - 1.55), 0.45, 3.10,
                             facecolor="#E4E8EC", edgecolor=NEAR_BLACK,
                             linewidth=1.6, hatch="////")
    ax.add_patch(wall)
    # ground/anchor ticks
    ax.plot([wall_x - 0.45, wall_x - 0.45], [y0 - 1.55, y0 + 1.55],
            color=NEAR_BLACK, lw=2.2)
    ax.text(wall_x - 0.22, y0 - 1.95, "fixed\nsupport", fontsize=9.5,
            ha="center", va="top", color=GREY)

    # ----- undeformed beam (solid, blue) -----
    ax.add_patch(patches.Rectangle((wall_x, y0 - beam_th), L_fig, 2 * beam_th,
                                   facecolor=BEAM_FILL, edgecolor=BLUE,
                                   linewidth=1.6))

    # ----- deflected shape (dashed curve) -----
    s = np.linspace(0, 1, 100)
    x_curve = wall_x + s * L_fig
    drop = delta_vis * (s**2 * (3 - s)) / 2.0
    ax.plot(x_curve, y0 - drop, color=BLUE, lw=2.0, linestyle=(0, (5, 3)),
            alpha=0.85)
    # faint reference line at original tip height
    ax.plot([x_tip, x_tip + 0.55], [y0, y0], color=GREY_LIGHT, lw=1.0)
    ax.plot([x_tip, x_tip + 0.55], [y0 - delta_vis, y0 - delta_vis],
            color=GREY_LIGHT, lw=1.0)

    # ----- deflection delta marker -----
    dx = x_tip + 0.40
    ax.annotate("", xy=(dx, y0 - delta_vis), xytext=(dx, y0),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
    ax.text(dx + 0.18, y0 - delta_vis / 2, r"$\delta$", fontsize=15,
            ha="left", va="center", color=RED, style="italic")

    # ----- tip load P (red downward arrow) -----
    load_x = x_tip
    ax.annotate("", xy=(load_x, y0 - beam_th - 0.05), xytext=(load_x, y0 + 1.30),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=3.0,
                                 mutation_scale=22))
    ax.text(load_x + 0.18, y0 + 1.15, r"$P$", fontsize=16,
            ha="left", color=GREEN, style="italic", fontweight="bold")
    ax.text(load_x + 0.18, y0 + 0.80, "tip load", fontsize=9.5,
            ha="left", color=GREEN)

    # ----- length dimension L -----
    dim_y = y0 - 1.65
    ax.annotate("", xy=(wall_x, dim_y), xytext=(x_tip, dim_y),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.6))
    ax.plot([wall_x, wall_x], [dim_y - 0.12, dim_y + 0.12], color=GREEN, lw=1.2)
    ax.plot([x_tip, x_tip], [dim_y - 0.12, dim_y + 0.12], color=GREEN, lw=1.2)
    ax.text((wall_x + x_tip) / 2, dim_y - 0.40, r"$L$  (length)", fontsize=12,
            ha="center", color=GREEN, fontweight="bold")

    # ===== top-left role legend (design + conditions; the rest are formulas below) =====
    leg_y = 5.62
    ax.add_patch(patches.Rectangle((0.60, leg_y), 0.20, 0.20,
                                   facecolor=BLUE, edgecolor="none"))
    ax.text(0.92, leg_y + 0.10, r"design  $(h, b)$", fontsize=11,
            ha="left", va="center", color=GREY)
    ax.add_patch(patches.Rectangle((3.05, leg_y), 0.20, 0.20,
                                   facecolor=GREEN, edgecolor="none"))
    ax.text(3.37, leg_y + 0.10, r"conditions  $(P, L)$", fontsize=11,
            ha="left", va="center", color=GREY)

    # ===== cross-section inset (right) =====
    cs_x0, cs_y0 = 8.55, 3.05
    cs_b, cs_h = 1.15, 1.85   # visual width (b) and height (h)
    ax.add_patch(patches.Rectangle((cs_x0, cs_y0), cs_b, cs_h,
                                   facecolor=BEAM_FILL, edgecolor=BLUE,
                                   linewidth=1.8))
    ax.text(cs_x0 + cs_b / 2, cs_y0 + cs_h + 0.55, "cross-section",
            fontsize=12, ha="center", color=NEAR_BLACK, fontweight="bold")
    ax.text(cs_x0 + cs_b / 2, cs_y0 + cs_h + 0.22, "= the design",
            fontsize=10, ha="center", color=BLUE, style="italic")

    # height h dimension (left of the rectangle)
    hx = cs_x0 - 0.30
    ax.annotate("", xy=(hx, cs_y0), xytext=(hx, cs_y0 + cs_h),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.6))
    ax.text(hx - 0.18, cs_y0 + cs_h / 2, r"$h$", fontsize=15,
            ha="right", va="center", color=BLUE, style="italic", fontweight="bold")
    # width b dimension (below the rectangle)
    by = cs_y0 - 0.30
    ax.annotate("", xy=(cs_x0, by), xytext=(cs_x0 + cs_b, by),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.6))
    ax.text(cs_x0 + cs_b / 2, by - 0.38, r"$b$", fontsize=15,
            ha="center", color=BLUE, style="italic", fontweight="bold")

    # ===== objective + constraints caption block (bottom) =====
    cap_y = 1.05
    # objective
    ax.add_patch(patches.Rectangle((wall_x - 0.45, cap_y - 0.02), 0.22, 0.22,
                                   facecolor=ORANGE, edgecolor="none"))
    ax.text(wall_x - 0.10, cap_y + 0.09,
            r"minimize  mass  $m = \rho\,L\,b\,h$",
            fontsize=12, ha="left", va="center", color=NEAR_BLACK)

    # constraints
    cstr_y = cap_y - 0.55
    ax.add_patch(patches.Rectangle((wall_x - 0.45, cstr_y - 0.02), 0.22, 0.22,
                                   facecolor=RED, edgecolor="none"))
    ax.text(wall_x - 0.10, cstr_y + 0.09,
            r"subject to   $\sigma = \dfrac{6PL}{bh^2} \leq 250\,$MPa"
            r"      and      $\delta = \dfrac{4PL^3}{E\,bh^3} \leq \dfrac{L}{250}$",
            fontsize=11.5, ha="left", va="center", color=NEAR_BLACK)

    fig.savefig(out_path, dpi=140, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    build(OUT / "cantilever_problem.png")
