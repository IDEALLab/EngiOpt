"""Super-basic conceptual schematic: GenAI for the beams2d inverse-design problem.

   [ 50x100 noise ]  +  [ conditions ]  --->  ( GenAI model )  --->  [ new beam ]

Produces ``beams2d_genai_concept.png`` for the DCC26 intro deck. The output beam
is a real dataset design used purely as a stand-in for a generated, out-of-sample
result (no trained model needed for the concept slide).

Run inside the EngiBench312 conda env:

    python workshops/dcc26/slides/build_genai_concept.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from engibench.utils.all_problems import BUILTIN_PROBLEMS

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "assets"))

SEED = 3
INK = "#1a1a1a"
BOX_BLACK = "#111418"
CHIP_FACE = "#eef3f8"
CHIP_EDGE = "#2c5f8a"

# target conditions for the demo (shown on the chip + used to pick the output beam)
TGT = {"volfrac": 0.30, "rmin": 2.0, "forcedist": 0.50}


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=28,
        lw=3, color=INK, shrinkA=0, shrinkB=0,
    ))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    problem = BUILTIN_PROBLEMS["beams2d"]()
    problem.reset(seed=0)
    nely, nelx = problem.design_space.shape

    tr = problem.dataset["train"]
    designs = np.array(tr["optimal_design"])
    vf = np.array(tr["volfrac"]); rmin = np.array(tr["rmin"]); fd = np.array(tr["forcedist"])
    rep = int(np.argmin(
        np.abs(vf - TGT["volfrac"]) + np.abs(rmin - TGT["rmin"]) + np.abs(fd - TGT["forcedist"])
    ))
    out_beam = designs[rep]

    noise = rng.random((nely, nelx))

    fig = plt.figure(figsize=(13, 4.6))

    # full-figure overlay for arrows / box / labels
    over = fig.add_axes((0, 0, 1, 1))
    over.set_xlim(0, 1); over.set_ylim(0, 1); over.axis("off")

    over.text(0.5, 0.93, "Generative AI for inverse design",
              ha="center", va="center", fontsize=21, fontweight="bold", color=INK)

    # ---- input: noise image (2:1 aspect) -----------------------------
    axn = fig.add_axes((0.045, 0.50, 0.205, 0.296))
    axn.imshow(noise, cmap="gray", aspect="auto")
    axn.set_xticks([]); axn.set_yticks([])
    for s in axn.spines.values():
        s.set_edgecolor(INK); s.set_linewidth(1.5)
    over.text(0.1475, 0.45, r"random noise  $z$   (50$\times$100)",
              ha="center", va="center", fontsize=12.5, color=INK)

    # ---- input: condition chip ---------------------------------------
    chip = FancyBboxPatch(
        (0.045, 0.205), 0.205, 0.13,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        linewidth=1.8, edgecolor=CHIP_EDGE, facecolor=CHIP_FACE,
    )
    over.add_patch(chip)
    over.text(0.1475, 0.30, "conditions", ha="center", va="center",
              fontsize=12, fontweight="bold", color=CHIP_EDGE)
    over.text(0.1475, 0.245,
              f"vf={TGT['volfrac']:.2f}   rmin={TGT['rmin']:.1f}   force@{TGT['forcedist']:.2f}",
              ha="center", va="center", fontsize=11.5, color=INK)

    # "+" between the two inputs
    over.text(0.1475, 0.41, "+", ha="center", va="center", fontsize=20,
              fontweight="bold", color="#888")

    # ---- arrows from inputs into the box -----------------------------
    arrow(over, 0.265, 0.50, 0.385, 0.49)   # noise -> box
    arrow(over, 0.265, 0.27, 0.385, 0.44)   # conditions -> box

    # ---- the black box (GenAI model) ---------------------------------
    box = FancyBboxPatch(
        (0.39, 0.31), 0.175, 0.30,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        linewidth=0, facecolor=BOX_BLACK,
    )
    over.add_patch(box)
    over.text(0.4775, 0.49, "GenAI", ha="center", va="center",
              fontsize=22, fontweight="bold", color="white")
    over.text(0.4775, 0.40, "model", ha="center", va="center",
              fontsize=15, color="#cfd6dd")
    over.text(0.4775, 0.345, "(black box)", ha="center", va="center",
              fontsize=10, style="italic", color="#9aa3ac")

    # ---- arrow from box to output ------------------------------------
    arrow(over, 0.575, 0.46, 0.715, 0.49)

    # ---- output: generated beam --------------------------------------
    axo = fig.add_axes((0.72, 0.45, 0.245, 0.354))
    axo.imshow(out_beam, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axo.set_xticks([]); axo.set_yticks([])
    for s in axo.spines.values():
        s.set_edgecolor(INK); s.set_linewidth(1.5)
    over.text(0.8425, 0.40, "new beam design", ha="center", va="center",
              fontsize=13, fontweight="bold", color=INK)
    over.text(0.8425, 0.35, "(out-of-sample conditions)", ha="center", va="center",
              fontsize=11.5, style="italic", color="#555")

    fig.savefig(os.path.join(OUT_DIR, "beams2d_genai_concept.png"),
                dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT_DIR, "beams2d_genai_concept.png"))


if __name__ == "__main__":
    main()
