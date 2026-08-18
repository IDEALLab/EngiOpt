"""Export the two image components for the GenAI concept slide as standalone PNGs.

Writes ``assets/genai_components/noise.png`` (grayscale 50x100 noise) and
``assets/genai_components/beam.png`` (viridis out-of-sample beam). These are
inserted as editable pictures by ``build_genai_concept_pptx.py``.

Run inside the EngiBench312 conda env:

    python workshops/dcc26/slides/export_genai_components.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from engibench.utils.all_problems import BUILTIN_PROBLEMS

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "assets", "genai_components")
)

SEED = 3
TGT = {"volfrac": 0.30, "rmin": 2.0, "forcedist": 0.50}
UPSCALE = 8  # nearest-neighbour upsample so the PNGs stay crisp on a slide


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
    beam = designs[rep]
    noise = rng.random((nely, nelx))

    # nearest-neighbour upsample -> crisp pixels when scaled in PowerPoint
    beam_up = np.repeat(np.repeat(beam, UPSCALE, axis=0), UPSCALE, axis=1)
    noise_up = np.repeat(np.repeat(noise, UPSCALE, axis=0), UPSCALE, axis=1)

    plt.imsave(os.path.join(OUT_DIR, "noise.png"), noise_up, cmap="gray", vmin=0, vmax=1)
    plt.imsave(os.path.join(OUT_DIR, "beam.png"), beam_up, cmap="viridis", vmin=0, vmax=1)
    print("wrote", os.path.join(OUT_DIR, "noise.png"))
    print("wrote", os.path.join(OUT_DIR, "beam.png"))
    print(f"output beam conditions: vf={vf[rep]:.2f} rmin={rmin[rep]:.1f} force@{fd[rep]:.2f}")


if __name__ == "__main__":
    main()
