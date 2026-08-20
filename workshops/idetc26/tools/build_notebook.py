"""Generate the challenge notebook from reviewable source.

A `.ipynb` is JSON, and JSON diffs are unreadable, so the notebook's prose and
code live here as ordinary Python literals and the notebook is a build product.
Edit this file, run it, commit both.

    python workshops/idetc26/tools/build_notebook.py

**The notebook runs in one direction and does not double back**: the data, then
the models as pictures, then the families of questions in pixels, then the same
questions in a fitted space, then the board, then free work. Each section is
about the same objects as the one before it, so nothing needs re-introducing.

There are exactly **two commands** in the whole notebook, `case.evaluate` for
numbers and `case.show` for pictures, because a participant who has to remember
which of `compute`/`evaluate`/`board`/`score` they wanted is spending their
attention on the API rather than on the argument.

**The notebook does not document the API.** Argument tables in markdown went
stale every time an argument moved, and they made every section twice as long as
the idea in it. Every optional flag lives in `case.help()`, which is run as the
sixth cell and pointed at again in the free-work section; the prose here uses
the plain forms and stops. If you are about to add a table of keyword arguments
to a cell, add it to `HELP` in `engiopt/workshops/idetc26/case.py` instead.

**Neither command has a blanket form**, so no cell draws all ten suspects or
scores every column at once. Both are read rather than thought about, and which
three columns a participant chose is the entire session.

The accusation at the end is spoken out loud in the room, not recorded by code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"

BRANCH = "feat/idetc26-workshop"
"""The EngiOpt branch Colab installs from."""

ENGIBENCH_REF = "main"
"""The EngiBench ref Colab installs from, rather than the PyPI release.

PyPI still carries 0.2.0, whose `photonics2d` is v0: three conditions and the v0
dataset. The frozen spec is defined against v1 -- six conditions and the v1
dataset -- so `Case.open("photonics2d")` fails its own definition check on a
released install, which is the guard working rather than a bug. `main` has v1
and matches what the spec was frozen against; pinning it here costs one clone at
setup and needs no release to land before the session.

It has to be forced. The build from `main` reports version 0.2.0, the same
string PyPI carries, so a plain `pip install engibench @ git+...` reports the
requirement already satisfied and changes nothing -- verified in a clean 3.12
venv, where it left `photonics2d` at v0 and the spec check still failed."""
"""The branch Colab installs EngiOpt from.

One constant, referenced everywhere. The DCC'26 notebooks hardcoded their branch
in fifteen URLs, which is why changing it was a chore nobody wanted to do.
"""


def cell_id(source: str) -> str:
    """A stable id for a cell, derived from what is in it.

    nbformat 4.5 requires one, and Jupyter invents a random id for any cell that
    arrives without it -- which rewrites the whole file the first time somebody
    opens it. Deriving the id from the source keeps a rebuild byte-identical and
    changes only the ids of cells whose text actually changed.
    """
    return hashlib.blake2b(source.encode(), digest_size=4).hexdigest()


def md(text: str) -> dict:
    """A markdown cell."""
    source = text.strip("\n")
    return {"cell_type": "markdown", "id": cell_id(source), "metadata": {}, "source": source.splitlines(keepends=True)}


def code(text: str) -> dict:
    """A code cell."""
    source = text.strip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id(source),
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS = [
    # ---- the generative model murder mystery ----
    md(
        """
# The generative model murder mystery

**IDETC-CIE 2026 · EngiBench hands-on workshop**

*Ten generative models were trained on the same topology optimization problem. You must put your detective hat on and find the best performing model. That is our killer.*
"""
    ),
    md("""**Before you edit anything:** File → Save a copy in Drive. Opens read-only from GitHub."""),
    # ---- setup ----
    md(
        """
---
## Setup
"""
    ),
    code(
        f"""
import sys

if "google.colab" in sys.modules:
    %pip install -q "git+https://github.com/IDEALLab/EngiOpt.git@{BRANCH}"
    # PyPI's engibench is 0.2.0 and so is the build from main, so pip calls the
    # requirement satisfied and leaves photonics2d at v0. --force-reinstall is
    # what actually swaps the code; --no-deps because main declares none that
    # 0.2.0 did not.
    %pip install -q --force-reinstall --no-deps "engibench[all] @ git+https://github.com/IDEALLab/EngiBench.git@{ENGIBENCH_REF}"

    import sysconfig
    from pathlib import Path
    installed = Path(sysconfig.get_paths()["purelib"]) / "engibench/problems/photonics2d/v1.py"
    print(f"photonics2d v1 present: {{installed.exists()}}")
    print("Runtime -> Restart session, then carry on from the next cell.")
"""
    ),
    code(
        """
from engiopt.workshops.idetc26 import Case

case = Case.open("beams2d")     # <- the problem you are working on
"""
    ),
    md("""The cheat sheet — refer to this anytime you want to call a function"""),
    code("""case.help()"""),
    # ---- 1 · the scene of the crime ----
    md(
        """
---
# 1 · The scene of the crime (the dataset)

`"train"` — what the models were fitted on. `"test"` — the held-out set they are
scored against. Sliders move the conditions.
"""
    ),
    code("""case.show("train")"""),
    code("""case.problem.render(case.designs("test")[0])"""),
    # ---- 2 · the suspects ----
    md(
        """
---
# 2 · The suspects

The ten models in our lineup -- which one is best?
"""
    ),
    code("""case.models()"""),
    md("""Each model has a case file explaining who they are. Pull it with `case.explain()`"""),
    code("""case.explain("knn_retrieval")"""),
    # ---- the evidence ----
    md(
        """
### The evidence (visualizing our suspects)

Show generated outputs from a given model
"""
    ),
    code("""case.show("diffusion", n=12)"""),
    md("""Two suspects, same brief. Either side can be a model, `"test"`, or `"train"`."""),
    code("""case.show("cgan_cnn_2d", "test")"""),
    md(
        """
`how=` swaps the picture:

| `how=` | draws |
|---|---|
| `"designs"` *(default)* | `n` designs from the suspect you name |
| `"compare"` | the suspects you name on the same briefs, real optimum on top |
| `"conditions"` | each design captioned `asked 0.30 / got 0.41` |
| `"nearest_training"` | each design above its closest training design |
| `"space_map"` | where a suspect's designs sit in a fitted space |
"""
    ),
    code("""case.show("knn_retrieval", "vqgan", how="compare", n=3)"""),
    # ---- 3 · the interrogation/questions ----
    md(
        """
---
# 3 · The interrogation/questions

We can look at many metrics to assist in our final verdict. Metrics provide insight into how well the model performs in various categories: cost, similarity, novelty, diversity, feasibility, and performance
"""
    ),
    code("""case.metrics()"""),
    # ---- cost ----
    md(
        """
## Cost — how expensive is the model?

If the training/inferencing the model is nearly as expensive as the traditional optimizer, is it worth it?

Cost metrics: `train_minutes`, `gen_seconds`, `params`
"""
    ),
    code("""case.evaluate("train_minutes").round(3)"""),
    # ---- similarity ----
    md(
        """
## Similarity — does it look like the real thing?

How close do generated designs resemble traditional optimized designs?

Similarity metrics: `mmd`, `pixel_paired_distance`
"""
    ),
    code("""case.evaluate(["mmd", "pixel_paired_distance"]).round(4)"""),
    md(
        """
Who won? Every column in this family is maximized by handing back the training
data — and it is the family almost every paper reports, because it is the one
you can afford.
"""
    ),
    # ---- novelty ----
    md(
        """
## Novelty — Are the designs new or memorized?

`novelty_ratio` is a scaled distance from each generated design to the nearest design
it could have copied. Around 1 is as novel as a real held-out design, near 0 is
memorized, far above 1 looks like nothing in the data — which random pixels also
achieve.
"""
    ),
    code("""case.evaluate("novelty_ratio").round(3)"""),
    md("""By eye — each design above the closest thing to it in the training set."""),
    code("""case.show("knn_retrieval", how="nearest_training")"""),
    # ---- diversity ----
    md(
        """
## Diversity — How different are our generated designs from each other?

Diversity scores are hard to interpret without a relative scale. We can add some control data augmentations for reference with `controls=True`

| control | what it means |
|---|---|
| `collapsed` | an averaged design repeated 50 times |
| `noise_doped` | real optima plus Gaussian noise |
| `volume_only` | random blob that hits volume budget exactly|

Diversity metrics: `vendi`, `dpp`
"""
    ),
    code("""case.evaluate("pixel_vendi", controls=True).round(3)"""),
    code("""case.show("noise_doped")   # controls can be looked at like any other model"""),
    # ---- obedience ----
    md(
        """
## Obedience — did it meet the constraints?

The design is not valid if it doesn't meet its budget

`cond_err` — Average distance from specified conditions (volume)
`viol` — a proportion of designs that missed by more than a tolerance
"""
    ),
    code("""case.evaluate(["cond_err", "viol"], controls=True).round(4)"""),
    code("""case.show("gan_cnn_2d", how="conditions")"""),
    # ---- performance ----
    md(
        """
## Performance — is the design actually any good?

Is the design close to optimum on generation (iog), does it require little effort to reach optimum on warmstart (cog), or does it converge to better/worse design when warmstarting (fog)?
"""
    ),
    code("""case.evaluate("cog", models=["knn_retrieval", "cgan_cnn_2d"]).round(3)"""),
    # ---- 4 · the same questions, somewhere other than pixels ----
    md(
        """
---
# 4 · The same questions, somewhere other than pixels

Every similarity/distance so far compared designs in pixel space. We can also project to featural spaces. Here we do so with PCA or through a learned autoencoder

| the question | pixels | PCA subspace | learned latent |
|---|---|---|---|
| does it look real? | `mmd` | `pca_mmd` | `lv_mmd` |
| how close to the right answer? | `pixel_paired_distance` | `pca_paired_distance` | `lv_paired_distance` |
| is it copying? | `novelty_ratio` | `pca_novelty` | `lv_novelty` |
| how many distinct designs? | `pixel_vendi` | `pca_vendi` | `lv_vendi` |
| did it cover the data? | — | `pca_coverage` | `lv_coverage` |
"""
    ),
    code("""case.show(case.evaluate(["mmd", "pca_mmd", "lv_mmd"]))"""),
    md(
        """PCA and LV project to the same dimensionality for this notebook (determined through least volume). LV refers to the encoded latent space from `constrained_plvae_2d`"""
    ),
    code("""case.latent_space()     # which autoencoder, how wide, and the PCA width matched to it"""),
    # ---- looking at the space ----
    md("""### Looking at the space"""),
    code("""case.show("knn_retrieval", "test", how="space_map")"""),
    md("""Same designs, linear space. Nothing about the model changed; the picture can."""),
    code("""case.show("knn_retrieval", "test", how="space_map", space="pca")"""),
    # ---- 5 · visualizing a board ----
    md(
        """
---
# 5 · Visualizing a board

"""
    ),
    code(
        """
board = case.evaluate(["mmd", "pixel_vendi", "iog_median"])
case.show(board)
"""
    ),
    # ---- 6 · your accusation ----
    md(
        """
---
# 6 · Your accusation

Out loud, to the room:

1. **Who did it.** The model you think is the best.
2. **On what evidence.** Why do you think so?
3. **What you could not rule out.** What might have the decision easier?

The cell below is yours to build with.

Remember `case.help()`
"""
    ),
    code(""""""),
]


def main() -> None:
    """Write the notebook."""
    notebook = {
        "cells": CELLS,
        # Key order matches what Jupyter writes, so opening and saving the
        # notebook does not reorder the file under the next build.
        # `accelerator` is what Colab reads to attach a runtime, so the notebook
        # opens on a GPU without anybody visiting Runtime -> Change runtime type.
        # Nothing in the session needs one -- the designs are cached and the
        # physics is read from the Hub -- but a cache miss samples from a
        # checkpoint, and that is the one place a participant would otherwise
        # sit and wait.
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    destination = NOTEBOOK_DIR / "01_find_the_best_model.ipynb"
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"Wrote {destination} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
