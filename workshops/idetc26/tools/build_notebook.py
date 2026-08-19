"""Generate the challenge notebook from reviewable source.

A `.ipynb` is JSON, and JSON diffs are unreadable, so the notebook's prose and
code live here as ordinary Python literals and the notebook is a build product.
Edit this file, run it, commit both.

    python workshops/idetc26/tools/build_notebook.py

**The notebook is a detective story in two acts.** Act 1 introduces the
suspects; Act 2 is the interrogation, one line of questioning at a time. That is
a deliberate reversal of the previous "toolbox with no order" build: a catalogue
of independent tools is a reference manual, and nobody learns a subject from a
reference manual in ninety minutes. The story is the pedagogy -- it gives a
participant a reason to run the next cell, and it gives the disagreement between
metrics somewhere to land.

There are exactly **two commands** in the whole notebook, `case.evaluate` for
numbers and `case.show` for pictures, because a participant who has to remember
which of `compute`/`evaluate`/`board`/`score` they wanted is spending their
attention on the API rather than on the argument.

The accusation at the end is spoken out loud in the room, not recorded by code.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"

BRANCH = "feat/idetc26-workshop"
"""The branch Colab installs EngiOpt from.

One constant, referenced everywhere. The DCC'26 notebooks hardcoded their branch
in fifteen URLs, which is why changing it was a chore nobody wanted to do.
"""


def md(text: str) -> dict:
    """A markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    """A code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


CELLS = [
    md(
        """
# Which one of these is the best model?

**IDETC-CIE 2026 · EngiBench hands-on workshop**

Ten models. The same 50 design briefs. One of them is best.

Your job: find which — and say **what you had to measure to be entitled to it.**

Two commands, and that is the whole interface:

| | |
|---|---|
| `case.show(...)` | look at what a model produced |
| `case.evaluate(...)` | put a question to the models |
"""
    ),
    md("**Before you edit anything:** File → Save a copy in Drive. This notebook opens read-only from GitHub."),
    md("---\n## Setup\n\nRun this once, then restart the runtime when it tells you to."),
    code(
        f"""
import sys

if "google.colab" in sys.modules:
    %pip install -q "git+https://github.com/IDEALLab/EngiOpt.git@{BRANCH}"
    print("Installed. Runtime -> Restart session, then carry on from the next cell.")
"""
    ),
    code(
        """
from engiopt.workshops.idetc26 import Case

case = Case.open("beams2d")     # <- the problem you are working on
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
# 1 · The scene of the crime

What a real solution looks like. Every comparison later is against this.

- `"train"` — what every model was fitted on
- `"test"` — held-out, what they are *scored* against
- Drag the sliders: watch the design change with the brief

**Knobs on `case.show`:** `n=12` (how many), `seed=2` (which draw),
`how=` (`"copying"`, `"conditions"`, `"map"`), `fresh=True` (resample).
"""
    ),
    code(
        """
case.show("train")
"""
    ),
    md(
        """
That is the view the whole notebook uses: one grey ramp, dark where there is material.

**The problem has its own opinion about what matters.** `case.problem.render`
draws the physics the objective is about — for photonics, the fields. No density
plot shows that. It solves to draw, so give it a few seconds.
"""
    ),
    code(
        """
case.problem.render(case.designs("test")[0])
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
# 2 · The suspects

Ten of them. Named for what they claim to be.

- Reach any one by a fragment — `"diffusion"`, `"knn"`, `"plvae"`
- **A name is a claim, not a fact**
"""
    ),
    code("""case.models()"""),
    md(
        """
### Rap sheets

The table truncates. `case.explain` gives the full record for one suspect —
what it is, how it works, what it cost, what to look at next.

- `case.explain()` with no name lists who you can ask about
"""
    ),
    code(
        """
case.explain("knn_retrieval")
"""
    ),
    md(
        """
### Mugshots

Four seconds each. The thing every practitioner does, every paper figures, and
no paper reports as a number.
"""
    ),
    code(
        """
case.show()                      # a few designs from every suspect
"""
    ),
    code(
        """
case.show("diffusion", n=12)     # one suspect, as many as you like
"""
    ),
    code(
        """
# Two suspects, same brief. Either side can be a model, "test", or "train".
case.show("cgan_cnn_2d", "test")
"""
    ),
    md(
        """
**Write down your ranking now, before any metric.** Best to worst, on paper.

You will want to revise it in ten minutes. Whether you *should* is the session.
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
# 3 · The question briefings

You may not ask "are you the best model?" Only questions with numeric answers.

Every column below belongs to one line of questioning.

**Knobs on `case.evaluate`:** `models=["knn", "vqgan"]` (who to ask),
`controls=True` (add the scale bar), `ranks=True` (order, not values),
`sigma=0.5` (kernel bandwidth), `n_samples=10`, `fresh=True`, `confirm=True`.
"""
    ),
    code("""case.metrics()"""),
    md(
        """
### Read the `space` column

The same question can be asked in three places, and they disagree:

| question | pixels | PCA | learned latent |
|---|---|---|---|
| does it look real? | `mmd` | `pca_mmd` | `lv_mmd` |
| is it copying? | `novelty_ratio` | — | `lv_novelty` |
| did it cover the modes? | — | `pca_coverage` | `lv_coverage` |
| how many distinct designs? | `pixel_vendi` | `pca_vendi` | `lv_vendi` |
| did it answer the brief? | `pixel_paired_distance` | — | `lv_paired_distance` |

- The disagreement is about the **spaces**, not the models
- **Someone fitted each space.** `lv_*` uses an autoencoder the spec pins, and
  `constrained_plvae_2d` — a suspect — is its sibling. `pca_*` is fitted on the
  training split. `mmd` is fitted on nothing
- Before reporting an `lv_` column: who fitted it, and were they in the room?

One question, three spaces. Darkest is rank 1. **Count the disagreements.**
"""
    ),
    code(
        """
case.show(case.evaluate(["mmd", "pca_mmd", "lv_mmd"]))
"""
    ),
    # ---- cost ----
    md(
        """
## Cost — what did it take to get in the room?

- Free to measure, and it decides whether a method is worth adopting
- A 2% win that costs 200× the compute is not a win
- Put last in every results table, which is how it gets skipped
- Also here: `train_minutes`, `params`
"""
    ),
    code(
        """
case.evaluate("gen_seconds").round(3)
"""
    ),
    md(
        """
Seconds per design, across four orders of magnitude. Hold it: every win below
has to be worth **this**.

(Replayed from the machine that built the design cache — it says so. `fresh=True`
re-times here instead.)
"""
    ),
    # ---- realism ----
    md(
        """
## Realism — does it look like the real thing?

- Does the generated set look like the real one?
- The family almost every paper reports, because it is the one you can afford
- **Every column here is optimized by handing back the training data**
- Also here: `pca_mmd`, `lv_mmd`, `pca_coverage`, `lv_coverage`, `lv_residual`
"""
    ),
    code(
        """
case.evaluate("mmd").round(4)
"""
    ),
    md(
        """
Who is on top?

If it is `knn_retrieval`, you have found the defect at the centre of this family:
**a model that memorizes the dataset scores perfectly.** So ask something else.
"""
    ),
    # ---- memorization ----
    md(
        """
## Memorization — inventing, or copying the case files?

- Distance from each design to the nearest thing it could have copied
- ≈ 1 as far off as a real held-out design · ≈ 0 memorized · ≫ 1 unlike anything
- **It cannot tell invention from garbage.** Random pixels also score enormous
- Also here: `lv_novelty`
"""
    ),
    code(
        """
case.evaluate("novelty_ratio").round(3)
"""
    ),
    md("""Check it by eye — each design beside the closest thing in the training set:"""),
    code(
        """
case.show("knn_retrieval", how="copying")
"""
    ),
    # ---- diversity ----
    md(
        """
## Diversity — one answer, or one story on repeat?

- More than one answer, or one story repeated?
- **A diversity number means nothing alone.** What does `pixel_vendi = 23` tell you?
- Nothing — until you score models whose answer you already know
- Also here: `dpp_geometric`, `pca_vendi`, `lv_vendi`
"""
    ),
    code(
        """
case.evaluate("pixel_vendi", controls=True).round(3)
"""
    ),
    md(
        """
`controls=True` adds a scale bar, the way one belongs on a micrograph. None is a
suspect; none is ranked.

- `collapsed` — one design, repeated
- `noise_doped` — real optimal designs, corrupted
- `volume_only` — hits the budget with material carrying no load

**Noise cannot improve an optimal design.** Watch what it does here, then to
`lv_vendi`.

**A diversity metric that rewards corruption is measuring entropy. Entropy is free.**
"""
    ),
    md(
        """
One look at `dpp_geometric`:

- The classic DPP is the determinant of a 50×50 kernel
- Fifty numbers below one, multiplied — it lands between `1e0` and `1e-290`
- At any precision a paper prints, distinct models come out as identical zeros
- Not wrong. **Unreportable** — a failure mode invisible in the code
- `dpp_geometric` is the same quantity as its n-th root
"""
    ),
    # ---- obedience ----
    md(
        """
## Obedience — did it answer the question asked?

- A model should answer *the question asked*, not just produce something plausible
- This is where a model that ignores its conditions gives itself away
- Also here: `pixel_paired_distance`, `lv_paired_distance`
"""
    ),
    code(
        """
case.evaluate("cond_err").round(4)
"""
    ),
    code(
        """
# The same thing without a metric: what was asked for, against what came back.
case.show("gan_cnn_2d", how="conditions")
"""
    ),
    md(
        """
`gan_cnn_2d` is unconditional — it never sees the brief.

**Does that difference show up anywhere in realism or diversity?** If not, those
families cannot tell a model that answered your question from one that ignored it.
"""
    ),
    # ---- legality ----
    md(
        """
## Legality — does it obey the rules?

- Does it obey the problem's constraints and budgets?
- A floor, not evidence of quality
- Only column in this family here: `viol`
"""
    ),
    code(
        """
case.evaluate("viol", controls=True).round(4)
"""
    ),
    md(
        """
Look where `volume_only` lands. It hits the budget exactly, with material
arranged so it carries no load whatsoever.

**Feasibility is a floor, not evidence of quality.**
"""
    ),
    # ---- performance ----
    md(
        """
## Performance — is the design actually any good?

- The one that matters, and the one nobody can afford
- How far each design is from optimal, before and after re-optimization
- Every sample runs one optimization and two simulations
- Nothing runs without `confirm=True` — ask, and you get the price first
- Also here: `iog`, `fog`
"""
    ),
    code(
        """
case.evaluate("cog")     # this does NOT run anything -- it quotes you
"""
    ),
    code(
        """
# Pick how long you are willing to wait for.
mine = case.evaluate("cog", models=["knn_retrieval", "cgan_cnn_2d"], n_samples=2, confirm=True)
mine.round(3)
"""
    ),
    md(
        """
Now multiply: fifty configurations × five seeds × three problems.

**That is why every paper you have read reports `mmd` and not `cog`.**
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
# 4 · The board

A full physics board over every suspect at all 50 briefs, computed ahead of time
— hours of optimizer per model — and published into each checkpoint's own
`metrics.json` on the Hub, beside the weights.

Read, not recomputed. A suspect with no published run shows blank rather than
being dropped.
"""
    ),
    code(
        """
physics = case.physics()      # read from the Hub, beside the weights
physics.round(3)
"""
    ),
    code(
        """
cheap = case.evaluate()          # every cheap question, every suspect
case.show(cheap, physics)        # the two boards side by side, as ranks
"""
    ),
    md(
        """
**Find two suspects the cheap columns rank in the opposite order to `cog`.**

If you can, every cheap column above is — for that pair — actively misleading.
And the cheap columns are the only ones anyone reports.
"""
    ),
    md(
        """
### The plants

Some suspects were **built for this session, not trained**. No weights, written
in an afternoon, ranked beside the checkpoints — several near the top.

- One looks reasonable in pixels, bad in the learned space
- The other looks bad in pixels, reasonable in the fitted ones
- **Whichever space you trusted, one of them would have got past you**
- Built from the training split only. Never saw the held-out designs
- Source: `engiopt/baselines/planted.py`. It is short, and that is the point
"""
    ),
    md(
        """
---
## The alibi you cannot check

Every number came from **one trained checkpoint each**.

- A different *sampling* seed only redraws noise — the 50 briefs are frozen
- The question with teeth needs several **training** seeds per model
- If two runs of one model straddle another, the gap you just ranked was never
  a property of the method
- Those checkpoints exist, at seeds 1–10

**Say how confident you are it survives a retrain — and that you have no
evidence either way.** Almost every results table is in that position silently.
"""
    ),
    md(
        """
---
## Your accusation

Out loud, to the room:

1. **Who did it** — which model you would ship
2. **On what evidence** — the three columns you would report, and what each
   catches that the other two miss
3. **What you could not rule out** — the question you cannot afford

*"I would not ship any of these, because ___"* is an accepted answer, often the
best one.

Blank cell below. Worth trying:

- `case.show(<mmd winner>, how="copying")` — is your favourite copying?
- `case.evaluate("lv_mmd")`, then re-read who fitted that space
- `case.show("vqgan", "test", how="map")` — a space you do not trust
- `case.evaluate("mmd", sigma=0.5)` — how much of the ranking was a default
  nobody reported?
- `case.explain(<your pick>)` — can you say what it actually does?
"""
    ),
    code(
        """
"""
    ),
    md(
        """
---

**Take it further:** every `case.evaluate` call printed the `engiopt` command
that reproduces it outside this notebook, on your own models.
`python -m engiopt.evaluate --list-metrics` shows every column the benchmark can
compute, and `BRING_YOUR_OWN_PROBLEM.md` walks through a problem of your own.
"""
    ),
    code("""case.help()"""),
]


def main() -> None:
    """Write the notebook."""
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    destination = NOTEBOOK_DIR / "01_find_the_best_model.ipynb"
    destination.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Wrote {destination} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
