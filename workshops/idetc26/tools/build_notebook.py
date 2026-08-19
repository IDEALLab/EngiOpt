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
# The model murder mystery

*Ten generative models, one topology optimization problem, and a set of metrics
that do not agree with each other.*

**IDETC-CIE 2026 · EngiBench hands-on workshop**

Ten models were trained on the same dataset, and each was asked for a design at
the same 50 sets of conditions. One of them is the model you would want to ship.
Your job over the next hour is to work out which, and to be able to say what you
measured to get there.

There are two commands:

| | |
|---|---|
| `case.show(...)` | look at what a model produced |
| `case.evaluate(...)` | ask the models a question that has a numeric answer |
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

Start with the data, since every comparison later is made against it. `"train"`
is what the models were fitted on, and `"test"` is the held-out set they are
scored against. Drag the sliders to see how the design changes with the
conditions.
"""
    ),
    code(
        """
case.show("train")
"""
    ),
    md(
        """
That is the view used everywhere in this notebook: a grey ramp, dark where there
is material.

The problem itself has an opinion about what matters. `case.problem.render`
draws the physics the objective is about, which for photonics is the field, and
no density plot shows that. It runs a solve in order to draw, so give it a few
seconds.
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

Ten models, each named after what it claims to be. You can refer to any of them
by a fragment of the name, such as `"diffusion"`, `"knn"` or `"plvae"`. The name
is a claim about the method, not evidence that it works.
"""
    ),
    code("""case.models()"""),
    md(
        """
### Rap sheets

The table above truncates. `case.explain` prints the full record for one model:
what it is, how it works, what it cost to train, and what to look at next. Call
it without a name to see who you can ask about.
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

A few seconds per model. Looking at the designs is the first thing most people
do and the thing most papers put in a figure, and it is also the check nobody
reports as a number.
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
### Ways of looking

`how=` chooses the picture, and nothing else changes.

| `how=` | what it draws | what it is for |
|---|---|---|
| `"designs"` *(default)* | `n` designs from one suspect, in a grid | the first look: collapse, noise, blur |
| `"compare"` | one row per suspect, one brief per column, with the real optimum on the top row | who answered this particular brief better |
| `"conditions"` | each design captioned `asked 0.30 / got 0.41`, sorted by what was asked | a model that ignores the brief it was given |
| `"nearest_training"` | each design with its closest training design beneath it, and the distance between them | memorization: copying rather than generating |
| `"space_map"` | a scatter of two sources in the top two dimensions of a fitted space, with the training set faded behind (`space="lv"` [default] or `"pca"`) | coverage and collapse, the shape a single number throws away |

```python
case.show("diffusion", how="designs", n=12)
case.show(how="compare", n=3)                  # every suspect, three briefs
case.show("gan_cnn_2d", how="conditions")
case.show("knn_retrieval", how="nearest_training")
case.show("vqgan", "test", how="space_map")
```

The last four all come up again later, at the point where a number needs
checking by eye.

**The other knobs on `case.show`.** All optional; the default is in brackets.

| | |
|---|---|
| `n=12` [4] | how many designs to draw |
| `seed=2` [1] | which sampling draw. It redraws the model's noise, never the 50 briefs |
| `space="pca"` [`"lv"`] | which fitted space `how="space_map"` draws in |
| `fresh=True` [off] | ignore the cache and sample from the checkpoint now. Slow |

The 50 designs from each model are already computed and cached, which is why
looking at one is instant. `fresh=True` ignores the cache and runs the model
here instead.
"""
    ),
    md(
        """
Write your ranking down now, before you have seen a single metric: best to
worst, on paper.

Most people want to change it ten minutes later. Whether you should is what the
rest of the session is about.
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
# 3 · The question briefings

You cannot ask a model whether it is the best one. You can only ask questions
that have numeric answers, and every column below belongs to one of those
questions.

The sections that follow work in raw pixels, one question at a time. The `space`
column in the catalogue is about the rest of them, and there is a section on it
once the questions themselves are familiar.

**Knobs on `case.evaluate`.** All optional, all off by default.

| | |
|---|---|
| `models=["knn", "vqgan"]` | who to ask. Any unambiguous fragment of a name. [all of them] |
| `ranks=True` | print each column as a placing, 1 = best, instead of its value |
| `controls=True` | add the known-answer rows underneath, as a scale bar |
| `n_samples=10` | score on 10 of the 50 briefs instead of all of them |
| `random_conditions=True` | take those 10 at random rather than the first 10. One draw, shared by every suspect |
| `sigma=0.5` | set the kernel width that `mmd`, `dpp` and `vendi` compare at, instead of the median-distance default |
| `fresh=True` | ignore the cached designs and sample from the checkpoints now. The only way `gen_seconds` times this machine |
"""
    ),
    code("""case.metrics()"""),
    # ---- cost ----
    md(
        """
## Cost — what did it take to train and run?

Cost is free to measure and it decides whether a method is worth adopting: a 2%
win that costs 200x the compute is not a win. It usually goes last in a results
table, which is how it ends up being skipped. Also in this family:
`train_minutes` and `params`.
"""
    ),
    code(
        """
case.evaluate("gen_seconds").round(3)
"""
    ),
    md(
        """
Seconds per design, spread over four orders of magnitude. Keep it in mind: every
win further down has to be worth this.

These timings were replayed from the machine that built the design cache, and
the note above says so. `fresh=True` re-times the models here instead.
"""
    ),
    # ---- similarity ----
    md(
        """
## Similarity — does it look like the real thing?

Two versions of one comparison. `mmd` holds the generated *set* against the real
one, and `pixel_paired_distance` holds each design against the reference optimum
for its own brief. This is the family almost every paper reports, because it is
the one you can afford, and every column in it is maximized by handing back the
training data.
"""
    ),
    code(
        """
case.evaluate(["mmd", "pixel_paired_distance"]).round(4)
"""
    ),
    md(
        """
Who came out on top?

If it is `knn_retrieval`, you have found the flaw at the centre of this family: a
model that memorizes the dataset scores perfectly. That is a reason to ask a
different question, not a reason to ship it.
"""
    ),
    # ---- memorization ----
    md(
        """
## Memorization — inventing, or copying the case files?

`novelty_ratio` is the distance from each generated design to the nearest design
it could have copied. Around 1 means it is as far from the training set as a real
held-out design is, near 0 means memorized, and much greater than 1 means it
looks like nothing in the data. It cannot tell invention from garbage: random
pixels also score enormous.
"""
    ),
    code(
        """
case.evaluate("novelty_ratio").round(3)
"""
    ),
    md("""Check it by eye. Each design sits above the closest thing to it in the training set:"""),
    code(
        """
case.show("knn_retrieval", how="nearest_training")
"""
    ),
    # ---- diversity ----
    md(
        """
## Diversity — one answer, or the same story on repeat?

Did the model give you more than one answer, or the same one over and over? A
diversity number on its own says nothing. `pixel_vendi = 23` means nothing until
you have scored a few models whose answer you already know. Also here:
`dpp_geometric`.
"""
    ),
    code(
        """
case.evaluate("pixel_vendi", controls=True).round(3)
"""
    ),
    md(
        """
`controls=True` added three rows marked `[control]`. These are calibration
standards rather than suspects, like the scale bar on a micrograph. They are not
in the line-up and they are never ranked.

Each one is built so that you know the answer before the column is computed.

| control | what it does | what you already know |
|---|---|---|
| `collapsed` | returns one design, the validation design nearest the middle of the condition space, for all 50 briefs, ignoring what was asked | it has zero diversity by construction, so whatever a diversity column reads here is that column's floor |
| `noise_doped` | takes a real optimal design, the held-out one whose conditions are closest to the brief, and adds Gaussian noise to every pixel (σ = 0.25) | it is strictly worse than the real designs, because noise cannot improve an optimum |
| `volume_only` | thresholds a smooth random blob field so that exactly the requested fraction of the domain is material | it is perfectly feasible and structurally useless: it hits the budget and carries nothing |

Read them as the ends of the scale, and read each model as a position between
them:

- if every model sits between `collapsed` and `noise_doped`, that column has not separated anything you care about
- if `noise_doped` scores above the models, that column is rewarding damage
- if `volume_only` looks respectable, that column is not measuring quality

They work like any other name here: `case.explain("noise_doped")` for the full
construction, `case.show("volume_only")` to look at one.

Adding noise cannot improve an optimal design, so watch where `noise_doped`
lands here, and then where it lands in `lv_vendi`. A diversity metric that
rewards corruption is measuring entropy, and entropy is free.
"""
    ),
    code(
        """
case.show("noise_doped")   # controls can be looked at like any other model
"""
    ),
    md(
        """
A note on `dpp_geometric`. The classic DPP diversity score is the determinant of
a 50x50 kernel: fifty numbers below one, multiplied together, landing somewhere
between `1e0` and `1e-290`. At the precision a paper prints, models that differ
come out as identical zeros. The quantity is not wrong, it is unreportable, and
nothing in the code looks broken. `dpp_geometric` is the same quantity rescaled
by its n-th root.
"""
    ),
    # ---- obedience ----
    md(
        """
## Obedience — did it answer the brief it was given?

A conditional model is supposed to answer the specific brief, not just produce
something plausible. This is where a model that ignores its conditions gives
itself away. `viol` is here too, and the next-but-one cell says why.
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
`gan_cnn_2d` is unconditional: it never sees the brief at all.

Does that difference show up anywhere in the realism or diversity columns? If it
does not, then those families cannot tell a model that answered your question
from one that ignored it.
"""
    ),
    md(
        """
### The same reading, twice

`viol` is the fraction of designs that missed the volume budget by more than a
tolerance. `cond_err` is the average amount by which they missed it. On this
problem the budget is one of the conditions, so the two are a rate and a
magnitude of one measurement rather than two separate questions: a model that is
1% over on every design and one that is 500% over get the same `viol` and very
different `cond_err`.

Report one of each if you want both readings, but not as independent evidence.
"""
    ),
    code(
        """
case.evaluate(["cond_err", "viol"], controls=True).round(4)
"""
    ),
    md(
        """
Look at where `volume_only` lands. It hits the budget exactly, with the material
arranged so that it carries no load at all. Answering the brief is a floor to
clear, not evidence that the design works.
"""
    ),
    # ---- spaces ----
    md(
        """
---
## The same questions, somewhere other than pixels

Every column so far compared designs pixel by pixel. The same questions can be
asked after projecting the designs into a fitted space, and the prefix on a
column name says which one it used:

| the question | pixels | PCA subspace | learned latent |
|---|---|---|---|
| does it look real? | `mmd` | `pca_mmd` | `lv_mmd` |
| how close to the right answer? | `pixel_paired_distance` | `pca_paired_distance` | `lv_paired_distance` |
| is it copying? | `novelty_ratio` | `pca_novelty` | `lv_novelty` |
| how many distinct designs? | `pixel_vendi` | `pca_vendi` | `lv_vendi` |
| did it cover the data? | — | `pca_coverage` | `lv_coverage` |

Same question, three answers, and they disagree. That disagreement is about the
spaces rather than about the models.

**How wide is each space?** The latent one is however many dimensions the pinned
autoencoder still uses after pruning, and the PCA control is fitted to *the same
number* — deliberately, because a linear subspace of some other width would make
"the latent space wins" a claim about dimensionality rather than about the
manifold. Both widths are on the board as `lv_active_dims` and `pca_dims`, and
`case.instrument()` names the autoencoder they came from.

Someone had to fit the two right-hand columns. `pca_*` is fitted on the dataset
split. `lv_*` uses an autoencoder that the spec pins — and `constrained_plvae_2d`,
one of the suspects, is its sibling. So before reporting an `lv_` column, ask who
fitted that space and whether they were in the room.

Darkest is rank 1. Count how often the three rows disagree.
"""
    ),
    code(
        """
case.instrument()      # which autoencoder, how wide, and the PCA width matched to it
"""
    ),
    code(
        """
case.show(case.evaluate(["mmd", "pca_mmd", "lv_mmd"]))
"""
    ),
    # ---- performance ----
    md(
        """
## Performance — is the design actually any good?

This is the family that matters and the one almost nobody can afford. It measures
how far each design is from optimal, before and after re-optimization, and every
sample costs one optimization and two simulations. The cell prints the estimated
time before it starts, and you can interrupt it like any other cell. Keep
`n_samples` and the number of models small. Also here: `iog` and `fog`.
"""
    ),
    code(
        """
# Two models, two briefs. Asking for all ten at all fifty would run for hours.
mine = case.evaluate("cog", models=["knn_retrieval", "cgan_cnn_2d"], n_samples=2)
mine.round(3)
"""
    ),
    md(
        """
Now multiply that by fifty configurations, five seeds and three problems. That is
why the papers you have read report `mmd` and not `cog`.
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
# 4 · The board

The simulator columns for every model, at all 50 briefs, were computed ahead of
time, at hours of optimizer per model, and published into each checkpoint's own
`metrics.json` on the Hub, next to the weights.

The cell below reads them rather than recomputing them. A model with no published
run shows up blank instead of being dropped.
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
See whether you can find two models that the cheap columns rank in the opposite
order to `cog`.

If you can, then for that pair every cheap column above is actively misleading,
and the cheap columns are the ones people report.
"""
    ),
    md(
        """
---
## The alibi you cannot check

Every number here came from one trained checkpoint per model.

Changing the sampling seed only redraws the noise, since the 50 briefs are
frozen, so it cannot answer the question you actually care about: would this
ranking survive retraining? That needs several training seeds per model, and if
two runs of one model straddle another model, the gap you just ranked was never
a property of the method. Those checkpoints exist on the Hub, at seeds 1–10.

So when you give your answer, say how confident you are that it survives a
retrain, and say that you have no evidence either way. Almost every results table
you have read is in the same position without mentioning it.
"""
    ),
    md(
        """
---
## Your accusation

Out loud, to the room:

1. **Who did it.** Which model you would ship.
2. **On what evidence.** The three columns you would report, and what each one
   catches that the other two miss.
3. **What you could not rule out.** The question you could not afford to ask.

"I would not ship any of these, because ..." is a perfectly good answer, and
often the best one.

There is an empty cell below. Things worth trying:

- `case.show(<your mmd winner>, how="nearest_training")` — is your favourite copying?
- `case.evaluate("lv_mmd")`, and then re-read who fitted that space
- `case.show("vqgan", "test", how="space_map")` — a space you have no particular reason to trust
- `case.evaluate("mmd", sigma=0.5)` — how much of the ranking came from a default nobody reports?
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

**Taking it further.** Every `case.evaluate` call printed the `engiopt` command
that reproduces it outside this notebook, on your own models.
`python -m engiopt.evaluate --list-metrics` lists every column the benchmark can
compute, and `BRING_YOUR_OWN_PROBLEM.md` walks through setting up a problem of
your own.
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
