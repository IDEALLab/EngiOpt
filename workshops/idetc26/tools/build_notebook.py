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

Eight suspects are in the room. Each one claims to be a good generative model
for the same engineering design problem. Each was handed the **same 50 design
briefs** and produced 50 designs. One of them is the best.

Your job is to find out which — and, much harder, to be able to say **what you
had to measure before you were entitled to claim it.**

You cannot simply look at the answer. What you *can* do is put questions to the
suspects. There are seven kinds of question, none of them conclusive on its own,
several of which contradict each other, and one of which is expensive enough
that most published papers never ask it.

There are two commands. That is the whole interface:

| | |
|---|---|
| `case.evaluate(...)` | put a question to the suspects |
| `case.show(...)` | look at what a suspect actually produced |
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
    md(
        """
---
# ACT 1 · The suspects

### First, the scene of the crime

**Before you meet any suspect, look at what a real solution to this problem
looks like.** Drag the sliders and watch the nearest real design change with the
brief. Every comparison you make for the rest of the session is against this, so
it is worth a minute.

`"train"` is what every suspect was fitted on; `"test"` is the held-out set they
are *scored* against. Look at both — whether a model can tell them apart is the
question underneath half of Act 2.
"""
    ),
    code(
        """
case.show("train")     # the designs every suspect was fitted on
"""
    ),
    md(
        """
That shows the designs the way the rest of this notebook will: one grey ramp,
dark where there is material, so ten models stay comparable at a glance.

**The problem itself has an opinion about what is worth showing.** `case.problem`
is the EngiBench problem, and its own `render` draws whatever matters for *this*
problem — for a photonics problem, the field magnitudes at each wavelength beside
the structure. That is the physics the objective is about, and no density plot
shows it.

For some problems this solves the fields in order to draw them, so give it a few
seconds.
"""
    ),
    code(
        """
case.problem.render(case.designs("test")[0])   # EngiBench's own renderer
"""
    ),
    md("### The line-up"),
    code("""case.models()"""),
    md(
        """
Name any suspect by any unambiguous part of its name — `"diffusion"`, `"knn"`,
`"plvae"`. If a fragment matches more than one, the error tells you which.

**They are named for what they are, and that is not a spoiler.** "Why is the
lookup table beating the diffusion model" is the most useful question this
session can produce, and it cannot be asked of *Suspect C*.

Two of them are not generative models at all, and they are here as serious
entries rather than as jokes:

- **`knn_retrieval`** hands back *the single nearest training design*, rescaled
  to the requested budget. It cannot produce anything that is not already in the
  dataset — which is exactly what makes it interesting when it wins.
- **`deconv_regression`** is supervised: conditions in, one design out, trained
  with a pixel loss. It shares its upsampling stack with the conditional GANs,
  so what differs between them is the *training objective*, not the
  architecture.

Those two are the two sides of Habibi et al. (*J. Mech. Des.* 148(6):061704,
2026), who found on this exact task that k-nearest-neighbours beats
deconvolutional networks at limited data, once you count the cost of generating
that data. Both are in the room, so you can check that rather than cite it.

Neither has a noise input, so neither can offer a second answer to the same
brief. Watch what that does to the diversity questions — and whether it costs
them anything on the questions you actually care about.
"""
    ),
    md(
        """
### Look at them

Before any number, spend four seconds per suspect on the thing every
practitioner actually does, every paper includes as a figure, and no paper
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
# Two suspects answering the *same* brief. Drag the sliders.
# Either side can be a suspect, or "test" (the real optimum for that brief),
# or "train" (the nearest training design).
case.show("cgan_cnn_2d", "test")
"""
    ),
    md(
        """
**Write down your ranking now, before any metric.** Best to worst, on paper.

You will be tempted to revise it in ten minutes. Whether you *should* is the
question the rest of the session is about: visual inspection is unreproducible,
unaggregatable and trivially cherry-picked, and it also catches, instantly,
things no column below catches. Hold both of those at once.
"""
    ),
    md(
        """
---
# ACT 2 · The interrogation

You may not ask "are you the best model?" You may only ask questions that have
numeric answers. There are seven **lines of questioning**, and every column the
benchmark computes belongs to one of them.
"""
    ),
    code("""case.metrics()"""),
    md(
        """
Read that table by its **line of questioning** column, not top to bottom.

Ask a whole line of questioning at once by naming it, or one specific question
by naming the metric:

```python
case.evaluate("diversity")          # every question in that line
case.evaluate("mmd")                # one specific question
case.evaluate(["cost", "realism"])  # several, mixed freely
```

And notice the **space** column. Several questions are asked in more than one
space — `mmd`, `pca_mmd` and `lv_mmd` are *one question* asked in raw pixels, in
a PCA subspace, and in a learned latent space:

| the question | in pixels | in a PCA subspace | in a learned latent space |
|---|---|---|---|
| does this look like real data? | `mmd` | `pca_mmd` | `lv_mmd` |
| is it copying the training set? | `novelty_ratio` | — | `lv_novelty` |
| did it cover the real modes? | — | `pca_coverage` | `lv_coverage` |
| how many distinct designs? | `pixel_vendi`, `dpp_geometric` | `pca_vendi` | `lv_vendi` |
| did it answer the brief? | `pixel_paired_distance` | — | `lv_paired_distance` |

**Those rows disagree with each other**, and the disagreement is a fact about
the *spaces*, not about the suspects. Two designs differing by a one-pixel shift
are nearly identical structurally and far apart in pixels; whether that counts
as a difference is a modelling choice that no results table declares.
"""
    ),
    md(
        """
### Line of questioning 1 — cost

Start here, because it is free, and because putting it last is how it gets
skipped.
"""
    ),
    code(
        """
case.evaluate("cost").round(2)
"""
    ),
    md(
        """
Three minutes for a GAN, two and a half hours for the autoencoder, six seconds
for the lookup table. Hold that number in mind for the rest of the session: any
win you find later has to be worth **this** much compute, and no leaderboard has
a column for it.

(`gen_seconds` is sampling time, replayed from the machine that built the design
cache — the notebook says so when it does that. `fresh=True` re-times the
suspects here instead.)
"""
    ),
    md(
        """
### Line of questioning 2 — realism

Do the designs look like real ones? This is the family almost every generative
paper reports, because it is the family you can afford.
"""
    ),
    code(
        """
answers = case.evaluate("realism")
answers.round(4)
"""
    ),
    code(
        """
# A table of raw values is hard to read: the columns span orders of magnitude.
case.show(answers)
"""
    ),
    md(
        """
Darkest is rank 1. **Look at how often the rows disagree** — and remember that
`mmd`, `pca_mmd` and `lv_mmd` are the same question in three spaces.

Now: which suspect is on top? If it is `knn_retrieval`, you have just discovered
the defect at the centre of this whole family. **Every column here is *optimized*
by handing back the training data.** A model that memorizes the dataset scores
perfectly on all of them.

So you have to ask a different question.
"""
    ),
    md(
        """
### Line of questioning 3 — memorization

Distance from each generated design to the nearest thing the model could have
copied. The only cheap family that separates a model which *learned* the
manifold from one that *memorized points on it*.
"""
    ),
    code(
        """
case.evaluate("memorization").round(3)
"""
    ),
    md(
        """
`novelty_ratio` divides the raw distance by the same measurement taken on real
held-out designs, which answer the same briefs and are also not in the training
set. So:

- **≈ 1** — as far from the training data as a genuine held-out design is
- **≈ 0** — memorization
- **≫ 1** — further from the data than real designs are, which is as likely to
  be garbage as invention

That last case matters: **nothing in this column distinguishes invention from
garbage.** Random pixels are extremely far from the training data.

Check it by eye — each design beside the closest thing in the training set:
"""
    ),
    code(
        """
case.show("knn_retrieval", how="copying")
"""
    ),
    md(
        """
### Line of questioning 4 — diversity

Can it give more than one answer, or does it have one story it repeats?
"""
    ),
    code(
        """
case.evaluate("diversity").round(3)
"""
    ),
    md(
        """
Two things worth knowing here.

**A diversity number means nothing on its own.** What does `pixel_vendi = 23`
tell you? Nothing, until you know what the column reads at a known input. So
score models whose answer is already known — `collapsed` is one design repeated,
`noise_doped` is real optimal designs with noise added, `volume_only` hits the
budget with material that carries no load. None is a suspect; none is ever
ranked. They are a scale bar under the board, the way one belongs on a
micrograph.
"""
    ),
    code(
        """
case.evaluate("diversity", controls=True).round(3)
"""
    ),
    md(
        """
Read `noise_doped` against the suspects. Adding noise to a real optimal design
**cannot** make it a better design. Look at what it does to `pixel_vendi`, and
then at what it does to `lv_vendi`.

**A diversity metric that rewards corruption is measuring entropy, and entropy
is free.**

The second thing: the classic DPP diversity is the determinant of a 50×50 kernel
matrix — fifty numbers below one, multiplied — so it lands anywhere between `1e0`
and `1e-290`, and at any precision a paper's table would print, distinct models
come out as identical zeros. It is not wrong; it is *unreportable*, which is a
way for a metric to fail that is invisible in the code. `dpp_geometric` is the
same quantity as its n-th root.
"""
    ),
    md(
        """
### Line of questioning 5 — obedience

A generative model is supposed to answer *the question it was asked*, not just
produce something plausible. This is where a model that ignores its conditions
gives itself away.
"""
    ),
    code(
        """
case.evaluate("obedience").round(4)
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
`gan_cnn_2d` is unconditional — it never sees the brief at all. Compare it to a
conditional model on the same view, and then check whether that difference shows
up anywhere in the realism or diversity families. If it does not, then those
families cannot tell a model that answered your question from one that ignored
it.
"""
    ),
    md(
        """
### Line of questioning 6 — legality

Does it obey the problem's constraints and budgets?
"""
    ),
    code(
        """
case.evaluate("legality", controls=True).round(4)
"""
    ),
    md(
        """
Look at where `volume_only` lands. It hits the budget exactly, with material
arranged so it carries no load whatsoever. **Feasibility is a floor, not
evidence of quality.**
"""
    ),
    md(
        """
### Line of questioning 7 — performance

The one that matters, and the one nobody can afford. `iog`, `cog` and `fog`
measure how far each design is from an optimal one, before and after
re-optimization. Every sample runs one optimization and two simulations.

Ask, and you get the price first. Nothing runs without `confirm=True`.
"""
    ),
    code(
        """
case.evaluate("performance")     # this does NOT run anything -- it quotes you
"""
    ),
    code(
        """
# Pick how long you are willing to wait for.
mine = case.evaluate("performance", models=["knn_retrieval", "cgan_cnn_2d"], n_samples=2, confirm=True)
mine.round(3)
"""
    ),
    md(
        """
Now multiply. A hyperparameter sweep is fifty configurations, five seeds each,
three problems. At the rate you just watched — **that is why every paper you have
read reports `mmd` and not `cog`.**

A full board over every suspect at all 50 briefs was computed ahead of time and
sealed into the repository, with its plaintext hash published beside it, so it
can be checked afterwards that the numbers were fixed before anyone saw them.
Your facilitator has the passphrase.
"""
    ),
    code(
        """
# START FILL
passphrase = "..."
# END FILL

physics = case.physics(passphrase)
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
**Find two suspects the cheap columns rank in the opposite order to
`iog`/`cog`/`fog`.** If you can, then every cheap column above is, for that pair,
actively misleading — and the cheap columns are the only ones anyone reports.
"""
    ),
    md(
        """
---
## What you could not ask

Every number above came from **one trained checkpoint each**. The obvious next
question — *would this ranking survive retraining the same model?* — is the one
this notebook cannot answer, and it is worth knowing why.

Re-running a model with a different *sampling* seed only redraws its noise: the
50 briefs are frozen by the evaluation spec, so nothing about the comparison
moves except the particular designs each model happened to draw. That is a much
weaker question than it looks, and answering it would tell you almost nothing.

The question with teeth needs several **training** seeds per suspect — the same
architecture, the same hyperparameters, retrained from scratch. If two training
runs of one model straddle another model entirely, then the gap you have spent
the last hour ranking was never a property of the method.

Those checkpoints exist for some of these suspects, at seeds 1–10. Pulling them
is the natural next version of this session.

**So when you make your accusation, say how confident you are that it would
survive a retrain — and note that you have no evidence either way.** That is the
honest position, and it is the one almost every results table in the literature
is also in, without saying so.
"""
    ),
    md(
        """
---
## Your accusation

Out loud, to the room:

1. **Who did it** — which suspect you would ship.
2. **On what evidence** — the three columns you would actually report, and what
   each one catches that the other two miss.
3. **What you could not rule out** — the question you would need to ask, and
   cannot afford to.

*"I would not ship any of these, because ___"* is an accepted answer, and often
the best one. A team that says **these questions cannot separate this line-up, I
need better questions** has got the point of the session.

Blank cell below. Some things worth trying:

- Take whichever suspect tops `mmd` and run `case.show(..., how="copying")` on it.
- Find a column where `cgan_cnn_2d` and `cgan_cnn_2d_tuned` — same architecture,
  different hyperparameters — differ by more than two *architectures* do.
- Ask a question in a space you do not trust: `case.show("vqgan", "test", how="map")`.
- Change the kernel bandwidth (`case.evaluate("mmd", sigma=0.5)`) and see how
  much of the ranking was a property of a default nobody reported.
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
that reproduces it outside this notebook — on your own models.
`python -m engiopt.evaluate --list-metrics` shows every column the benchmark can
compute, and `BRING_YOUR_OWN_PROBLEM.md` walks through running all of this on a
problem of your own.

`case.help()` prints the whole toolbox on one card.
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
