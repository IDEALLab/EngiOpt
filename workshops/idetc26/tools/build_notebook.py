"""Generate the challenge notebook from reviewable source.

A `.ipynb` is JSON, and JSON diffs are unreadable, so the notebook's prose and
code live here as ordinary Python literals and the notebook is a build product.
Edit this file, run it, commit both.

    python workshops/idetc26/tools/build_notebook.py
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
# Find the Best Model

**IDETC-CIE 2026 · EngiBench hands-on workshop**

You have a bank of generative models for the same engineering design problem.
They are labelled `Model A`, `Model B`, ... and nothing else. Your job is to work
out which one you would ship, and to be able to defend it.

You have three metrics to work with. They are the three that generative-design
papers report most often, and you will notice they are also the three that are
cheapest to compute. That is not a coincidence, and it is most of what this
session is about.

**The rules:**
1. Look at the designs before you compute anything.
2. Compute the metrics you can afford.
3. Commit to a winner, in writing, with a reason. Your commitment gets hashed.
4. *Then* we start revealing things.

A team that says *"these metrics don't separate this bank, I need more"* has
won the exercise. The commitment cell needs a pick, but
**"I would not ship any of these because ___" is an accepted answer.**
"""
    ),
    md("**Before you edit anything:** File → Save a copy in Drive. This notebook opens read-only from GitHub."),
    md("---\n## Step 0 · Setup\n\nRun this once, then restart the runtime when it tells you to."),
    code(
        f"""
import subprocess, sys

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    def pip_install(pkgs):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

    pip_install(["engibench[all]", "cryptography", "matplotlib", "pandas", "scikit-learn"])
    pip_install(["git+https://github.com/IDEALLab/EngiOpt.git@{BRANCH}#egg=engiopt"])
    print("Install complete. Runtime → Restart session, then continue from the next cell.")
else:
    print("Using the current environment.")
"""
    ),
    code(
        """
import pandas as pd
import torch as th

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# A GPU speeds up sampling from the trained models in the bank. It does *not*
# speed up the physics: this problem's optimizer is CPU-bound topology
# optimization, so the expensive metrics cost the same either way.
print("GPU available:", th.cuda.is_available())
"""
    ),
    md(
        """
---
## Step 1 · Open the challenge

Put your team's name in. It fixes which model gets which letter, so your bank is
ordered differently from the team next to you — you cannot shortcut this by
comparing letters across tables, and the same name always gives you the same
bank if you rerun.
"""
    ),
    code(
        """
from engiopt.workshops.idetc26 import Challenge

TEAM = "orange"          # <- your team name
PROBLEM = "beams2d"      # <- your problem

ch = Challenge.open(PROBLEM, team=TEAM)
"""
    ),
    md(
        """
### What the problem is

`beams2d` is cantilever topology optimization. You are given a volume budget and
a few other conditions, and you must lay out material in a 50×100 grid so the
beam is as stiff as possible. Every model in your bank has been asked for
designs under the **same 50 sets of conditions**, drawn from a frozen evaluation
spec — so anything you see is a difference between models, never a difference in
what they were asked.
"""
    ),
    md(
        """
---
## Round 1 · The eyeball test

Before any number exists, look at the designs.

Visual inspection is the metric every practitioner actually uses, every paper
includes as a figure, and no paper reports as a number. It takes four seconds
and it catches things no column in this notebook catches. It is also
unreproducible, unaggregatable, trivially cherry-picked, and you are seeing four
designs out of fifty chosen by whoever wrote this cell.

Hold both of those thoughts. Rank the bank by eye anyway.
"""
    ),
    code("""fig = ch.gallery(n_per_model=4)"""),
    md("**Fill in:** your ranking by eye, best first. Two minutes, no discussion with other teams."),
    code(
        """
print("Your bank, in order:", ch.bank.labels)

# START FILL -- reorder this list, best first
eyeball_ranking = list(ch.bank.labels)
# END FILL

assert sorted(eyeball_ranking) == sorted(ch.bank.labels), "List every model exactly once."
print("Recorded. You will be asked about this again.")
"""
    ),
    md(
        """
---
## Round 2 · The metrics you can afford

Three columns:

| metric | asks | direction |
|---|---|---|
| `mmd` | Does the set of generated designs look like the set of real ones? | lower is better |
| `dpp` | Is the model producing varied designs, or repeating itself? | higher is better |
| `viol` | What fraction of designs break a constraint or miss the volume budget? | lower is better |

**No simulator runs in this cell.** That is why it takes seconds instead of
twenty minutes, and it is why these are the metrics you see in papers.
"""
    ),
    code(
        """
board = ch.board()
board.round(4)
"""
    ),
    md("Which model tops each column?"),
    code("""ch.winners(board)"""),
    md(
        """
**Fill in:** commit to a winner.

Your pick and your reason get hashed and written to `verdict.json`. Read the
hash out when your team presents — it is how the room knows you did not revise
after the reveal. That is the practice this whole session is arguing for, so we
may as well run it on ourselves.
"""
    ),
    code(
        """
# START FILL
winner = "Model A"
why = "..."
metric_ranking = list(ch.bank.labels)   # reorder, best first
# END FILL

ch.submit(winner=winner, why=why, ranking=metric_ranking, eyeball_ranking=eyeball_ranking)
"""
    ),
    md(
        """
---
## Reveal 0 · Your own two rankings

Nothing has been unsealed. This uses only data your team produced in the last
fifteen minutes.
"""
    ),
    code(
        """
comparison = pd.DataFrame({
    "by eye": {label: i + 1 for i, label in enumerate(eyeball_ranking)},
    "by metric": {label: i + 1 for i, label in enumerate(metric_ranking)},
}).sort_values("by metric")
comparison["moved"] = (comparison["by eye"] - comparison["by metric"]).abs()
comparison
"""
    ),
    md(
        """
If those two columns disagree, one of your two procedures is wrong and you do
not yet know which. Most teams find they disagree badly. **Discuss for two
minutes: which one do you actually trust, and what would settle it?**
"""
    ),
    md(
        """
---
## Reveal 1 · The seed lottery

Same models. Same three metrics. Same conditions. The only thing that changes is
the random seed the models sample at.

Still nothing unsealed, still no simulator.
"""
    ),
    code(
        """
lottery = ch.seed_lottery()
lottery
"""
    ),
    md(
        """
Look along a row. If a model's rank moves between seeds, then **every
single-seed ranking is a coin flip you did not know you were making** — and most
published comparisons report one seed.

This is the cheapest possible criticism of your own result and almost nobody
runs it.
"""
    ),
    md(
        """
---
## Reveal 2 · The columns you were not given

Five more metrics. **Every one of them is cheap.** No simulator, no cluster, no
waiting. They were affordable the entire time; you simply were not handed them.

| metric | asks | direction |
|---|---|---|
| `novelty` | How far is each design from the nearest *training* design? | higher is better |
| `cond_err` | Did the design hit the volume fraction it was asked for? | lower is better |
| `pixel_vendi` | How many *effectively distinct* designs are in the set? | higher is better |
| `pca_mmd` | `mmd`, but measured in a PCA subspace instead of raw pixels | lower is better |
| `gen_seconds` | What did it cost to generate the set? | lower is better |
"""
    ),
    code(
        """
withheld = ch.withheld()
withheld.round(4)
"""
    ),
    code(
        """
full_board = pd.concat([board, withheld], axis=1)
ch.winners(full_board)
"""
    ),
    md(
        """
Two things to notice, and they are different kinds of problem.

**`novelty` is the memorization check.** Every distribution metric — `mmd`,
`pca_mmd`, and the precision/coverage family too — is *optimized* by copying the
training set. A lookup table beats every one of them. Only distance-to-training
separates a model that learned the manifold from one that memorized points on
it, and it is one line of code.

**`dpp` and `pixel_vendi` both claim to measure diversity, and only one of them
is readable.** Most of the `dpp` column prints as `0.0000`. Those are not ties —
run the cell below and look at the actual values.
"""
    ),
    code(
        """
board[["dpp"]].map(lambda v: f"{v:.3e}").join(withheld[["pixel_vendi"]].round(2))
"""
    ),
    md(
        """
`dpp` is the determinant of a 50×50 kernel matrix — a product of fifty numbers
below one — so it lands anywhere between `1e0` and `1e-290`. It is not wrong.
It is **unreportable**: at any precision a paper's table would use, distinct
models render as identical zeros. `pixel_vendi` measures the same property as an
effective sample count (1 means "one design repeated", 50 means "fifty distinct
designs"), on a scale that survives being written down.

A metric can fail by being uninformative. It can also fail by being
uncommunicable, and that failure is invisible in the code.
"""
    ),
    md(
        """
---
## Reveal 3 · The physics

Now the simulator. `iog`, `cog` and `fog` measure how far each design is from an
optimal one, before and after re-optimization — the thing you actually care
about, and the thing nobody can afford to put in a rebuttal.

This board was computed ahead of time and sealed into the repository with its
hash published beside it, so you can check afterwards that the answer was fixed
before yours was.

Your facilitator will read out the passphrase.
"""
    ),
    code(
        """
# START FILL
passphrase = "..."
# END FILL

physics = ch.unseal_physics(passphrase)
physics.round(3)
"""
    ),
    code(
        """
final = pd.concat([full_board, physics], axis=1)
ch.winners(final)
"""
    ),
    md(
        """
---
## Reveal 4 · Who they actually were
"""
    ),
    code("""ch.identities()"""),
    md(
        """
Several of the models in your bank were **not trained**. They were constructed,
each in under fifty lines, each aimed at exactly one metric:

- a **lookup table** that returns the nearest training design, rescaled to hit
  the requested volume budget
- a **one-trick pony** that emits a single good design and never reads its input
- a **volume cheater** that hits the budget exactly with blobs that carry no load
- a **noise-doped** model — real optimal designs with noise added, which *raises*
  pixel diversity
- a **checkerboard** — the classic topology-optimization artifact, which looks
  broken and scores well
- a **linear regression**, with no generative model in it at all

None of them required a GPU. Several of them beat models that people trained.

The disclosure is the lesson, not a trick: **if a benchmark can be topped by a
fifty-line fraud, the benchmark is the problem.**
"""
    ),
    md(
        """
---
## Debrief · Write the recipe you would actually trust

You now have eleven columns and a demonstration that any one of them can be
gamed. So write down what you would require before believing a claim that model
X is better than model Y.
"""
    ),
    code(
        """
# START FILL
recipe = {
    "metrics": ["..."],              # which columns, and why each earns its place
    "n_seeds": 1,                    # how many seeds before you believe a ranking
    "aggregate": "median",           # median or mean, and what you report alongside it
    "gates": ["..."],                # what disqualifies a model outright, regardless of score
    "visual_protocol": "...",        # how many designs, chosen how, shown to whom
}
# END FILL

import json
print(json.dumps(recipe, indent=2))
"""
    ),
    md(
        """
Trade recipes with the team next to you and try to break theirs. Specifically:

1. Which of the models in this bank would still pass their gates?
2. What would it cost to run their recipe on a new problem?
3. Their `visual_protocol` is the interesting one. "We looked at some designs" is
   exactly the unreproducible part, and it is what every paper does. Can they
   write one you could actually replicate?

---

**Take it further:** `python -m engiopt.evaluate --list-metrics` shows every
column the benchmark can compute. `BRING_YOUR_OWN_PROBLEM.md` walks through
running this arc on a problem of your own.
"""
    ),
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
