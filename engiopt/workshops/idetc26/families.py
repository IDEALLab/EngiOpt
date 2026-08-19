"""The lines of questioning a suspect can be put under.

Nobody walks into this session knowing what `pca_coverage` is, and a list of
thirty metric names sorted alphabetically teaches them nothing. What they *can*
hold in their head is a handful of **questions a detective would ask**, so every
column the benchmark computes is filed under one of six:

    cost           what did it take to put this model in the room?
    similarity     do its designs look like the real ones?
    memorization   is it inventing, or copying out of the case files?
    diversity      has it got more than one answer, or one story it repeats?
    obedience      did it answer the question it was actually asked?
    performance    are the designs any good? (this is the one that costs)

That is the whole vocabulary. `case.evaluate("diversity")` asks a line of
questioning; `case.evaluate("mmd")` asks one specific question inside it. Both
work, and a participant who only ever learns the seven has still done the
session.

**The families here are not the registry's `MetricSpec.family`.** The registry
files `novelty` under `distribution`, because that is what it is computed from.
But `mmd` and `novelty` ask *opposite* questions -- `mmd` is minimized by
copying the training set and `novelty` is the column that catches you doing it
-- and filing them together is precisely what lets a board look coherent while
containing its own refutation. So `memorization` is split out, and the split is
declared here rather than derived.

`viol` is filed with `cond_err` for the opposite reason. The registry calls it
feasibility, and an earlier version of this file gave it a line of questioning
of its own -- but on a problem whose budget is a *condition*, missing the budget
and missing the brief are one reading with a threshold between them. `cond_err`
says by how much a design missed the volume fraction it was asked for and `viol`
says how often it missed by more than the tolerance, which is a rate and a
magnitude of the same measurement. Two families would have been two names for
one question.

The other regrouping: the registry files every `lv_*` column under `latent`,
which describes what it *depends on* rather than what it asks. `lv_mmd` asks
what `mmd` asks, in a different space. Here it joins `mmd`, and the space it
measures in becomes a separate column of the catalogue. Two metrics that
disagree while asking the same question is the single most useful thing in the
suite, and it is invisible if they are filed apart.
"""

from __future__ import annotations

from dataclasses import dataclass

from engiopt.evaluation.registry import METRICS


@dataclass(frozen=True)
class Family:
    """One line of questioning.

    Attributes:
        key: What you type -- `case.evaluate("diversity")`.
        question: The question in the words a detective would use. This is the
            string the catalogue prints, and it is meant to be readable by
            somebody who has never seen a generative-model paper.
        detail: What the answer does and does not entitle you to conclude.
            Every family has a way of being satisfied by a model that is
            obviously bad, and that is what this says.
    """

    key: str
    question: str
    detail: str


FAMILIES: dict[str, Family] = {
    "cost": Family(
        key="cost",
        question="What did it take to put this model in the room?",
        detail=(
            "Training time, parameter count, seconds per design. The group nobody reports, and the one that "
            "decides whether a method is worth adopting: a 2% win that costs 200x the compute is not a win."
        ),
    ),
    "similarity": Family(
        key="similarity",
        question="Do its designs look like the real ones?",
        detail=(
            "Two forms of one comparison. `mmd` and the coverage columns hold the generated *set* against the "
            "real one; the paired distances hold each design against the reference optimum for its own brief. "
            "Every column here is *optimized* by handing back the training data, so a good score is not "
            "evidence of a good model until the memorization question has been asked too."
        ),
    ),
    "memorization": Family(
        key="memorization",
        question="Is it inventing, or copying out of the case files?",
        detail=(
            "Distance from each generated design to the nearest thing the model could have memorized. The only "
            "cheap family that separates a model which learned the manifold from one that memorized points on "
            "it. Note that it cannot tell invention from garbage: random pixels are also very far from the "
            "training data."
        ),
    ),
    "diversity": Family(
        key="diversity",
        question="Has it got more than one answer, or one story it repeats?",
        detail=(
            "How much the generated set varies. Satisfied perfectly by noise, so a high score means nothing on "
            "its own -- adding corruption to real designs raises most of these columns."
        ),
    ),
    "obedience": Family(
        key="obedience",
        question="Did it answer the question it was actually asked?",
        detail=(
            "Each design compared against the specific conditions requested of it, rather than against the "
            "dataset as a whole. A model that ignores its inputs entirely can still look excellent on similarity "
            "and diversity; this is the group that notices. `cond_err` says by how much a design missed the "
            "volume fraction it was asked for, and `viol` says how often it missed by more than the tolerance "
            "-- a magnitude and a rate of one reading, so report at most one of each. Passing is a floor "
            "rather than evidence of quality: material placed to hit the budget exactly and carry no load "
            "scores perfectly on both."
        ),
    ),
    "performance": Family(
        key="performance",
        question="Are the designs actually any good?",
        detail=(
            "How the designs do under the physics, against a real optimizer. The thing you actually care about "
            "and the only group that needs the simulator -- which is why it is missing from most papers, and "
            "why every other family in this list exists."
        ),
    ),
    "latent_space": Family(
        key="latent_space",
        question="What is the latent space itself doing?",
        detail=(
            "Diagnostics of the fitted space the lv_ columns are measured in, not properties of a model. "
            "Never ranked, because there is no better or worse -- they are here so a reader can check the "
            "space was behaving before trusting what was measured in it. `case.latent_space()` names the "
            "autoencoder they describe."
        ),
    ),
}
"""Every line of questioning, in the order the catalogue prints them.

Cost comes first on purpose. It is the cheapest question, the one every team can
answer without running anything, and the one that most often settles the
argument -- and putting it last, where results tables put it, is how it gets
skipped.
"""

SPACE_PREFIXES = {
    "lv_": "learned latent",
    "lvoff_": "latent (no perf. constraint)",
    "pca_": "PCA subspace",
    "pixel_": "pixels",
}
"""Metric-name prefix to the space that metric measures distances in.

A prefix is a *space*, never a question. `lv_mmd` and `mmd` are one question in
two spaces, and the whole reason the catalogue has a `space` column is so that a
team who finds them disagreeing knows to blame the space rather than the models.
"""

_BASE_FAMILY = {
    "mmd": "similarity",
    "coverage": "similarity",
    "residual": "similarity",
    "paired_distance": "similarity",
    "novelty": "memorization",
    "novelty_ratio": "memorization",
    "vendi": "diversity",
    "dpp": "diversity",
    "dpp_geometric": "diversity",
    "dpp_logdet": "diversity",
    "cond_err": "obedience",
    "cond_sens": "obedience",
    "viol": "obedience",
    "params": "cost",
    "train_minutes": "cost",
    "gen_seconds": "cost",
    "sample_seconds": "cost",
    "dual_gap": "latent_space",
    "active_dims": "latent_space",
    "dims": "latent_space",
}
"""Family for a metric with its space prefix stripped off.

Keyed on the *stem* rather than the full name, so a column added later in a new
space -- `pca_novelty`, say -- is filed correctly without this table being
touched. `performance` is absent because the registry already agrees with us
there and `iog`/`cog`/`fog` have no space variants.
"""


_NO_SPACE = {"viol"}
"""Stems that count rather than measure, so no space applies.

A violation rate is counted, not measured in a space. It sits in `obedience`
beside `cond_err`, which does measure one, so the exemption has to be per metric
rather than per family.
"""


def strip_space(metric: str) -> tuple[str, str]:
    """Split a metric name into the space it measures in and what it asks.

    Args:
        metric: A registered metric name.

    Returns:
        `(space, stem)` -- a human-readable space, and the name with its prefix
        removed. An unprefixed design metric measures in pixels.
    """
    for prefix, space in SPACE_PREFIXES.items():
        if metric.startswith(prefix):
            return space, metric[len(prefix) :]
    return "pixels", metric


def family_of(metric: str) -> str:
    """Which line of questioning a metric belongs to.

    Falls back to the registry's own family when the stem is unrecognised, so a
    metric registered after this file was written lands somewhere sensible
    rather than raising in the middle of a workshop.

    Args:
        metric: A registered metric name.

    Returns:
        A key of `FAMILIES`.
    """
    _, stem = strip_space(metric)
    if stem in _BASE_FAMILY:
        return _BASE_FAMILY[stem]
    if metric in _BASE_FAMILY:
        return _BASE_FAMILY[metric]
    registry_family = METRICS[metric].family if metric in METRICS else "latent_space"
    return registry_family if registry_family in FAMILIES else "latent_space"


def space_of(metric: str) -> str:
    """The space a metric measures distances in, or `--` when it measures none.

    A parameter count and a wall-clock second are not measured in a space, and
    printing "pixels" for them would be noise dressed as information.

    Args:
        metric: A registered metric name.

    Returns:
        A human-readable space name, or `"--"`.
    """
    if family_of(metric) in {"cost", "performance"} or strip_space(metric)[1] in _NO_SPACE:
        return "--"
    return strip_space(metric)[0]
