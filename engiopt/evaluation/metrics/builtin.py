"""Built-in evaluation metrics.

These wrap the numerics in `engiopt.metrics` and declare, for each one, the
question it answers, whether it needs a simulator, and which direction is
better. Behavior is identical to the per-model `evaluate_*.py` scripts these
replaced, so numbers remain comparable.

Adding a metric means adding one decorated function here (or in your own module
-- importing it is enough to register it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from engiopt import metrics as metrics_mod
from engiopt.evaluation.registry import register_metric

if TYPE_CHECKING:
    from engiopt.evaluation.context import EvaluationContext


# ----------------------------------------------------------------------
# Distribution: does the generator match the reference design distribution?
# ----------------------------------------------------------------------


@register_metric(
    "mmd",
    family="distribution",
    cost="cheap",
    higher_is_better=False,
)
def mmd(ctx: EvaluationContext) -> float:
    """Distance between the generated designs and the reference optima, compared as whole sets. Zero means the two sets are indistinguishable."""
    return float(metrics_mod.mmd(ctx.gen_flat, ctx.ref_flat, sigma=ctx.kernel_sigma))


# ----------------------------------------------------------------------
# Diversity: is the generator producing varied designs, or repeating itself?
# ----------------------------------------------------------------------


@register_metric(
    "dpp",
    family="diversity",
    cost="cheap",
    higher_is_better=True,
)
def dpp(ctx: EvaluationContext) -> float:
    """Spread of the generated set, as the geometric mean of its similarity-kernel eigenvalues. One is perfectly spread, near zero is collapsed.

    The n-th root of the DPP kernel determinant, i.e. the geometric mean of the
    kernel's eigenvalues. The raw determinant that earlier boards published under
    this name is a product of n numbers below one and reads 1e-20 on every real
    board; the n-th root is the same quantity on a scale that is bounded in
    (0, 1] and comparable across sample sizes. Rows from spec v1 carry the raw
    form and are not comparable with this column.

    A determinant still rises when a collapsed set is jittered, so it rewards
    noise as diversity. `vendi` does not, and is the column to prefer.
    """
    return float(metrics_mod.dpp_geometric_mean(ctx.gen_flat, sigma=ctx.kernel_sigma))


# ----------------------------------------------------------------------
# Memorization: did the model generate this, or retrieve it?
# ----------------------------------------------------------------------


@register_metric(
    "train_distance",
    family="memorization",
    cost="cheap",
    higher_is_better=None,
    pixel_only=True,
    outputs=("train_distance", "train_distance_ratio"),
)
def train_distance(ctx: EvaluationContext) -> dict[str, float]:
    """Per-element distance from each generated design to the nearest design in the training set. Zero means the model reproduces its training data.

    Two readings of one measurement. `train_distance` is the distance itself,
    per element, so the number means the same thing on problems of different
    resolution. `train_distance_ratio` divides it by the same measurement taken
    on the withheld reference designs, which is how far real optima the model
    never saw sit from the training sweep: one means the model's designs are as
    far from its training data as real unseen designs are, near zero means it
    reproduces the training set, and well above one means further from the data
    than real designs, which noise also achieves. No tolerance anywhere; the
    reference designs supply the scale and nothing else.

    Diagnostic, with no direction: zero is retrieval, but large is not good.
    Read beside `mmd` and `viol`. NaN when the problem has no training split.
    """
    if ctx.train_designs is None:
        return {"train_distance": float("nan"), "train_distance_ratio": float("nan")}
    generated = ctx.reduce(ctx.nearest_train_distance(ctx.gen_flat))
    reference = ctx.reduce(ctx.nearest_train_distance(ctx.ref_flat))
    return {"train_distance": generated, "train_distance_ratio": generated / reference if reference > 0 else float("nan")}


# ----------------------------------------------------------------------
# Conditions: does the model actually use the conditions it was given?
# ----------------------------------------------------------------------


@register_metric(
    "cond_sens",
    pixel_only=True,
    family="conditions",
    cost="cheap",
    higher_is_better=None,
)
def cond_sens(ctx: EvaluationContext) -> float:
    """How much each design changes, per element, when the model is re-run under another sample's conditions. Zero means the conditions are ignored.

    Mean per-element RMS change in output when each sample is given another's conditions.

    Sampled from the same seed, so the latent draw is held fixed and the only
    thing that varies is the conditions. A model that ignores them returns
    the identical batch and scores 0; a model that responds to them scores the
    size of that response.

    Diagnostic rather than ranked. Being unconditional is a legitimate thing for
    a model to be -- the contract says so, and such models are still handed
    conditions so that this can be measured -- while a large response is not by
    itself a good one, since responding *wrongly* also moves the output.
    Its job is to stop an unconditional model quietly collecting a conditional
    model's `mmd` score, not to be maximized.

    `cheap` here means what it means everywhere in this registry: it cannot
    reach a simulator or optimizer. It is not free, though -- it draws the batch
    a second time, so selecting it doubles generation cost. Immaterial for a
    GAN's tenth of a second; noticeable for a diffusion model, and worth knowing
    before pointing it at one that samples a pixel at a time.

    Returns NaN when there is nothing to compare: an unconditional problem, or
    fewer than two samples.
    """
    permuted = ctx.permuted_designs
    if permuted is None:
        return float("nan")
    deltas = np.linalg.norm(ctx.gen_flat - permuted, axis=1) / np.sqrt(ctx.gen_flat.shape[1])
    return ctx.reduce(deltas)


# ----------------------------------------------------------------------
# Feasibility: is the design admissible, and does it respect its budget?
# ----------------------------------------------------------------------


@register_metric(
    "viol",
    pixel_only=True,
    family="feasibility",
    cost="cheap",
    higher_is_better=False,
)
def viol(ctx: EvaluationContext) -> float:
    """Fraction of the generated designs that break a constraint of the problem. Zero means every design is admissible; blank when the designs' conditions were not supplied.

    Defined for every problem: `problem.check_constraints` always applies, and
    the spec's `volume_condition` adds the volume-fraction budget for problems
    that have one.

    Cheap, and deliberately so. Feasibility describes the design as generated,
    so it is judged before any solver runs -- which means it still reports when
    the optimizer refuses to start from an invalid design, the case where the
    answer matters most.
    """
    if ctx.conditions is None:
        return float("nan")
    values = ctx.feasibility
    return float(np.mean(values)) if values else float("nan")


# ----------------------------------------------------------------------
# Performance: how good are the designs once physics is applied?
# ----------------------------------------------------------------------


@register_metric(
    "iog",
    family="performance",
    cost="expensive",
    higher_is_better=False,
)
def iog(ctx: EvaluationContext) -> float:
    """Optimality gap of each generated design as generated, before any re-optimization. Zero means already optimal."""
    return ctx.reduce(ctx.optimization.iog)


@register_metric(
    "cog",
    family="performance",
    cost="expensive",
    higher_is_better=False,
)
def cog(ctx: EvaluationContext) -> float:
    """Optimality gap summed over every optimizer step when re-optimizing from each generated design, the area under its gap curve. Zero means no optimizer effort was wasted.

    For each generated design, the optimizer is started from that design and
    `objective(step) - objective(reference optimum)` is summed over every step it
    takes; that sum is the area under the design's optimality-gap curve. The
    column is the mean of those sums over the generated designs.

    Because it sums a whole trajectory it mixes two things: how far from optimal
    the start was, and how many steps the optimizer needed. `iog` isolates the
    first and `fog` the end point; read `cog` beside both.
    """
    return ctx.reduce(ctx.optimization.cog)


@register_metric(
    "fog",
    family="performance",
    cost="expensive",
    higher_is_better=False,
)
def fog(ctx: EvaluationContext) -> float:
    """Optimality gap of each design once re-optimization from it has finished. Zero means the warm start reached the reference optimum."""
    return ctx.reduce(ctx.optimization.fog)


# ----------------------------------------------------------------------
# Conditions: did each design answer the conditions it was given?
# ----------------------------------------------------------------------


@register_metric(
    "per_condition_distance",
    family="conditions",
    cost="cheap",
    higher_is_better=False,
)
def per_condition_distance(ctx: EvaluationContext) -> float:
    """Per-element distance from each generated design to the reference optimum for its own conditions. Zero means every design matches its own optimum.

    Design `i` is compared with reference design `i`, which answers the same
    conditions. That is a sharper question than any set-level metric can ask:
    not "does the set look right" but "is *this* design what *these* conditions
    called for". A model that returns the correct designs in the wrong order
    scores a perfect `mmd` and a large value here. Per-element RMS.
    """
    if ctx.gen_flat.shape != ctx.ref_flat.shape:
        return float("nan")
    distances = np.linalg.norm(ctx.gen_flat - ctx.ref_flat, axis=1) / np.sqrt(ctx.gen_flat.shape[1])
    return ctx.reduce(distances)


@register_metric(
    "volume_error",
    family="conditions",
    cost="cheap",
    higher_is_better=False,
    pixel_only=True,
)
def volume_error(ctx: EvaluationContext) -> float:
    """Gap between each design's material fraction and the fraction its conditions requested. Zero means every design hit its volume target.

    The volume fraction of a density field is its mean, so this needs nothing
    fitted: it is the exact error on the one condition that can be read straight
    off the design. `viol` reports how many designs missed the budget by more
    than the tolerance; this reports by how much. NaN on problems with no volume
    condition.
    """
    if ctx.volume_condition is None or ctx.conditions is None:
        return float("nan")
    requested = np.asarray(ctx.conditions[ctx.volume_condition], dtype=np.float64)
    realized = ctx.gen_flat.mean(axis=1)
    return ctx.reduce(np.abs(realized - requested))


# ----------------------------------------------------------------------
# Distribution and diversity: does the set reach the reference, and is it varied?
# ----------------------------------------------------------------------

COVERAGE_QUANTILE = 0.05
"""The radius around each reference design is set from the reference set's own
nearest-neighbor spacing, at this upper quantile, so it adapts to the space."""


@register_metric(
    "coverage",
    family="distribution",
    cost="cheap",
    higher_is_better=True,
)
def coverage(ctx: EvaluationContext) -> float:
    """Fraction of the reference optima that have at least one generated design close to them. One means every reference design is reached.

    Distribution distance can look healthy while whole regions go unvisited, so
    this counts reference designs directly: a mode the generator never produces
    leaves its neighborhood empty. Bounded in [0, 1].
    """
    from scipy.spatial.distance import cdist

    within_reference = cdist(ctx.ref_flat, ctx.ref_flat)
    np.fill_diagonal(within_reference, np.inf)
    radius = float(np.quantile(within_reference.min(axis=1), 1.0 - COVERAGE_QUANTILE))
    nearest_generated = cdist(ctx.ref_flat, ctx.gen_flat).min(axis=1)
    return float((nearest_generated <= radius).mean())


@register_metric(
    "vendi",
    family="diversity",
    cost="cheap",
    higher_is_better=True,
)
def vendi(ctx: EvaluationContext) -> float:
    """Effective number of distinct designs in the generated set. One means all identical, the sample size means all mutually distinct.

    The exponential of the entropy of the similarity kernel's eigenvalues, so it
    reads as a count: n identical designs score 1, n mutually dissimilar ones
    score n. Unlike a determinant it does not rise when a collapsed set is
    jittered, which is why it is the diversity column to prefer.
    Friedman & Dieng (2023), The Vendi Score.
    """
    return float(metrics_mod.vendi_score(ctx.gen_flat, sigma=ctx.kernel_sigma))


# ----------------------------------------------------------------------
# Performance: what does the optimizer still have to do after the warm start?
# ----------------------------------------------------------------------
#
# `iog` asks whether the generated design is already good. That is the wrong
# question whenever a defect is cheap to repair, and many are: a design at the
# wrong filter length scale is corrected by the first few updates. The metrics
# below price a defect in the currency an engineer pays, which is optimizer
# calls, and read the same trajectory `iog`, `cog` and `fog` summarize.

NEAR_OPTIMUM_FRACTION = 0.05
"""A gap within this fraction of the reference optimum's objective counts as near optimal."""

GAP_AFTER_CALLS = (1, 2, 5, 10)
"""Call budgets to report the remaining gap at: is a defect gone immediately, or not?"""


def _calls_to_near_optimum(path: np.ndarray, reference_objective: float) -> float:
    """Calls after which the gap stays within five percent of the reference optimum's objective.

    Last exit rather than first touch, so a path that dips into the band and
    leaves again is not credited. Zero means the design was near optimal from
    its first call. A path still outside the band at its last call never got
    there and counts as one more than the budget, so a design that never
    arrives ranks worse than one that is merely slow.
    """
    band = NEAR_OPTIMUM_FRACTION * max(abs(reference_objective), 1e-12)
    if float(path[-1]) > band:
        return float(path.size + 1)
    outside = np.flatnonzero(path > band)
    return float(outside[-1] + 1) if outside.size else 0.0


@register_metric(
    "calls_to_near_optimum",
    family="performance",
    cost="expensive",
    higher_is_better=False,
)
def calls_to_near_optimum(ctx: EvaluationContext) -> float:
    """Optimizer calls from each generated design until its gap stays within five percent of the reference optimum for its conditions. Fewer is faster; one more than the budget means it never got there.

    The band is set by the optimum for the design's own conditions, not by the
    design's starting point, so a design that starts absurdly far away is not
    credited with settling merely because its first call removed most of the
    absurdity. Each call is one simulate plus one sensitivity evaluation.
    """
    results = ctx.optimization
    if not results.trajectories:
        return float("nan")
    calls = [_calls_to_near_optimum(path, scale) for path, scale in zip(results.trajectories, results.reference_objectives)]
    return ctx.reduce(calls)


@register_metric(
    "gap_after_calls",
    family="performance",
    cost="expensive",
    higher_is_better=False,
    outputs=tuple(f"gap_after_{k}_calls" for k in GAP_AFTER_CALLS),
)
def gap_after_calls(ctx: EvaluationContext) -> dict[str, float]:
    """Optimality gap remaining after one, two, five and ten optimizer calls from each generated design. Zero means the defect was repaired within that budget.

    A path shorter than the budget has converged early, so its final value is
    carried forward: the optimizer would have spent the remaining calls and
    changed nothing.
    """
    paths = ctx.optimization.trajectories
    if not paths:
        return {f"gap_after_{k}_calls": float("nan") for k in GAP_AFTER_CALLS}
    return {
        f"gap_after_{k}_calls": ctx.reduce([float(path[min(k, path.size) - 1]) for path in paths]) for k in GAP_AFTER_CALLS
    }


@register_metric(
    "reaches_reference_rate",
    family="performance",
    cost="expensive",
    higher_is_better=True,
)
def reaches_reference_rate(ctx: EvaluationContext) -> float:
    """Fraction of generated designs whose re-optimization reaches or beats the reference optimum. One means every warm start gets there.

    The gap is measured against the reference optimum, so reaching it means the
    gap touches zero at some call. A rate rather than a call count, because a
    count is censored by designs that never get there, and a copier reaches
    parity in zero calls. Bounded in [0, 1].
    """
    paths = ctx.optimization.trajectories
    if not paths:
        return float("nan")
    return float(np.mean([bool(np.any(np.asarray(path) <= 0.0)) for path in paths]))


# ----------------------------------------------------------------------
# Cost: what does the model cost to run, and what did it cost to make?
# ----------------------------------------------------------------------


@register_metric(
    "generation_seconds",
    family="cost",
    cost="cheap",
    higher_is_better=False,
    pixel_only=True,
)
def generation_seconds(ctx: EvaluationContext) -> float:
    """Wall-clock seconds the model took to generate the evaluation batch. Comparable only between models run on the same machine.

    Wall-clock, so it compares models within one run on one machine, not across
    machines. NaN when the designs did not come from a timed `Generator.sample`.
    """
    return float("nan") if ctx.sample_seconds is None else float(ctx.sample_seconds)


@register_metric(
    "n_parameters",
    family="cost",
    cost="cheap",
    higher_is_better=False,
    pixel_only=True,
)
def n_parameters(ctx: EvaluationContext) -> float:
    """Number of trainable parameters in the model. Blank for a model with no network.

    The hardware-independent companion to `generation_seconds`. Neither is a
    complete account of cost alone: a small diffusion model can be slower than
    a large GAN because it samples iteratively. NaN for a model with no network.
    """
    return float("nan") if ctx.model_params is None else float(ctx.model_params)


@register_metric(
    "train_minutes",
    family="cost",
    cost="cheap",
    higher_is_better=False,
    pixel_only=True,
)
def train_minutes(ctx: EvaluationContext) -> float:
    """Wall-clock minutes the model took to train, when the checkpoint records it. Blank where no checkpoint does.

    Prices the decision to adopt the method rather than a forward pass; it is
    where a lookup table and a diffusion model differ by orders of magnitude.
    NaN unless the checkpoint records it, which today none do -- that gap is
    itself a finding.
    """
    return float("nan") if ctx.train_minutes is None else float(ctx.train_minutes)
