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
    description="Taken as a set, how different are the generated designs from the reference optimal designs?",
)
def mmd(ctx: EvaluationContext) -> float:
    """Maximum Mean Discrepancy between generated and reference designs."""
    return float(metrics_mod.mmd(ctx.gen_flat, ctx.ref_flat, sigma=ctx.sigma))


# ----------------------------------------------------------------------
# Diversity: is the generator producing varied designs, or repeating itself?
# ----------------------------------------------------------------------


@register_metric(
    "dpp",
    family="diversity",
    cost="cheap",
    higher_is_better=None,
    description="How spread out are the generated designs, measured as the volume their similarity kernel spans?",
)
def dpp(ctx: EvaluationContext) -> float:
    """Determinantal Point Process diversity of the generated designs."""
    return float(metrics_mod.dpp_diversity(ctx.gen_flat, sigma=ctx.sigma))


# ----------------------------------------------------------------------
# Memorization: did the model generate this, or retrieve it?
# ----------------------------------------------------------------------


@register_metric(
    "novelty",
    pixel_only=True,
    family="memorization",
    cost="cheap",
    higher_is_better=None,
    description="How close is each generated design to the nearest design it could have copied (a training design or a reference optimum)?",
    outputs=("novelty", "copy_rate"),
)
def novelty(ctx: EvaluationContext) -> dict[str, float]:
    """How far the generated designs sit from the corpus a model could copy from.

    The evaluation protocol is public: which conditions are scored, and the
    dataset-optimal design for each one, can be recomputed by anyone from the
    committed spec. A lookup table keyed on the condition vector therefore tops
    `mmd`, `iog`, and `fog` -- not by cheating the implementation, but because
    those metrics are *defined* as closeness to exactly the designs it returns.
    No amount of care in the evaluator changes that; the only defense available
    to a public board is to measure retrieval and say so.

    Two columns, because they answer different questions:

    - `novelty` -- mean per-element RMS distance to the nearest corpus design.
      Diagnostic, deliberately: it has no good direction. Zero means the model
      is a retrieval system, but large means only that the output is far from
      the data, which pure noise also achieves. Read it next to `mmd` and
      `viol`, never on its own.
    - `copy_rate` -- the fraction of designs closer than `copy_tol`, i.e. the
      share of this batch that is a reproduction rather than a generation. This
      is the one that gates a submission.

    Returns NaN when no corpus is available, rather than claiming novelty that
    was never checked.
    """
    distances = ctx.nearest_corpus_distance
    if distances is None:
        return {"novelty": float("nan"), "copy_rate": float("nan")}
    return {
        "novelty": ctx.reduce(distances),
        "copy_rate": float(np.mean(distances < ctx.copy_tol)),
    }


# ----------------------------------------------------------------------
# Conditions: does the model actually use the conditions it was given?
# ----------------------------------------------------------------------


@register_metric(
    "cond_sens",
    pixel_only=True,
    family="conditions",
    cost="cheap",
    higher_is_better=None,
    description="When the same model is given different conditions, does its output change?",
)
def cond_sens(ctx: EvaluationContext) -> float:
    """Mean per-element RMS change in output when each sample is given another's conditions.

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
    description="What fraction of the generated designs violate the problem's constraints?",
)
def viol(ctx: EvaluationContext) -> float:
    """Fraction of infeasible designs; see `EvaluationContext.is_infeasible`.

    Defined for every problem: `problem.check_constraints` always applies, and
    the spec's `volume_condition` adds the volume-fraction budget for problems
    that have one.

    Cheap, and deliberately so. Feasibility describes the design as generated,
    so it is judged before any solver runs -- which means it still reports when
    the optimizer refuses to start from an invalid design, the case where the
    answer matters most.
    """
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
    description="How much worse than the reference optimum is the generated design, exactly as generated?",
)
def iog(ctx: EvaluationContext) -> float:
    """Mean initial optimality gap, before any re-optimization."""
    return ctx.reduce(ctx.optimization.iog)


@register_metric(
    "cog",
    family="performance",
    cost="expensive",
    higher_is_better=False,
    description="Starting the optimizer from the generated design, how much worse than optimal is it, summed over every optimizer step?",
)
def cog(ctx: EvaluationContext) -> float:
    """Mean cumulative optimality gap.

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
    description="Starting the optimizer from the generated design, how much worse than optimal is it once the optimizer finishes?",
)
def fog(ctx: EvaluationContext) -> float:
    """Mean final optimality gap."""
    return ctx.reduce(ctx.optimization.fog)
