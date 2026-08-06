"""Built-in evaluation metrics.

These wrap the numerics in `engiopt.metrics` and declare, for each one, the
question it answers, whether it needs a simulator, and which direction is
better. Behaviour is identical to the per-model `evaluate_*.py` scripts these
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
    description="Maximum Mean Discrepancy between generated and reference designs (pixel space).",
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
    higher_is_better=True,
    description="Determinantal Point Process diversity of the generated set.",
)
def dpp(ctx: EvaluationContext) -> float:
    """Determinantal Point Process diversity of the generated designs."""
    return float(metrics_mod.dpp_diversity(ctx.gen_flat, sigma=ctx.sigma))


# ----------------------------------------------------------------------
# Feasibility: is the design admissible, and does it respect its budget?
# ----------------------------------------------------------------------


@register_metric(
    "viol",
    family="feasibility",
    cost="cheap",
    higher_is_better=False,
    description="Fraction of designs violating the problem's constraints or their volume budget.",
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
    description="Mean initial optimality gap: how far generated designs start from the reference optimum.",
)
def iog(ctx: EvaluationContext) -> float:
    """Mean initial optimality gap, before any re-optimization."""
    return float(np.mean(ctx.optimization.iog))


@register_metric(
    "cog",
    family="performance",
    cost="expensive",
    higher_is_better=False,
    description="Mean cumulative optimality gap over re-optimization from each generated design.",
)
def cog(ctx: EvaluationContext) -> float:
    """Mean cumulative optimality gap."""
    return float(np.mean(ctx.optimization.cog))


@register_metric(
    "fog",
    family="performance",
    cost="expensive",
    higher_is_better=False,
    description="Mean final optimality gap after re-optimization.",
)
def fog(ctx: EvaluationContext) -> float:
    """Mean final optimality gap."""
    return float(np.mean(ctx.optimization.fog))
