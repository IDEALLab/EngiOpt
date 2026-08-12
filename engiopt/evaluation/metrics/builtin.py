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
    return float(metrics_mod.mmd(ctx.gen_flat, ctx.ref_flat, sigma=ctx.pixel_sigma))


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
    """Determinantal Point Process diversity of the generated designs.

    Kept in its published raw-determinant form on purpose. It is already frozen
    into `beams2d/v2` and recorded in every `metrics.json` written against it,
    and silently changing what a column *means* is the failure the whole spec
    mechanism exists to prevent. `dpp_geometric` is the repaired column; report
    that one and leave this where it is.
    """
    return float(metrics_mod.dpp_diversity(ctx.gen_flat, sigma=ctx.pixel_sigma))


@register_metric(
    "dpp_geometric",
    family="diversity",
    cost="cheap",
    higher_is_better=True,
    description="DPP diversity as the n-th root of the determinant: bounded in (0, 1] and comparable across n.",
)
def dpp_geometric(ctx: EvaluationContext) -> float:
    """`dpp` on a scale that survives being written down.

    Same kernel, same determinant, `n`-th root taken. See
    `metrics.dpp_geometric_mean` for why that is the form worth reporting.
    """
    return float(metrics_mod.dpp_geometric_mean(ctx.gen_flat, sigma=ctx.pixel_sigma))


@register_metric(
    "dpp_logdet",
    family="diversity",
    cost="cheap",
    higher_is_better=True,
    description="Log-determinant form of DPP diversity; unbounded below and scales with the sample count.",
)
def dpp_logdet(ctx: EvaluationContext) -> float:
    """The log-determinant, reported alongside so the three forms can be compared.

    Not the recommended column -- it is not comparable across sample sizes --
    but it is what "fix DPP with slogdet" usually means, and showing it beside
    `dpp_geometric` is what makes the difference between the two visible.
    """
    return float(metrics_mod.log_dpp_diversity(ctx.gen_flat, sigma=ctx.pixel_sigma))


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
