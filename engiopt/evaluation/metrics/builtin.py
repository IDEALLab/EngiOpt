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
    into `beams2d/v1` and recorded in every `metrics.json` written against it,
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
    requires=("volume_condition",),
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


# Median counterparts. The per-design gap is unbounded above -- a single generated
# design that the optimizer cannot rescue carries an effectively infinite
# compliance -- so a mean over ~50 samples is set by its worst member. On the
# beams2d board three models sit within 5% of each other on MMD and report mean
# IOG of 818, 50 and 1.5e8 while all three finish at FOG = -2.2: the optimizer
# converges them to the same place, and the 1.5e8 is one starting design, not a
# worse model. Rank correlations are unaffected (Spearman only sees order), but
# any statement about *magnitude* needs these.
@register_metric(
    "iog_median",
    family="performance",
    cost="expensive",
    higher_is_better=False,
    description="Median initial optimality gap; robust to a single unrecoverable design.",
)
def iog_median(ctx: EvaluationContext) -> float:
    """Median initial optimality gap."""
    return float(np.median(ctx.optimization.iog))


@register_metric(
    "cog_median",
    family="performance",
    cost="expensive",
    higher_is_better=False,
    description="Median cumulative optimality gap; robust to a single unrecoverable design.",
)
def cog_median(ctx: EvaluationContext) -> float:
    """Median cumulative optimality gap."""
    return float(np.median(ctx.optimization.cog))


@register_metric(
    "fog_median",
    family="performance",
    cost="expensive",
    higher_is_better=False,
    description="Median final optimality gap; robust to a single unrecoverable design.",
)
def fog_median(ctx: EvaluationContext) -> float:
    """Median final optimality gap."""
    return float(np.median(ctx.optimization.fog))


# ----------------------------------------------------------------------
# Convergence: what did the warm start cost the optimizer?
# ----------------------------------------------------------------------
#
# `iog`, `cog` and `fog` are three summaries of one trajectory, and none of them
# answers "how many function calls did this design save me" -- the question the
# warm-start literature is actually about. `cog` looks like it should: it is the
# only one that sees the whole path. But it is an unnormalized *sum* of gaps
# that may be negative, so every extra iteration spent below the reference makes
# it better. A design that lingers at gap -37 for 200 calls scores -7400; one
# that converges in three calls to -3 scores -9. It rewards slowness precisely
# where the design is already good.
#
# Nor does the raw iteration count answer it. beams2d stops on
# ``norm(x_new - x, inf) < 0.025`` -- a test on the *design variables*, not the
# objective -- so one pixel still drifting keeps the loop alive long after the
# compliance has settled. Counting iterations would punish a design that landed
# near-optimal on call one and then jittered.
#
# Both metrics below therefore read the objective path and ignore the stopping
# rule, which is also what makes them portable: photonics2d runs a fixed 200-step
# schedule with no early exit, and there `settle_calls` is the only way to see
# that a design was finished at call 40.

SETTLE_BAND = 0.05
"""Fraction of a design's own achievable improvement that counts as "settled".

Relative to the design, not absolute, because gaps span 1e-3 to 1e10 across this
pool and no single tolerance is meaningful over that range.
"""


def _settle_calls(path: np.ndarray, band_fraction: float = SETTLE_BAND) -> float:
    """Calls after which `path` stays within a band of its converged value.

    Last-exit rather than first-touch: a trajectory that dips into the band and
    leaves again has not converged, and first-touch would credit it for a lucky
    excursion. This is settling time in the control-theory sense.

    Returns 0.0 when the design starts already inside the band -- the warm start
    needed no calls at all, which is the best possible answer and has to be
    representable.
    """
    if path.size < 2:  # noqa: PLR2004 - a one-step path has no convergence to measure
        return 0.0
    final = float(path[-1])
    # Floored on |final| so a design that starts at its converged value gets a
    # band of sensible width rather than one of width zero, which nothing clears.
    reach = max(abs(float(path[0]) - final), abs(final), 1e-12)
    outside = np.flatnonzero(np.abs(path - final) > band_fraction * reach)
    return float(outside[-1] + 1) if outside.size else 0.0


@register_metric(
    "settle_calls",
    family="performance",
    cost="expensive",
    higher_is_better=False,
    description="Mean optimizer calls after which the objective stays within 5% of its converged value.",
)
def settle_calls(ctx: EvaluationContext) -> float:
    """Mean settling time over the generated designs, in optimizer calls.

    Each call is one simulate plus one sensitivity evaluation, so this is the
    quantity an engineer pays in. Unlike the raw iteration count it is unmoved
    by a design that has effectively converged and is still being nudged.
    """
    paths = ctx.optimization.trajectories
    return float(np.mean([_settle_calls(path) for path in paths])) if paths else float("nan")


@register_metric(
    "first_call_yield",
    family="performance",
    cost="expensive",
    higher_is_better=True,
    description="Fraction of the achievable improvement the optimizer's first call delivers.",
)
def first_call_yield(ctx: EvaluationContext) -> float:
    """How much of the whole re-optimization one call buys, averaged over designs.

    Near 1 means the design's defects clean up immediately -- a checkerboard or a
    floating member that a single step resolves -- and the remaining calls are
    refinement. Near 0 means the warm start bought nothing.

    Designs with nothing to gain (the path starts where it ends) are skipped
    rather than scored: the fraction is undefined there, and calling it 0 would
    read as failure when it is the opposite.
    """
    yields = []
    for path in ctx.optimization.trajectories:
        if path.size < 2:  # noqa: PLR2004 - needs a first step to have a first-step yield
            continue
        achievable = float(path[0]) - float(path[-1])
        if abs(achievable) < 1e-12:  # noqa: PLR2004 - nothing to improve, so no fraction of it exists
            continue
        yields.append((float(path[0]) - float(path[1])) / achievable)
    return float(np.mean(yields)) if yields else float("nan")
