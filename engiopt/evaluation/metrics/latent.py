"""Latent-space metrics, per `claude_markdowns/metric_suite.md`.

Three axes, each answering a different question about a generated design:

- **Is it a valid design?** -- projection residual `r`, and optionally the
  dual-LVAE gap. Both are measured in *pixel* space on purpose: latent-space
  per-sample distances are uninformative here, because even a bad design gets
  mapped to a normal-looking latent code.
- **Does it match its stated condition?** -- paired latent distance `l_perf`
  and condition-recovery error `e`.
- **Is the set as a whole right?** -- LV-MMD, coverage, and LV-Vendi diversity.

Two deliberate exclusions, both from the same document:

- **Conditional MMD** is degenerate here. `p(design | c)` is close to a point
  mass -- one optimum per condition -- so estimating a per-condition
  distribution is ill-posed. `l_perf` and condition-recovery replace it.
- **DPP diversity** is inflated by noise, which is the failure mode the suite
  exists to catch. LV-Vendi is the diversity measure instead.

Pixel-space equivalents are registered alongside as the baselines each latent
metric has to beat, not as competing recommendations.

The bandwidth is calibrated on the validation split, so nothing reported is
tuned on the reference set it scores against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from engiopt import metrics as metrics_mod
from engiopt.evaluation.registry import register_metric

if TYPE_CHECKING:
    import numpy.typing as npt

    from engiopt.evaluation.context import EvaluationContext

COVERAGE_QUANTILE = 0.05
"""Neighbour radius tau, as a quantile of reference-to-reference distances."""


def _residuals(designs: npt.NDArray, projected: npt.NDArray) -> npt.NDArray:
    """Per-sample L2 distance between designs and their manifold projections."""
    flat = np.asarray(designs).reshape(len(designs), -1)
    return np.linalg.norm(flat - np.asarray(projected).reshape(len(projected), -1), axis=1)


# ----------------------------------------------------------------------
# Axis 1 -- is it a valid design?
# ----------------------------------------------------------------------


@register_metric(
    "lv_residual",
    family="latent",
    cost="cheap",
    higher_is_better=False,
    outputs=("lv_residual_mean", "lv_residual_p90"),
    description="Pixel-space distance from each generated design to its projection on the LVAE manifold.",
)
def lv_residual(ctx: EvaluationContext) -> dict[str, float]:
    """How far each generated design sits off the manifold of real optima.

    Measured in pixel space rather than latent space: the encoder maps an
    invalid design to a perfectly ordinary-looking code, so the distance only
    becomes visible after decoding back out.

    The p90 is reported alongside the mean because a generator that is usually
    fine and occasionally far off-manifold is a different failure from one that
    is uniformly mediocre, and the mean alone cannot tell them apart.
    """
    residuals = _residuals(ctx.gen_designs, ctx.gen_projected)
    return {
        "lv_residual_mean": float(residuals.mean()),
        "lv_residual_p90": float(np.percentile(residuals, 90)),
    }


@register_metric(
    "lv_dual_gap",
    family="latent",
    cost="cheap",
    higher_is_better=False,
    description="Distance between the performance-constrained and reconstruction-only reconstructions.",
)
def lv_dual_gap(ctx: EvaluationContext) -> float:
    """What the performance constraint changes about a design's reconstruction.

    Two autoencoders trained to the same reconstruction budget, one of which
    also had to predict performance, will reconstruct the same design
    differently exactly where performance-relevant structure lives. The size of
    that disagreement is the signal.

    `metric_suite.md` marks this provisional: keep it only if it earns its place
    against the residual in a `COG ~ r + gap` regression, and drop it if the two
    turn out to be strongly correlated.
    """
    perf = ctx.require_latent_lvae()
    recon = ctx.require_recon_only_lvae()

    designs = np.asarray(ctx.gen_designs)
    perf_projection = perf.project(designs).reshape(len(designs), -1)
    recon_projection = recon.project(designs).reshape(len(designs), -1)
    return float(np.linalg.norm(perf_projection - recon_projection, axis=1).mean())


# ----------------------------------------------------------------------
# Axis 2 -- does it match the condition it was asked for?
# ----------------------------------------------------------------------


@register_metric(
    "lv_paired_distance",
    family="conditions",
    cost="cheap",
    higher_is_better=False,
    description="Latent distance between each generated design and the known optimum for its condition.",
)
def lv_paired_distance(ctx: EvaluationContext) -> float:
    """Distance to the design that condition should have produced.

    Every generated design is paired with the dataset optimum for the same
    condition, so this asks a sharper question than any distribution metric: not
    "does this set look right" but "is *this* design what *this* condition
    called for". Restricted to the performance-carrying latent dimensions, and
    standardized per dimension so no single axis dominates by scale alone.
    """
    generated, reference = ctx.latent_codes
    perf_dim = min(ctx.require_latent_lvae().config.perf_dim, generated.shape[1])

    gen_slice = generated[:, :perf_dim]
    ref_slice = reference[:, :perf_dim]

    scale = ref_slice.std(axis=0)
    scale[scale == 0] = 1.0
    return float(np.linalg.norm((gen_slice - ref_slice) / scale, axis=1).mean())


@register_metric(
    "cond_err",
    family="conditions",
    cost="cheap",
    higher_is_better=False,
    description="Exact error between a design's realized volume fraction and the one requested.",
)
def cond_err(ctx: EvaluationContext) -> float:
    """Did the design hit the volume fraction it was asked for?

    Read analytically -- the volume fraction of a density field *is* its mean,
    so there is nothing to fit and nothing to approximate. `metric_suite.md`
    calls for the analytic readout wherever one exists, and this is the case
    where one does.

    Returns NaN on problems with no volume budget (photonics2d), which is
    reported rather than papered over.
    """
    if ctx.volume_condition is None or ctx.conditions is None:
        return float("nan")

    requested = np.asarray(ctx.conditions[ctx.volume_condition], dtype=np.float64)
    realized = ctx.gen_flat.mean(axis=1)
    return float(np.abs(realized - requested).mean())


@register_metric(
    "cond_recovery",
    family="conditions",
    cost="cheap",
    outputs=("cond_recovery_err", "cond_probe_r2"),
    higher_is_better=False,
    description="Probe-based condition readout, for conditions with no closed form. See cond_err first.",
)
def cond_recovery(ctx: EvaluationContext) -> dict[str, float]:
    """The fallback for conditions that cannot be read off a design directly.

    Where a closed form exists, use it: `cond_err` computes the volume-fraction
    error exactly, and no fitted probe can improve on an identity. This metric
    exists for the rest -- `rmin`, `forcedist`, `overhang_constraint` -- where
    the only way to ask "does this design still reflect its condition" is to
    learn the readout.

    A least-squares matrix is fitted on training codes, deliberately not a
    network: the question is whether the condition is *linearly readable* from a
    frozen encoder, and a model with its own capacity could recover a condition
    the codes barely encode, answering something else.

    Costs a pass over the full training split, so it is worth selecting only
    when those non-analytic conditions matter. `cond_probe_r2` decides whether
    the error column means anything at all -- a low training R-squared says the
    conditions are not linearly readable here, and the error then describes the
    probe rather than the generator.
    """
    weights, r_squared = ctx.condition_probe
    generated, _ = ctx.latent_codes

    requested = ctx.requested_conditions
    predicted = np.hstack([generated, np.ones((len(generated), 1))]) @ weights

    error = float(np.linalg.norm(predicted - requested, axis=1).mean())
    return {"cond_recovery_err": error, "cond_probe_r2": float(np.nanmean(r_squared))}


# ----------------------------------------------------------------------
# Axis 3 -- is the set as a whole right?
# ----------------------------------------------------------------------


@register_metric(
    "lv_mmd",
    family="latent",
    cost="cheap",
    higher_is_better=False,
    description="Maximum Mean Discrepancy in the active latent subspace of the pinned instrument.",
)
def lv_mmd(ctx: EvaluationContext) -> float:
    """Distribution distance measured in latent rather than pixel space."""
    generated, reference = ctx.latent_codes
    return float(metrics_mod.mmd(generated, reference, sigma=ctx.latent_sigma))


@register_metric(
    "lv_coverage",
    family="latent",
    cost="cheap",
    higher_is_better=True,
    description="Fraction of reference optima with a generated design within tau in latent space.",
)
def lv_coverage(ctx: EvaluationContext) -> float:
    """How much of the reference set the generator actually reaches.

    Distribution distance can look healthy while whole regions go unvisited, so
    this counts reference optima directly: a mode the generator never produces
    leaves its neighbourhood empty. `tau` is set from the reference set's own
    nearest-neighbour spacing, so it adapts to the space rather than importing
    an arbitrary radius.
    """
    generated, reference = ctx.latent_codes
    from scipy.spatial.distance import cdist

    within_reference = cdist(reference, reference)
    np.fill_diagonal(within_reference, np.inf)
    tau = float(np.quantile(within_reference.min(axis=1), 1.0 - COVERAGE_QUANTILE))

    nearest_generated = cdist(reference, generated).min(axis=1)
    return float((nearest_generated <= tau).mean())


@register_metric(
    "lv_vendi",
    family="diversity",
    cost="cheap",
    higher_is_better=True,
    description="Vendi score (effective number of distinct designs) over latent codes.",
)
def lv_vendi(ctx: EvaluationContext) -> float:
    """Effective number of distinct designs in the generated set.

    The exponential of the von Neumann entropy of the normalized kernel matrix,
    which reads directly as a count: a set of `n` identical designs scores 1, a
    set of `n` mutually dissimilar ones scores `n`.

    Preferred over DPP diversity because it is not inflated by noise -- adding
    Gaussian noise to a collapsed set raises a determinant-based score while
    leaving the effective count where it belongs.
    """
    generated, _ = ctx.latent_codes
    return metrics_mod.vendi_score(generated, sigma=ctx.latent_sigma)


# ----------------------------------------------------------------------
# Pixel-space baselines -- what the latent metrics have to beat
# ----------------------------------------------------------------------


@register_metric(
    "pixel_paired_distance",
    family="conditions",
    cost="cheap",
    higher_is_better=False,
    description="Pixel-space distance between each generated design and the optimum for its condition.",
)
def pixel_paired_distance(ctx: EvaluationContext) -> float:
    """The baseline `lv_paired_distance` has to beat."""
    return float(np.linalg.norm(ctx.gen_flat - ctx.ref_flat, axis=1).mean())


@register_metric(
    "pixel_vendi",
    family="diversity",
    cost="cheap",
    higher_is_better=True,
    description="Vendi score over raw pixels, the baseline for lv_vendi.",
)
def pixel_vendi(ctx: EvaluationContext) -> float:
    """The baseline `lv_vendi` has to beat."""
    return metrics_mod.vendi_score(ctx.gen_flat, sigma=ctx.sigma)
