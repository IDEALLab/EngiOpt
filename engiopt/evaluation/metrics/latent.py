"""Latent-space metrics.

Pixel-space distances treat every cell of a design as an independent coordinate,
which makes them sensitive to things a designer does not care about -- a
one-pixel translation, a little additive noise -- and insensitive to things they
do, like whether a structure is connected. Measuring in the active latent
subspace of a least-volume autoencoder instead puts distance in a space where
directions correspond to variation the data actually contains.

The cost is that these metrics depend on a fitted instrument as well as on the
designs, so the spec must pin one. See `engiopt.evaluation.spec.LatentInstrument`.

Bandwidths use the median heuristic rather than the spec's fixed `sigma`: the
latent space and pixel space differ in scale by orders of magnitude, and a
bandwidth chosen for one saturates in the other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from engiopt import metrics as metrics_mod
from engiopt.evaluation.registry import register_metric

if TYPE_CHECKING:
    from engiopt.evaluation.context import EvaluationContext


@register_metric(
    "lv_mmd",
    family="latent",
    cost="cheap",
    higher_is_better=False,
    description="Maximum Mean Discrepancy in the active latent subspace of the pinned instrument.",
)
def lv_mmd(ctx: EvaluationContext) -> float:
    """Distribution distance measured in latent rather than pixel space."""
    gen, ref = ctx.latent_codes
    sigma = metrics_mod.compute_median_sigma(ref)
    return float(metrics_mod.mmd(gen, ref, sigma=sigma))


@register_metric(
    "lv_dpp",
    family="latent",
    cost="cheap",
    higher_is_better=True,
    description="Log-determinant DPP diversity in the active latent subspace.",
)
def lv_dpp(ctx: EvaluationContext) -> float:
    """Diversity measured in latent space, on a log scale.

    Uses the log-determinant form because the raw determinant of a 50x50 kernel
    matrix underflows toward the limit of double precision, at which point it
    reports rounding error rather than diversity.
    """
    gen, ref = ctx.latent_codes
    sigma = metrics_mod.compute_median_sigma(ref)
    return metrics_mod.log_dpp_diversity(gen, sigma=sigma)


@register_metric(
    "lv_prdc",
    family="latent",
    cost="cheap",
    higher_is_better=True,
    outputs=("lv_precision", "lv_recall", "lv_density", "lv_coverage"),
    description="Precision, recall, density, and coverage in the active latent subspace.",
)
def lv_prdc(ctx: EvaluationContext) -> dict[str, float]:
    """Split latent-space fidelity from latent-space diversity.

    A single distance conflates generating implausible designs with generating
    too few distinct ones; these four separate the two failures.
    """
    gen, ref = ctx.latent_codes
    scores = metrics_mod.compute_prdc(ref, gen)
    return {f"lv_{name}": value for name, value in scores.items()}
