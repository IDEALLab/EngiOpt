"""Novelty, and the linear baseline the latent metrics are measured against.

Two metrics that sit outside the latent suite but exist because of it.

`novelty` closes a gap no distribution metric can: MMD is *minimized* by
returning the training set verbatim, so a model that memorizes scores perfectly
on the thing most papers report. Nothing else in the suite notices.

`pca_mmd` is the control for the claim that latent-space measurement helps. PCA
fitted to the same number of components as the instrument's active subspace is
the closest linear analogue of that subspace, so a latent metric that fails to
beat it is not buying anything a rotation could not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import cdist

from engiopt import metrics as metrics_mod
from engiopt.evaluation.metrics.latent import COVERAGE_QUANTILE
from engiopt.evaluation.registry import register_metric

if TYPE_CHECKING:
    from engiopt.evaluation.context import EvaluationContext


def _pca_codes(ctx: EvaluationContext) -> tuple[np.ndarray, np.ndarray]:
    """Generated and reference designs in a PCA subspace matched to the instrument.

    Thin wrapper over `EvaluationContext.pca_codes`, which fits the components on
    the validation split and caches them so the decomposition is not repeated
    once per metric.
    """
    generated, reference, _ = ctx.pca_codes
    return generated, reference


@register_metric(
    "pca_paired_distance",
    family="conditions",
    cost="cheap",
    higher_is_better=False,
    description="PCA-subspace distance between each generated design and the optimum for its condition.",
)
def pca_paired_distance(ctx: EvaluationContext) -> float:
    """The middle term `lv_paired_distance` has to beat.

    `lv_paired_distance` is the strongest cheap predictor of downstream
    optimization quality on the boards run so far, but on its own that number
    cannot separate two explanations: that the *latent space* is the right place
    to measure, or that *pairing each generated design against the optimum for
    its own condition* is simply a good idea in any space. `pixel_paired_distance`
    supplies one end of that comparison; this supplies the middle -- the same
    pairing in a linear subspace matched to the instrument's active
    dimensionality. Without both, a manifold claim rests on an uncontrolled
    comparison, which is the trap the corruption battery already fell into when
    matched-dimension PCA turned out to reproduce most of the effect.
    """
    generated, reference = _pca_codes(ctx)
    return float(np.linalg.norm(generated - reference, axis=1).mean())


@register_metric(
    "novelty",
    family="distribution",
    cost="cheap",
    higher_is_better=True,
    description="Mean distance from each generated design to its nearest training design.",
)
def novelty(ctx: EvaluationContext) -> float:
    """How far the generated designs sit from anything the model was trained on.

    This is the memorization check, and it is the reason it cannot be inferred
    from the other columns: MMD, precision, coverage and the rest all *improve*
    as generated designs approach the training distribution, and are optimal
    when the model reproduces it exactly. A lookup table beats every one of
    them. Only distance-to-training-set falls, and only it separates a model
    that learned the manifold from one that memorized points on it.

    Higher is better only up to a point -- a model producing garbage also scores
    high -- so it qualifies the distribution metrics rather than replacing them.
    Read it as a floor: near zero means the result is not a model's output in
    any useful sense.
    """
    if ctx.train_designs is None:
        return float("nan")

    train_flat = np.asarray(ctx.train_designs).reshape(len(ctx.train_designs), -1)
    return float(cdist(ctx.gen_flat, train_flat).min(axis=1).mean())


@register_metric(
    "pca_mmd",
    family="distribution",
    cost="cheap",
    higher_is_better=False,
    description="MMD in a PCA subspace matched to the instrument's active dimensionality.",
)
def pca_mmd(ctx: EvaluationContext) -> float:
    """The linear control for latent-space measurement.

    Projects designs onto principal components fitted on the validation split --
    never on the reference set the metric then scores against -- and takes as
    many components as the instrument keeps active, so the comparison is at
    matched dimensionality rather than matched effort.

    If `lv_mmd` does not beat this, the autoencoder is contributing nothing a
    linear projection does not already provide, and the latent machinery is not
    worth its instrument.
    """
    generated, reference = _pca_codes(ctx)
    return float(metrics_mod.mmd(generated, reference, sigma=ctx.pca_sigma))


@register_metric(
    "pca_vendi",
    family="diversity",
    cost="cheap",
    higher_is_better=True,
    description="Vendi score in a PCA subspace matched to the instrument's active dimensionality.",
)
def pca_vendi(ctx: EvaluationContext) -> float:
    """The linear control for `lv_vendi`.

    Exists to separate two explanations of why latent diversity behaves better
    than pixel diversity: that any low-dimensional projection suppresses the
    high-frequency noise a pixel-space score mistakes for variety, or that the
    performance constraint specifically is what does it. Only a matched linear
    projection can tell those apart.
    """
    generated, reference = _pca_codes(ctx)
    return metrics_mod.vendi_score(generated, sigma=ctx.pca_sigma)


@register_metric(
    "pca_coverage",
    family="distribution",
    cost="cheap",
    higher_is_better=True,
    description="Fraction of reference optima with a generated design within tau in a matched PCA subspace.",
)
def pca_coverage(ctx: EvaluationContext) -> float:
    """The linear control for `lv_coverage`, with the same tau rule."""
    generated, reference = _pca_codes(ctx)
    within_reference = cdist(reference, reference)
    np.fill_diagonal(within_reference, np.inf)
    tau = float(np.quantile(within_reference.min(axis=1), 1.0 - COVERAGE_QUANTILE))
    return float((cdist(reference, generated).min(axis=1) <= tau).mean())


@register_metric(
    "novelty_ratio",
    family="distribution",
    cost="cheap",
    higher_is_better=None,
    description="Novelty as a fraction of the reference designs' own novelty; 1.0 = as novel as real data.",
)
def novelty_ratio(ctx: EvaluationContext) -> float:
    """`novelty`, divided by the same measurement taken on real held-out designs.

    Raw novelty is a distance in whatever units the design space happens to
    have, and part of what it measures is how far the *evaluation conditions*
    sit from the training conditions -- a property of the split, not of the
    model. That makes an absolute value nearly uninterpretable: nobody can say
    whether 0.6 is memorization or healthy variation.

    The reference designs answer the same conditions, are real, and are not in
    the training split, so their distance-to-training is exactly the scale the
    model's should be read against:

    - ~1.0: as far from the training set as genuine held-out designs are
    - ~0.0: memorization
    - >>1.0: further from the data than real designs, which is as likely to be
      garbage as invention

    Deliberately declared with no direction. The good answer is *near one*, and
    a leaderboard that sorted by it would reward the models furthest from the
    data. Read it beside `novelty`, not instead of it.
    """
    if ctx.train_designs is None:
        return float("nan")

    train_flat = np.asarray(ctx.train_designs).reshape(len(ctx.train_designs), -1)
    generated = float(cdist(ctx.gen_flat, train_flat).min(axis=1).mean())
    reference = float(cdist(ctx.ref_flat, train_flat).min(axis=1).mean())
    return generated / reference if reference > 0 else float("nan")
