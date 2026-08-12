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
