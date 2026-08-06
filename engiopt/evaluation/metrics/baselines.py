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
from engiopt.evaluation.registry import register_metric

if TYPE_CHECKING:
    from engiopt.evaluation.context import EvaluationContext

DEFAULT_PCA_COMPONENTS = 16
"""Used when no instrument is pinned to match dimensionality against."""


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
    from sklearn.decomposition import PCA

    fit_designs = ctx.sigma_designs if ctx.sigma_designs is not None else ctx.ref_designs
    fit_flat = np.asarray(fit_designs).reshape(len(fit_designs), -1)

    n_components = DEFAULT_PCA_COMPONENTS
    if ctx.latent_lvae is not None:
        from engiopt.lvae.encode import get_active_mask

        n_components = int(get_active_mask(ctx.latent_lvae.encoder).sum())
    n_components = max(1, min(n_components, *fit_flat.shape))

    pca = PCA(n_components=n_components).fit(fit_flat)
    generated = pca.transform(ctx.gen_flat)
    reference = pca.transform(ctx.ref_flat)

    return float(metrics_mod.mmd(generated, reference, sigma=metrics_mod.compute_median_sigma(reference)))
