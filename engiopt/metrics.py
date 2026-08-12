"""Numerical primitives used by the evaluation metrics.

Kernel-based distribution and diversity measures, plus the optimality-gap
definition. The evaluator in `engiopt.evaluation` composes these; see
`engiopt/evaluation/metrics/builtin.py` for the registered metrics themselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from engibench import OptiStep

MIN_PRDC_SAMPLES = 2
"""Below two samples per set there is no neighbourhood to measure."""

EIGENVALUE_FLOOR = 1e-12
"""Eigenvalues below this are treated as zero: a PSD matrix can return
slightly negative ones, and exact zeros would send the entropy to -inf."""


def mmd(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Compute the Maximum Mean Discrepancy (MMD) between two sets of samples.

    Args:
        x (np.ndarray): Array of shape (n, l, w) for generative model designs.
        y (np.ndarray): Array of shape (m, l, w) for dataset designs.
        sigma (float): Bandwidth parameter for the Gaussian kernel.

    Returns:
        float: The MMD value.
    """
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)

    k_xx = np.exp(-cdist(x_flat, x_flat, "sqeuclidean") / (2 * sigma**2))
    k_yy = np.exp(-cdist(y_flat, y_flat, "sqeuclidean") / (2 * sigma**2))
    k_xy = np.exp(-cdist(x_flat, y_flat, "sqeuclidean") / (2 * sigma**2))

    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()


def dpp_diversity(x: np.ndarray, sigma: float = 1.0) -> float:
    """Compute the Determinantal Point Process (DPP) diversity for a set of samples.

    Args:
        x (np.ndarray): Array of shape (n, l, w) for generative model designs.
        sigma (float): Bandwidth parameter for the Gaussian kernel.

    Returns:
        float: The DPP diversity value.
    """
    x_flat = x.reshape(x.shape[0], -1)
    pairwise_sq_dists = cdist(x_flat, x_flat, "sqeuclidean")
    similarity_matrix = np.exp(-pairwise_sq_dists / (2 * sigma**2))

    # Regularize the matrix slightly to avoid numerical issues
    reg_matrix = similarity_matrix + 1e-6 * np.eye(x.shape[0])

    try:
        return np.linalg.det(reg_matrix)
    except np.linalg.LinAlgError:
        return 0.0  # fallback in case of numerical issues


def log_dpp_diversity(x: np.ndarray, sigma: float = 1.0) -> float:
    """Log-determinant form of `dpp_diversity`, for spaces where the raw determinant underflows.

    The determinant of an `n x n` kernel matrix is a product of `n` numbers below
    one, so at `n = 50` it routinely lands near 1e-11 and can reach the limit of
    double precision -- at which point the metric stops distinguishing models and
    starts reporting rounding error. `slogdet` measures the same quantity on a
    scale that survives.

    Args:
        x: Samples of shape `(n, ...)`; flattened internally.
        sigma: Bandwidth of the Gaussian kernel.

    Returns:
        The log-determinant. Larger means more diverse. Returns `-inf` for a
        singular matrix, which is the honest limit of a degenerate sample set.
    """
    x_flat = x.reshape(x.shape[0], -1)
    similarity = np.exp(-cdist(x_flat, x_flat, "sqeuclidean") / (2 * sigma**2))
    reg_matrix = similarity + 1e-6 * np.eye(x.shape[0])

    # On a near-singular kernel matrix NumPy's slogdet warns about intermediate
    # overflow while still returning a correct result -- which is precisely the
    # case this function exists to handle, so the warnings are noise here.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sign, logabsdet = np.linalg.slogdet(reg_matrix)

    if sign <= 0 or not np.isfinite(logabsdet):
        return float("-inf")
    return float(logabsdet)


def dpp_geometric_mean(x: np.ndarray, sigma: float = 1.0) -> float:
    r"""`n`-th root of the DPP determinant: the geometric mean of its eigenvalues.

    The same quantity `dpp_diversity` reports, on the only scale of the three
    that is readable *and* comparable across sample sizes.

    - The raw determinant is a product of `n` numbers below one, so it underflows
      (1e-11 at n=50, and worse) and distinct models render as identical zeros.
    - The log-determinant fixes the underflow but is unbounded below and still
      scales with `n`, so a value means nothing without knowing the sample count
      it was computed at -- and two papers reporting "log-DPP" at n=50 and n=200
      are not comparable.
    - `det(K)^(1/n) = exp(logdet / n)` is the geometric mean of the eigenvalues.
      Since `K` has a unit diagonal its eigenvalues sum to `n`, so their
      arithmetic mean is exactly 1 and the geometric mean lands in `(0, 1]` by
      AM-GM: **1 means a perfectly diverse set (`K = I`), and values approach 0
      as samples collapse onto each other.** That bound holds at every `n`.

    Interpretation is therefore fixed rather than relative: 0.5 is the same
    statement about a set of 50 designs as about a set of 500.

    This does **not** fix the pathology `vendi_score` exists to resist -- a
    determinant still rewards any perturbation that pushes samples apart, so
    adding noise to a collapsed set raises this too. It fixes the *numerical*
    failure only, which is a different and more embarrassing one.

    Args:
        x: Samples of shape `(n, ...)`; flattened internally.
        sigma: Bandwidth of the Gaussian kernel.

    Returns:
        The geometric mean of the kernel eigenvalues, in `(0, 1]`. Returns 0.0
        for a degenerate set, which is the honest limit rather than `-inf`.
    """
    n = x.shape[0]
    if n == 0:
        return 0.0

    logdet = log_dpp_diversity(x, sigma=sigma)
    if not np.isfinite(logdet):
        return 0.0
    return float(np.exp(logdet / n))


def vendi_score(x: np.ndarray, sigma: float = 1.0) -> float:
    """Effective number of distinct samples in a set.

    The exponential of the von Neumann entropy of the normalized similarity
    matrix, which reads directly as a count: `n` identical samples score 1, and
    `n` mutually dissimilar ones score `n`.

    Preferred over `dpp_diversity` for diversity. A determinant rewards any
    perturbation that makes samples less similar, so adding noise to a collapsed
    set *raises* it -- exactly the gaming this measure is meant to resist. The
    entropy of the eigenvalue spectrum does not move that way, because noise
    spreads eigenvalues without adding modes.

    See Friedman & Dieng (2023), "The Vendi Score".

    Args:
        x: Samples of shape `(n, ...)`; flattened internally.
        sigma: Bandwidth of the Gaussian kernel.

    Returns:
        The effective sample count, in `[1, n]`.
    """
    x_flat = x.reshape(x.shape[0], -1)
    n = x_flat.shape[0]
    if n == 0:
        return 0.0

    kernel = np.exp(-cdist(x_flat, x_flat, "sqeuclidean") / (2 * sigma**2)) / n
    eigenvalues = np.linalg.eigvalsh(kernel)

    # Clip: eigenvalues of a PSD matrix can come back slightly negative, and
    # zero eigenvalues contribute nothing to entropy but would produce -inf.
    positive = eigenvalues[eigenvalues > EIGENVALUE_FLOOR]
    if positive.size == 0:
        return 1.0
    return float(np.exp(-np.sum(positive * np.log(positive))))


def compute_median_sigma(x: np.ndarray, y: np.ndarray | None = None) -> float:
    """Choose a kernel bandwidth by the median heuristic.

    A fixed bandwidth cannot serve two spaces at once: pixel space and a pruned
    latent space differ in scale by orders of magnitude, and a kernel sized for
    one saturates in the other. The median pairwise distance adapts to whichever
    space it is handed.

    Sampling is capped and seeded so the bandwidth is reproducible.

    Args:
        x: Samples of shape `(n, d)`.
        y: Optional second sample set; distances are then cross-set.

    Returns:
        The bandwidth, floored at 1e-6 so a degenerate set cannot divide by zero.
    """
    x = x.reshape(x.shape[0], -1)
    n_sample = min(500, len(x))
    rng = np.random.default_rng(42)
    idx_x = rng.choice(len(x), n_sample, replace=len(x) < n_sample)

    if y is not None:
        y = y.reshape(y.shape[0], -1)
        idx_y = rng.choice(len(y), n_sample, replace=len(y) < n_sample)
        dists = cdist(x[idx_x], y[idx_y], "sqeuclidean")
    else:
        dists = cdist(x[idx_x], x[idx_x], "sqeuclidean")
        dists = dists[np.triu_indices_from(dists, k=1)]

    sigma = np.sqrt(np.median(dists) / 2) if len(dists) > 0 else 1.0
    return max(float(sigma), 1e-6)


def compute_prdc(real_features: np.ndarray, fake_features: np.ndarray, nearest_k: int = 5) -> dict[str, float]:
    """Compute precision, recall, density, and coverage between two sample sets.

    The four metrics of Naeem et al. 2020, "Reliable Fidelity and Diversity
    Metrics for Generative Models" (ICML). They exist because a single
    distribution distance conflates two different failures: generating
    implausible designs, and generating too few distinct ones. MMD reports one
    number for both; these separate them.

    All four lie in `[0, 1]`, higher is better.

    - **precision**: fraction of generated samples inside some real sample's
      k-NN ball -- are the generations plausible?
    - **recall**: fraction of real samples inside some generated sample's ball
      -- is the data manifold covered?
    - **density**: smoothed precision, robust to real outliers that would
      otherwise inflate it.
    - **coverage**: fraction of real samples whose own ball contains a generated
      sample; more robust than recall when generations are noisy outliers.

    Args:
        real_features: Reference samples, shape `(n_real, d)`.
        fake_features: Generated samples, shape `(n_fake, d)`.
        nearest_k: Neighbours defining the ball radius, clamped to fit the sets.

    Returns:
        Mapping with keys `precision`, `recall`, `density`, `coverage`. All NaN
        when either set has fewer than two samples.
    """
    real = real_features.reshape(real_features.shape[0], -1)
    fake = fake_features.reshape(fake_features.shape[0], -1)
    n_real, n_fake = len(real), len(fake)

    if n_real < MIN_PRDC_SAMPLES or n_fake < MIN_PRDC_SAMPLES:
        return dict.fromkeys(("precision", "recall", "density", "coverage"), float("nan"))

    k = max(1, min(nearest_k, n_real - 1, n_fake - 1))

    # Self-distance sits at index 0 of the partition, so index k is the k-th
    # non-self neighbour.
    real_radii = np.partition(cdist(real, real, "euclidean"), k, axis=1)[:, k]
    fake_radii = np.partition(cdist(fake, fake, "euclidean"), k, axis=1)[:, k]
    d_rf = cdist(real, fake, "euclidean")

    inside_real = d_rf <= real_radii[:, None]
    inside_fake = d_rf <= fake_radii[None, :]

    return {
        "precision": float(inside_real.any(axis=0).mean()),
        "recall": float(inside_fake.any(axis=1).mean()),
        "density": float(inside_real.sum(axis=0).mean()) / k,
        "coverage": float((d_rf.min(axis=1) <= real_radii).mean()),
    }


def optimality_gap(opt_history: list[OptiStep], baseline: float) -> list[float]:
    """Compute the optimality gap of an optimization history.

    Args:
        opt_history (list[OptiStep]): The optimization history.
        baseline (float): The baseline value to compare against.

    Returns:
        list[float]: The optimality gap at each step in opt_history.
    """
    return [opt.obj_values - baseline for opt in opt_history]
