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


def optimality_gap(opt_history: list[OptiStep], baseline: float) -> list[float]:
    """Compute the optimality gap of an optimization history.

    Args:
        opt_history (list[OptiStep]): The optimization history.
        baseline (float): The baseline value to compare against.

    Returns:
        list[float]: The optimality gap at each step in opt_history.
    """
    return [opt.obj_values - baseline for opt in opt_history]


# Return the failure ratio
