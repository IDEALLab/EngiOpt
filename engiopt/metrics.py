"""This module provides metrics for evaluating generative model designs.

Maximum Mean Discrepancy (MMD), Determinantal Point Process (DPP) diversity,
and optimality gap calculations.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import traceback
from typing import Any, TYPE_CHECKING

from gymnasium import spaces
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from datasets import Dataset
    from engibench import OptiStep
    from engibench.core import Problem


if sys.platform != "win32":  #  only set fork on non-Windows
    multiprocessing.set_start_method("fork", force=True)
else:
    multiprocessing.set_start_method("spawn", force=True)


def apply_importance_weights(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Weight each dimension by sqrt(variance) computed from a reference set.

    This makes distance calculations sensitive to dimensions that vary more,
    whether those dimensions are pixels, latent codes, or any other features.

    Args:
        x: Array of shape (n, d) to be weighted.
        ref: Reference array of shape (m, d) used to compute per-dimension variance.

    Returns:
        Weighted copy of x with shape (n, d).
    """
    var = np.var(ref, axis=0)
    weights = np.sqrt(var + 1e-12)
    return x * weights


def mmd(x: np.ndarray, y: np.ndarray, sigma: float | None = 1.0, importance_weighted: bool = False) -> float:
    """Compute the Maximum Mean Discrepancy (MMD) between two sets of samples.

    Args:
        x: Array of shape (n, ...) for generated designs (or latent codes).
        y: Array of shape (m, ...) for dataset/reference designs (or latent codes).
        sigma: Bandwidth parameter for the Gaussian kernel. If None, uses median heuristic
            on the reference data y only (so sigma is consistent across different generators).
        importance_weighted: If True, weight each dimension by sqrt(variance) of y
            before computing distances. Useful for latent spaces where some
            dimensions are more informative than others.

    Returns:
        float: The MMD value.
    """
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)

    if importance_weighted:
        x_flat = apply_importance_weights(x_flat, y_flat)
        y_flat = apply_importance_weights(y_flat, y_flat)

    if sigma is None:
        sigma = compute_median_sigma(y_flat)

    k_xx = np.exp(-cdist(x_flat, x_flat, "sqeuclidean") / (2 * sigma**2))
    k_yy = np.exp(-cdist(y_flat, y_flat, "sqeuclidean") / (2 * sigma**2))
    k_xy = np.exp(-cdist(x_flat, y_flat, "sqeuclidean") / (2 * sigma**2))

    return float(k_xx.mean() + k_yy.mean() - 2 * k_xy.mean())


def dpp_diversity(x: np.ndarray, sigma: float | None = 1.0, importance_weighted: bool = False) -> float:
    """Compute the Determinantal Point Process (DPP) diversity for a set of samples.

    Args:
        x: Array of shape (n, ...) for generated designs (or latent codes).
        sigma: Bandwidth parameter for the Gaussian kernel. If None, uses median heuristic.
        importance_weighted: If True, weight each dimension by sqrt(variance) of x
            before computing distances.

    Returns:
        float: The DPP diversity value.
    """
    x_flat = x.reshape(x.shape[0], -1)

    if importance_weighted:
        x_flat = apply_importance_weights(x_flat, x_flat)

    if sigma is None:
        sigma = compute_median_sigma(x_flat)

    pairwise_sq_dists = cdist(x_flat, x_flat, "sqeuclidean")
    similarity_matrix = np.exp(-pairwise_sq_dists / (2 * sigma**2))

    # Regularize the matrix slightly to avoid numerical issues
    reg_matrix = similarity_matrix + 1e-6 * np.eye(x.shape[0])

    try:
        return float(np.linalg.det(reg_matrix))
    except np.linalg.LinAlgError:
        return 0.0  # fallback in case of numerical issues


def compute_median_sigma(x: np.ndarray, y: np.ndarray | None = None) -> float:
    """Compute sigma using median heuristic on pairwise distances.

    Args:
        x: First set of samples of shape (n, d).
        y: Second set of samples (optional, uses x if None).

    Returns:
        Computed sigma value.
    """
    x = x.reshape(x.shape[0], -1)
    if y is not None:
        y = y.reshape(y.shape[0], -1)

    n_sample = min(500, len(x))
    rng = np.random.default_rng(42)
    idx_x = rng.choice(len(x), n_sample, replace=len(x) < n_sample)

    if y is not None:
        idx_y = rng.choice(len(y), n_sample, replace=len(y) < n_sample)
        dists = cdist(x[idx_x], y[idx_y], "sqeuclidean")
    else:
        dists = cdist(x[idx_x], x[idx_x], "sqeuclidean")
        # Use upper triangle (exclude diagonal)
        dists = dists[np.triu_indices_from(dists, k=1)]

    sigma = np.sqrt(np.median(dists) / 2) if len(dists) > 0 else 1.0
    return max(sigma, 1e-6)


def optimality_gap(opt_history: list[OptiStep], baseline: float) -> list[float]:
    """Compute the optimality gap of an optimization history.

    Args:
        opt_history (list[OptiStep]): The optimization history.
        baseline (float): The baseline value to compare against.

    Returns:
        list[float]: The optimality gap at each step in opt_history.
    """
    return [opt.obj_values - baseline for opt in opt_history]


def simulate_failure_ratio(  # noqa: C901
    problem: Problem,
    gen_designs: npt.NDArray,
    sampled_conditions: Dataset | None = None,
) -> float:
    """Compute the failure ratio of generated designs. This is designed for the airfoil problem.

    Args:
        problem: The optimization problem to evaluate.
        gen_designs (np.ndarray): Array of shape (n_samples, l, w) for generative model designs.
        sampled_conditions (Dataset): Dataset of sampled conditions for optimization. If None, no conditions are used.

    Returns:
        float: The failure ratio of generated designs.
    """
    failure_count = 0
    for idx, design in enumerate(gen_designs):
        if isinstance(problem.design_space, spaces.Dict):
            # Need to unflatten the design to be used for optimization or simulation
            unflattened_design = spaces.unflatten(problem.design_space, design)
            unflattened_design["angle_of_attack"] = unflattened_design["angle_of_attack"][0]
        else:
            unflattened_design = design

        def worker(idx, config, return_queue):
            try:
                objs = problem.simulate(unflattened_design, config=config, mpicores=10)  # noqa: B023
                if np.isnan(objs[0]) or np.isnan(objs[1]):
                    print(f"Simulation returned NaN values for design {idx}")
                    raise Exception("Simulation returned NaN values")  # noqa: TRY002, TRY301
                return_queue.put(("ok", objs))
            except Exception:  # noqa: BLE001
                return_queue.put(("error", traceback.format_exc()))

        # Attempt to simulate the design
        def run_with_timeout(idx, timeout=30):
            config = sampled_conditions[idx] if sampled_conditions is not None else None
            q = multiprocessing.Queue()
            p = multiprocessing.Process(target=worker, args=(idx, config, q))
            p.start()
            p.join(timeout)

            if p.is_alive():
                p.terminate()  # force-kill the child process
                p.join()
                os.system("docker stop machaero")  # noqa: S605
                raise RuntimeError(f"Simulation timed out for design {idx}")

            if not q.empty():
                status, payload = q.get()
                if status == "ok":
                    print(f"Simulation successful for design {idx}: {payload}")
                else:
                    raise RuntimeError(f"Simulation error for design {idx}:\n{payload}")
            else:
                raise RuntimeError("Simulation process ended without reporting back")

        try:
            run_with_timeout(idx, timeout=30)
        except RuntimeError as e:
            failure_count += 1
            print(e)

    return failure_count / len(gen_designs)  # Return the failure ratio


def metrics(
    problem: Problem,
    gen_designs: npt.NDArray,
    dataset_designs: npt.NDArray,
    sampled_conditions: Dataset | None = None,
    sigma: float | None = None,
) -> dict[str, Any]:
    """Compute various metrics for evaluating generative model designs.

    Args:
        problem: The optimization problem to evaluate.
        gen_designs (np.ndarray): Array of shape (n_samples, l, w) for generative model designs (potentially flattened for dict spaces).
        dataset_designs (np.ndarray): Array of shape (n_samples, l, w) for dataset designs (these are not flattened).
        sampled_conditions (Dataset): Dataset of sampled conditions for optimization. If None, no conditions are used.
        sigma: Bandwidth parameter for the Gaussian kernel (in mmd and dpp calculation).
            If None, uses median heuristic on the reference (dataset) designs.

    Returns:
        dict[str, Any]: A dictionary containing the computed metrics:
            - "iog": Average Initial Optimality Gap (float).
            - "cog": Average Cumulative Optimality Gap (float).
            - "fog": Average Final Optimality Gap (float).
            - "mmd": Maximum Mean Discrepancy (float).
            - "dpp": Determinantal Point Process diversity (float).
            - "mmd_sigma": The actual sigma used for MMD/DPP (float).
    """
    n_samples = len(gen_designs)

    cog_list = []
    iog_list = []
    fog_list = []
    viol_list = []
    for i in range(n_samples):
        conditions = sampled_conditions[i] if sampled_conditions is not None else None
        if isinstance(problem.design_space, spaces.Dict):
            # Need to unflatten the design to be used for optimization or simulation
            unflattened_design = spaces.unflatten(problem.design_space, gen_designs[i])
        else:
            unflattened_design = gen_designs[i]
        _, opt_history = problem.optimize(unflattened_design, config=conditions)
        reference_optimum = problem.simulate(dataset_designs[i], config=conditions)
        opt_history_gaps = optimality_gap(opt_history, reference_optimum)

        iog_list.append(problem.simulate(unflattened_design, config=conditions))
        cog_list.append(np.sum(opt_history_gaps))
        fog_list.append(opt_history_gaps[-1])

        # Check if conditions dict has 'volfrac' or 'volume' key and compare with design mean
        if conditions:
            tol = 0.01  # Tolerance for equality constraint deviation
            target_vol = conditions.get("volfrac") or conditions.get("volume")
            if target_vol is not None:
                viol = np.abs(np.mean(unflattened_design) - target_vol) >= tol
                viol_list.append(viol)

    # Compute the average Initial Optimality Gap (IOG), Cumulative Optimality Gap (COG), and Final Optimality Gap (FOG)
    average_iog: float = float(np.mean(iog_list))  # Average of initial optimality gaps
    average_cog: float = float(np.mean(cog_list))  # Average of cumulative optimality gaps
    average_fog: float = float(np.mean(fog_list))  # Average of final optimality gaps
    average_viol: float = float(np.mean(viol_list))  # Average of violation ratios

    # Compute the Maximum Mean Discrepancy (MMD) between generated and dataset designs
    # We compute the MMD on the flattened designs
    flattened_ds_designs: list[npt.NDArray] = []
    for design in dataset_designs:
        if isinstance(problem.design_space, spaces.Dict):
            flattened = spaces.flatten(problem.design_space, design)
            flattened_ds_designs.append(np.array(flattened))
        else:
            flattened_ds_designs.append(design)
    flattened_ds_designs_array: npt.NDArray = np.array(flattened_ds_designs)

    # Resolve sigma: if None, compute median heuristic from reference (dataset) designs
    if sigma is None:
        sigma = compute_median_sigma(flattened_ds_designs_array)

    mmd_value: float = mmd(gen_designs, flattened_ds_designs_array, sigma=sigma)

    # Compute the Determinantal Point Process (DPP) diversity for generated designs
    # We compute the DPP on the flattened designs
    dpp_value: float = dpp_diversity(gen_designs, sigma=sigma)

    # Return all computed metrics as a dictionary
    return {
        "iog": average_iog,
        "cog": average_cog,
        "fog": average_fog,
        "mmd": mmd_value,
        "dpp": dpp_value,
        "mmd_sigma": sigma,
        "viol": average_viol,
    }
