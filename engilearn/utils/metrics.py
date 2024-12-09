"""This module contains utility functions for computing metrics."""

from engibench import OptiStep
import numpy as np


def optimality_gap(opt_history: list[OptiStep]) -> float:
    """Compute the optimality gap of an optimization history.

    Args:
        opt_history (list[OptiStep]): The optimization history.

    Returns:
        float: The optimality gap.
    """
    return np.abs(opt_history[-1].obj_values - opt_history[0].obj_values)


def cumulative_optimality_gap(opt_history: list[OptiStep]) -> float:
    """Compute the cumulative optimality gap of an optimization history.

    Args:
        opt_history (list[OptiStep]): The optimization history.

    Returns:
        float: The cumulative optimality gap.
    """
    return np.sum([np.abs(opt.obj_values - opt_history[0].obj_values) for opt in opt_history[1:]], axis=0)
