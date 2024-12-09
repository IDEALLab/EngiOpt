from engibench.core import OptiStep
import numpy as np

from engilearn.utils.metrics import cumulative_optimality_gap
from engilearn.utils.metrics import optimality_gap


def test_opt_gap() -> None:
    """Test the optimality_gap function."""
    opt_history = [OptiStep(obj_values=np.array([1.0]), step=0), OptiStep(obj_values=np.array([0.0]), step=1)]
    assert optimality_gap(opt_history) == 1.0


def test_cum_opt_gap() -> None:
    """Test the cumulative_optimality_gap function."""
    opt_history = [
        OptiStep(obj_values=np.array([2.0]), step=0),
        OptiStep(obj_values=np.array([1.0]), step=1),
        OptiStep(obj_values=np.array([0.0]), step=2),
    ]
    assert cumulative_optimality_gap(opt_history)[0] == 3.0  # noqa: PLR2004


def test_cum_opt_gap_multiobj() -> None:
    """Test the cumulative_optimality_gap function with multiple objectives."""
    opt_history = [
        OptiStep(obj_values=np.array([2.0, 1.0]), step=0),
        OptiStep(obj_values=np.array([1.0, 0.0]), step=1),
        OptiStep(obj_values=np.array([0.0, 0.0]), step=2),
    ]
    assert np.all(cumulative_optimality_gap(opt_history) == np.array([3.0, 1.0]))
