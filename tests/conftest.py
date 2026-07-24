"""Shared fixtures.

The evaluation tests deliberately avoid EngiBench problems: they need a design
space and some conditions, not a physics solver, and pulling a real dataset
would make the suite slow and network-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gymnasium import spaces
import numpy as np
import pytest


@dataclass
class FakeOptiStep:
    """Stand-in for `engibench.OptiStep`, which carries one objective value."""

    obj_values: Any


class FakeProblem:
    """A minimal stand-in exposing the parts of `Problem` the evaluator touches.

    The fake objective is `mean(design)`, so the optimum is known analytically
    and the optimality gaps can be asserted exactly.
    """

    def __init__(
        self,
        shape: tuple[int, ...] = (8, 10),
        conditions_keys: tuple[str, ...] = ("volfrac", "rmin"),
        *,
        n_objectives: int = 1,
        directions: tuple[str, ...] | None = None,
    ):
        self.design_space = spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float64)
        self.conditions_keys = list(conditions_keys)
        self.n_objectives = n_objectives
        # Mirrors `Problem.objectives`: (name, direction) pairs in simulate order.
        directions = directions or ("MINIMIZE",) * n_objectives
        self.objectives = tuple((f"objective_{i}", directions[i]) for i in range(n_objectives))
        self.reset_calls = 0
        self.optimize_calls = 0

    def reset(self, seed: int | None = None) -> None:
        """Record that the evaluator reset us between solver calls."""
        self.reset_calls += 1

    def random_design(self) -> tuple[Any, dict[str, Any]]:
        """Return an arbitrary in-space design."""
        return self.design_space.sample(), {}

    def simulate(self, design: Any, config: Any = None) -> np.ndarray:
        """Objective is the design's mean, repeated per objective."""
        return np.full(self.n_objectives, float(np.mean(np.asarray(design))))

    def optimize(self, design: Any, config: Any = None) -> tuple[Any, list[FakeOptiStep]]:
        """Walk the objective down toward zero in three steps."""
        self.optimize_calls += 1
        start = float(np.mean(np.asarray(design)))
        history = [FakeOptiStep(np.full(self.n_objectives, start * factor)) for factor in (1.0, 0.5, 0.25)]
        return design, history


@pytest.fixture
def fake_problem() -> FakeProblem:
    """A small 2D problem with two scalar conditions."""
    return FakeProblem()


class FakeDataset:
    """A tiny stand-in for a HuggingFace dataset split."""

    def __init__(self, columns: dict[str, list[Any]]):
        self._columns = columns
        self.column_names = list(columns)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            return self._columns[item]
        return {name: values[item] for name, values in self._columns.items()}

    def __len__(self) -> int:
        return len(next(iter(self._columns.values())))


@pytest.fixture
def mixed_condition_dataset() -> FakeDataset:
    """A dataset mixing scalar, array-valued, and constant conditions.

    Mirrors the two problems that previously broke condition sampling:
    photonics2d (solver-only keys absent from the dataset) and thermoelastic2d
    (array-valued boundary conditions).
    """
    return FakeDataset(
        {
            "volfrac": [0.3, 0.4, 0.5],
            "rmin": [2.0, 2.0, 2.0],  # constant
            "fixed_elements": [np.zeros((4, 4)) for _ in range(3)],  # array-valued
        }
    )
