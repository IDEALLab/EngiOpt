"""`Generator` contract for the ridge-regression baseline."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import torch as th

from engiopt.core import ConditionBatch
from engiopt.core import Generator
from engiopt.generators.knn_retrieval.adapter import match_volume_fraction
from engiopt.generators.linear_regression.linear_regression import polynomial_features

if TYPE_CHECKING:
    from engibench.core import Problem
    import numpy.typing as npt

    from engiopt.checkpoint_store import ResolvedCheckpoint


class LinearRegression(Generator):
    """Closed-form ridge map from conditions to pixels.

    Deterministic by construction: one condition in, one design out, the same
    one every time. The model has no way to express that several different
    structures might satisfy the same brief, which is precisely what the
    diversity family measures and what a generative model is supposed to add.
    """

    algo_id = "linear_regression"
    conditional = True
    design_kinds = ("2d",)
    checkpoint_files = ("linear_regression.pth",)
    primary_state_key = "weights"
    output_clip = (1e-3, 1.0)

    volume_condition = "volfrac"

    def __init__(
        self,
        weights: npt.NDArray[Any],
        design_shape: tuple[int, ...],
        degree: int,
        *,
        match_volume: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.weights = weights
        # Held as a module so the `params` column reports the real count. Small
        # enough that the number is the point: this is what the rest of the
        # leaderboard has to justify itself against.
        self.net = th.nn.Module()
        self.net.coefficients = th.nn.Parameter(th.from_numpy(weights).float(), requires_grad=False)
        self.fitted_shape = design_shape
        self.degree = degree
        self.match_volume = match_volume

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> LinearRegression:
        """Load the fitted weight matrix from its checkpoint package."""
        payload = th.load(resolved.files["linear_regression.pth"], map_location="cpu", weights_only=False)
        config = resolved.run_config
        return cls(
            weights=payload["weights"].numpy().astype(np.float64),
            design_shape=tuple(payload["design_shape"]),
            degree=int(payload.get("degree", config.get("degree", 2))),
            match_volume=bool(payload.get("match_volume", False)),
            problem=problem,
            device=device,
            **base,
        )

    def _sample(self, conditions: ConditionBatch, n: int) -> npt.NDArray[Any]:
        """Evaluate the fitted map at the requested conditions."""
        requested = conditions.require_tensor(self.algo_id).detach().cpu().numpy().astype(np.float64)[:n]
        predicted = polynomial_features(requested, self.degree) @ self.weights
        designs = predicted.reshape(n, *self.fitted_shape)

        if not self.match_volume:
            return designs
        keys = conditions.keys or self.condition_keys
        if self.volume_condition not in keys:
            return designs
        return match_volume_fraction(designs, requested[:, keys.index(self.volume_condition)])
