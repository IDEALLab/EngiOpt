"""`Generator` contract for the k-nearest-neighbour retrieval baseline."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import torch as th

from engiopt.core import ConditionBatch
from engiopt.core import Generator

if TYPE_CHECKING:
    from engibench.core import Problem
    import numpy.typing as npt

    from engiopt.checkpoint_store import ResolvedCheckpoint


def match_volume_fraction(
    designs: npt.NDArray[Any], targets: npt.NDArray[Any], *, iterations: int = 40
) -> npt.NDArray[Any]:
    """Rescale each design's densities so its mean hits the requested volume fraction.

    A per-design gain found by bisection, because clipping at 1 makes the mean a
    non-linear function of it. The structure is untouched -- the same material
    stays in the same places -- so this is a calibration step, not a redesign.

    Args:
        designs: `(n, ...)` density fields.
        targets: `(n,)` requested volume fractions.
        iterations: Bisection steps.

    Returns:
        The rescaled designs, clipped to `[0, 1]`.
    """
    out = np.empty_like(designs, dtype=np.float64)
    for i, target in enumerate(np.asarray(targets, dtype=np.float64)):
        design = np.asarray(designs[i], dtype=np.float64)
        low, high = 0.0, 1e3
        for _ in range(iterations):
            gain = 0.5 * (low + high)
            if np.clip(design * gain, 0.0, 1.0).mean() < target:
                low = gain
            else:
                high = gain
        out[i] = np.clip(design * (0.5 * (low + high)), 0.0, 1.0)
    return out


class KNNRetrieval(Generator):
    """Retrieval baseline: average the `k` training designs with the nearest conditions.

    Competitive rather than decorative -- Habibi et al. (J. Mech. Des. 148(6):061704,
    2026) found kNN beats deconvolutional networks for topology-optimization
    warm-starting at limited data sizes, once data-generation cost is counted.

    It has no latent variable, so it is deterministic given a condition: every
    seed returns the same design. That is a real limitation and the diversity
    columns are where it shows up.
    """

    algo_id = "knn_retrieval"
    conditional = True
    design_kinds = ("2d",)
    checkpoint_files = ("knn.pth",)
    primary_state_key = "designs"
    output_clip = (1e-3, 1.0)

    volume_condition = "volfrac"

    def __init__(
        self,
        conditions: npt.NDArray[Any],
        designs: npt.NDArray[Any],
        scale: npt.NDArray[Any],
        k: int,
        *,
        distance_weighted: bool = False,
        match_volume: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.train_conditions = conditions
        self.train_designs = designs
        self.scale = scale
        # The retrieval set is this model's parameters -- there is nothing else
        # to learn -- so it is registered as such and lands in the `params`
        # column beside a diffusion model's weight count. Putting a kNN's data
        # cost on the same axis as a network's parameter cost is the comparison
        # Habibi et al. argue nobody makes.
        self.store = th.nn.Module()
        self.store.retrieval_set = th.nn.Parameter(th.from_numpy(designs).float(), requires_grad=False)
        self.k = k
        self.distance_weighted = distance_weighted
        self.match_volume = match_volume

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> KNNRetrieval:
        """Load the fitted retrieval index from its checkpoint package."""
        payload = th.load(resolved.files["knn.pth"], map_location="cpu", weights_only=False)
        config = resolved.run_config
        return cls(
            conditions=payload["conditions"].numpy().astype(np.float64),
            designs=payload["designs"].numpy().astype(np.float32),
            scale=payload["scale"].numpy().astype(np.float64),
            k=int(payload.get("k", config.get("k", 5))),
            distance_weighted=bool(payload.get("distance_weighted", False)),
            match_volume=bool(payload.get("match_volume", True)),
            problem=problem,
            device=device,
            **base,
        )

    def _sample(self, conditions: ConditionBatch, n: int) -> npt.NDArray[Any]:
        """Retrieve, blend, and optionally rescale to the requested volume budget."""
        requested = conditions.require_tensor(self.algo_id).detach().cpu().numpy().astype(np.float64)[:n]

        pool = self.train_conditions / self.scale
        query = requested / self.scale
        distances = np.linalg.norm(query[:, None, :] - pool[None, :, :], axis=2)
        neighbours = np.argsort(distances, axis=1)[:, : self.k]

        picked = self.train_designs[neighbours]
        if self.distance_weighted:
            # Inverse-distance weights, floored so an exact match does not divide by zero.
            weights = 1.0 / np.maximum(np.take_along_axis(distances, neighbours, axis=1), 1e-8)
            weights = weights / weights.sum(axis=1, keepdims=True)
            designs = (picked * weights[..., None, None]).sum(axis=1)
        else:
            designs = picked.mean(axis=1)

        if not self.match_volume:
            return designs
        column = self._volume_column(conditions)
        return designs if column is None else match_volume_fraction(designs, requested[:, column])

    def _volume_column(self, conditions: ConditionBatch) -> int | None:
        """Position of the volume-fraction condition in the input tensor, if present."""
        keys = conditions.keys or self.condition_keys
        return keys.index(self.volume_condition) if self.volume_condition in keys else None
