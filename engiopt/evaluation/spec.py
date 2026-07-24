"""Versioned evaluation specifications.

A leaderboard is only meaningful if every row was produced under identical
conditions. An `EvalSpec` freezes that contract: which problem, which test
conditions, how many samples, which metrics, and the kernel bandwidth. Every
leaderboard row records the `spec_version` it was produced under, so a change to
the protocol shows up as a new version rather than silently invalidating history.

Specs live in `engiopt/specs/<problem_id>/<version>.json` and are committed, so
the contract is reviewable in the diff.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch as th

from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.transforms import get_scalar_condition_keys

if TYPE_CHECKING:
    from datasets import Dataset
    from engibench.core import Problem
    import numpy.typing as npt

SPEC_ROOT = Path(__file__).resolve().parent.parent / "specs"
"""Directory holding committed specs, `engiopt/specs/<problem_id>/<version>.json`."""


@dataclass(frozen=True)
class EvalSpec:
    """The frozen contract for evaluating models on one problem.

    Attributes:
        problem_id: EngiBench problem registry key.
        version: Spec version, e.g. `v1`. Bump on any change below.
        n_samples: Number of conditions each model is scored on.
        condition_seed: Seed used to draw the test conditions.
        metrics: Metric names to compute, in leaderboard column order.
        sigma: Gaussian-kernel bandwidth for MMD and DPP.
        volfrac_tol: Tolerance for the volume-fraction violation check.
        objective_weights: Fixed weights over `problem.objectives`, for
            multi-objective problems. Summing or averaging objectives with
            different units is not a valid default, so one of these two fields
            is required whenever a problem has more than one objective.
        objective_weight_condition: Name of a per-sample condition carrying the
            trade-off instead of fixing it. thermoelastic2d's `weight` splits
            the first two objectives as `(w, 1 - w)`, so a purely structural
            sample (`w = 1`) is unaffected by thermal compliance.
        condition_digest: Hash of the drawn conditions and reference designs.
            Recomputed at evaluation time and compared, so an upstream dataset
            change is caught instead of silently shifting every number.
        engibench_version: EngiBench version the digest was recorded under.
        notes: Free-text rationale for this version.
    """

    problem_id: str
    version: str = "v1"
    n_samples: int = 50
    condition_seed: int = 1
    metrics: tuple[str, ...] = ("mmd", "dpp", "viol", "iog", "cog", "fog")
    sigma: float = 10.0
    volfrac_tol: float = 0.01
    objective_weights: tuple[float, ...] | None = None
    objective_weight_condition: str | None = None
    condition_digest: str | None = None
    engibench_version: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        """Normalize sequence fields to tuples.

        JSON has no tuple type, so a spec read back from disk would otherwise
        carry lists and compare unequal to the spec that produced it.
        """
        object.__setattr__(self, "metrics", tuple(self.metrics))
        if self.objective_weights is not None:
            object.__setattr__(self, "objective_weights", tuple(self.objective_weights))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, reference: str, *, root: Path | None = None) -> EvalSpec:
        """Load a spec from `"<problem_id>/<version>"` or a path to a JSON file."""
        path = Path(reference)
        if path.suffix == ".json" and path.exists():
            return cls(**json.loads(path.read_text()))
        problem_id, _, version = reference.partition("/")
        version = version or "v1"
        spec_path = (root or SPEC_ROOT) / problem_id / f"{version}.json"
        if not spec_path.exists():
            raise FileNotFoundError(
                f"No eval spec at {spec_path}. Create one with `EvalSpec(problem_id={problem_id!r}).freeze(problem)`."
            )
        return cls(**json.loads(spec_path.read_text()))

    def save(self, *, root: Path | None = None) -> Path:
        """Write this spec to its canonical committed location."""
        spec_path = (root or SPEC_ROOT) / self.problem_id / f"{self.version}.json"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        return spec_path

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, problem: Problem, *, device: th.device | None = None) -> ResolvedSpec:
        """Draw this spec's test conditions and reference designs.

        Raises:
            ValueError: If the drawn data no longer matches `condition_digest`,
                meaning the upstream dataset changed and the spec must be
                re-frozen under a new version.
        """
        device = device or th.device("cpu")
        conditions_tensor, conditions, ref_designs, indices = sample_conditions(
            problem=problem, n_samples=self.n_samples, device=device, seed=self.condition_seed
        )
        digest = _digest(indices, ref_designs)
        if self.condition_digest is not None and digest != self.condition_digest:
            raise ValueError(
                f"Eval spec {self.problem_id}/{self.version} no longer reproduces its frozen conditions "
                f"(expected {self.condition_digest}, got {digest}). The dataset or sampling changed: "
                "freeze a new spec version rather than comparing across the change."
            )
        return ResolvedSpec(
            spec=self,
            conditions_tensor=conditions_tensor,
            conditions=conditions,
            ref_designs=ref_designs,
            indices=np.asarray(indices),
            condition_keys=tuple(get_scalar_condition_keys(problem, problem.dataset["test"])),
        )

    def freeze(self, problem: Problem) -> EvalSpec:
        """Return a copy with `condition_digest` and `engibench_version` filled in."""
        import engibench

        _, _, ref_designs, indices = sample_conditions(
            problem=problem, n_samples=self.n_samples, device=th.device("cpu"), seed=self.condition_seed
        )
        return EvalSpec(
            **{
                **asdict(self),
                "condition_digest": _digest(indices, ref_designs),
                "engibench_version": getattr(engibench, "__version__", None),
            }
        )


@dataclass(frozen=True)
class ResolvedSpec:
    """An `EvalSpec` with its conditions and reference designs materialized."""

    spec: EvalSpec
    conditions_tensor: th.Tensor
    conditions: Dataset
    ref_designs: npt.NDArray[Any]
    indices: npt.NDArray[Any]
    condition_keys: tuple[str, ...] = ()
    """Names of the columns in `conditions_tensor`, in order.

    A subset of `problem.conditions_keys`: solver-only and array-valued
    conditions cannot travel in a dense tensor. See
    `engiopt.transforms.get_scalar_condition_keys`.
    """

    @property
    def n_samples(self) -> int:
        """Number of test conditions in this spec."""
        return self.spec.n_samples


def _digest(indices: npt.NDArray[Any], ref_designs: npt.NDArray[Any]) -> str:
    """Stable hash of the drawn sample indices and their reference designs."""
    hasher = hashlib.sha256()
    hasher.update(np.asarray(indices).astype(np.int64).tobytes())
    hasher.update(np.asarray(ref_designs, dtype=np.float64).round(8).tobytes())
    return hasher.hexdigest()[:16]


def freeze_spec(
    problem_id: str,
    *,
    version: str = "v1",
    n_samples: int = 50,
    condition_seed: int = 1,
    metrics: tuple[str, ...] = ("mmd", "dpp", "viol", "iog", "cog", "fog"),
    sigma: float = 10.0,
    objective_weights: tuple[float, ...] | None = None,
    objective_weight_condition: str | None = None,
) -> Path:
    """Draw a problem's test conditions once and commit them as a spec.

    Run this once per problem, then never again for that version -- the point of
    a spec is that it stops moving.

    Returns:
        Path to the written spec file.
    """
    from engibench.utils.all_problems import BUILTIN_PROBLEMS

    problem = BUILTIN_PROBLEMS[problem_id]()
    problem.reset(seed=condition_seed)
    spec = EvalSpec(
        problem_id=problem_id,
        version=version,
        n_samples=n_samples,
        condition_seed=condition_seed,
        metrics=metrics,
        sigma=sigma,
        objective_weights=objective_weights,
        objective_weight_condition=objective_weight_condition,
    ).freeze(problem)
    path = spec.save()
    print(f"Froze {problem_id}/{version}: digest={spec.condition_digest} n={spec.n_samples} -> {path}")
    return path


if __name__ == "__main__":
    import tyro

    tyro.cli(freeze_spec)
