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
import subprocess
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


class ProblemDefinitionMismatchError(ValueError):
    """Raised when the installed EngiBench defines a problem differently than a spec expects.

    Distinct from a digest mismatch: the data did not change underneath us, the
    *problem* did. Nothing about the spec can be reproduced until the EngiBench
    versions agree, so this says which side moved rather than reporting two
    hashes that will not match.
    """

    def __init__(
        self,
        *,
        problem_id: str,
        version: str,
        expected_conditions: tuple[str, ...],
        found_conditions: tuple[str, ...],
        expected_dataset: str | None,
        found_dataset: str | None,
        engibench_version: str | None,
    ) -> None:
        self.expected_conditions = expected_conditions
        self.found_conditions = found_conditions
        details = [
            f"Eval spec {problem_id}/{version} was frozen against a different definition of {problem_id!r}.",
            f"  conditions expected: {list(expected_conditions)}",
            f"  conditions found:    {list(found_conditions)}",
        ]
        if expected_dataset != found_dataset:
            details.append(f"  dataset expected:    {expected_dataset}")
            details.append(f"  dataset found:       {found_dataset}")
        details.append(
            f"The spec was frozen under EngiBench {engibench_version or 'unknown'}. Install an EngiBench whose "
            f"{problem_id!r} matches, or freeze a new spec version against the one you have."
        )
        super().__init__("\n".join(details))


@dataclass(frozen=True)
class LatentInstrument:
    """The trained autoencoder a spec measures latent-space metrics with.

    Latent metrics differ from every other family in the registry: they depend
    on a fitted model as well as on the designs. The same generator scored
    against two autoencoders -- different seed, different reconstruction
    threshold -- yields two different numbers, because the active subspace those
    autoencoders find is not the same subspace. `n_active` has been observed to
    range from 3 to 100 across threshold settings on one problem.

    So a latent column is only comparable across leaderboard rows if the
    instrument is pinned exactly. That is what this records, and
    `expected_n_active` is checked at evaluation time so a substituted
    instrument fails loudly rather than quietly producing incomparable numbers.

    Args:
        algo: Generator family holding the instrument, e.g. `constrained_plvae_2d`.
        seed: Training seed of the instrument checkpoint.
        config_fingerprint: Configuration fingerprint pinning one sweep member.
            Without it, `seed_N` resolves to whatever the default config was.
        revision: HuggingFace commit the package is read at.
        expected_n_active: Active latent dimensions the instrument should report.
        recon_only_config_fingerprint: The companion autoencoder trained at the
            *same* reconstruction threshold but without the performance
            constraint. The dual-LVAE gap is the distance between what these two
            reconstruct, so it isolates the effect of the performance constraint
            alone -- which only holds if the reconstruction budget matches.
        recon_only_seed: Training seed of the companion; defaults to `seed`.
        hf_entity: HF org/user holding the checkpoint repo.
        hf_repo_prefix: Prefix of the per-model-family repo.
    """

    algo: str
    seed: int = 1
    config_fingerprint: str | None = None
    revision: str | None = None
    expected_n_active: int | None = None
    recon_only_config_fingerprint: str | None = None
    recon_only_seed: int | None = None
    hf_entity: str = "IDEALLab"
    hf_repo_prefix: str = "engiopt"

    @property
    def has_recon_only(self) -> bool:
        """Whether a companion is pinned, which is what the dual gap needs."""
        return self.recon_only_config_fingerprint is not None


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
        volume_condition: Name of the condition holding the volume-fraction
            budget a design must hit, e.g. beams2d's `volfrac`. Feasibility is
            otherwise `problem.check_constraints` alone, which is the right
            answer for problems with no volume budget (photonics2d).
        objective_weights: Fixed weights over `problem.objectives`, for
            multi-objective problems. Summing or averaging objectives with
            different units is not a valid default, so one of these two fields
            is required whenever a problem has more than one objective.
        objective_weight_condition: Name of a per-sample condition carrying the
            trade-off instead of fixing it. thermoelastic2d's `weight` splits
            the first two objectives as `(w, 1 - w)`, so a purely structural
            sample (`w = 1`) is unaffected by thermal compliance.
        condition_digest: Hash of the drawn indices, condition values, and
            reference designs. Recomputed at evaluation time and compared, so an
            upstream dataset change is caught instead of silently shifting every
            number.
        dataset_id: HuggingFace dataset the conditions were drawn from.
        dataset_revision: The dataset commit they were drawn at. A spec loads
            the dataset at exactly this revision, so a new upload to the dataset
            repo cannot change what a leaderboard row means. Leave it empty to
            follow the dataset's current main branch.
        problem_conditions: The problem's full condition list when the spec was
            frozen. Pinning the dataset is not enough on its own: which
            conditions exist, and which dataset a problem points at, are defined
            by the installed EngiBench, so a spec frozen against a different
            EngiBench draws different columns. Checked before the digest, so
            that case reports itself instead of surfacing as an opaque hash
            mismatch.
        engibench_version: EngiBench version the spec was frozen under. Recorded
            for provenance; `problem_conditions` is what actually gets checked,
            since a version string does not pin a problem definition either.
        notes: Free-text rationale for this version.
    """

    problem_id: str
    version: str = "v1"
    n_samples: int = 50
    condition_seed: int = 1
    latent_instrument: LatentInstrument | None = None
    metrics: tuple[str, ...] = ("mmd", "dpp", "viol", "iog", "cog", "fog")
    sigma: float = 10.0
    volfrac_tol: float = 0.01
    volume_condition: str | None = None
    objective_weights: tuple[float, ...] | None = None
    objective_weight_condition: str | None = None
    condition_digest: str | None = None
    dataset_id: str | None = None
    dataset_revision: str | None = None
    problem_conditions: tuple[str, ...] | None = None
    engibench_version: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        """Normalize fields that JSON cannot represent directly.

        JSON has no tuple type, so a spec read back from disk would otherwise
        carry lists and compare unequal to the spec that produced it. Nested
        dataclasses come back as plain dicts for the same reason.
        """
        object.__setattr__(self, "metrics", tuple(self.metrics))
        if self.objective_weights is not None:
            object.__setattr__(self, "objective_weights", tuple(self.objective_weights))
        if self.problem_conditions is not None:
            object.__setattr__(self, "problem_conditions", tuple(self.problem_conditions))
        if isinstance(self.latent_instrument, dict):
            object.__setattr__(self, "latent_instrument", LatentInstrument(**self.latent_instrument))

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
        self.check_problem_definition(problem)
        self.pin_dataset(problem)
        conditions_tensor, conditions, ref_designs, indices = sample_conditions(
            problem=problem, n_samples=self.n_samples, device=device, seed=self.condition_seed
        )
        digest = _digest(indices, conditions, ref_designs)
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

    def check_problem_definition(self, problem: Problem) -> None:
        """Verify the installed EngiBench defines this problem the way the spec was frozen against.

        A pinned dataset revision fixes the *data*; it does not fix which
        conditions the problem declares or which dataset it points at, both of
        which live in EngiBench. A spec frozen against a different EngiBench
        draws different condition columns and so computes a different digest --
        which, without this check, surfaces only as an unexplained hash
        mismatch.

        Raises:
            ProblemDefinitionMismatchError: If the installed problem declares
                different conditions, or points at a different dataset.
        """
        if self.problem_conditions is None:
            return
        current = tuple(problem.conditions_keys)
        current_dataset = getattr(problem, "dataset_id", None)
        dataset_moved = self.dataset_id is not None and current_dataset is not None and current_dataset != self.dataset_id
        if current != self.problem_conditions or dataset_moved:
            raise ProblemDefinitionMismatchError(
                problem_id=self.problem_id,
                version=self.version,
                expected_conditions=self.problem_conditions,
                found_conditions=current,
                expected_dataset=self.dataset_id,
                found_dataset=current_dataset,
                engibench_version=self.engibench_version,
            )

    def pin_dataset(self, problem: Problem) -> None:
        """Point `problem.dataset` at this spec's frozen dataset revision.

        `Problem.dataset` lazily loads `dataset_id` from the Hub's main branch,
        with no hook for a revision, so the spec loads the pinned revision and
        seeds the problem's cache before anything reads it.
        """
        if not self.dataset_revision:
            return
        from datasets import load_dataset

        dataset_id = self.dataset_id or getattr(problem, "dataset_id", None)
        if dataset_id is None:
            raise ValueError(f"Spec {self.problem_id}/{self.version} pins a dataset revision but names no dataset.")
        problem._dataset = load_dataset(dataset_id, revision=self.dataset_revision)  # noqa: SLF001

    def freeze(self, problem: Problem) -> EvalSpec:
        """Return a copy with the digest, dataset revision, and versions filled in."""
        dataset_id = self.dataset_id or getattr(problem, "dataset_id", None)
        revision = self.dataset_revision or _dataset_revision(dataset_id)
        frozen = EvalSpec(**{**asdict(self), "dataset_id": dataset_id, "dataset_revision": revision})
        frozen.pin_dataset(problem)
        _, conditions, ref_designs, indices = sample_conditions(
            problem=problem, n_samples=self.n_samples, device=th.device("cpu"), seed=self.condition_seed
        )
        return EvalSpec(
            **{
                **asdict(frozen),
                "condition_digest": _digest(indices, conditions, ref_designs),
                "problem_conditions": tuple(problem.conditions_keys),
                "engibench_version": _engibench_version(),
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


def _digest(indices: npt.NDArray[Any], conditions: Dataset, ref_designs: npt.NDArray[Any]) -> str:
    """Stable hash of everything a model is scored against.

    Covers the drawn indices, the condition columns *and their values*, and the
    reference designs. Hashing only the indices would let an upstream edit
    change a condition value -- and so every score -- without tripping the
    check, since row 7 is still row 7.
    """
    hasher = hashlib.sha256()
    hasher.update(np.asarray(indices).astype(np.int64).tobytes())
    for name in sorted(conditions.column_names):
        hasher.update(name.encode())
        hasher.update(_condition_bytes(conditions[name]))
    hasher.update(np.asarray(ref_designs, dtype=np.float64).round(8).tobytes())
    return hasher.hexdigest()[:16]


def _condition_bytes(values: Any) -> bytes:
    """Stable byte representation of one condition column.

    Numeric columns (including the array-valued ones, e.g. thermoelastic2d's
    boundary matrices) are rounded before hashing so that float noise below the
    tolerance the metrics themselves use cannot invalidate a spec; anything
    non-numeric falls back to its JSON form.
    """
    array = np.asarray(values)
    if array.dtype.kind in "fiub":
        return np.ascontiguousarray(array, dtype=np.float64).round(8).tobytes()
    return json.dumps(values, sort_keys=True, default=str).encode()


def _engibench_version() -> str:
    """The EngiBench that produced a spec: its version, plus a git sha from a source checkout.

    The release version alone does not identify a problem definition -- 0.2.0 on
    PyPI and 0.2.0 from `main` point `thermoelastic2d` at different datasets. A
    change to `simulate` or `optimize` moves the optimality gaps without
    touching any condition name, and only the revision records that.
    """
    import engibench

    version = getattr(engibench, "__version__", "unknown")
    source_root = Path(engibench.__file__).resolve().parent.parent
    try:
        sha = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    # Installed as a wheel rather than a checkout: the version is all there is.
    except (OSError, subprocess.SubprocessError):
        return version
    return f"{version}+{sha}" if sha else version


def _dataset_revision(dataset_id: str | None) -> str | None:
    """Current commit of a HuggingFace dataset repo, or None if it cannot be read."""
    if dataset_id is None:
        return None
    from huggingface_hub import HfApi

    return HfApi().dataset_info(dataset_id).sha


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
    notes: str = "",
) -> Path:
    """Draw a problem's test conditions once and commit them as a spec.

    Run this once per problem, then never again for that version -- the point of
    a spec is that it stops moving. The dataset revision in force is recorded,
    so later uploads to the dataset repo cannot change what the spec means.

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
        notes=notes,
    ).freeze(problem)
    path = spec.save()
    print(
        f"Froze {problem_id}/{version}: digest={spec.condition_digest} n={spec.n_samples} "
        f"dataset={spec.dataset_id}@{spec.dataset_revision} -> {path}"
    )
    return path


if __name__ == "__main__":
    import tyro

    tyro.cli(freeze_spec)
