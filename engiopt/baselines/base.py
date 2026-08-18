"""The shared half of a dataset-fitted generator: dataset access and the contract.

A dataset-fitted generator carries no trained weights. What it carries instead
is the dataset split the evaluation spec already pins, which makes it exactly as
reproducible as a checkpoint and rather more portable.
`DesignBank` gives every constructed model the same view of it (designs plus
their conditions, per split, with a nearest-condition lookup), loaded once per
problem so that building eight bank members does not read the dataset eight
times.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar, TYPE_CHECKING

import numpy as np

from engiopt.core import ConditionBatch
from engiopt.core import design_shape_of
from engiopt.core import Generator
from engiopt.core import pick_device
from engiopt.transforms import condition_keys as scalar_condition_keys

if TYPE_CHECKING:
    from engibench.core import Problem
    import numpy.typing as npt
    import torch as th

    from engiopt.checkpoint_store import ResolvedCheckpoint


class NoCheckpointError(NotImplementedError):
    """Raised when a constructed generator is asked to load from a checkpoint.

    These models are fitted at load time from the pinned dataset split, so
    there is nothing to fetch. Build them with `from_problem`.
    """

    def __init__(self, algo_id: str) -> None:
        super().__init__(
            f"{algo_id!r} is fitted from the dataset, not from a checkpoint. "
            f"Build it from the problem instead: {algo_id}.from_problem(problem, problem_id=...)."
        )


@dataclass(frozen=True)
class Split:
    """One dataset split, as the arrays a constructed model needs.

    Attributes:
        designs: `(n, *design_shape)` optimal designs.
        conditions: `(n, n_conds)` scalar conditions, in `keys` order.
        keys: Condition names, matching the generator's input-tensor order.
    """

    designs: npt.NDArray[Any]
    conditions: npt.NDArray[Any]
    keys: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.designs)


class DesignBank:
    """The dataset, in the form every constructed model reads it.

    Cached per `(problem_id, split set)` so a whole bank of models shares one
    read. The nearest-condition lookup is standardized by the training split's
    per-column spread, so a condition measured in thousands does not dominate
    one measured in hundredths.
    """

    _cache: ClassVar[dict[str, DesignBank]] = {}

    def __init__(self, problem: Problem, problem_id: str) -> None:
        self.problem = problem
        self.problem_id = problem_id
        self.keys = tuple(scalar_condition_keys(problem))

    @classmethod
    def for_problem(cls, problem: Problem, problem_id: str) -> DesignBank:
        """Return the cached bank for this problem, building it on first use."""
        if problem_id not in cls._cache:
            cls._cache[problem_id] = cls(problem, problem_id)
        return cls._cache[problem_id]

    def split(self, name: str) -> Split:
        """Load one split's designs and conditions, falling back to train."""
        return self._splits[name] if name in self._splits else self._splits["train"]

    @cached_property
    def _splits(self) -> dict[str, Split]:
        dataset = self.problem.dataset
        out: dict[str, Split] = {}
        for name in dataset:
            designs = np.asarray(dataset[name]["optimal_design"], dtype=np.float32)
            conditions = np.stack(
                [np.asarray(dataset[name][key], dtype=np.float64) for key in self.keys],
                axis=1,
            )
            out[name] = Split(designs=designs, conditions=conditions, keys=self.keys)
        return out

    @cached_property
    def _scale(self) -> npt.NDArray[Any]:
        """Per-condition spread used to make the nearest-neighbour distance dimensionless."""
        scale = self.split("train").conditions.std(axis=0)
        scale[scale == 0] = 1.0
        return scale

    def nearest(self, conditions: npt.NDArray[Any], split: str = "train", k: int = 1) -> npt.NDArray[Any]:
        """Indices of the `k` closest designs in `split` for each requested condition.

        Args:
            conditions: `(n, n_conds)` requested conditions.
            split: Which split to search.
            k: Number of neighbours. At `k = 1` the trailing axis is dropped, so
                the common case still indexes as `designs[idx]`.

        Returns:
            `(n,)` indices at `k = 1`, otherwise `(n, k)`.
        """
        indices, _ = self.neighbours(conditions, split=split, k=k)
        return indices

    def neighbours(
        self, conditions: npt.NDArray[Any], split: str = "train", k: int = 1
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """The `k` closest designs in `split`, with how far away each one is.

        `nearest` throws the distances away, which is all a retrieval model
        needs. A model that *weights* its neighbours needs them, and recomputing
        the distance matrix outside this class would mean a second copy of the
        standardization -- the one thing here that must not be duplicated, since
        two different scalings would make two models' "nearest" mean different
        things.

        Args:
            conditions: `(n, n_conds)` requested conditions.
            split: Which split to search.
            k: Number of neighbours. At `k = 1` the neighbour axis is dropped
                from the indices, matching `nearest`.

        Returns:
            `(indices, distances)`, standardized distances in the same shape as
            the indices.
        """
        pool = self.split(split).conditions / self._scale
        query = np.asarray(conditions, dtype=np.float64) / self._scale
        distances = np.linalg.norm(query[:, None, :] - pool[None, :, :], axis=2)
        if k == 1:
            picked = np.asarray(distances.argmin(axis=1))
            return picked, distances[np.arange(len(picked)), picked]
        order = np.argsort(distances, axis=1)[:, :k]
        return order, np.take_along_axis(distances, order, axis=1)

    def column(self, name: str) -> int | None:
        """Position of a named condition in the tensor, or None if absent."""
        return self.keys.index(name) if name in self.keys else None

    def warm(self) -> None:
        """Materialize the splits and the distance scaling ahead of any sampling."""
        _ = self._splits
        _ = self._scale


def match_volume_fraction(
    designs: npt.NDArray[Any], targets: npt.NDArray[Any], *, iterations: int = 40
) -> npt.NDArray[Any]:
    """Rescale each design's densities so its mean hits the requested volume fraction.

    A per-design gain, found by bisection because clipping at 1 makes the mean a
    non-linear function of it. The structure is untouched -- the same material is
    in the same places -- so this changes what the feasibility check sees without
    changing what the design *is*, which is the whole trick.

    Args:
        designs: `(n, ...)` density fields.
        targets: `(n,)` requested volume fractions.
        iterations: Bisection steps; 40 is far past float precision on `[0, 1e3]`.

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


class DatasetGenerator(Generator):
    """A generator fitted from the dataset at load time rather than from a checkpoint.

    Subclasses implement `_sample` exactly as a trained adapter does. The only
    difference is where the model comes from: `from_problem` instead of
    `from_pretrained`, because there is no checkpoint.

    Class attributes:
        bank_eligible: Whether this model may sit in a workshop bank alongside
            trained checkpoints. True for published baselines like kNN; False
            for the reference instruments, which exist to calibrate what a
            metric value means and would be dishonest as contestants.
        tuning: Names of the attributes that change what this model outputs --
            a blend width, a noise scale, a coarsening factor. They do two jobs.

            They go into the model's **cache key**, because a checkpoint's
            weights change its fingerprint and a dataset-fitted model has no
            fingerprint to change. Without that, retuning a construction and
            re-scoring it silently replays the designs the old settings
            produced, and the board reports the measurement you thought you had
            just replaced.

            And they are **settable per instance**, from a bank entry or a
            keyword, so one class covers a whole severity ladder: four entries
            declaring `temperature` 0.05 / 0.08 / 0.12 / 0.15 are four rungs
            that cache separately and score independently. That is what lets a
            workshop draw two rungs and a distortion study sweep all of them
            out of the same class, the same checkpoints and the same cache.
        planted: Whether this model was built to top a column it does not
            deserve. Planted models *are* ranked in a line-up, under a name that
            implies a method, and are disclosed at the reveal -- which is a
            different thing from both a baseline and a reference instrument, and
            has to be readable off the class rather than inferred from which
            catalogue it happens to be in.
        summary: One line describing what the model does.
        reference: Citation, where the method is one from the literature.
        wins: Metric columns this model is expected to top.
        loses: Metric columns it is expected to bottom.
    """

    design_kinds: ClassVar[tuple[str, ...]] = ("2d",)
    checkpoint_files: ClassVar[tuple[str, ...]] = ()
    output_clip: tuple[Any, Any] | None = (1e-3, 1.0)

    bank_eligible: ClassVar[bool] = False
    planted: ClassVar[bool] = False
    tuning: ClassVar[tuple[str, ...]] = ()
    summary: ClassVar[str] = ""
    reference: ClassVar[str] = ""
    wins: ClassVar[tuple[str, ...]] = ()
    loses: ClassVar[tuple[str, ...]] = ()

    def __init__(self, bank: DesignBank, **kwargs: Any) -> None:
        # Tuned knobs arrive as ordinary keywords and land on the instance,
        # shadowing the class default. Consumed here rather than passed on
        # because `Generator.__init__` rejects what it does not recognise --
        # which is right for a typo and wrong for a declared severity.
        for name in self.tuning:
            if name in kwargs:
                setattr(self, name, kwargs.pop(name))
        super().__init__(**kwargs)
        self.bank = bank

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> DatasetGenerator:  # noqa: ARG003
        """Never valid for a dataset-fitted model.

        The signature has to match the contract every trained adapter
        implements -- that uniformity is what lets the evaluator score a kNN
        baseline and a diffusion model the same way -- so the arguments are
        unused here by design rather than by oversight.

        Raises:
            NoCheckpointError: Always; use `from_problem`.
        """
        raise NoCheckpointError(cls.algo_id)

    @classmethod
    def from_problem(
        cls,
        problem: Problem,
        *,
        problem_id: str,
        seed: int = 1,
        device: th.device | None = None,
        **kwargs: Any,
    ) -> DatasetGenerator:
        """Fit the model from the problem's dataset.

        The counterpart of `from_pretrained`, and deliberately the same shape:
        once built, the evaluator cannot tell the two apart.

        Args:
            problem: The problem to build against.
            problem_id: Its registry key.
            seed: Sampling seed, so a dataset-fitted model varies across seeds
                the way a trained one does.
            device: Torch device; these models compute in numpy, so this only
                affects how conditions arrive.
            **kwargs: Passed through to the subclass constructor.

        Returns:
            The constructed generator, ready to `sample`.
        """
        bank = DesignBank.for_problem(problem, problem_id)
        # Read the dataset now rather than inside the first `sample`, so the
        # `gen_seconds` column times generation and not a one-off disk read.
        bank.warm()
        return cls(
            bank=bank,
            problem=problem,
            problem_id=problem_id,
            seed=seed,
            device=device or pick_device(),
            run_config={"fitted_from_dataset": cls.algo_id},
            condition_keys=bank.keys,
            **kwargs,
        )

    @classmethod
    def settings(cls) -> dict[str, Any]:
        """The declared defaults for this model's knobs, for the cache key.

        Class-level rather than per-instance because the key has to be known
        before anything is built -- a bank member loads lazily, and a member
        whose designs are cached never constructs its model at all. Whatever a
        bank entry declares is merged over this by the caller, so the key
        describes the rung that will actually be sampled.

        Returns:
            `{attribute: value}` over `tuning`, empty when nothing is tunable.
        """
        return {name: getattr(cls, name) for name in cls.tuning}

    @classmethod
    def package_fingerprint(cls, settings: dict[str, Any] | None = None) -> str:
        """The identifier this model's published metrics are filed under.

        A trained package is addressed on the Hub by `cfg_<hyperparameter
        fingerprint>`; this is the constructed equivalent, and it has to cover
        **both** halves of what makes a construction what it is -- the
        mechanism and the knobs it was run at.

        Covering only the mechanism is a silent data-loss bug rather than a
        cosmetic one: every rung of a severity ladder shares one source, so
        `temperature` 0.05 and 0.15 would resolve to the same `cfg_<digest>/`
        path and the second run's physics would overwrite the first's. Physics
        is the expensive half -- hours per model -- and nothing downstream would
        report anything wrong.

        Args:
            settings: Effective knob values. Defaults to the class declarations,
                which is right for a model built without overrides.

        Returns:
            Eight hex characters, or `""` when the source cannot be read.
        """
        import hashlib

        mechanism = cls.mechanism_digest()
        if not mechanism:
            return ""
        knobs = sorted((settings if settings is not None else cls.settings()).items())
        return hashlib.sha256(f"{mechanism}|{knobs}".encode()).hexdigest()[:8]

    @classmethod
    def mechanism_digest(cls) -> str:
        """A short hash of this model's own source, standing in for a weight file.

        A trained model's cache is keyed by the fingerprint of its weights. A
        dataset-fitted model has no weights: **its mechanism is its source**, so
        that is what gets hashed. Without this, editing how a construction
        samples and re-scoring it serves the designs the previous version
        produced -- which cost two rounds of tuning here before it was noticed,
        both times looking like "the change had no effect on the metrics".

        Hashes this class and any construction base between it and
        `DatasetGenerator`, so a shared helper like `_shift_to_budget` counts
        too. It does **not** cover `DatasetGenerator` itself or module-level
        helpers such as `match_volume_fraction`: changing those invalidates
        nothing automatically, and the cache has to be cleared by hand.

        Returns:
            Eight hex characters, or `""` if the source cannot be read (a REPL,
            a frozen build), in which case the key simply omits it.
        """
        import hashlib
        import inspect

        sources = []
        for klass in cls.__mro__:
            if klass is DatasetGenerator:
                break
            try:
                sources.append(inspect.getsource(klass))
            except (OSError, TypeError):
                return ""
        return hashlib.sha256("".join(sources).encode()).hexdigest()[:8]

    # ------------------------------------------------------------------
    # Helpers shared by the constructions
    # ------------------------------------------------------------------

    def requested(self, conditions: ConditionBatch, n: int) -> npt.NDArray[Any]:
        """The requested conditions as an `(n, n_conds)` float array.

        Raises:
            ValueError: If a construction that needs conditions got none.
        """
        tensor = conditions.tensor
        if tensor is None:
            raise ValueError(f"{self.algo_id} needs conditions and received none.")
        values = tensor.detach().cpu().numpy().astype(np.float64)
        return values[:n]

    def rng(self) -> np.random.Generator:
        """Seeded RNG, so a dataset-fitted model is as reproducible as a trained one."""
        return np.random.default_rng(self.seed)

    @cached_property
    def flat_shape(self) -> int:
        """Number of values in one design."""
        return int(np.prod(design_shape_of(self.problem)))
