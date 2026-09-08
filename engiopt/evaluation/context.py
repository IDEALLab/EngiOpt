"""Shared state for one model-vs-reference comparison.

The context exists so that metrics compose without recomputing. Ten metrics that
each need the flattened designs flatten once; the whole optimality-gap family
shares a single pass of the optimizer instead of running it once per metric.

Anything expensive is a `cached_property`, so it is paid for only if some
selected metric actually asks for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from functools import cached_property
from typing import Any, TYPE_CHECKING

from gymnasium import spaces
import numpy as np

from engiopt import metrics as metrics_mod

if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset
    from engibench.core import Problem
    import numpy.typing as npt


def _as_arrays(conditions: dict[str, Any] | None) -> dict[str, Any]:
    """Give array-valued conditions back their array type before constraint checking.

    A HuggingFace dataset hands back thermoelastic2d's boundary matrices as
    nested lists, and EngiBench's bound checks compare them numerically, which
    raises `TypeError: '<=' not supported between 'float' and 'list'`.
    """
    return {key: np.asarray(value) if isinstance(value, list) else value for key, value in (conditions or {}).items()}


class MultiObjectiveScalarizationError(ValueError):
    """Raised when a multi-objective problem has no declared scalarization.

    Averaging or summing objectives with different units and different intended
    priorities produces a number that looks fine and means nothing, so the
    evaluator refuses rather than guessing.
    """

    def __init__(self, problem_id: str, objective_names: list[str]) -> None:
        super().__init__(
            f"{problem_id!r} has {len(objective_names)} objectives ({', '.join(objective_names)}) but the eval "
            "spec does not say how to combine them. Set `objective_weights` for fixed weights, or "
            "`objective_weight_condition` when the trade-off is a per-sample condition (thermoelastic2d's "
            "`weight`). Summing or averaging objectives with different units is not a valid default."
        )


@dataclass
class OptimizationResults:
    """Per-sample outputs of the simulator/optimizer pass.

    Computing these is the expensive part of evaluation -- one optimizer run and
    two simulations per sample -- so every performance metric reads from one
    shared instance.
    """

    iog: list[float] = field(default_factory=list)
    """Initial optimality gap: how far each generated design starts from the reference optimum."""
    cog: list[float] = field(default_factory=list)
    """Sum of the optimality gaps over each design's re-optimization history."""
    fog: list[float] = field(default_factory=list)
    """Final optimality gap at the end of re-optimization."""


@dataclass
class EvaluationContext:
    """Everything a metric may read about one generator's output.

    Attributes:
        problem: The EngiBench problem being evaluated against.
        problem_id: Registry key for the problem, used in leaderboard rows.
        gen_designs: Generated designs, `(n_samples, *design_shape)`.
        ref_designs: Reference (dataset-optimal) designs for the same conditions.
        conditions: The conditions each design was asked to satisfy.
        sigma: Gaussian-kernel bandwidth for MMD and DPP.
        volfrac_tol: Tolerance used by the volume-fraction violation check.
        volume_condition: Name of the condition holding the volume-fraction
            budget a design must hit, when the problem has one. Declared by the
            spec rather than guessed, so a problem without a volume budget
            (photonics2d) is not silently scored against a missing column.
        sample_seconds: Wall-clock seconds the generator took to produce
            `gen_designs`, recorded as leaderboard provenance.
        objective_weights: Fixed weights over `problem.objectives`, used to
            reduce a multi-objective result to one number.
        objective_weight_condition: Name of a per-sample condition carrying the
            trade-off instead. thermoelastic2d's `weight` splits the first two
            objectives as `(w, 1 - w)`; any remaining objectives get zero.
        copy_corpus_fn: Returns the designs a model could plausibly have copied
            -- the dataset designs it may have trained on, plus the reference
            optima it is being scored against. Deferred behind a callable so
            that fetching a training split is paid for only when a memorization
            metric actually runs, and shared across every model in a sweep.
        copy_tol: RMS per-element distance below which a generated design counts
            as a copy of something in the corpus.
        resample_permuted: Re-runs the generator on a permutation of the same
            conditions, with the same latent draw. `None` when the comparison is
            unavailable -- an unconditional problem, or a context built directly
            from arrays rather than from a live generator.
    """

    problem: Problem
    problem_id: str
    gen_designs: npt.NDArray[Any]
    ref_designs: npt.NDArray[Any]
    conditions: Dataset | None = None
    sigma: float = 10.0
    volfrac_tol: float = 0.01
    volume_condition: str | None = None
    sample_seconds: float | None = None
    objective_weights: tuple[float, ...] | None = None
    objective_weight_condition: str | None = None
    copy_corpus_fn: Callable[[], npt.NDArray[Any]] | None = None
    copy_tol: float = 0.01
    resample_permuted: Callable[[npt.NDArray[Any]], npt.NDArray[Any]] | None = None

    @property
    def n_samples(self) -> int:
        """Number of generated designs under comparison."""
        return len(self.gen_designs)

    @property
    def is_dict_space(self) -> bool:
        """Whether the problem uses a `spaces.Dict` design space needing flatten/unflatten."""
        return isinstance(self.problem.design_space, spaces.Dict)

    @cached_property
    def gen_flat(self) -> npt.NDArray[Any]:
        """Generated designs flattened to `(n_samples, -1)`."""
        return np.asarray(self.gen_designs).reshape(self.n_samples, -1)

    @cached_property
    def ref_flat(self) -> npt.NDArray[Any]:
        """Reference designs flattened to `(n_samples, -1)`, unpacking dict spaces."""
        if not self.is_dict_space:
            return np.asarray(self.ref_designs).reshape(len(self.ref_designs), -1)
        flattened = [np.asarray(spaces.flatten(self.problem.design_space, design)) for design in self.ref_designs]
        return np.asarray(flattened)

    @cached_property
    def copy_corpus(self) -> npt.NDArray[Any] | None:
        """Flattened designs a generator could have memorized, or None if there are none.

        The scored reference designs are always in here, even though they are
        also `mmd`'s comparison target. They are the single most attractive
        thing to copy precisely *because* the protocol is public and names them,
        so a memorization check that omitted them would miss the easiest attack
        on this board. `copy_corpus_fn` widens the corpus to the training split
        the model was actually fitted on.

        Parts whose flattened width does not match the generated designs are
        dropped rather than raising: a mismatched corpus makes the check
        unavailable, and that is not a reason to fail an evaluation.
        """
        width = self.gen_flat.shape[1]
        parts = [part for part in (self.ref_flat,) if part.shape[1] == width]
        if self.copy_corpus_fn is not None:
            extra = np.asarray(self.copy_corpus_fn())
            if len(extra):
                flattened = extra.reshape(len(extra), -1)
                if flattened.shape[1] == width:
                    parts.append(flattened)
        return np.concatenate(parts) if parts else None

    @cached_property
    def nearest_corpus_distance(self) -> npt.NDArray[Any] | None:
        """Per-design RMS distance to the closest design in `copy_corpus`.

        Normalized by the square root of the design dimension so the number is a
        *per-element* deviation. That makes one tolerance meaningful across
        problems of different resolution, where a raw L2 norm would not be.
        """
        corpus = self.copy_corpus
        if corpus is None or not len(corpus):
            return None
        from scipy.spatial.distance import cdist

        distances = cdist(self.gen_flat, corpus, "euclidean")
        return np.asarray(distances.min(axis=1) / np.sqrt(self.gen_flat.shape[1]))

    @cached_property
    def permuted_designs(self) -> npt.NDArray[Any] | None:
        """Designs the generator produces when the conditions are shuffled between samples.

        Same latent draw, same model, different brief. A model that reads its
        conditions produces something different; a model that ignores them
        produces the identical batch in a different order at best, and an
        identical batch outright at worst. Comparing the two is what turns "this
        model declares itself conditional" into a measurement.
        """
        if self.resample_permuted is None or self.n_samples < 2:  # noqa: PLR2004 - a permutation needs two rows
            return None
        # Derangement by rotation: every sample is scored against a condition
        # that belongs to some *other* sample, with no fixed points to dilute
        # the signal, and no RNG to make the metric irreproducible.
        permutation = np.roll(np.arange(self.n_samples), 1)
        permuted = self.resample_permuted(permutation)
        return None if permuted is None else np.asarray(permuted).reshape(self.n_samples, -1)

    def condition_at(self, index: int) -> dict[str, Any] | None:
        """Conditions for sample `index`, or None when the problem is unconditional."""
        return self.conditions[index] if self.conditions is not None else None

    def design_for_solver(self, index: int) -> Any:
        """Generated design `index` in the shape the simulator and optimizer expect."""
        if self.is_dict_space:
            return spaces.unflatten(self.problem.design_space, self.gen_designs[index])
        return self.gen_designs[index]

    @property
    def objective_names(self) -> list[str]:
        """Names of the problem's objectives, in the order the simulator returns them."""
        return [name for name, _direction in self.problem.objectives]

    @cached_property
    def objective_signs(self) -> npt.NDArray[Any]:
        """`+1` for objectives to minimize, `-1` for objectives to maximize.

        A raw `objective - baseline` difference only means "worse than the
        reference" when the objective is being minimized. photonics2d maximizes
        `total_overlap`, so there a *positive* difference means the design is
        better; flipping its sign keeps every gap pointing the same way and lets
        `iog`, `cog`, and `fog` stay honestly registered as lower-is-better.
        """
        signs = []
        for _name, direction in self.problem.objectives:
            label = getattr(direction, "name", str(direction))
            signs.append(-1.0 if "MAXIMIZE" in label.upper() else 1.0)
        return np.asarray(signs, dtype=float)

    def weights_at(self, index: int) -> npt.NDArray[Any] | None:
        """Weights used to scalarize sample `index`'s objective vector.

        Returns None for single-objective problems, where no choice is needed.

        Raises:
            MultiObjectiveScalarizationError: If the problem has several
                objectives and the spec declares no way to combine them.
        """
        names = self.objective_names
        if len(names) <= 1:
            return None

        if self.objective_weight_condition is not None:
            conditions = self.condition_at(index) or {}
            trade_off = float(conditions[self.objective_weight_condition])
            # The declared convention: w on the first objective, (1 - w) on the
            # second, nothing on the rest. At w = 1 the second objective cannot
            # influence the gap at all, which is the whole point.
            weights = np.zeros(len(names))
            weights[0] = trade_off
            weights[1] = 1.0 - trade_off
            return weights

        if self.objective_weights is not None:
            if len(self.objective_weights) != len(names):
                msg = (
                    f"{self.problem_id!r} has {len(names)} objectives but the spec declares "
                    f"{len(self.objective_weights)} weights."
                )
                raise ValueError(msg)
            return np.asarray(self.objective_weights, dtype=float)

        raise MultiObjectiveScalarizationError(self.problem_id, names)

    def scalarize_gap(self, gap: Any, index: int) -> float:
        """Reduce one sample's raw `objective - baseline` vector to a single number.

        Two corrections are applied, in this order:

        1. Objective direction, so a positive result always means "worse than
           the reference optimum" even for maximized objectives.
        2. The declared weights, so objectives the sample does not care about
           cannot influence its score.

        Both are linear, so applying them to the gap is equivalent to applying
        them to the objectives before differencing.
        """
        array = np.atleast_1d(np.asarray(gap, dtype=float))
        signed = array * self.objective_signs[: array.size]
        if array.size == 1:
            return float(signed[0])
        weights = self.weights_at(index)
        if weights is None:  # more values than objectives should not happen
            raise MultiObjectiveScalarizationError(self.problem_id, self.objective_names)
        return float(np.dot(weights[: array.size], signed))

    @cached_property
    def optimization(self) -> OptimizationResults:
        """Run the optimizer and simulator over every sample (expensive).

        All three optimality gaps are measured against the same baseline -- the
        simulated objective of the reference (dataset-optimal) design for the
        same conditions:

        - `iog` is the gap *before* any re-optimization: how good the generated
          design already is.
        - `cog` accumulates the gap over the whole re-optimization trajectory.
        - `fog` is the gap once re-optimization has finished.

        Every gap is passed through `scalarize_gap`, which flips maximized
        objectives and applies the spec's declared weights. All three metrics
        are therefore lower-is-better on every problem, and an objective a
        sample does not care about cannot influence its score.
        """
        results = OptimizationResults()
        for i in range(self.n_samples):
            conditions = self.condition_at(i)
            design = self.design_for_solver(i)

            self.problem.reset()
            _, opt_history = self.problem.optimize(design, config=conditions)
            self.problem.reset()
            reference_optimum = self.problem.simulate(self.ref_designs[i], config=conditions)
            gaps = metrics_mod.optimality_gap(opt_history, reference_optimum)
            self.problem.reset()
            generated_objective = self.problem.simulate(design, config=conditions)

            results.iog.append(self.scalarize_gap(np.asarray(generated_objective) - np.asarray(reference_optimum), i))
            results.cog.append(sum(self.scalarize_gap(step_gap, i) for step_gap in gaps))
            results.fog.append(self.scalarize_gap(gaps[-1], i))
        return results

    @cached_property
    def feasibility(self) -> list[bool]:
        """Whether each generated design is infeasible, judged before any solver runs.

        Deliberately independent of `optimization`. Feasibility is a property of
        the design as generated, so it must survive the optimizer refusing to
        start from an invalid one -- exactly the case where the answer matters
        most. Keeping it separate also means asking for `viol` alone costs a
        constraint check rather than a full optimization pass.
        """
        return [self.is_infeasible(self.design_for_solver(i), self.condition_at(i)) for i in range(self.n_samples)]

    def is_infeasible(self, design: Any, conditions: dict[str, Any] | None) -> bool:
        """Whether one generated design fails the problem's declared feasibility.

        Two sources, both of which the problem itself defines:

        1. `problem.check_constraints`, EngiBench's own contract -- the design
           must lie in the design space and satisfy every declared constraint.
           This is what a new problem gets for free.
        2. The volume-fraction budget named by the spec's `volume_condition`,
           when the problem has one. Missing that target is a design failing to
           honor its brief rather than an invalid design, and no EngiBench
           constraint covers it.

        A problem with neither -- photonics2d has no volume budget -- is scored
        on (1) alone, rather than reporting NaN.
        """
        violations = self.problem.check_constraints(self._as_space_dtype(design), _as_arrays(conditions))
        if getattr(violations, "violations", None):
            return True
        if self.volume_condition is not None and conditions is not None:
            target = conditions.get(self.volume_condition)
            # An explicit None test, so a legitimate target of 0.0 still counts.
            if target is not None:
                return bool(np.abs(np.mean(design) - float(target)) >= self.volfrac_tol)
        return False

    def _as_space_dtype(self, design: Any) -> Any:
        """Cast a design to the design space's own dtype before the membership check.

        `Box.contains` requires a safely castable dtype, so a float64 design is
        rejected by a float32 space no matter what its values are -- which would
        mark thermoelastic2d's own dataset-optimal designs infeasible. The cast
        makes the check about the values, which is what it is meant to test.
        """
        space = self.problem.design_space
        dtype = getattr(space, "dtype", None)
        if self.is_dict_space or dtype is None:
            return design
        return np.asarray(design).astype(dtype, copy=False)
