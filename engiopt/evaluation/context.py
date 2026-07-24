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
    from datasets import Dataset
    from engibench.core import Problem
    import numpy.typing as npt


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
    viol: list[bool] = field(default_factory=list)
    """Whether each design missed its volume-fraction target beyond tolerance."""


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
        sample_seconds: Wall-clock seconds the generator took to produce
            `gen_designs`, recorded as leaderboard provenance.
        objective_weights: Fixed weights over `problem.objectives`, used to
            reduce a multi-objective result to one number.
        objective_weight_condition: Name of a per-sample condition carrying the
            trade-off instead. thermoelastic2d's `weight` splits the first two
            objectives as `(w, 1 - w)`; any remaining objectives get zero.
    """

    problem: Problem
    problem_id: str
    gen_designs: npt.NDArray[Any]
    ref_designs: npt.NDArray[Any]
    conditions: Dataset | None = None
    sigma: float = 10.0
    volfrac_tol: float = 0.01
    sample_seconds: float | None = None
    objective_weights: tuple[float, ...] | None = None
    objective_weight_condition: str | None = None

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

            if conditions:
                target_vol = conditions.get("volfrac") or conditions.get("volume")
                if target_vol is not None:
                    results.viol.append(bool(np.abs(np.mean(design) - target_vol) >= self.volfrac_tol))
        return results
