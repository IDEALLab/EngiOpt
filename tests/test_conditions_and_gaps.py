"""Tests for condition selection and the optimality-gap definitions.

Both were real defects: condition sampling could not handle two of the four
benchmark problems, and `iog` reported a raw objective rather than a gap.
"""

from __future__ import annotations

from typing import Any

from gymnasium import spaces
import numpy as np
import pytest

from engiopt.evaluation.context import EvaluationContext
from engiopt.evaluation.context import MultiObjectiveScalarizationError
from engiopt.transforms import get_image_condition_keys
from engiopt.transforms import get_scalar_condition_keys
from tests.conftest import FakeViolations


class _Problem:
    """Just enough of `Problem` for the condition-key helpers."""

    def __init__(self, conditions_keys: list[str]):
        self.conditions_keys = conditions_keys


# ----------------------------------------------------------------------
# Condition selection
# ----------------------------------------------------------------------


def test_scalar_keys_drop_conditions_absent_from_the_dataset(mixed_condition_dataset: Any) -> None:
    """photonics2d declares solver settings that are not dataset columns."""
    problem = _Problem(["volfrac", "num_elems_x", "num_optimization_steps"])
    assert get_scalar_condition_keys(problem, mixed_condition_dataset) == ["volfrac"]


def test_scalar_keys_drop_array_valued_conditions(mixed_condition_dataset: Any) -> None:
    """thermoelastic2d's boundary conditions are matrices; they cannot be stacked."""
    problem = _Problem(["volfrac", "fixed_elements"])
    assert get_scalar_condition_keys(problem, mixed_condition_dataset) == ["volfrac"]


def test_scalar_keys_preserve_conditions_key_order(mixed_condition_dataset: Any) -> None:
    """Column order defines the tensor layout, so it must follow the contract."""
    problem = _Problem(["rmin", "volfrac"])
    assert get_scalar_condition_keys(problem, mixed_condition_dataset) == ["rmin", "volfrac"]


def test_scalar_keys_can_drop_constant_columns(mixed_condition_dataset: Any) -> None:
    """A constant condition carries no signal for a conditional model."""
    problem = _Problem(["volfrac", "rmin"])
    assert get_scalar_condition_keys(problem, mixed_condition_dataset, drop_constants=True) == ["volfrac"]


def test_image_keys_are_the_complement(mixed_condition_dataset: Any) -> None:
    """Array conditions still reach the simulator, so they must be identifiable."""
    problem = _Problem(["volfrac", "rmin", "fixed_elements"])
    scalars = get_scalar_condition_keys(problem, mixed_condition_dataset)
    images = get_image_condition_keys(problem, mixed_condition_dataset)
    assert images == ["fixed_elements"]
    assert not set(scalars) & set(images)


# ----------------------------------------------------------------------
# Optimality gaps
# ----------------------------------------------------------------------


def _context(problem: Any, gen: np.ndarray) -> EvaluationContext:
    return EvaluationContext(problem=problem, problem_id="fake", gen_designs=gen, ref_designs=gen.copy())


def test_iog_is_zero_when_the_generated_design_is_the_reference(fake_problem: Any) -> None:
    """The defining property: no gap between a design and itself.

    Before the fix `iog` stored the raw simulated objective, so this returned the
    design's objective (~0.5 here) instead of 0.
    """
    designs = np.full((3, *fake_problem.design_space.shape), 0.5)
    assert np.allclose(_context(fake_problem, designs).optimization.iog, 0.0)


def test_iog_grows_as_the_design_gets_worse(fake_problem: Any) -> None:
    """A worse design must show a larger initial gap."""
    shape = fake_problem.design_space.shape
    reference = np.full((2, *shape), 0.2)
    near = EvaluationContext(
        problem=fake_problem, problem_id="fake", gen_designs=np.full((2, *shape), 0.3), ref_designs=reference
    )
    far = EvaluationContext(
        problem=fake_problem, problem_id="fake", gen_designs=np.full((2, *shape), 0.9), ref_designs=reference
    )
    assert np.mean(far.optimization.iog) > np.mean(near.optimization.iog) > 0


def test_all_three_gaps_share_one_baseline(fake_problem: Any) -> None:
    """iog, cog, and fog must all be measured against the reference optimum.

    With the fake solver the objective decays 1.0 -> 0.5 -> 0.25 of its start, so
    for a design at 0.8 against a reference at 0.4 the gaps are exactly known.
    """
    shape = fake_problem.design_space.shape
    ctx = EvaluationContext(
        problem=fake_problem,
        problem_id="fake",
        gen_designs=np.full((1, *shape), 0.8),
        ref_designs=np.full((1, *shape), 0.4),
    )
    results = ctx.optimization
    assert results.iog[0] == pytest.approx(0.8 - 0.4)
    assert results.cog[0] == pytest.approx((0.8 - 0.4) + (0.4 - 0.4) + (0.2 - 0.4))
    assert results.fog[0] == pytest.approx(0.2 - 0.4)


def test_optimization_is_computed_once_and_cached(fake_problem: Any) -> None:
    """Every performance metric shares one solver pass; that is the point of the context."""
    designs = np.full((2, *fake_problem.design_space.shape), 0.5)
    ctx = _context(fake_problem, designs)
    ctx.optimization
    ctx.optimization
    assert fake_problem.optimize_calls == 2  # one per sample, not per access


# ----------------------------------------------------------------------
# Multi-objective scalarization
# ----------------------------------------------------------------------


def _multi_context(**kwargs: Any) -> EvaluationContext:
    """A 3-objective context whose objective vector is `mean(design)` repeated."""
    from tests.conftest import FakeDataset
    from tests.conftest import FakeProblem

    problem = FakeProblem(n_objectives=3, conditions_keys=("weight",))
    designs = np.full((1, *problem.design_space.shape), 0.8)
    return EvaluationContext(
        problem=problem,
        problem_id="thermo_like",
        gen_designs=designs,
        ref_designs=np.full_like(designs, 0.5),
        conditions=FakeDataset({"weight": [kwargs.pop("weight", 0.5)]}),
        **kwargs,
    )


def test_multi_objective_without_a_declared_weighting_is_refused() -> None:
    """Silently averaging objectives with different units produces a meaningless number."""
    with pytest.raises(MultiObjectiveScalarizationError, match="does not say how to combine"):
        _multi_context().optimization


def test_a_purely_structural_sample_ignores_the_thermal_objective() -> None:
    """With `weight = 1` the second objective must not influence the gap at all.

    This is the property that makes the weighted scalarization correct and both
    summing and averaging wrong: at w=1 a change in thermal compliance has to
    leave every gap untouched.
    """
    ctx = _multi_context(weight=1.0, objective_weight_condition="weight")
    weights = ctx.weights_at(0)
    assert weights is not None
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.0)

    baseline = ctx.scalarize_gap(np.array([0.3, 7.0, 99.0]), 0)
    perturbed = ctx.scalarize_gap(np.array([0.3, 1000.0, -5.0]), 0)
    assert baseline == pytest.approx(perturbed)
    assert baseline == pytest.approx(0.3)


def test_a_purely_thermal_sample_ignores_the_structural_objective() -> None:
    """The mirror case: at `weight = 0` only the second objective counts."""
    ctx = _multi_context(weight=0.0, objective_weight_condition="weight")
    assert ctx.scalarize_gap(np.array([500.0, 0.7, 3.0]), 0) == pytest.approx(0.7)


def test_weighted_gaps_use_the_per_sample_trade_off(fake_problem: Any) -> None:
    """Each gap is scalarized with that sample's own weight, not a global one."""
    ctx = _multi_context(weight=0.25, objective_weight_condition="weight")
    results = ctx.optimization
    # Objective vector is mean(design) repeated, so every component's gap is 0.3.
    # Weights (0.25, 0.75, 0.0) sum to 1, leaving the gap at 0.3.
    assert results.iog[0] == pytest.approx(0.8 - 0.5)


def test_fixed_objective_weights_are_honoured() -> None:
    """A spec may declare constant weights instead of a per-sample condition."""
    ctx = _multi_context(objective_weights=(0.5, 0.5, 0.0))
    assert ctx.scalarize_gap(np.array([2.0, 4.0, 1000.0]), 0) == pytest.approx(3.0)


def test_mismatched_weight_count_is_rejected() -> None:
    """A weight vector that does not line up with the objectives is a spec bug."""
    ctx = _multi_context(objective_weights=(1.0,))
    with pytest.raises(ValueError, match="3 objectives but the spec declares 1 weights"):
        ctx.weights_at(0)


def test_single_objective_problems_need_no_weighting(fake_problem: Any) -> None:
    """The common case stays configuration-free."""
    ctx = EvaluationContext(
        problem=fake_problem,
        problem_id="fake",
        gen_designs=np.full((1, *fake_problem.design_space.shape), 0.5),
        ref_designs=np.full((1, *fake_problem.design_space.shape), 0.5),
    )
    assert ctx.weights_at(0) is None
    assert ctx.scalarize_gap(np.array([2.5]), 0) == pytest.approx(2.5)


# ----------------------------------------------------------------------
# Objective direction
# ----------------------------------------------------------------------


def _directed_context(direction: str, gen: float, ref: float) -> EvaluationContext:
    from tests.conftest import FakeProblem

    problem = FakeProblem(n_objectives=1, directions=(direction,))
    return EvaluationContext(
        problem=problem,
        problem_id="fake",
        gen_designs=np.full((1, *problem.design_space.shape), gen),
        ref_designs=np.full((1, *problem.design_space.shape), ref),
    )


def test_minimized_objective_keeps_its_sign() -> None:
    """Worse (higher) than the reference must give a positive gap."""
    assert _directed_context("MINIMIZE", gen=0.8, ref=0.5).optimization.iog[0] == pytest.approx(0.3)


def test_maximized_objective_flips_sign() -> None:
    """photonics2d maximizes total_overlap, so a *higher* value is better.

    Without the flip its gap would be +0.3 for a design that beats the
    reference, and every lower-is-better metric would rank it last.
    """
    assert _directed_context("MAXIMIZE", gen=0.8, ref=0.5).optimization.iog[0] == pytest.approx(-0.3)


def test_worse_designs_score_higher_under_both_directions() -> None:
    """The invariant the leaderboard depends on: bigger gap == worse design."""
    minimize_worse = _directed_context("MINIMIZE", gen=0.9, ref=0.5).optimization.iog[0]
    minimize_better = _directed_context("MINIMIZE", gen=0.6, ref=0.5).optimization.iog[0]
    maximize_worse = _directed_context("MAXIMIZE", gen=0.1, ref=0.5).optimization.iog[0]
    maximize_better = _directed_context("MAXIMIZE", gen=0.4, ref=0.5).optimization.iog[0]
    assert minimize_worse > minimize_better
    assert maximize_worse > maximize_better


def test_signs_are_per_objective_for_mixed_directions() -> None:
    """A problem may mix directions; each objective flips independently."""
    from tests.conftest import FakeDataset
    from tests.conftest import FakeProblem

    problem = FakeProblem(n_objectives=2, conditions_keys=("weight",), directions=("MINIMIZE", "MAXIMIZE"))
    designs = np.zeros((1, *problem.design_space.shape))
    ctx = EvaluationContext(
        problem=problem,
        problem_id="mixed",
        gen_designs=designs,
        ref_designs=designs,
        conditions=FakeDataset({"weight": [0.5]}),
        objective_weight_condition="weight",
    )
    assert list(ctx.objective_signs) == [1.0, -1.0]
    # Both objectives beat the reference by 1.0, so both should read as -0.5.
    assert ctx.scalarize_gap(np.array([-1.0, 1.0]), 0) == pytest.approx(-1.0)


# ----------------------------------------------------------------------
# Feasibility
# ----------------------------------------------------------------------


def _feasibility_context(problem: Any, design_value: float, **kwargs: Any) -> EvaluationContext:
    designs = np.full((1, *problem.design_space.shape), design_value)
    return EvaluationContext(
        problem=problem,
        problem_id="fake",
        gen_designs=designs,
        ref_designs=designs.copy(),
        **kwargs,
    )


def test_feasibility_falls_back_to_the_problems_own_constraints(fake_problem: Any) -> None:
    """Every problem has `check_constraints`, so `viol` is defined even without a volume budget.

    photonics2d has no volume-fraction condition; before this it returned NaN.
    """
    ctx = _feasibility_context(fake_problem, 0.5)
    assert ctx.is_infeasible(np.full(fake_problem.design_space.shape, 0.5), {"lambda1": 1.5}) is False

    fake_problem.infeasible = True
    assert ctx.is_infeasible(np.full(fake_problem.design_space.shape, 0.5), {"lambda1": 1.5}) is True


def test_volume_budget_is_checked_when_the_spec_names_one(fake_problem: Any) -> None:
    """Missing the declared volume target is a violation of the design's brief."""
    ctx = _feasibility_context(fake_problem, 0.5, volume_condition="volfrac", volfrac_tol=0.01)
    design = np.full(fake_problem.design_space.shape, 0.5)
    assert ctx.is_infeasible(design, {"volfrac": 0.5}) is False
    assert ctx.is_infeasible(design, {"volfrac": 0.9}) is True


def test_a_zero_volume_target_is_still_a_target(fake_problem: Any) -> None:
    """`conditions.get(a) or conditions.get(b)` silently skipped a legitimate 0.0."""
    ctx = _feasibility_context(fake_problem, 0.5, volume_condition="volfrac", volfrac_tol=0.01)
    assert ctx.is_infeasible(np.full(fake_problem.design_space.shape, 0.5), {"volfrac": 0.0}) is True


def test_an_unnamed_volume_condition_is_not_guessed(fake_problem: Any) -> None:
    """Without a declared `volume_condition`, only the problem's constraints apply."""
    ctx = _feasibility_context(fake_problem, 0.5)
    assert ctx.is_infeasible(np.full(fake_problem.design_space.shape, 0.5), {"volfrac": 0.9}) is False


def test_a_design_is_judged_on_its_values_not_its_dtype(fake_problem: Any) -> None:
    """`Box.contains` rejects an uncastable dtype outright.

    thermoelastic2d's space is float32 while its designs are float64, so without
    the cast every design -- including the dataset's own optima -- is infeasible.
    """
    fake_problem.design_space = spaces.Box(low=0.0, high=1.0, shape=(8, 10), dtype=np.float32)
    ctx = _feasibility_context(fake_problem, 0.5)
    assert ctx.is_infeasible(np.full(fake_problem.design_space.shape, 0.5, dtype=np.float64), {}) is False
    assert ctx.is_infeasible(np.full(fake_problem.design_space.shape, 5.0, dtype=np.float64), {}) is True


def test_array_valued_conditions_survive_the_constraint_check(fake_problem: Any) -> None:
    """A HuggingFace dataset hands matrices back as nested lists, which bound checks cannot compare."""
    seen: dict[str, Any] = {}

    def record(design: Any, config: dict[str, Any]) -> Any:
        seen.update(config)
        return FakeViolations(violations=[])

    fake_problem.check_constraints = record
    ctx = _feasibility_context(fake_problem, 0.5)
    ctx.is_infeasible(np.full(fake_problem.design_space.shape, 0.5), {"fixed_elements": [[0, 1], [1, 0]], "rmin": 2.0})

    assert isinstance(seen["fixed_elements"], np.ndarray)
    assert seen["rmin"] == 2.0
