"""The ported metrics: each one on the toy structure every EngiBench dataset shares."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from engiopt.evaluation.board import Board
from engiopt.evaluation.board import register_space
from engiopt.evaluation.board import SPACES
from engiopt.evaluation.context import EvaluationContext
from engiopt.evaluation.context import OptimizationResults
from engiopt.evaluation.registry import METRICS
from engiopt.evaluation.spec import EvalSpec

# Importing the metrics package registers the built-ins.
import engiopt.evaluation.metrics  # noqa: F401  # isort: skip


def _ctx(problem: Any, gen: np.ndarray, ref: np.ndarray, **kwargs: Any) -> EvaluationContext:
    return EvaluationContext(problem=problem, problem_id="fake", gen_designs=gen, ref_designs=ref, **kwargs)


@pytest.fixture
def sets(fake_problem: Any) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    shape = fake_problem.design_space.shape
    return rng.random((12, *shape)), rng.random((12, *shape))


# ---------------------------------------------------------------- diversity


def test_vendi_counts_distinct_designs(fake_problem: Any, sets: Any) -> None:
    gen, _ = sets
    collapsed = np.repeat(gen[:1], 12, axis=0)
    assert METRICS["vendi"].fn(_ctx(fake_problem, collapsed, gen)) == pytest.approx(1.0)
    spread = METRICS["vendi"].fn(_ctx(fake_problem, gen, gen, sigma=0.05))
    assert spread == pytest.approx(12.0, rel=0.05), "mutually dissimilar designs count as n"


def test_dpp_is_on_a_unit_scale_and_prefers_spread(fake_problem: Any, sets: Any) -> None:
    gen, _ = sets
    collapsed = np.repeat(gen[:1], 12, axis=0) + np.random.default_rng(1).normal(0, 1e-3, gen.shape)
    low = METRICS["dpp"].fn(_ctx(fake_problem, collapsed, gen, sigma=1.0))
    high = METRICS["dpp"].fn(_ctx(fake_problem, gen, gen, sigma=1.0))
    assert 0.0 <= low < high <= 1.0


# ------------------------------------------------------------- distribution


def test_coverage_is_one_for_the_reference_itself_and_zero_far_away(fake_problem: Any, sets: Any) -> None:
    _, ref = sets
    assert METRICS["coverage"].fn(_ctx(fake_problem, ref.copy(), ref)) == pytest.approx(1.0)
    assert METRICS["coverage"].fn(_ctx(fake_problem, ref + 50.0, ref)) == pytest.approx(0.0)


# --------------------------------------------------------------- conditions


def test_per_condition_distance_sees_what_mmd_cannot(fake_problem: Any, sets: Any) -> None:
    """The correct designs in the wrong order: a perfect set, every one for the wrong conditions."""
    _, ref = sets
    permuted = ref[np.random.default_rng(2).permutation(len(ref))]
    assert METRICS["mmd"].fn(_ctx(fake_problem, permuted, ref)) == pytest.approx(0.0, abs=1e-12)
    assert METRICS["per_condition_distance"].fn(_ctx(fake_problem, permuted, ref)) > 0.1
    assert METRICS["per_condition_distance"].fn(_ctx(fake_problem, ref.copy(), ref)) == pytest.approx(0.0)


def test_volume_error_reads_the_material_fraction_off_the_design(fake_problem: Any) -> None:
    shape = fake_problem.design_space.shape
    gen = np.stack([np.full(shape, 0.3), np.full(shape, 0.5)])
    ctx = _ctx(fake_problem, gen, gen, conditions={"volfrac": np.array([0.3, 0.4])}, volume_condition="volfrac")
    assert METRICS["volume_error"].fn(ctx) == pytest.approx(0.05)
    assert np.isnan(METRICS["volume_error"].fn(_ctx(fake_problem, gen, gen)))


# ------------------------------------------------------------- memorization


def test_train_distance_reads_zero_for_copies_and_one_for_data_like_designs(fake_problem: Any, sets: Any) -> None:
    """The ratio's scale comes from the reference designs, so a real held-out design scores about one."""
    gen, ref = sets
    train = np.random.default_rng(3).random((20, *fake_problem.design_space.shape))
    copies = METRICS["train_distance"].fn(_ctx(fake_problem, train[:12], ref, train_designs_fn=lambda: train))
    assert copies["train_distance"] == pytest.approx(0.0)
    assert copies["train_distance_ratio"] == pytest.approx(0.0)
    fresh = METRICS["train_distance"].fn(_ctx(fake_problem, gen, ref, train_designs_fn=lambda: train))
    assert fresh["train_distance"] > 0.1
    assert fresh["train_distance_ratio"] == pytest.approx(1.0, abs=0.25), "as far from training as real unseen designs"
    without_split = METRICS["train_distance"].fn(_ctx(fake_problem, gen, ref))
    assert np.isnan(without_split["train_distance"])
    assert np.isnan(without_split["train_distance_ratio"])


def test_the_bandwidth_defaults_to_the_training_designs_median(fake_problem: Any, sets: Any) -> None:
    """No number anywhere: the kernel scale comes from the training split, or the reference set without one."""
    from engiopt import metrics as metrics_mod

    gen, ref = sets
    train = np.random.default_rng(5).random((30, *fake_problem.design_space.shape))
    with_train = _ctx(fake_problem, gen, ref, train_designs_fn=lambda: train)
    assert with_train.kernel_sigma == pytest.approx(metrics_mod.compute_median_sigma(train.reshape(30, -1)))
    without = _ctx(fake_problem, gen, ref)
    assert without.kernel_sigma == pytest.approx(metrics_mod.compute_median_sigma(ref.reshape(len(ref), -1)))
    assert _ctx(fake_problem, gen, ref, sigma=3.0).kernel_sigma == 3.0, "a pinned value wins"


def test_the_v2_specs_pin_no_bandwidth() -> None:
    assert EvalSpec.load("beams2d/v2").sigma is None


# -------------------------------------------------------------- performance


@pytest.fixture
def trajectories(fake_problem: Any, sets: Any) -> EvaluationContext:
    """Two re-optimization paths: one reaches the reference optimum, one never does."""
    gen, ref = sets
    ctx = _ctx(fake_problem, gen[:2], ref[:2])
    reaches = np.array([5.0, 4.0, 3.0, 0.0, 0.0])
    never = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    # Reference objective 10 for both, so "near optimal" means a gap of 0.5 or less.
    ctx.__dict__["optimization"] = OptimizationResults(trajectories=[reaches, never], reference_objectives=[10.0, 10.0])
    return ctx


def test_calls_to_near_optimum_is_set_by_the_optimum_not_the_start(trajectories: EvaluationContext) -> None:
    # reaches: last outside the 0.5 band at call 3; never: still outside at its last call, so budget + 1 = 6.
    assert METRICS["calls_to_near_optimum"].fn(trajectories) == pytest.approx((3 + 6) / 2)


def test_an_absurd_start_is_not_credited_with_settling(fake_problem: Any, sets: Any) -> None:
    """A gap of 1e9 that drops to 1e3 in one call has not arrived; the old relative band said it had."""
    gen, ref = sets
    ctx = _ctx(fake_problem, gen[:1], ref[:1])
    path = np.array([1e9, 1e3, 1e2, 10.0, 3.0])
    ctx.__dict__["optimization"] = OptimizationResults(trajectories=[path], reference_objectives=[10.0])
    assert METRICS["calls_to_near_optimum"].fn(ctx) == pytest.approx(6.0), "never within 0.5 of the optimum"


def test_gap_after_calls_carries_a_short_path_forward(trajectories: EvaluationContext) -> None:
    gaps = METRICS["gap_after_calls"].fn(trajectories)
    assert gaps["gap_after_1_calls"] == pytest.approx(5.0)
    assert gaps["gap_after_2_calls"] == pytest.approx(4.0)
    assert gaps["gap_after_5_calls"] == pytest.approx(0.5)
    assert gaps["gap_after_10_calls"] == pytest.approx(0.5), "a five-call path is done at call ten"


def test_reaches_reference_rate_is_a_rate_not_a_censored_count(trajectories: EvaluationContext) -> None:
    assert METRICS["reaches_reference_rate"].fn(trajectories) == pytest.approx(0.5)


def test_trajectory_metrics_respect_the_aggregation_policy(trajectories: EvaluationContext) -> None:
    trajectories.aggregation = "median"
    assert METRICS["calls_to_near_optimum"].fn(trajectories) == pytest.approx(4.5), "median of two equals their mean"


# --------------------------------------------------------------------- cost


def test_cost_metrics_are_nan_without_a_live_model(fake_problem: Any, sets: Any) -> None:
    gen, ref = sets
    ctx = _ctx(fake_problem, gen, ref)
    assert all(np.isnan(METRICS[name].fn(ctx)) for name in ("generation_seconds", "n_parameters", "train_minutes"))
    timed = _ctx(fake_problem, gen, ref, sample_seconds=1.5, model_params=1000, train_minutes=30.0)
    assert METRICS["generation_seconds"].fn(timed) == 1.5
    assert METRICS["n_parameters"].fn(timed) == 1000
    assert METRICS["train_minutes"].fn(timed) == 30.0


# -------------------------------------------------------------------- board


def test_a_space_is_a_registered_projection(fake_problem: Any, sets: Any) -> None:
    gen, ref = sets
    register_space("halves", lambda reference, width: lambda designs: designs.reshape(len(designs), -1)[:, ::2])
    frame = Board(fake_problem, reference=ref).evaluate({"m": gen}, space="halves")
    assert "mmd@halves" in frame.columns
    assert "viol@halves" not in frame.columns
    del SPACES["halves"]
    with pytest.raises(KeyError, match="Unknown space"):
        Board(fake_problem, reference=ref).evaluate({"m": gen}, space="halves")


def test_the_reference_row_leaves_per_condition_metrics_blank(fake_problem: Any, sets: Any) -> None:
    gen, ref = sets
    frame = Board(fake_problem, reference=ref).evaluate({"m": gen})
    assert np.isnan(frame.loc["reference (split-half)", "per_condition_distance"])
    assert not np.isnan(frame.loc["m", "per_condition_distance"])


def test_from_evaluator_scores_models_and_saved_designs_under_one_spec() -> None:
    class StubContext:
        def __init__(self, designs: Any) -> None:
            self.gen_designs = designs

    class StubEvaluator:
        problem = None
        registry = METRICS
        spec = type("Spec", (), {"sigma": 1.0})()
        resolved = type("Resolved", (), {"ref_designs": np.zeros((2, 3)), "conditions": None})()

        def context_for(self, generator: Any) -> StubContext:
            return StubContext(generator)

        def context_for_designs(self, designs: Any) -> StubContext:
            return StubContext(designs)

        def score_context(self, ctx: Any, *, only: Any, include_expensive: bool) -> dict[str, float]:
            return {"mmd": float(np.mean(ctx.gen_designs)), "iog": 1.0 if include_expensive else float("nan")}

    board = Board.from_evaluator(
        StubEvaluator(), {"a": np.full(3, 0.2), "b": np.full(3, 0.1)}, designs={"probe": np.full(3, 0.3)}, expensive=True
    )
    assert list(board.frame.index) == ["a", "b", "probe"]
    assert board.rank("mmd").index[0] == "b"
    assert board.frame.loc["probe", "iog"] == 1.0
    assert set(board.designs) == {"a", "b", "probe"}


def test_from_evaluator_keeps_the_evaluators_registry() -> None:
    """A custom metric the evaluator scored must be readable and rankable on the returned board."""
    from engiopt.evaluation.registry import MetricRegistry
    from engiopt.evaluation.registry import register_metric

    custom = MetricRegistry()
    register_metric("mine", family="diversity", cost="cheap", higher_is_better=True, registry=custom)(lambda ctx: 0.0)

    class StubEvaluator:
        problem = None
        registry = custom
        spec = type("Spec", (), {"sigma": None})()
        resolved = type("Resolved", (), {"ref_designs": np.zeros((2, 3)), "conditions": None})()

        def context_for(self, generator: Any) -> Any:
            return type("Ctx", (), {"gen_designs": generator})()

        def score_context(self, ctx: Any, *, only: Any, include_expensive: bool) -> dict[str, float]:
            return {"mine": float(np.mean(ctx.gen_designs))}

    board = Board.from_evaluator(StubEvaluator(), {"a": np.full(3, 0.2), "b": np.full(3, 0.4)})
    assert board.rank("mine").index[0] == "b"
    assert "mine" in board.explain().index


def test_viol_is_blank_without_the_designs_conditions(fake_problem: Any, sets: Any) -> None:
    """A constraint check that reads the conditions cannot run on bare designs; blank beats raising."""
    gen, ref = sets
    assert np.isnan(METRICS["viol"].fn(_ctx(fake_problem, gen, ref)))


def test_the_default_spec_metrics_are_all_registered() -> None:
    assert set(EvalSpec(problem_id="beams2d").metrics) <= set(METRICS)


# -------------------------------------------------------------------- specs


@pytest.mark.parametrize("problem_id", ["beams2d", "heatconduction2d", "photonics2d", "thermoelastic2d"])
def test_every_current_spec_names_only_registered_metrics(problem_id: str) -> None:
    spec = EvalSpec.load(problem_id)
    assert spec.aggregation == "mean"
    assert set(spec.metrics) <= set(METRICS)


def test_the_default_spec_is_the_newest_committed_for_the_problem() -> None:
    assert EvalSpec.load("beams2d").version == "v2"
    assert EvalSpec.load("thermoelastic2d").version == "v3"
