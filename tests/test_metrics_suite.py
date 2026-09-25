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


def test_train_distance_and_copy_rate_ask_different_questions(fake_problem: Any, sets: Any) -> None:
    gen, ref = sets
    train = np.random.default_rng(3).random((20, *fake_problem.design_space.shape))
    copies_train = _ctx(fake_problem, train[:12], ref, copy_corpus_fn=lambda: train)
    assert METRICS["train_distance"].fn(copies_train) == pytest.approx(0.0)
    assert METRICS["copy_rate"].fn(copies_train) == pytest.approx(1.0)
    copies_reference = _ctx(fake_problem, ref.copy(), ref, copy_corpus_fn=lambda: train)
    assert METRICS["train_distance"].fn(copies_reference) > 0.1, "the withheld optima are not training designs"
    assert METRICS["copy_rate"].fn(copies_reference) == pytest.approx(1.0), "but they are in the copyable corpus"
    assert np.isnan(METRICS["train_distance"].fn(_ctx(fake_problem, gen, ref)))


# -------------------------------------------------------------- performance


@pytest.fixture
def trajectories(fake_problem: Any, sets: Any) -> EvaluationContext:
    """Two re-optimization paths: one reaches the reference optimum, one never does."""
    gen, ref = sets
    ctx = _ctx(fake_problem, gen[:2], ref[:2])
    reaches = np.array([5.0, 4.0, 3.0, 0.0, 0.0])
    never = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    ctx.__dict__["optimization"] = OptimizationResults(trajectories=[reaches, never])
    return ctx


def test_calls_to_settle_is_last_exit_from_the_band(trajectories: EvaluationContext) -> None:
    # reaches: settles after call 3 (band 0.25 around 0); never: after call 4 (band 0.2 around 1).
    assert METRICS["calls_to_settle"].fn(trajectories) == pytest.approx(3.5)


def test_gap_after_calls_carries_a_short_path_forward(trajectories: EvaluationContext) -> None:
    gaps = METRICS["gap_after_calls"].fn(trajectories)
    assert gaps["gap_after_1_calls"] == pytest.approx(5.0)
    assert gaps["gap_after_2_calls"] == pytest.approx(4.0)
    assert gaps["gap_after_5_calls"] == pytest.approx(0.5)
    assert gaps["gap_after_10_calls"] == pytest.approx(0.5), "a five-call path is done at call ten"


def test_reaches_reference_rate_is_a_rate_not_a_censored_count(trajectories: EvaluationContext) -> None:
    assert METRICS["reaches_reference_rate"].fn(trajectories) == pytest.approx(0.5)


def test_first_call_gain_is_the_first_step_over_the_whole_improvement(trajectories: EvaluationContext) -> None:
    assert METRICS["first_call_gain"].fn(trajectories) == pytest.approx((0.2 + 0.25) / 2)


def test_trajectory_metrics_respect_the_aggregation_policy(trajectories: EvaluationContext) -> None:
    trajectories.aggregation = "median"
    assert METRICS["calls_to_settle"].fn(trajectories) == pytest.approx(3.5), "median of two equals their mean"


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


def test_from_generators_scores_through_an_evaluator() -> None:
    class StubEvaluator:
        def score(self, generator: Any, *, only: Any, include_expensive: bool) -> dict[str, float]:
            return {"mmd": generator, "iog": 1.0 if include_expensive else float("nan")}

    board = Board.from_generators(StubEvaluator(), {"a": 0.2, "b": 0.1}, expensive=True)
    assert board.rank("mmd").index[0] == "b"
    assert board.frame.loc["a", "iog"] == 1.0


# -------------------------------------------------------------------- specs


@pytest.mark.parametrize("problem_id", ["beams2d", "heatconduction2d", "photonics2d", "thermoelastic2d"])
def test_every_v2_spec_names_only_registered_metrics(problem_id: str) -> None:
    spec = EvalSpec.load(f"{problem_id}/v2")
    assert spec.version == "v2"
    assert spec.aggregation == "mean"
    assert set(spec.metrics) <= set(METRICS)


def test_the_default_spec_is_v2() -> None:
    assert EvalSpec.load("beams2d").version == "v2"
