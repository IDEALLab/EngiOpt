"""`Board`: from saved designs to a readable table in three calls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from engiopt.evaluation.board import Board
from engiopt.evaluation.board import REFERENCE_ROW
from engiopt.evaluation.context import EvaluationContext
from engiopt.evaluation.registry import METRICS

# Importing the metrics package registers the built-ins.
import engiopt.evaluation.metrics  # noqa: F401  # isort: skip


@pytest.fixture
def designs(fake_problem: Any) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    shape = fake_problem.design_space.shape
    train = rng.random((20, *shape))
    ref = rng.random((6, *shape))
    models = {"honest": np.clip(ref + rng.normal(0, 0.01, ref.shape), 0, 1), "copier": ref.copy()}
    return models, ref, train


def test_evaluate_scores_every_model_on_the_cheap_metrics(fake_problem: Any, designs: Any) -> None:
    models, ref, train = designs
    frame = Board(fake_problem, reference=ref, train=train).evaluate(models)
    assert list(frame.index) == ["honest", "copier", REFERENCE_ROW]
    assert {"mmd", "copy_rate", "viol"} <= set(frame.columns)


def test_explain_names_a_pick_only_for_ranked_columns(fake_problem: Any, designs: Any) -> None:
    models, ref, train = designs
    board = Board(fake_problem, reference=ref, train=train)
    board.evaluate(models)
    explained = board.explain()
    assert explained.loc["mmd", "picks"] == "copier", "identical sets score the best MMD"
    assert explained.loc["copy_rate", "picks"] == "", "a diagnostic picks nobody"
    assert explained.loc["mmd", "question"] == METRICS["mmd"].description
    assert explained.loc["mmd", "real designs score"] == board.frame.loc[REFERENCE_ROW, "mmd"]


def test_the_reference_row_is_measured_and_never_picked(fake_problem: Any, designs: Any) -> None:
    models, ref, train = designs
    board = Board(fake_problem, reference=ref, train=train)
    board.evaluate(models)
    assert board.frame.loc[REFERENCE_ROW, "mmd"] > 0.0, "half of the data against the other half is not identical"
    assert REFERENCE_ROW not in set(board.explain()["picks"])
    assert REFERENCE_ROW not in board.evaluate(models, reference_row=False).index


def test_rank_refuses_a_diagnostic(fake_problem: Any, designs: Any) -> None:
    models, ref, train = designs
    board = Board(fake_problem, reference=ref, train=train)
    board.evaluate(models)
    assert board.rank("mmd").index[0] == "copier"
    with pytest.raises(ValueError, match="diagnostic"):
        board.rank("copy_rate")


def test_space_is_an_argument_not_a_metric(fake_problem: Any, designs: Any) -> None:
    """The same metric, asked in PCA space, gets a suffixed column; pixel-only metrics are skipped."""
    models, ref, train = designs
    frame = Board(fake_problem, reference=ref, train=train).evaluate(models, space="pca", pca_dims=3)
    assert "mmd@pca" in frame.columns
    assert "viol@pca" not in frame.columns, "a constraint check on PCA codes would be meaningless"
    assert frame.loc["copier", "mmd@pca"] == pytest.approx(0.0)


def test_aggregation_is_a_policy_on_the_context(fake_problem: Any) -> None:
    """One diverged design moves the mean and not the median."""
    values = [0.1, 0.1, 0.1, 100.0]
    zeros = np.zeros((4, 8, 10))
    mean_ctx = EvaluationContext(problem=fake_problem, problem_id="f", gen_designs=zeros, ref_designs=zeros)
    median_ctx = EvaluationContext(
        problem=fake_problem, problem_id="f", gen_designs=zeros, ref_designs=zeros, aggregation="median"
    )
    assert mean_ctx.reduce(values) == pytest.approx(25.075)
    assert median_ctx.reduce(values) == pytest.approx(0.1)
