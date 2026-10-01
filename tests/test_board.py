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
    assert {"mmd", "train_distance", "viol"} <= set(frame.columns)


def test_explain_names_a_pick_only_for_ranked_columns(fake_problem: Any, designs: Any) -> None:
    models, ref, train = designs
    board = Board(fake_problem, reference=ref, train=train)
    board.evaluate(models)
    explained = board.explain()
    assert explained.loc["mmd", "picks"] == "copier", "identical sets score the best MMD"
    assert explained.loc["train_distance", "picks"] == "", "a diagnostic picks nobody"
    assert explained.loc["mmd", "question"] == METRICS["mmd"].description
    assert explained.loc["mmd", "split-half reference"] == board.frame.loc[REFERENCE_ROW, "mmd"]


def test_the_reference_row_is_measured_and_never_picked(fake_problem: Any, designs: Any) -> None:
    models, ref, train = designs
    board = Board(fake_problem, reference=ref, train=train)
    board.evaluate(models)
    assert board.frame.loc[REFERENCE_ROW, "mmd"] > 0.0, "half of the data against the other half is not identical"
    assert REFERENCE_ROW not in set(board.explain()["picks"])
    assert REFERENCE_ROW not in board.rank("mmd").index, "a scale reference is never ranked"
    assert REFERENCE_ROW not in board.evaluate(models, reference_row=False).index


def test_rank_refuses_a_diagnostic(fake_problem: Any, designs: Any) -> None:
    models, ref, train = designs
    board = Board(fake_problem, reference=ref, train=train)
    board.evaluate(models)
    assert board.rank("mmd").index[0] == "copier"
    with pytest.raises(ValueError, match="diagnostic"):
        board.rank("train_distance")


def test_space_is_an_argument_not_a_metric(fake_problem: Any, designs: Any) -> None:
    """The same metric, asked in PCA space, gets a suffixed column; pixel-only metrics are skipped."""
    models, ref, train = designs
    frame = Board(fake_problem, reference=ref, train=train).evaluate(models, space="pca", width=3)
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


def test_volume_error_reaches_saved_designs(fake_problem: Any, designs: Any) -> None:
    """A board told which condition is the volume budget scores it; one not told leaves it blank."""
    models, ref, _ = designs
    conditions = {"volfrac": [float(design.mean()) for design in ref]}
    told = Board(fake_problem, reference=ref, conditions=conditions, volume_condition="volfrac")
    frame = told.evaluate({"exact": ref.copy()}, metrics=["volume_error"])
    assert frame.loc["exact", "volume_error"] == pytest.approx(0.0), "designs at their own budgets have no error"
    untold = Board(fake_problem, reference=ref, conditions=conditions)
    assert np.isnan(untold.evaluate(models, metrics=["volume_error"]).loc["honest", "volume_error"])


def test_dict_conditions_survive_a_full_evaluate(fake_problem: Any, designs: Any) -> None:
    """A plain dict of columns must feed row-reading metrics too.

    `viol` asks for one design's brief at a time (`conditions[0]`), which a bare
    dict answers with `KeyError: 0`. The board wraps the dict, so the same input
    that fed `volume_error` survives a default evaluate.
    """
    models, ref, _ = designs
    conditions = {"volfrac": [float(design.mean()) for design in ref]}
    board = Board(fake_problem, reference=ref, conditions=conditions, volume_condition="volfrac")
    frame = board.evaluate(models)
    assert not np.isnan(frame.loc["honest", "viol"]), "the row-access path must work, not just the column path"
    assert not np.isnan(frame.loc["honest", "volume_error"])


def test_a_dataset_slice_is_designs_and_briefs_at_once(fake_problem: Any) -> None:
    """Passing the test rows keeps design i and brief i aligned by construction.

    The briefs already sit beside the optimal designs in the dataset; slicing
    the designs into an array is what loses them. Handing the board the rows
    themselves means nothing is lost and nothing can be misaligned.
    """
    rows = fake_problem.dataset["train"]
    board = Board(fake_problem, reference=rows, volume_condition="volfrac")
    assert board.reference.shape == (3, 8, 10), "the design column became the reference array"
    assert board.conditions is not None
    assert board.conditions["volfrac"] == [0.5, 0.5, 0.5], "the remaining columns became the briefs"
    frame = board.evaluate({"exact": board.reference.copy()}, metrics=["volume_error"])
    # The dataset's designs are flat fields at 0.1, 0.2 and 0.3 against a 0.5 budget.
    assert frame.loc["exact", "volume_error"] == pytest.approx(np.mean([0.4, 0.3, 0.2]))


def test_misaligned_conditions_are_refused(fake_problem: Any, designs: Any) -> None:
    """Conditions are an answer key, not a filter; a short one is wrong labels, not a subset."""
    _, ref, _ = designs
    with pytest.raises(ValueError, match="do not select"):
        Board(fake_problem, reference=ref, conditions={"volfrac": [0.2, 0.3]})


def test_every_row_shares_one_inferred_bandwidth(fake_problem: Any, designs: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The split-half row must not size its kernel from its own half.

    Left to each context, a half landing in one cluster of the reference set
    infers a degenerate bandwidth (the 1e-6 floor), so its kernel columns would
    differ from the model rows in bandwidth as well as in sample size.
    """
    from engiopt import metrics as metrics_mod

    models, ref, _ = designs
    calls: list[int] = []
    real = metrics_mod.compute_median_sigma

    def spy(x: Any, y: Any = None) -> float:
        calls.append(len(x))
        return real(x, y)

    monkeypatch.setattr(metrics_mod, "compute_median_sigma", spy)
    Board(fake_problem, reference=ref).evaluate(models)
    assert calls == [len(ref)], "one bandwidth, sized from the full reference set, for every row"


def test_the_default_bandwidth_lets_diversity_metrics_see_anything(fake_problem: Any, designs: Any) -> None:
    """At a fixed sigma=10 every design looks identical to every other; the training-data median does not."""
    models, ref, _ = designs
    collapsed = np.repeat(models["honest"][:1], len(ref), axis=0)
    frame = Board(fake_problem, reference=ref).evaluate({"collapsed": collapsed, "spread": models["honest"]})
    assert frame.loc["collapsed", "vendi"] == pytest.approx(1.0)
    assert frame.loc["spread", "vendi"] > 3.0
    saturated = Board(fake_problem, reference=ref, sigma=10.0).evaluate({"spread": models["honest"]})
    assert saturated.loc["spread", "vendi"] < 1.5, "a pinned, too-wide kernel is still honored"
