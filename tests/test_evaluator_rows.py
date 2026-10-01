"""`Evaluator.for_rows`: score against a dataset slice you chose.

The spec's own draw stays the leaderboard contract. This path swaps only the
draw: your rows' conditions drive the sampling, your rows' optimal designs are
the reference, and everything else the spec declares stays in force.
"""

from __future__ import annotations

import numpy as np
import pytest

from engiopt.evaluation.evaluator import Evaluator
from engiopt.evaluation.spec import EvalSpec
from tests.conftest import FakeDataset
from tests.conftest import FakeProblem


@pytest.fixture
def rows() -> FakeDataset:
    """Two hand-picked test rows with distinct briefs and known optima."""
    shape = (8, 10)
    return FakeDataset(
        {
            "volfrac": [0.2, 0.3],
            "rmin": [2.0, 2.0],
            "optimal_design": [np.full(shape, 0.2), np.full(shape, 0.3)],
        }
    )


@pytest.fixture
def evaluator(rows: FakeDataset, monkeypatch: pytest.MonkeyPatch) -> Evaluator:
    from engibench.utils.all_problems import BUILTIN_PROBLEMS

    monkeypatch.setitem(BUILTIN_PROBLEMS, "fake2d", FakeProblem)
    spec = EvalSpec(problem_id="fake2d", volume_condition="volfrac")
    return Evaluator.for_rows("fake2d", rows, spec=spec)


def test_the_rows_become_the_reference_and_the_briefs(evaluator: Evaluator, rows: FakeDataset) -> None:
    assert evaluator.resolved.ref_designs.shape == (2, 8, 10)
    assert evaluator.resolved.ref_designs[0].mean() == pytest.approx(0.2)
    assert evaluator.resolved.conditions["volfrac"] == [0.2, 0.3]
    assert evaluator.resolved.conditions_tensor.shape == (2, 2), "one row per design, one column per scalar condition"
    assert evaluator.spec.n_samples == len(rows), "generators sample one design per chosen row"


def test_a_custom_slice_is_not_the_frozen_contract(evaluator: Evaluator) -> None:
    """The digest certifies the spec's own draw; a custom draw must not inherit it."""
    assert evaluator.spec.condition_digest is None
    assert evaluator.spec.volume_condition == "volfrac", "everything but the draw survives from the spec"


def test_designs_are_scored_against_the_chosen_rows(evaluator: Evaluator) -> None:
    """A batch equal to the rows' own optima has zero volume error under their briefs."""
    ctx = evaluator.context_for_designs(evaluator.resolved.ref_designs.copy())
    row = evaluator.score_context(ctx, only=["volume_error"], include_expensive=False)
    assert row["volume_error"] == pytest.approx(0.0)
