"""`Evaluator.for_rows`: score against a dataset slice you chose.

The spec's own draw stays the leaderboard contract. This path swaps only the
draw: your rows' conditions drive the sampling, your rows' optimal designs are
the reference, and everything else the spec declares stays in force.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engiopt.evaluation.evaluator import Evaluator
from engiopt.evaluation.leaderboard import merge_rows
from engiopt.evaluation.spec import EvalSpec
from engiopt.evaluation.submission import CUSTOM_ROWS_MARKER
from engiopt.evaluation.submission import prepare_submission
from engiopt.evaluation.submission import SubmissionRejectedError
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


OFFICIAL_VERSION = EvalSpec(problem_id="fake2d").version
"""The protocol name the fixture's spec starts from, before any custom rows."""


def _row_scored_by(evaluator: Evaluator, **overrides: object) -> dict[str, object]:
    """A row with a fetchable checkpoint, under whatever protocol name this evaluator reports."""
    row = {
        "problem_id": "fake2d",
        "algo_id": "cgan_cnn_2d",
        "config_fingerprint": "023dd1fb",
        "seed": 1,
        "spec_version": evaluator.spec.version,
        "checkpoint_repo": "someone/engiopt-cgan-cnn-2d",
        "checkpoint_path": "fake2d/seed_1",
        "checkpoint_revision": "cafe1234",
        "checkpoint_hash": "0011223344556677",
        "mmd": 0.001,
    }
    row.update(overrides)
    return row


def test_a_custom_slice_reports_its_own_protocol_name(evaluator: Evaluator) -> None:
    """Two evaluation sets must not share one protocol name, so the chosen rows change it."""
    assert evaluator.spec.version == OFFICIAL_VERSION + CUSTOM_ROWS_MARKER


def test_a_custom_slice_row_is_refused_by_the_public_board(evaluator: Evaluator) -> None:
    """A score on rows the submitter picked is not the frozen protocol, so it cannot be published."""
    with pytest.raises(SubmissionRejectedError, match="rows the submitter chose"):
        prepare_submission(pd.DataFrame([_row_scored_by(evaluator)]))


def test_a_custom_slice_row_cannot_replace_the_official_one(evaluator: Evaluator) -> None:
    """Merged into one table, the official row survives beside the custom one."""
    official = _row_scored_by(evaluator, spec_version=OFFICIAL_VERSION, mmd=0.04)

    merged = merge_rows(pd.DataFrame([official]), pd.DataFrame([_row_scored_by(evaluator)]))

    assert sorted(merged["spec_version"]) == [OFFICIAL_VERSION, OFFICIAL_VERSION + CUSTOM_ROWS_MARKER]
    assert merged.loc[merged["spec_version"] == OFFICIAL_VERSION, "mmd"].item() == pytest.approx(0.04)
