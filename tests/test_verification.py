"""Tests for re-running a published row from its checkpoint address.

Verification is the only defence a public board has against a number that no
model produced, so the properties here are about what it refuses to accept:
weights that have moved since they were scored, addresses that lead nowhere, and
models nobody can rebuild. The successful path matters less than those, because
a verifier that stamps rows it did not actually reproduce is worse than none --
it launders the claim it was supposed to check.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from engiopt.evaluation import verify as verify_mod
from engiopt.evaluation.spec import EvalSpec
from engiopt.evaluation.submission import FLAG_MEMORIZED
from engiopt.evaluation.verify import verify_board
from engiopt.evaluation.verify import verify_row


class _FakeGenerator:
    """Stands in for a rebuilt model; only its checkpoint hash is consulted."""

    algo_id = "cgan_cnn_2d"
    conditional = True
    checkpoint_files = ("generator.pth",)

    def __init__(self, content_hash: str = "0011223344556677"):
        self.checkpoint_hash = content_hash


class _FakeEvaluator:
    """Returns fixed scores, standing in for a real scoring pass."""

    def __init__(self, scores: dict[str, Any] | None = None):
        self.spec = EvalSpec(problem_id="beams2d")
        self.scores = scores or {"mmd": 0.04, "copy_rate": 0.0}
        self.calls = 0

    def score(self, generator: Any, *, include_expensive: bool = False) -> dict[str, Any]:
        del generator, include_expensive
        self.calls += 1
        return dict(self.scores)


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "problem_id": "beams2d",
        "algo_id": "cgan_cnn_2d",
        "config_fingerprint": "023dd1fb",
        "seed": 1,
        "spec_version": "v1",
        "checkpoint_repo": "someone/engiopt-cgan-cnn-2d",
        "checkpoint_path": "beams2d/seed_1",
        "checkpoint_revision": "cafe1234",
        "checkpoint_hash": "0011223344556677",
        "mmd": 0.04,
        "verified": False,
    }
    row.update(overrides)
    return row


@pytest.fixture
def loads_fake_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rebuild any row into a `_FakeGenerator`, so no network or weights are needed."""
    monkeypatch.setattr(verify_mod, "_load_at_recorded_revision", lambda row, cls, device=None: _FakeGenerator())


# ----------------------------------------------------------------------
# What verification refuses
# ----------------------------------------------------------------------


def test_a_model_with_no_registered_adapter_cannot_be_verified() -> None:
    """Scores for a model nobody can rebuild are unfalsifiable by construction."""
    result = verify_row(_row(algo_id="a_model_nobody_shipped"), verifier="ci")

    assert result.status == "unknown_generator"
    assert not result.ok
    assert result.row is None


def test_weights_that_moved_since_scoring_are_not_re_scored_into_place(monkeypatch: pytest.MonkeyPatch) -> None:
    """The swap attack: score a good checkpoint, then overwrite the path.

    Re-scoring whatever is at the address now would quietly attach a fresh,
    honest score to a row whose provenance is a lie. The row has to fail
    instead, because what it claims to describe no longer exists.
    """
    monkeypatch.setattr(
        verify_mod, "_load_at_recorded_revision", lambda row, cls, device=None: _FakeGenerator("ffffffffffffffff")
    )
    evaluator = _FakeEvaluator()

    result = verify_row(_row(), verifier="ci", evaluator=evaluator)

    assert result.status == "weights_moved"
    assert evaluator.calls == 0, "a moved checkpoint must not be scored at all"
    assert "no longer exist at that address" in result.detail


def test_an_unreachable_checkpoint_fails_rather_than_being_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A row that cannot be fetched is a failed verification, not an absent one."""

    def _explode(row: Any, cls: Any, device: Any = None) -> None:
        raise FileNotFoundError("HF package path not found")

    monkeypatch.setattr(verify_mod, "_load_at_recorded_revision", _explode)

    result = verify_row(_row(), verifier="ci", evaluator=_FakeEvaluator())

    assert result.status == "unresolvable"
    assert "not found" in result.detail


def test_a_row_without_an_address_has_nothing_to_fetch() -> None:
    """Admission should have caught this, but verification must not assume it did."""
    result = verify_row(_row(checkpoint_repo=None, checkpoint_path=None), verifier="ci")

    assert result.status == "unresolvable"
    assert "no checkpoint address" in result.detail


# ----------------------------------------------------------------------
# What verification publishes
# ----------------------------------------------------------------------


def test_the_verified_row_carries_the_runners_numbers(loads_fake_generator: None) -> None:
    """Not a pass/fail on the submitted ones.

    Sampling from one seed on different hardware genuinely produces different
    designs, so a comparison would need a tolerance loose enough to admit real
    fudging. Re-scoring sidesteps that: the runner's number is the number, and
    the submitted one becomes a claim reported as corroborated or not.
    """
    evaluator = _FakeEvaluator({"mmd": 0.42, "copy_rate": 0.0})

    result = verify_row(_row(mmd=0.001), verifier="ideallab-ci", evaluator=evaluator)

    assert result.ok
    assert result.row is not None
    assert result.row["mmd"] == 0.42
    assert result.row["verified"] is True
    assert result.row["verified_by"] == "ideallab-ci"
    assert result.row["verified_at"]


def test_an_unreproduced_claim_is_reported_without_failing_the_row(loads_fake_generator: None) -> None:
    """The row still verifies -- the weights are real and were scored.

    What is not true is the submitted number, and saying so is more useful than
    refusing the row, because the honest explanation (different hardware) and
    the dishonest one produce the same signal.
    """
    result = verify_row(_row(mmd=0.001), verifier="ci", evaluator=_FakeEvaluator({"mmd": 0.42}))

    assert result.ok
    assert "mmd" in result.uncorroborated
    assert result.uncorroborated["mmd"] == (0.001, 0.42)
    assert "NOT reproduced" in result.detail


def test_a_reproduced_claim_says_so(loads_fake_generator: None) -> None:
    """Small differences are noise, not evidence, and must not read as an accusation."""
    result = verify_row(_row(mmd=0.0400), verifier="ci", evaluator=_FakeEvaluator({"mmd": 0.0401}))

    assert not result.uncorroborated
    assert "corroborated" in result.detail


def test_verification_re_derives_the_flags_rather_than_trusting_them(loads_fake_generator: None) -> None:
    """A submitter who cleared their own `memorized` flag must not keep it cleared."""
    evaluator = _FakeEvaluator({"mmd": 0.04, "copy_rate": 0.95})

    result = verify_row(_row(flags=""), verifier="ci", evaluator=evaluator)

    assert result.row is not None
    assert FLAG_MEMORIZED in result.row["flags"]


# ----------------------------------------------------------------------
# Fetching the package the row actually names
# ----------------------------------------------------------------------


def test_the_recorded_revision_is_what_gets_fetched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not the current head.

    Pinning the revision is what stops a later upload to the same path from
    laundering itself into an old row's score. Without it, "verified" would mean
    "some checkpoint at this path scores well today".
    """
    captured: dict[str, Any] = {}

    def _fake_resolve(**kwargs: Any) -> Any:
        captured.update(kwargs)
        raise FileNotFoundError("stop here; the call is what is under test")

    monkeypatch.setattr(verify_mod, "resolve_checkpoint_reference", _fake_resolve)

    verify_row(_row(checkpoint_revision="deadbeef"), verifier="ci", evaluator=_FakeEvaluator())

    assert captured["revision"] == "deadbeef"
    assert captured["model_ref"] == "hf://someone/engiopt-cgan-cnn-2d/beams2d/seed_1"


# ----------------------------------------------------------------------
# Sweeping a whole board
# ----------------------------------------------------------------------


def test_already_verified_rows_are_left_alone_by_default(
    loads_fake_generator: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running everything on every pass would make the runner cost grow with the board."""
    monkeypatch.setattr(verify_mod.Evaluator, "for_problem", classmethod(lambda cls, *a, **k: _FakeEvaluator()))
    board = pd.DataFrame([_row(seed=1, verified=True), _row(seed=2, verified=False)])

    _, results = verify_board(board, verifier="ci")

    assert len(results) == 1
    assert results[0].key["seed"] == 2


def test_one_unverifiable_row_does_not_stop_the_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unattended runner has to get through the board and report at the end."""
    monkeypatch.setattr(verify_mod.Evaluator, "for_problem", classmethod(lambda cls, *a, **k: _FakeEvaluator()))
    monkeypatch.setattr(verify_mod, "_load_at_recorded_revision", lambda row, cls, device=None: _FakeGenerator())
    board = pd.DataFrame([_row(seed=1, algo_id="not_a_real_model"), _row(seed=2)])

    verified, results = verify_board(board, verifier="ci")

    assert {result.status for result in results} == {"unknown_generator", "verified"}
    assert len(verified) == 1


def test_an_unloadable_spec_fails_its_rows_and_not_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A row naming a spec version that was never committed is a bad row, not a crash."""

    def _no_such_spec(cls: Any, *args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError("No eval spec at beams2d/v9.json")

    monkeypatch.setattr(verify_mod.Evaluator, "for_problem", classmethod(_no_such_spec))

    _, results = verify_board(pd.DataFrame([_row(spec_version="v9")]), verifier="ci")

    assert results[0].status == "unresolvable"
    assert "could not load spec" in results[0].detail
