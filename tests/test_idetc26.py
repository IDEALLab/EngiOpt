"""Verification for the IDETC'26 challenge harness.

The tests that matter here are the ones whose failure mode is "in front of a
room": the cheap board silently invoking a simulator and stalling the live
segment, the reveal firing before anyone has committed, or the config resolving
differently depending on which directory a notebook was launched from.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from engiopt.workshops.idetc26 import Challenge
from engiopt.workshops.idetc26 import ModelBank
from engiopt.workshops.idetc26 import WorkshopConfig
from engiopt.workshops.idetc26.challenge import NotCommittedError
from engiopt.workshops.idetc26.seal import seal
from engiopt.workshops.idetc26.seal import SealError
from engiopt.workshops.idetc26.seal import unseal

PROBLEM_ID = "beams2d"


@pytest.fixture(scope="module")
def challenge(tmp_path_factory: pytest.TempPathFactory) -> Challenge:
    """One opened challenge, shared: opening it reads the dataset."""
    return Challenge.open(PROBLEM_ID, team="pytest", artifact_dir=tmp_path_factory.mktemp("idetc"))


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


def test_config_round_trips() -> None:
    """Every declared metric must be one the registry actually knows."""
    from engiopt.evaluation.registry import METRICS

    config = WorkshopConfig.load(PROBLEM_ID)
    assert config.problem_id == PROBLEM_ID
    for name in (*config.cheap_metrics, *config.expensive_metrics):
        assert name in METRICS, f"{name} is configured but not registered"


def test_cheap_and_expensive_are_declared_consistently_with_the_registry() -> None:
    """A column in `opening_metrics` that secretly runs a simulator would stall the session."""
    from engiopt.evaluation.registry import METRICS

    config = WorkshopConfig.load(PROBLEM_ID)
    assert all(METRICS[name].cost == "cheap" for name in config.cheap_metrics)
    assert all(METRICS[name].cost == "expensive" for name in config.expensive_metrics)


def test_config_resolves_the_same_from_any_working_directory(tmp_path: Path) -> None:
    """The DCC'26 bug, tested away: config paths anchor on `__file__`, never on the cwd."""
    original = Path.cwd()
    try:
        from_repo = WorkshopConfig.load(PROBLEM_ID)
        os.chdir(tmp_path)
        from_elsewhere = WorkshopConfig.load(PROBLEM_ID)
    finally:
        os.chdir(original)
    assert from_repo == from_elsewhere


def test_unavailable_metrics_are_derived_rather_than_declared() -> None:
    """beams2d has a volume budget, so nothing is unavailable -- and that answer comes from the spec."""
    from engiopt.evaluation.spec import EvalSpec

    config = WorkshopConfig.load(PROBLEM_ID)
    assert EvalSpec.load(config.spec).volume_condition is not None
    assert config.unavailable_metrics() == {}


# ----------------------------------------------------------------------
# The bank
# ----------------------------------------------------------------------


def test_team_permutation_is_deterministic_and_team_specific(challenge: Challenge) -> None:
    """A team reruns and sees the same letters; the next table sees different ones."""
    problem, config = challenge.evaluator.problem, challenge.config

    first = ModelBank.assemble(config, problem, team="orange")
    again = ModelBank.assemble(config, problem, team="orange")
    other = ModelBank.assemble(config, problem, team="blue")

    assert [m.key for m in first] == [m.key for m in again]
    assert [m.key for m in first] != [m.key for m in other]
    assert sorted(m.key for m in first) == sorted(m.key for m in other)


def test_the_bank_reports_what_it_could_not_load(challenge: Challenge) -> None:
    """Skipped members are surfaced, not swallowed -- and never silently.

    Which entries skip depends on what the Hub currently holds, so this asserts
    the accounting rather than a specific list: every configured entry either
    loaded or was reported.
    """
    declared = {str(entry["algo"]) for entry in challenge.config.bank}
    assert set(challenge.bank.skipped) <= declared
    assert len(challenge.bank) + len(challenge.bank.skipped) == len(challenge.config.bank)


def test_unknown_labels_fail_with_the_valid_ones(challenge: Challenge) -> None:
    """A typo in a notebook must say what the options were."""
    with pytest.raises(KeyError, match="Model A"):
        challenge.bank["Model Z"]


# ----------------------------------------------------------------------
# The board
# ----------------------------------------------------------------------


def test_the_opening_board_never_touches_the_simulator(challenge: Challenge, monkeypatch: pytest.MonkeyPatch) -> None:
    """The load-bearing guarantee of the live segment.

    Forty-five minutes of the session assume the board is instant. If any
    opening metric reaches for `simulate` or `optimize`, the room waits.
    """

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("the cheap board invoked the simulator")

    monkeypatch.setattr(challenge.evaluator.problem, "simulate", forbidden)
    monkeypatch.setattr(challenge.evaluator.problem, "optimize", forbidden)

    board = challenge.board()
    assert list(board.columns) == list(challenge.config.opening_metrics)
    assert board.notna().to_numpy().all()


def test_the_board_disagrees_with_itself(challenge: Challenge) -> None:
    """The premise of the whole session: no single model tops every column."""
    board = challenge.board(metrics=challenge.config.cheap_metrics)
    winners = challenge.winners(board)
    assert winners.nunique() >= 3, f"the bank does not make the point: {winners.to_dict()}"


def test_constant_columns_are_not_ranked(challenge: Challenge) -> None:
    """Ranking a column where every model ties would invent an ordering."""
    import pandas as pd

    frame = pd.DataFrame({"mmd": [0.5, 0.5, 0.5], "novelty": [1.0, 2.0, 3.0]}, index=["A", "B", "C"])
    ranked = challenge.rank(frame)
    assert "mmd" not in ranked.columns
    assert ranked["novelty"].tolist() == [3, 2, 1]


# ----------------------------------------------------------------------
# Commitment and reveal
# ----------------------------------------------------------------------


def test_reveals_are_gated_on_a_commitment(challenge: Challenge) -> None:
    """Reading the answer first teaches that the answer is surprising, not that you were wrong."""
    challenge.verdict = None
    for reveal in (challenge.seed_lottery, challenge.withheld, challenge.identities):
        with pytest.raises(NotCommittedError):
            reveal()


def test_submitting_writes_a_hashed_verdict(challenge: Challenge) -> None:
    """The commitment has to be checkable after the fact, or it is not a commitment."""
    verdict = challenge.submit(winner="Model A", why="testing", ranking=challenge.bank.labels)

    written = (challenge.artifact_dir / "verdict.json").read_text()
    assert verdict.digest in written
    assert len(verdict.digest) == 64

    # The same commitment hashes the same; a different one does not.
    assert challenge.submit(winner="Model A", why="testing", ranking=challenge.bank.labels).digest == verdict.digest
    assert challenge.submit(winner="Model B", why="testing", ranking=challenge.bank.labels).digest != verdict.digest


def test_submitting_an_unknown_model_fails(challenge: Challenge) -> None:
    """A team cannot commit to a model that is not in their bank."""
    with pytest.raises(KeyError):
        challenge.submit(winner="Model Q", why="typo")


# ----------------------------------------------------------------------
# Sealing
# ----------------------------------------------------------------------


def test_seal_round_trips(tmp_path: Path) -> None:
    """Identity, and the published digest matches the plaintext."""
    payload = "key,iog\nregurgitator#1,1.7\n"
    destination = tmp_path / "board.csv.enc"

    digest = seal(payload, "open sesame", destination)

    assert unseal(destination, "open sesame") == payload
    assert destination.with_suffix(destination.suffix + ".sha256").read_text().strip() == digest


def test_the_wrong_passphrase_is_refused(tmp_path: Path) -> None:
    """And says so in words a facilitator can act on."""
    destination = tmp_path / "board.csv.enc"
    seal("key,iog\na,1\n", "correct", destination)

    with pytest.raises(SealError, match="does not open"):
        unseal(destination, "incorrect")


def test_a_tampered_board_is_detected(tmp_path: Path) -> None:
    """The digest is the point: it catches the answer being changed after publication."""
    destination = tmp_path / "board.csv.enc"
    seal("key,iog\na,1\n", "phrase", destination)
    seal("key,iog\na,999\n", "phrase", destination)  # rewrite the ciphertext...
    destination.with_suffix(destination.suffix + ".sha256").write_text("0" * 64 + "\n")  # ...and a stale digest

    with pytest.raises(SealError, match="moved the goalposts"):
        unseal(destination, "phrase")


def test_a_missing_board_says_where_it_looked(tmp_path: Path) -> None:
    """A missing seal on the day must not surface as a stack trace."""
    with pytest.raises(SealError, match="No sealed board"):
        unseal(tmp_path / "absent.csv.enc", "phrase")
