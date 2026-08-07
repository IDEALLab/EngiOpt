"""Tests for publishing to the shared leaderboard.

Publishing is read-merge-write against a table other people are also writing to,
so the failure modes that matter are the ones that silently *delete* rows: a
download error mistaken for an empty board, and a concurrent publisher whose
commit gets overwritten.
"""

from __future__ import annotations

from typing import Any

from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.errors import RepositoryNotFoundError
import pandas as pd
import pytest
import requests

from engiopt.evaluation import leaderboard as lb


def hub_error(error_cls: type[Exception], message: str, status: int = 500) -> Exception:
    """Build a `huggingface_hub` error the way the library itself would.

    Versions disagree about `response`: newer ones require it as a keyword-only
    argument, and they read attributes off it (`request`, `headers`). A stub
    would raise `TypeError`/`AttributeError` in place of the error under test,
    turning a real assertion into a confusing failure -- so this passes a
    genuine `requests.Response`, which satisfies every version.
    """
    response = requests.Response()
    response.status_code = status
    response.reason = message
    response._content = message.encode()
    response.request = requests.Request(method="GET", url="https://huggingface.co/api").prepare()
    try:
        return error_cls(message, response=response)  # type: ignore[call-arg]
    except TypeError:
        return error_cls(message)


def _row(algo: str, mmd: float) -> dict[str, Any]:
    return {
        "problem_id": "p",
        "algo_id": algo,
        "config_fingerprint": "default",
        "seed": 1,
        "spec_version": "v1",
        "mmd": mmd,
    }


class _FakeApi:
    """Records uploads and can fail the first one, like a lost commit race."""

    def __init__(self, *, revision: str | None = "sha1", fail_first: bool = False):
        self.revision = revision
        self.fail_first = fail_first
        self.uploads: list[dict[str, Any]] = []

    def __call__(self, *_args: Any, **_kwargs: Any) -> _FakeApi:
        return self

    def create_repo(self, **_kwargs: Any) -> None:
        return None

    def repo_info(self, **_kwargs: Any) -> Any:
        if self.revision is None:
            raise hub_error(RepositoryNotFoundError, "no such repo", status=404)
        return type("Info", (), {"sha": self.revision})()

    def upload_file(self, **kwargs: Any) -> None:
        self.uploads.append({**kwargs, "content": pd.read_csv(kwargs["path_or_fileobj"])})
        if self.fail_first and len(self.uploads) == 1:
            # What the Hub returns when `parent_commit` is no longer the head.
            self.revision = "sha2"
            raise hub_error(HfHubHTTPError, "412 Client Error: parent_commit is out of date", status=412)


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    """Patch the Hub so `push_to_hub` runs against a local CSV."""
    board_file = tmp_path / "leaderboard.csv"
    pd.DataFrame([_row("existing", 0.5)]).to_csv(board_file, index=False)

    def fake_download(**_kwargs: Any) -> str:
        return str(board_file)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    return board_file


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------


def test_missing_board_reads_as_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first model to publish finds no board, and that is not an error."""
    api = _FakeApi(revision=None)
    monkeypatch.setattr("huggingface_hub.HfApi", api)
    assert lb.load_from_hub("org/board").empty


def test_missing_file_in_an_existing_repo_reads_as_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A repo created but not yet populated is also just an empty board."""
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi())

    def fake_download(**_kwargs: Any) -> str:
        raise hub_error(EntryNotFoundError, "no leaderboard.csv", status=404)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    assert lb.load_from_hub("org/board").empty


def test_transient_download_failure_is_not_an_empty_board(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swallowing this would let the next push upload only the new rows, deleting the rest."""
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi())

    def fake_download(**_kwargs: Any) -> str:
        raise hub_error(HfHubHTTPError, "503 Server Error: Service Unavailable", status=503)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    with pytest.raises(HfHubHTTPError):
        lb.load_from_hub("org/board")


def test_auth_failure_is_not_an_empty_board(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same reasoning: a private board we cannot read is not a board with no rows."""
    api = _FakeApi()

    def raise_auth(**_kwargs: Any) -> Any:
        raise hub_error(HfHubHTTPError, "401 Client Error: Unauthorized", status=401)

    api.repo_info = raise_auth  # type: ignore[method-assign]
    monkeypatch.setattr("huggingface_hub.HfApi", api)
    with pytest.raises(HfHubHTTPError):
        lb.load_from_hub("org/board")


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------


def test_push_merges_instead_of_replacing(monkeypatch: pytest.MonkeyPatch, published: Any) -> None:
    """Publishing one model must not require re-running anyone else's evaluation."""
    api = _FakeApi()
    monkeypatch.setattr("huggingface_hub.HfApi", api)

    merged = lb.push_to_hub(pd.DataFrame([_row("newcomer", 0.1)]), "org/board")

    assert set(merged["algo_id"]) == {"existing", "newcomer"}
    assert set(api.uploads[0]["content"]["algo_id"]) == {"existing", "newcomer"}


def test_push_is_conditional_on_the_revision_it_read(monkeypatch: pytest.MonkeyPatch, published: Any) -> None:
    """Without `parent_commit`, two concurrent jobs silently drop one another's rows."""
    api = _FakeApi()
    monkeypatch.setattr("huggingface_hub.HfApi", api)

    lb.push_to_hub(pd.DataFrame([_row("newcomer", 0.1)]), "org/board")

    assert api.uploads[0]["parent_commit"] == "sha1"


def test_push_retries_against_the_newer_board_after_losing_a_race(monkeypatch: pytest.MonkeyPatch, published: Any) -> None:
    """A rejected commit means someone else published; re-merge rather than overwrite."""
    api = _FakeApi(fail_first=True)
    monkeypatch.setattr("huggingface_hub.HfApi", api)

    merged = lb.push_to_hub(pd.DataFrame([_row("newcomer", 0.1)]), "org/board")

    assert len(api.uploads) == 2
    assert api.uploads[1]["parent_commit"] == "sha2"
    assert set(merged["algo_id"]) == {"existing", "newcomer"}


def test_push_gives_up_rather_than_forcing_a_write(monkeypatch: pytest.MonkeyPatch, published: Any) -> None:
    """Losing every attempt must surface, not resolve into a clobbering write."""

    class _AlwaysStale(_FakeApi):
        def upload_file(self, **kwargs: Any) -> None:
            self.uploads.append(kwargs)
            raise hub_error(HfHubHTTPError, "412 Client Error: parent_commit is out of date", status=412)

    monkeypatch.setattr("huggingface_hub.HfApi", _AlwaysStale())
    with pytest.raises(HfHubHTTPError):
        lb.push_to_hub(pd.DataFrame([_row("newcomer", 0.1)]), "org/board", max_attempts=2)


# ----------------------------------------------------------------------
# Distinguishing "no repo" from "no access" and "no file"
# ----------------------------------------------------------------------


def test_a_private_repo_is_not_an_empty_board(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Hub reports a repo you cannot see as not-found; that must not read as empty."""
    api = _FakeApi()

    def raise_private(**_kwargs: Any) -> Any:
        raise hub_error(RepositoryNotFoundError, "401 Client Error: Unauthorized", status=401)

    api.repo_info = raise_private  # type: ignore[method-assign]
    monkeypatch.setattr("huggingface_hub.HfApi", api)
    with pytest.raises(RepositoryNotFoundError):
        lb.load_from_hub("org/board")


def test_first_publish_into_an_existing_repo_keeps_its_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """A repo with no leaderboard.csv still has a head commit, and it guards the first write.

    Discarding it means two jobs both publishing for the first time can each
    overwrite the other.
    """
    api = _FakeApi(revision="sha1")
    monkeypatch.setattr("huggingface_hub.HfApi", api)

    def missing_file(**_kwargs: Any) -> str:
        raise hub_error(EntryNotFoundError, "no leaderboard.csv", status=404)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", missing_file)

    lb.push_to_hub(pd.DataFrame([_row("first", 0.1)]), "org/board")

    assert api.uploads[0]["parent_commit"] == "sha1"


# ----------------------------------------------------------------------
# Ranking stays inside a problem
# ----------------------------------------------------------------------


def test_ranks_restart_within_each_problem() -> None:
    """A beams2d score and a photonics2d score are not competitors."""
    frame = pd.DataFrame(
        [
            {"problem_id": "beams2d", "algo_id": "a", "config_fingerprint": "c", "seed": 1,
             "spec_version": "v1", "mmd": 0.1},
            {"problem_id": "beams2d", "algo_id": "b", "config_fingerprint": "c", "seed": 1,
             "spec_version": "v1", "mmd": 0.2},
            {"problem_id": "photonics2d", "algo_id": "a", "config_fingerprint": "c", "seed": 1,
             "spec_version": "v1", "mmd": 0.3},
            {"problem_id": "photonics2d", "algo_id": "b", "config_fingerprint": "c", "seed": 1,
             "spec_version": "v1", "mmd": 0.4},
        ]
    )  # fmt: skip
    ranked = lb.rank(frame, "mmd")
    for problem in ("beams2d", "photonics2d"):
        assert sorted(ranked[ranked["problem_id"] == problem]["rank"]) == [1, 2]


def test_ranks_restart_within_each_spec_version() -> None:
    """Two protocols are two boards, even for the same problem."""
    frame = pd.DataFrame(
        [
            {"problem_id": "p", "algo_id": "a", "config_fingerprint": "c", "seed": 1, "spec_version": "v1",
             "mmd": 0.1},
            {"problem_id": "p", "algo_id": "b", "config_fingerprint": "c", "seed": 1, "spec_version": "v2",
             "mmd": 0.2},
        ]
    )  # fmt: skip
    assert list(lb.rank(frame, "mmd")["rank"]) == [1, 1]


# ----------------------------------------------------------------------
# Skipping work already done
# ----------------------------------------------------------------------


def _published_row(**overrides: Any) -> pd.DataFrame:
    row = {**_row("a", 0.1), "checkpoint_hash": "hash_old"}
    row.update(overrides)
    return pd.DataFrame([row])


def test_retrained_weights_are_not_skipped() -> None:
    """Same config and seed, different weights: the old row does not describe them."""
    assert not lb.already_evaluated(
        _published_row(),
        problem_id="p",
        algo_id="a",
        config_fingerprint="default",
        seed=1,
        spec_version="v1",
        checkpoint_hash="hash_new",
    )


def test_the_same_weights_are_skipped() -> None:
    """Re-running an unchanged checkpoint is the work `--skip-existing` exists to avoid."""
    assert lb.already_evaluated(
        _published_row(),
        problem_id="p",
        algo_id="a",
        config_fingerprint="default",
        seed=1,
        spec_version="v1",
        checkpoint_hash="hash_old",
    )


def test_a_row_without_a_hash_is_not_treated_as_a_match() -> None:
    """A missing hash means the weights behind that row are unknown, not identical.

    Skipping on it lets one hashless row suppress every future evaluation of that
    configuration and seed -- including genuinely retrained weights -- until
    somebody deletes the row by hand. This schema has never shipped on `main`, so
    there is no historical board whose re-evaluation cost the old leniency was
    protecting.
    """
    assert not lb.already_evaluated(
        _published_row(checkpoint_hash=None),
        problem_id="p",
        algo_id="a",
        config_fingerprint="default",
        seed=1,
        spec_version="v1",
        checkpoint_hash="hash_new",
    )
