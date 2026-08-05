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

from engiopt.evaluation import leaderboard as lb


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
            raise RepositoryNotFoundError("no such repo")
        return type("Info", (), {"sha": self.revision})()

    def upload_file(self, **kwargs: Any) -> None:
        self.uploads.append({**kwargs, "content": pd.read_csv(kwargs["path_or_fileobj"])})
        if self.fail_first and len(self.uploads) == 1:
            # What the Hub returns when `parent_commit` is no longer the head.
            self.revision = "sha2"
            raise HfHubHTTPError("412 Client Error: parent_commit is out of date")


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
        raise EntryNotFoundError("no leaderboard.csv")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    assert lb.load_from_hub("org/board").empty


def test_transient_download_failure_is_not_an_empty_board(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swallowing this would let the next push upload only the new rows, deleting the rest."""
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi())

    def fake_download(**_kwargs: Any) -> str:
        raise HfHubHTTPError("503 Server Error: Service Unavailable")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    with pytest.raises(HfHubHTTPError):
        lb.load_from_hub("org/board")


def test_auth_failure_is_not_an_empty_board(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same reasoning: a private board we cannot read is not a board with no rows."""
    api = _FakeApi()

    def raise_auth(**_kwargs: Any) -> Any:
        raise HfHubHTTPError("401 Client Error: Unauthorized")

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
            raise HfHubHTTPError("412 Client Error: parent_commit is out of date")

    monkeypatch.setattr("huggingface_hub.HfApi", _AlwaysStale())
    with pytest.raises(HfHubHTTPError):
        lb.push_to_hub(pd.DataFrame([_row("newcomer", 0.1)]), "org/board", max_attempts=2)
