"""Tests for what `--list-generators` claims about a problem.

Registered, fitting, and available are three states, and the listing is the only
place a user sees the difference. Collapsing the last two invents migrations that
nobody owes: a 1D generator has no `beams2d` checkpoint because it cannot serve a
2D problem, not because someone failed to publish one.
"""

from __future__ import annotations

import dataclasses

import pytest

from engiopt import evaluate
from engiopt.utils.all_generators import BUILTIN_GENERATORS

PUBLISHED = {"cgan_cnn_2d", "diffusion_2d_cond", "gan_cnn_2d", "vqgan"}


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Answer package queries from a fixed set, and record who was asked."""
    from engiopt import checkpoint_store

    asked: list[str] = []

    def _list_packages(repo_id: str, problem_id: str | None = None) -> list[str]:
        algo = repo_id.split("engiopt-", 1)[1].replace("-", "_")
        asked.append(algo)
        return ["beams2d/seed_1"] if algo in PUBLISHED else []

    monkeypatch.setattr(checkpoint_store, "list_packages", _list_packages)
    return asked


def _listing(**overrides: object) -> evaluate.Args:
    return dataclasses.replace(evaluate.Args(), list_generators=True, **overrides)  # type: ignore[arg-type]


def test_only_generators_that_fit_the_problem_are_queried(fake_hub: list[str], capsys: pytest.CaptureFixture) -> None:
    """A repository holds nothing for a problem its model cannot serve, so do not ask."""
    evaluate._print_generators(_listing(check_availability=True))
    capsys.readouterr()

    fitting = evaluate._fitting_generator_names("beams2d")
    assert set(fake_hub) == fitting
    assert fitting < set(BUILTIN_GENERATORS), "beams2d must not fit every registered generator"


def test_an_incompatible_generator_is_not_reported_as_missing(
    fake_hub: list[str],
    capsys: pytest.CaptureFixture,
) -> None:
    """1D and 3D models must read as inapplicable, never as unpublished."""
    evaluate._print_generators(_listing(check_availability=True))
    out = capsys.readouterr().out

    assert "cgan_1d" in out, "every registered generator is still listed"
    incompatible = set(BUILTIN_GENERATORS) - evaluate._fitting_generator_names("beams2d")
    for name in incompatible:
        row = next(line for line in out.splitlines() if line.strip().startswith(name))
        assert "does not fit beams2d" in row
        assert "no published checkpoints" not in row


def test_the_missing_count_covers_only_generators_that_fit(
    fake_hub: list[str],
    capsys: pytest.CaptureFixture,
) -> None:
    """The number a reader takes away is the number of migrations actually owed."""
    evaluate._print_generators(_listing(check_availability=True))
    out = capsys.readouterr().out

    fitting = evaluate._fitting_generator_names("beams2d")
    owed = len(fitting - PUBLISHED)
    assert f"{owed} of the {len(fitting)} generators that fit beams2d have no published" in out
    assert owed < len(set(BUILTIN_GENERATORS) - PUBLISHED), "the whole point is that this is smaller"


def test_without_the_flag_nothing_touches_the_network(fake_hub: list[str], capsys: pytest.CaptureFixture) -> None:
    """`--list-generators` alone must stay usable offline."""
    evaluate._print_generators(_listing())
    out = capsys.readouterr().out

    assert fake_hub == []
    assert "--check-availability" in out
