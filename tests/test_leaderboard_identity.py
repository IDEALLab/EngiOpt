"""Tests for what identifies a leaderboard row and what may be skipped.

`--skip-existing` decides whether work has already been done. Getting it wrong in
either direction is expensive: too strict re-runs a finished sweep, too lenient
silently keeps a stale score for weights that have since been retrained.
"""

from __future__ import annotations

import pandas as pd

from engiopt.evaluate import _fingerprints_for
from engiopt.evaluation.leaderboard import already_evaluated


def _board(**overrides: object) -> pd.DataFrame:
    row = {
        "problem_id": "beams2d",
        "algo_id": "cgan_cnn_2d",
        "config_fingerprint": "023dd1fb",
        "seed": 1,
        "spec_version": "v1",
        "checkpoint_hash": "aaaaaaaaaaaaaaaa",
    }
    row.update(overrides)
    return pd.DataFrame([row])


_KEY = {
    "problem_id": "beams2d",
    "algo_id": "cgan_cnn_2d",
    "config_fingerprint": "023dd1fb",
    "seed": 1,
    "spec_version": "v1",
}


def test_the_same_weights_are_skipped() -> None:
    """The point of the flag: a finished sweep must not re-run."""
    assert already_evaluated(_board(), **_KEY, checkpoint_hash="aaaaaaaaaaaaaaaa")


def test_retrained_weights_are_not_skipped() -> None:
    """Same config and seed, new training run, different weights: score them."""
    assert not already_evaluated(_board(), **_KEY, checkpoint_hash="bbbbbbbbbbbbbbbb")


def test_a_row_without_a_hash_does_not_count_as_a_match() -> None:
    """Unknown identity is not the same as equal identity.

    Treating a hashless row as a match lets one such row suppress every future
    evaluation of that configuration and seed, including genuinely new weights,
    until somebody deletes it by hand.
    """
    assert not already_evaluated(_board(checkpoint_hash=None), **_KEY, checkpoint_hash="aaaaaaaaaaaaaaaa")


def test_a_different_seed_is_not_skipped() -> None:
    """Sanity: the ordinary key columns still discriminate."""
    assert not already_evaluated(_board(), **{**_KEY, "seed": 2}, checkpoint_hash="aaaaaaaaaaaaaaaa")


def test_an_empty_board_skips_nothing() -> None:
    assert not already_evaluated(pd.DataFrame(), **_KEY, checkpoint_hash="aaaaaaaaaaaaaaaa")


def test_a_board_predating_the_hash_column_still_matches_on_the_rest() -> None:
    """A board with no hash column at all is a schema difference, not a mismatch."""
    board = _board().drop(columns=["checkpoint_hash"])

    assert already_evaluated(board, **_KEY, checkpoint_hash="aaaaaaaaaaaaaaaa")


# ----------------------------------------------------------------------
# Config fingerprints belong to one algorithm
# ----------------------------------------------------------------------


def test_a_bare_fingerprint_applies_to_every_model() -> None:
    """Single-model runs are the common case and should not need scoping."""
    assert _fingerprints_for(("023dd1fb",), "cgan_cnn_2d") == ("023dd1fb",)
    assert _fingerprints_for(("023dd1fb",), "gan_cnn_2d") == ("023dd1fb",)


def test_a_scoped_fingerprint_reaches_only_its_own_model() -> None:
    """Otherwise a two-model run asks for packages that were never going to exist."""
    requested = ("cgan_cnn_2d:023dd1fb", "gan_cnn_2d:06d9a9a1")

    assert _fingerprints_for(requested, "cgan_cnn_2d") == ("023dd1fb",)
    assert _fingerprints_for(requested, "gan_cnn_2d") == ("06d9a9a1",)


def test_scoped_and_bare_entries_mix() -> None:
    """A shared fingerprint alongside a scoped one is a reasonable thing to ask for."""
    requested = ("cgan_cnn_2d:023dd1fb", "deadbeef")

    assert _fingerprints_for(requested, "cgan_cnn_2d") == ("023dd1fb", "deadbeef")
    assert _fingerprints_for(requested, "gan_cnn_2d") == ("deadbeef",)


def test_no_fingerprints_means_the_canonical_checkpoint() -> None:
    """`None` is what `from_pretrained` reads as "the default-hyperparameter package"."""
    assert _fingerprints_for((), "cgan_cnn_2d") == (None,)
