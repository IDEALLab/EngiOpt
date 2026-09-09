"""Tests for what a checkpoint package's identity covers, and for partial packages.

Two properties are load-bearing for the leaderboard:

- A checkpoint's hash identifies its *weights*. Attaching a score to a package
  must not change the identity of the thing that was scored, or `--skip-existing`
  chases its own tail forever.
- A package written by a run that never finished must say so. Multi-stage models
  upload after each stage, so an interrupted run leaves something that has every
  file a loader looks for except the weights.
"""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from engiopt import checkpoint_store


def _write_package(
    root: Path,
    *,
    weights: bytes = b"weights",
    files: dict[str, bytes] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "generator.pth").write_bytes(weights)
    for name, payload in (files or {}).items():
        (root / name).write_bytes(payload)
    (root / "run_config.json").write_text(json.dumps({"latent_dim": 32}))
    (root / "metadata.json").write_text(json.dumps(metadata or {"algo": "demo", "seed": 1}))
    return root


# ----------------------------------------------------------------------
# Attaching metrics must not change what the weights hash to
# ----------------------------------------------------------------------


def test_attaching_metrics_does_not_change_the_checkpoint_hash(tmp_path: Path) -> None:
    """`publish_checkpoint_metrics` writes into the package it describes.

    If that file counted as package content, evaluating a checkpoint would change
    its hash, `--skip-existing` would see new weights, re-evaluate, rewrite the
    metrics, and change the hash again -- an evaluation loop with no fixed point.
    """
    package = _write_package(tmp_path / "pkg")
    before = checkpoint_store.hash_package_contents(str(package))

    (package / checkpoint_store.METRICS_FILE).write_text(json.dumps({"mmd": 0.04, "dpp": 1e-29}))
    after = checkpoint_store.hash_package_contents(str(package))

    assert before == after

    # And a second, different evaluation still does not move it.
    (package / checkpoint_store.METRICS_FILE).write_text(json.dumps({"mmd": 0.99}))
    assert checkpoint_store.hash_package_contents(str(package)) == before


def test_the_hash_still_covers_undeclared_weight_files(tmp_path: Path) -> None:
    """Excluding metrics must not weaken the guarantee that weights are covered."""
    package = _write_package(tmp_path / "pkg", files={"cvqgan.pth": b"conditional encoder"})
    before = checkpoint_store.hash_package_contents(str(package))

    (package / "cvqgan.pth").write_bytes(b"a different conditional encoder")

    assert checkpoint_store.hash_package_contents(str(package)) != before


def test_metrics_are_not_served_as_a_checkpoint_file(tmp_path: Path) -> None:
    """A package that has been evaluated must not offer `metrics.json` as a weight file."""
    package = _write_package(tmp_path / "pkg")
    (package / checkpoint_store.METRICS_FILE).write_text(json.dumps({"mmd": 0.04}))

    resolved = checkpoint_store._load_package_from_directory(root_dir=str(package), required_files=[], source="local")

    assert checkpoint_store.METRICS_FILE not in resolved.files


# ----------------------------------------------------------------------
# A package from an unfinished run has to say so
# ----------------------------------------------------------------------


def test_an_incomplete_package_explains_itself(tmp_path: Path) -> None:
    """The VQGAN case: stage 0 uploaded, the run died, the package looks loadable."""
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "cvqgan.pth").write_bytes(b"condition autoencoder only")
    (package / "run_config.json").write_text(json.dumps({"conditional": True}))
    (package / "metadata.json").write_text(json.dumps({"stage": "cvqgan", checkpoint_store.PACKAGE_COMPLETE_FIELD: False}))

    with pytest.raises(FileNotFoundError) as excinfo:
        checkpoint_store._load_package_from_directory(
            root_dir=str(package), required_files=["vqgan.pth", "transformer.pth"], source="local"
        )

    message = str(excinfo.value)
    assert "incomplete" in message
    assert "did not finish" in message
    assert "cvqgan" in message
    # It also says what the package does contain, so the reader can tell how far it got.
    assert "cvqgan.pth" in message


def test_a_complete_package_missing_a_file_reports_what_is_there(tmp_path: Path) -> None:
    """A finished run missing a file is a different problem and must not blame the run."""
    package = _write_package(tmp_path / "pkg")

    with pytest.raises(FileNotFoundError) as excinfo:
        checkpoint_store._load_package_from_directory(
            root_dir=str(package), required_files=["generator.pth", "absent.pth"], source="local"
        )

    message = str(excinfo.value)
    assert "absent.pth" in message
    assert "generator.pth" in message
    assert "incomplete" not in message


def test_an_incomplete_stage_never_claims_the_canonical_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The canonical path is what a bare model name resolves to; a partial run must not own it.

    Without this, a multi-stage default-hyperparameter run publishes the bare
    model name at stage 0, and it stays unloadable for as long as the remaining
    stages run -- hours, for VQGAN -- or permanently if the run dies before the
    last one, as 4 of the pool's 102 VQGAN runs did.
    """
    uploaded: list[str] = []

    def fake_upload(*, package_path: str, **_: Any) -> str:
        uploaded.append(package_path)
        return "deadbeef"

    monkeypatch.setattr(checkpoint_store, "_upload_package_to_hf", fake_upload)
    monkeypatch.setattr(checkpoint_store.wandb, "run", None)

    info = checkpoint_store.save_checkpoint_package(
        checkpoint_backend="hf",
        hf_entity="TestOrg",
        hf_repo_prefix="engiopt",
        hf_private=False,
        problem_id="beams2d",
        algo="vqgan",
        seed=42,
        checkpoint_files={"cvqgan.pth": "cvqgan.pth"},
        run_config={"latent_dim": 16},
        config_fingerprint="abc12345",
        is_default_config=True,
        package_complete=False,
    )

    assert uploaded == ["beams2d/cfg_abc12345/seed_42"]
    assert info["hf_package_path"] is None


def test_a_finished_default_run_still_claims_the_canonical_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The completeness guard must not cost ordinary single-stage models their canonical path."""
    uploaded: list[str] = []

    def fake_upload(*, package_path: str, **_: Any) -> str:
        uploaded.append(package_path)
        return "deadbeef"

    monkeypatch.setattr(checkpoint_store, "_upload_package_to_hf", fake_upload)
    monkeypatch.setattr(checkpoint_store.wandb, "run", None)

    checkpoint_store.save_checkpoint_package(
        checkpoint_backend="hf",
        hf_entity="TestOrg",
        hf_repo_prefix="engiopt",
        hf_private=False,
        problem_id="beams2d",
        algo="cgan_cnn_2d",
        seed=1,
        checkpoint_files={"generator.pth": "generator.pth"},
        run_config={"latent_dim": 16},
        config_fingerprint="abc12345",
        is_default_config=True,
    )

    assert uploaded == ["beams2d/cfg_abc12345/seed_1", "beams2d/seed_1"]


def test_metadata_records_completeness_for_ordinary_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single-stage model should not have to opt in to being considered finished."""
    written: list[dict[str, Any]] = []

    def fake_upload(*, metadata: dict[str, Any], **_: Any) -> str:
        written.append(metadata)
        return "deadbeef"

    monkeypatch.setattr(checkpoint_store, "_upload_package_to_hf", fake_upload)
    monkeypatch.setattr(checkpoint_store.wandb, "run", None)

    checkpoint_store.save_checkpoint_package(
        checkpoint_backend="hf",
        hf_entity="TestOrg",
        hf_repo_prefix="engiopt",
        hf_private=False,
        problem_id="beams2d",
        algo="cgan_cnn_2d",
        seed=1,
        checkpoint_files={"generator.pth": "generator.pth"},
        run_config={},
        is_default_config=True,
    )

    assert all(entry[checkpoint_store.PACKAGE_COMPLETE_FIELD] is True for entry in written)


# ----------------------------------------------------------------------
# The hash must cover everything that changes what the model generates
# ----------------------------------------------------------------------


def test_changing_fitted_preprocessing_changes_the_hash(tmp_path: Path) -> None:
    """Normalizer bounds are part of the model even though they never reach a state dict.

    Several 1D models fit min/max bounds on the training split and denormalize
    through them on the way out, so a package with identical weights and
    different bounds *decodes differently*. If the hash missed that, a
    behaviourally different checkpoint would inherit the previous one's
    leaderboard row via `--skip-existing`.
    """
    package = _write_package(
        tmp_path / "pkg",
        metadata={"algo": "demo", "design_normalizer": {"min": [0.0], "max": [1.0]}},
    )
    before = checkpoint_store.hash_package_contents(str(package))

    (package / "metadata.json").write_text(json.dumps({"algo": "demo", "design_normalizer": {"min": [0.0], "max": [2.0]}}))

    assert checkpoint_store.hash_package_contents(str(package)) != before


def test_changing_condition_statistics_changes_the_hash(tmp_path: Path) -> None:
    """The same argument for the models that rescale their conditions instead."""
    package = _write_package(tmp_path / "pkg", metadata={"condition_stats": {"mean": [0.5], "std": [0.1]}})
    before = checkpoint_store.hash_package_contents(str(package))

    (package / "metadata.json").write_text(json.dumps({"condition_stats": {"mean": [0.9], "std": [0.1]}}))

    assert checkpoint_store.hash_package_contents(str(package)) != before


def test_the_same_weights_at_two_addresses_hash_alike(tmp_path: Path) -> None:
    """One run writes its package to both a config path and the canonical path.

    Those differ in the metadata recording where they sit, and they must not
    therefore look like two different models -- the hash answers "which
    weights", not "which location".
    """
    shared = {"algo": "demo", "seed": 1, "condition_keys": ["volfrac"]}
    config_package = _write_package(
        tmp_path / "cfg", metadata={**shared, "hf_package_path": "beams2d/cfg_abc12345/seed_1", "hf_revision": "aaa"}
    )
    canonical_package = _write_package(
        tmp_path / "canonical", metadata={**shared, "hf_package_path": "beams2d/seed_1", "hf_revision": "bbb"}
    )

    assert checkpoint_store.hash_package_contents(str(config_package)) == checkpoint_store.hash_package_contents(
        str(canonical_package)
    )


def test_a_promotion_does_not_change_what_the_weights_hash_to(tmp_path: Path) -> None:
    """Promotion records where a package came from; it does not alter the model."""
    package = _write_package(tmp_path / "pkg", metadata={"algo": "demo"})
    before = checkpoint_store.hash_package_contents(str(package))

    (package / "metadata.json").write_text(json.dumps({"algo": "demo", "promoted_from": "beams2d/cfg_abc12345/seed_1"}))

    assert checkpoint_store.hash_package_contents(str(package)) == before


# ----------------------------------------------------------------------
# An incomplete package must never load, however complete it looks
# ----------------------------------------------------------------------


def test_an_incomplete_package_is_refused_even_with_every_file_present(tmp_path: Path) -> None:
    """Completeness is checked before file discovery, not while explaining a missing file.

    A stage that happens to have written everything the *next* stage's loader
    asks for would otherwise load successfully, and a model from a run that
    never finished would reach a leaderboard row with nothing to indicate it.
    """
    package = _write_package(tmp_path / "pkg", metadata={"stage": "vqgan", checkpoint_store.PACKAGE_COMPLETE_FIELD: False})

    with pytest.raises(FileNotFoundError) as excinfo:
        checkpoint_store._load_package_from_directory(
            root_dir=str(package), required_files=["generator.pth"], source="local"
        )

    assert "incomplete" in str(excinfo.value)
    assert "did not finish" in str(excinfo.value)


def test_a_complete_package_still_loads(tmp_path: Path) -> None:
    """The guard must not cost ordinary packages their ability to load."""
    package = _write_package(tmp_path / "pkg", metadata={checkpoint_store.PACKAGE_COMPLETE_FIELD: True})

    resolved = checkpoint_store._load_package_from_directory(
        root_dir=str(package), required_files=["generator.pth"], source="local"
    )

    assert "generator.pth" in resolved.files


# ----------------------------------------------------------------------
# An upload must not blend two runs at one path
# ----------------------------------------------------------------------


def test_a_complete_upload_clears_whatever_was_at_that_path() -> None:
    """`upload_folder` overwrites but does not delete, so a re-train leaves strays.

    Re-training a configuration whose previous run wrote *more* files leaves the
    extras in place, and discovery then serves a package mixing two runs'
    weights -- which loads, because every filename a loader looks for is there.
    """
    assert checkpoint_store._stale_file_patterns({checkpoint_store.PACKAGE_COMPLETE_FIELD: True}) == ["*"]
    # Absent marker means an ordinary single-stage model, which is complete.
    assert checkpoint_store._stale_file_patterns({}) == ["*"]


def test_a_staged_upload_leaves_the_earlier_stages_alone() -> None:
    """VQGAN's second stage uploads `vqgan.pth` without re-uploading `cvqgan.pth`.

    Deleting there would destroy the first stage's work -- the very thing staged
    uploads exist to protect.
    """
    assert checkpoint_store._stale_file_patterns({checkpoint_store.PACKAGE_COMPLETE_FIELD: False}) is None
