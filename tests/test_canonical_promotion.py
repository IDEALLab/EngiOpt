"""Tests for reaching the canonical checkpoint path after a sweep.

`{problem_id}/seed_{seed}` is what a bare model name resolves to. Only a run
using the training script's default hyperparameters writes it, and a sweep has no
such run -- every arm varies something. So a 500-run sweep can publish hundreds of
packages and leave the documented `--seeds 1` command resolving nothing, which is
exactly the state the IDEALLab repos were in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from engiopt import checkpoint_store


def _package(root: Path, *, metadata: dict[str, Any] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "generator.pth").write_bytes(b"weights")
    (root / "run_config.json").write_text(json.dumps({"latent_dim": 32}))
    (root / "metadata.json").write_text(json.dumps(metadata or {"algo": "cgan_cnn_2d", "seed": 1}))
    return root


def test_a_missing_canonical_package_names_the_ones_that_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    """ "Not found" alone sends the reader looking for a broken path.

    The actual situation is that the canonical package was never written, and the
    configurations that *were* published are sitting right there. Say so.
    """
    monkeypatch.setattr(
        checkpoint_store,
        "list_packages",
        lambda _repo, _problem=None: ["beams2d/cfg_023dd1fb/seed_1", "beams2d/cfg_1763fb14/seed_1"],
    )

    hint = checkpoint_store._sibling_packages_hint("IDEALLab/engiopt-cgan-cnn-2d", "beams2d", 1, None)

    assert "cfg_023dd1fb" in hint
    assert "canonical beams2d/seed_1" in hint
    assert "--config-fingerprints" in hint
    assert "promote_checkpoint" in hint


def test_the_hint_is_quiet_when_a_configuration_was_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """Asking for a specific fingerprint that is absent is a different mistake."""
    monkeypatch.setattr(checkpoint_store, "list_packages", lambda _repo, _problem=None: ["beams2d/cfg_023dd1fb/seed_1"])

    hint = checkpoint_store._sibling_packages_hint("IDEALLab/engiopt-cgan-cnn-2d", "beams2d", 1, ["cfg_deadbeef"])

    assert "cfg_023dd1fb" in hint
    assert "promote_checkpoint" not in hint


def test_an_empty_repo_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checkpoint_store, "list_packages", lambda _repo, _problem=None: [])

    hint = checkpoint_store._sibling_packages_hint("IDEALLab/engiopt-vqgan", "photonics2d", 1, None)

    assert "No packages for 'photonics2d'" in hint


def test_the_hint_never_replaces_the_real_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hint that itself fails must not mask what the caller was actually told."""

    def explode(_repo: str, _problem: str | None = None) -> list[str]:
        raise ConnectionError("the Hub is down")

    monkeypatch.setattr(checkpoint_store, "list_packages", explode)

    assert checkpoint_store._sibling_packages_hint("repo", "beams2d", 1, None) == ""


# ----------------------------------------------------------------------
# Promotion
# ----------------------------------------------------------------------


def test_promotion_copies_a_configuration_to_the_canonical_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The escape hatch: make an already-trained arm the default without retraining."""
    source = _package(
        tmp_path / "snapshot" / "beams2d" / "cfg_023dd1fb" / "seed_1",
        metadata={"algo": "cgan_cnn_2d", "seed": 1, "config_fingerprint": "023dd1fb"},
    )
    (source / checkpoint_store.METRICS_FILE).write_text(json.dumps({"mmd": 0.04}))

    uploads: list[dict[str, Any]] = []

    class _Api:
        def __init__(self, **_: Any) -> None:
            pass

        def upload_folder(self, **kwargs: Any) -> None:
            staged = sorted(entry.name for entry in Path(kwargs["folder_path"]).iterdir())
            uploads.append({"path_in_repo": kwargs["path_in_repo"], "files": staged})

    monkeypatch.setattr(checkpoint_store, "HfApi", _Api)
    monkeypatch.setattr(checkpoint_store, "snapshot_download", lambda **_: str(tmp_path / "snapshot"))

    canonical = checkpoint_store.promote_to_canonical(
        hf_entity="IDEALLab",
        hf_repo_prefix="engiopt",
        problem_id="beams2d",
        algo="cgan_cnn_2d",
        seed=1,
        config_fingerprint="023dd1fb",
    )

    assert canonical == "beams2d/seed_1"
    assert uploads[0]["path_in_repo"] == "beams2d/seed_1"
    assert "generator.pth" in uploads[0]["files"]
    # The old address's scores do not describe the promoted copy.
    assert checkpoint_store.METRICS_FILE not in uploads[0]["files"]


def test_promotion_records_where_it_came_from(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A leaderboard row must still be traceable to the configuration that earned it."""
    _package(
        tmp_path / "snapshot" / "beams2d" / "cfg_023dd1fb" / "seed_1",
        metadata={"algo": "cgan_cnn_2d", "seed": 1, "config_fingerprint": "023dd1fb"},
    )
    written: dict[str, Any] = {}

    class _Api:
        def __init__(self, **_: Any) -> None:
            pass

        def upload_folder(self, **kwargs: Any) -> None:
            written.update(json.loads((Path(kwargs["folder_path"]) / "metadata.json").read_text()))

    monkeypatch.setattr(checkpoint_store, "HfApi", _Api)
    monkeypatch.setattr(checkpoint_store, "snapshot_download", lambda **_: str(tmp_path / "snapshot"))

    checkpoint_store.promote_to_canonical(
        hf_entity="IDEALLab",
        hf_repo_prefix="engiopt",
        problem_id="beams2d",
        algo="cgan_cnn_2d",
        seed=1,
        config_fingerprint="023dd1fb",
    )

    assert written["config_fingerprint"] == "023dd1fb"
    assert written["promoted_from"] == "beams2d/cfg_023dd1fb/seed_1"
    assert written["hf_package_path"] == "beams2d/seed_1"


def test_an_incomplete_package_is_never_promoted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Promoting a torn package would put an unloadable thing at the canonical path.

    That is precisely the state VQGAN was already in; promotion must not be a
    second way to reach it.
    """
    _package(
        tmp_path / "snapshot" / "beams2d" / "cfg_1151406c" / "seed_42",
        metadata={"stage": "cvqgan", checkpoint_store.PACKAGE_COMPLETE_FIELD: False},
    )

    monkeypatch.setattr(checkpoint_store, "HfApi", lambda **_: None)
    monkeypatch.setattr(checkpoint_store, "snapshot_download", lambda **_: str(tmp_path / "snapshot"))

    with pytest.raises(FileNotFoundError, match="marked incomplete"):
        checkpoint_store.promote_to_canonical(
            hf_entity="IDEALLab",
            hf_repo_prefix="engiopt",
            problem_id="beams2d",
            algo="vqgan",
            seed=42,
            config_fingerprint="1151406c",
        )


def test_promoting_an_absent_configuration_says_so(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(checkpoint_store, "HfApi", lambda **_: None)
    monkeypatch.setattr(checkpoint_store, "snapshot_download", lambda **_: str(tmp_path / "snapshot"))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        checkpoint_store.promote_to_canonical(
            hf_entity="IDEALLab",
            hf_repo_prefix="engiopt",
            problem_id="beams2d",
            algo="cgan_cnn_2d",
            seed=1,
            config_fingerprint="absent00",
        )
