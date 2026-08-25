from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

import engiopt.publish_selected_checkpoint as publisher
from engiopt.publish_selected_checkpoint import Args

if TYPE_CHECKING:
    from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recovers_identical_remote_selected_checkpoint(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_bytes(b"selected checkpoint")
    checksum = _sha256(checkpoint_path)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"selected_checkpoint_sha256": checksum}), encoding="utf-8")

    def fake_download(*, filename, **_kwargs):
        return str(metadata_path if filename.endswith("metadata.json") else checkpoint_path)

    monkeypatch.setattr(publisher, "hf_hub_download", fake_download)
    monkeypatch.setattr(
        publisher,
        "HfApi",
        lambda: SimpleNamespace(repo_info=lambda **_kwargs: SimpleNamespace(sha="abc123")),
    )

    info = publisher._recover_existing_remote_package(  # noqa: SLF001
        args=Args(bundle_dir="unused", hf_entity="IDEALLab", hf_repo_prefix="engiopt-engopt2026"),
        metadata={
            "algo": "flow_matching_2d_cond",
            "problem_id": "beams2d",
            "seed": 1,
            "release": "v1",
            "package_label": "euler_16",
        },
        selected_filename="model.pth",
        expected_sha256=checksum,
    )

    assert info is not None
    assert info["hf_revision"] == "abc123"
    assert info["recovered_existing_upload"] is True
    assert info["hf_package_path"] == "beams2d/selected/v1/euler_16/seed_1"


def test_rejects_conflicting_remote_selected_checkpoint(tmp_path, monkeypatch):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"selected_checkpoint_sha256": "different"}), encoding="utf-8")
    monkeypatch.setattr(publisher, "hf_hub_download", lambda **_kwargs: str(metadata_path))

    with pytest.raises(FileExistsError, match="new release label"):
        publisher._recover_existing_remote_package(  # noqa: SLF001
            args=Args(bundle_dir="unused", hf_entity="IDEALLab", hf_repo_prefix="engiopt-engopt2026"),
            metadata={
                "algo": "diffusion_2d_cond",
                "problem_id": "beams2d",
                "seed": 1,
                "release": "v1",
                "package_label": "timesteps_1000",
            },
            selected_filename="model.pth",
            expected_sha256="expected",
        )
