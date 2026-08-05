"""Tests for reading and writing checkpoint packages.

A package is the weight files plus `run_config.json` and `metadata.json`. What
matters here is that a package can be loaded from a plain directory (no Hub
account required to try a model), that its metadata describes the package it is
actually in, and that a loaded package identifies the exact weights it holds.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch as th

from engiopt import checkpoint_store
from engiopt.core import condition_keys_for
from engiopt.core import Generator


def _write_package(root: Path, *, weights: bytes = b"weights", metadata: dict[str, Any] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "generator.pth").write_bytes(weights)
    (root / "run_config.json").write_text(json.dumps({"latent_dim": 32}))
    (root / "metadata.json").write_text(json.dumps(metadata or {"algo": "demo", "seed": 1}))
    return root


# ----------------------------------------------------------------------
# Loading from a local directory
# ----------------------------------------------------------------------


def test_local_package_loads_without_the_hub(tmp_path: Path) -> None:
    """`--model-source local` has to be reachable, not just advertised."""
    package = _write_package(tmp_path / "pkg")

    resolved = checkpoint_store.resolve_named_checkpoint(
        model_source="local",
        problem_id="beams2d",
        algo="demo",
        seed=1,
        hf_entity="none",
        hf_repo_prefix="none",
        required_files=["generator.pth"],
        local_model_dir=str(package),
    )

    assert resolved.source == "local"
    assert resolved.run_config == {"latent_dim": 32}
    assert Path(resolved.files["generator.pth"]).read_bytes() == b"weights"


def test_from_pretrained_passes_the_local_directory_through() -> None:
    """The public API must accept the directory `resolve_checkpoint` needs."""
    assert "local_model_dir" in inspect.signature(Generator.from_pretrained).parameters
    assert "local_model_dir" in inspect.signature(Generator.resolve_checkpoint).parameters


def test_the_cli_exposes_the_local_directory() -> None:
    """An option nobody can point at a directory is not a usable source."""
    from engiopt.evaluate import Args

    assert "local_model_dir" in set(Args.__dataclass_fields__)


def test_a_missing_local_directory_says_so(tmp_path: Path) -> None:
    """A clear error beats a confusing failure deeper in loading."""
    with pytest.raises(FileNotFoundError, match="Local checkpoint directory not found"):
        checkpoint_store.resolve_named_checkpoint(
            model_source="local",
            problem_id="beams2d",
            algo="demo",
            seed=1,
            hf_entity="none",
            hf_repo_prefix="none",
            required_files=["generator.pth"],
            local_model_dir=str(tmp_path / "absent"),
        )


# ----------------------------------------------------------------------
# Identifying the exact weights
# ----------------------------------------------------------------------


def test_package_records_a_content_hash(tmp_path: Path) -> None:
    """Config and seed name a *configuration*; only a content hash names the weights."""
    first = checkpoint_store._load_package_from_directory(
        root_dir=str(_write_package(tmp_path / "a", weights=b"one")), required_files=["generator.pth"], source="local"
    )
    second = checkpoint_store._load_package_from_directory(
        root_dir=str(_write_package(tmp_path / "b", weights=b"two")), required_files=["generator.pth"], source="local"
    )
    same = checkpoint_store._load_package_from_directory(
        root_dir=str(_write_package(tmp_path / "c", weights=b"one")), required_files=["generator.pth"], source="local"
    )

    assert first.content_hash != second.content_hash
    assert first.content_hash == same.content_hash


def test_retrained_weights_get_a_new_hash_under_the_same_name(tmp_path: Path) -> None:
    """Re-training the same config and seed replaces a leaderboard row; the hash says so."""
    package = _write_package(tmp_path / "pkg", weights=b"first training")
    before = checkpoint_store._load_package_from_directory(
        root_dir=str(package), required_files=["generator.pth"], source="local"
    )
    (package / "generator.pth").write_bytes(b"second training")
    after = checkpoint_store._load_package_from_directory(
        root_dir=str(package), required_files=["generator.pth"], source="local"
    )
    assert before.content_hash != after.content_hash


def test_evaluation_rows_carry_the_checkpoint_identity() -> None:
    """Provenance has to reach the leaderboard, not just the loader."""
    from engiopt.evaluation.evaluator import PROVENANCE_COLUMNS

    assert {"checkpoint_revision", "checkpoint_hash", "code_version"} <= set(PROVENANCE_COLUMNS)


# ----------------------------------------------------------------------
# Metadata describes the package it sits in
# ----------------------------------------------------------------------


def test_each_uploaded_package_describes_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-default run writes only the config path, so its metadata must name that path.

    Sharing one metadata object made a config-only run's `metadata.json` point
    at a canonical package it never uploaded.
    """
    uploads: list[dict[str, Any]] = []

    def fake_upload(**kwargs: Any) -> str:
        uploads.append({"path": kwargs["package_path"], "metadata": kwargs["metadata"]})
        return "sha"

    monkeypatch.setattr(checkpoint_store, "_upload_package_to_hf", fake_upload)
    monkeypatch.setattr(checkpoint_store.wandb, "run", None)

    info = checkpoint_store.save_checkpoint_package(
        checkpoint_backend="hf",
        hf_entity="org",
        hf_repo_prefix="engiopt",
        hf_private=False,
        problem_id="beams2d",
        algo="demo",
        seed=1,
        checkpoint_files={},
        run_config={"latent_dim": 64},
        config_fingerprint="abc123",
        is_default_config=False,
    )

    assert len(uploads) == 1
    assert uploads[0]["path"] == "beams2d/cfg_abc123/seed_1"
    assert uploads[0]["metadata"]["hf_package_path"] == "beams2d/cfg_abc123/seed_1"
    assert info["hf_package_path"] is None


def test_a_default_run_claims_both_paths_with_matching_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each of the two packages describes itself, not the other."""
    uploads: list[dict[str, Any]] = []

    def fake_upload(**kwargs: Any) -> str:
        uploads.append({"path": kwargs["package_path"], "metadata": kwargs["metadata"]})
        return "sha"

    monkeypatch.setattr(checkpoint_store, "_upload_package_to_hf", fake_upload)
    monkeypatch.setattr(checkpoint_store.wandb, "run", None)

    checkpoint_store.save_checkpoint_package(
        checkpoint_backend="hf",
        hf_entity="org",
        hf_repo_prefix="engiopt",
        hf_private=False,
        problem_id="beams2d",
        algo="demo",
        seed=1,
        checkpoint_files={},
        run_config={},
        config_fingerprint="abc123",
        is_default_config=True,
    )

    assert [upload["path"] for upload in uploads] == ["beams2d/cfg_abc123/seed_1", "beams2d/seed_1"]
    for upload in uploads:
        assert upload["metadata"]["hf_package_path"] == upload["path"]


def test_condition_schema_travels_with_the_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A model is rebuilt for the conditions it saw, not the problem's current list."""
    uploads: list[dict[str, Any]] = []

    def fake_upload(**kwargs: Any) -> str:
        uploads.append(kwargs["metadata"])
        return "sha"

    monkeypatch.setattr(checkpoint_store, "_upload_package_to_hf", fake_upload)
    monkeypatch.setattr(checkpoint_store.wandb, "run", None)

    checkpoint_store.save_checkpoint_package(
        checkpoint_backend="hf",
        hf_entity="org",
        hf_repo_prefix="engiopt",
        hf_private=False,
        problem_id="thermoelastic2d",
        algo="demo",
        seed=1,
        checkpoint_files={},
        run_config={},
        condition_keys=["volume_fraction_target", "rmin", "weight"],
    )

    assert uploads[0]["condition_keys"] == ["volume_fraction_target", "rmin", "weight"]


def test_recorded_schema_survives_the_problem_gaining_a_condition(tmp_path: Path) -> None:
    """An old checkpoint keeps loading when the problem later adds a condition."""
    package = _write_package(tmp_path / "pkg", metadata={"condition_keys": ["volfrac", "rmin"]})
    resolved = checkpoint_store._load_package_from_directory(
        root_dir=str(package), required_files=["generator.pth"], source="local"
    )

    class _Problem:
        conditions_keys: ClassVar[list[str]] = ["volfrac", "rmin", "brand_new"]

    assert condition_keys_for(_Problem(), resolved) == ("volfrac", "rmin")


# ----------------------------------------------------------------------
# Sampling cost is measured, not dispatched
# ----------------------------------------------------------------------


def test_sample_timing_waits_for_the_device(fake_problem: Any) -> None:
    """CUDA and MPS queue work asynchronously; the timer must not stop mid-flight."""
    synchronized: list[bool] = []

    class _Timed(Generator):
        algo_id = "timed"
        conditional = False

        @classmethod
        def build(cls, *_args: Any, **_kwargs: Any) -> _Timed:
            raise NotImplementedError

        def _synchronize_device(self) -> None:
            synchronized.append(True)

        def _sample(self, conditions: Any, n: int) -> th.Tensor:
            return th.zeros((n, *self.design_shape))

    generator = _Timed(problem=fake_problem, problem_id="p", seed=1, device=th.device("cpu"))
    generator.sample(None, n=2)

    assert synchronized == [True]
    assert generator.last_sample_seconds is not None
