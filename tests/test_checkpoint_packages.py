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
# Saving to a local directory
# ----------------------------------------------------------------------


def _save_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, files: tuple[str, ...] = ("generator.pth",), **overrides: Any
) -> dict[str, Any]:
    """Save a package with the local backend, from `tmp_path`, with the Hub made unreachable."""

    def no_hub(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a local save must not touch the Hub")

    monkeypatch.setattr(checkpoint_store, "HfApi", no_hub)
    monkeypatch.chdir(tmp_path)
    for name in files:
        (tmp_path / name).write_bytes(name.encode())
    arguments: dict[str, Any] = {
        "checkpoint_backend": "local",
        "hf_entity": "nobody",
        "hf_repo_prefix": "engiopt",
        "hf_private": False,
        "problem_id": "beams2d",
        "algo": "demo",
        "seed": 1,
        "checkpoint_files": {name: str(tmp_path / name) for name in files},
        "run_config": {"latent_dim": 32},
    }
    return checkpoint_store.save_checkpoint_package(**{**arguments, **overrides})


def _load_locally(*, required_files: tuple[str, ...] = ("generator.pth",)) -> checkpoint_store.ResolvedCheckpoint:
    """Load the `demo` package by name, with no directory given."""
    return checkpoint_store.resolve_named_checkpoint(
        model_source="local",
        problem_id="beams2d",
        algo="demo",
        seed=1,
        hf_entity="nobody",
        hf_repo_prefix="engiopt",
        required_files=list(required_files),
    )


def test_a_local_save_is_found_by_a_local_load_with_no_path_typed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Training with the local backend and loading with the local source must meet without a directory argument."""
    info = _save_locally(tmp_path, monkeypatch, metadata={"n_train_steps": 200}, condition_keys=("volfrac",))

    resolved = _load_locally()

    assert Path(info["local_package_dir"]) == Path("local_checkpoints/demo/beams2d/seed_1")
    assert resolved.source == "local"
    assert resolved.run_config == {"latent_dim": 32}
    assert Path(resolved.files["generator.pth"]).read_bytes() == b"generator.pth"
    assert resolved.metadata["n_train_steps"] == 200, "what the training script recorded travels with a local package too"
    assert resolved.metadata["condition_keys"] == ["volfrac"]
    assert resolved.content_hash


def test_a_local_save_leaves_the_published_address_blank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A row scored from it must be refused by the board, so it must not look as if it were published."""
    info = _save_locally(tmp_path, monkeypatch)
    resolved = _load_locally()

    assert info["hf_repo_id"] is None
    assert info["hf_package_path"] is None
    assert resolved.repo_id is None
    assert resolved.revision is None


def test_a_second_local_save_replaces_the_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A rerun that saves fewer files must not load a mixture of two runs' weights."""
    _save_locally(tmp_path, monkeypatch, files=("generator.pth", "discriminator.pth"))
    _save_locally(tmp_path, monkeypatch, files=("generator.pth",))

    assert sorted(path.name for path in Path("local_checkpoints/demo/beams2d/seed_1").iterdir()) == [
        "generator.pth",
        "metadata.json",
        "run_config.json",
    ]


def test_a_local_save_refuses_a_directory_that_is_not_a_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replacing a package deletes what was there, so only a package may be replaced."""
    notes = tmp_path / "thesis"
    notes.mkdir()
    (notes / "chapter1.tex").write_text("irreplaceable")

    with pytest.raises(ValueError, match="not a checkpoint package"):
        _save_locally(tmp_path, monkeypatch, local_dir=str(notes))

    assert (notes / "chapter1.tex").read_text() == "irreplaceable"


def test_a_local_save_survives_its_weights_already_being_in_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saving over a package from inside it must not delete the file it is about to copy."""
    package = _write_package(tmp_path / "pkg", weights=b"retrained")

    _save_locally(
        tmp_path,
        monkeypatch,
        files=(),
        checkpoint_files={"generator.pth": str(package / "generator.pth")},
        local_dir=str(package),
    )

    assert (package / "generator.pth").read_bytes() == b"retrained"


def test_an_unfinished_local_stage_adds_to_the_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A multi-stage model saves stage by stage, and a later stage must not delete an earlier one's file."""
    _save_locally(tmp_path, monkeypatch, files=("stage_one.pth",), package_complete=False)
    _save_locally(tmp_path, monkeypatch, files=("stage_two.pth",), package_complete=False)

    present = {path.name for path in Path("local_checkpoints/demo/beams2d/seed_1").iterdir()}
    assert {"stage_one.pth", "stage_two.pth"} <= present


def test_the_template_trains_saves_and_loads_without_an_account(
    fake_problem: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The newcomer's first hour: save the template's network locally, load it by name, generate designs."""
    from engiopt.generators._template.adapter import MyModel
    from engiopt.generators._template.my_model import Generator as TemplateNet

    keys = ("volfrac", "rmin")
    trained = TemplateNet(latent_dim=8, n_conds=len(keys), design_shape=fake_problem.design_space.shape)
    monkeypatch.chdir(tmp_path)
    th.save({"generator": trained.state_dict()}, "generator.pth")
    _save_locally(
        tmp_path,
        monkeypatch,
        files=(),
        algo="my_model",
        problem_id="fake",
        checkpoint_files={"generator.pth": "generator.pth"},
        run_config={"latent_dim": 8},
        condition_keys=keys,
    )

    loaded = MyModel.from_pretrained(fake_problem, problem_id="fake", device=th.device("cpu"), model_source="local")

    for name, weights in trained.state_dict().items():
        assert th.equal(loaded.net.state_dict()[name], weights), name
    assert loaded.sample(th.full((4, len(keys)), 0.5), n=4, seed=0).shape == (4, *fake_problem.design_space.shape)


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
        condition_keys=["volfrac", "rmin", "weight"],
    )

    assert uploads[0]["condition_keys"] == ["volfrac", "rmin", "weight"]


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


def test_the_hash_covers_files_the_model_did_not_declare(tmp_path: Path) -> None:
    """VQGAN's conditional variant loads `cvqgan.pth`, which is not in `checkpoint_files`.

    Hashing only the declared list would give two packages with different
    condition encoders the same identity, even though they generate differently.
    """
    package = _write_package(tmp_path / "pkg")
    (package / "cvqgan.pth").write_bytes(b"encoder one")
    before = checkpoint_store._load_package_from_directory(
        root_dir=str(package), required_files=["generator.pth"], source="local"
    )

    (package / "cvqgan.pth").write_bytes(b"encoder two")
    after = checkpoint_store._load_package_from_directory(
        root_dir=str(package), required_files=["generator.pth"], source="local"
    )

    assert before.content_hash != after.content_hash


def test_the_hash_ignores_the_files_describing_the_package(tmp_path: Path) -> None:
    """`metadata.json` records the path and revision, which differ between the two
    locations one run writes -- so the same weights must not hash differently."""
    package = _write_package(tmp_path / "pkg", metadata={"hf_package_path": "beams2d/seed_1"})
    canonical = checkpoint_store._load_package_from_directory(
        root_dir=str(package), required_files=["generator.pth"], source="local"
    )

    (package / "metadata.json").write_text(json.dumps({"hf_package_path": "beams2d/cfg_abc/seed_1"}))
    config_scoped = checkpoint_store._load_package_from_directory(
        root_dir=str(package), required_files=["generator.pth"], source="local"
    )

    assert canonical.content_hash == config_scoped.content_hash


def test_condition_statistics_travel_with_the_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that rescales its conditions must record the scale it used."""
    uploads: list[dict[str, Any]] = []
    monkeypatch.setattr(
        checkpoint_store, "_upload_package_to_hf", lambda **kwargs: uploads.append(kwargs["metadata"]) or "sha"
    )
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
        condition_keys=["volfrac"],
        condition_stats={"mean": [0.4], "std": [0.2]},
    )

    assert uploads[0]["condition_stats"] == {"mean": [0.4], "std": [0.2]}
