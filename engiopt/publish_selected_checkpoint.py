"""Publish a staged selected-checkpoint bundle to HF and update its W&B run."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError
from huggingface_hub.utils import RepositoryNotFoundError
import tyro
import wandb

from engiopt.checkpoint_store import build_hf_package_path
from engiopt.checkpoint_store import build_hf_repo_id
from engiopt.checkpoint_store import save_checkpoint_package
from engiopt.selected_checkpoint_bundle import selected_extra_path_parts


@dataclass
class Args:
    """Command-line arguments for publishing one staged bundle."""

    bundle_dir: str
    """Directory containing metadata.json, run_config.json, and selected checkpoint files."""
    hf_entity: str
    """HF organization or user that owns the model repository."""
    hf_repo_prefix: str = "engiopt"
    """HF repository prefix."""
    hf_private: bool = False
    """Create/use a private HF repository."""
    force: bool = False
    """Upload again even if a local upload receipt already exists."""


def publish_staged_bundle(args: Args) -> dict[str, Any]:
    """Validate and publish one staged selected-checkpoint package."""
    bundle_dir = Path(args.bundle_dir)
    metadata = _read_json(bundle_dir / "metadata.json")
    run_config = _read_json(bundle_dir / "run_config.json")
    if metadata.get("bundle_type") != "selected_checkpoint":
        raise ValueError(f"Not a selected-checkpoint bundle: {bundle_dir}")

    receipt_path = bundle_dir / "upload_receipt.json"
    if receipt_path.exists() and not args.force:
        info = _read_json(receipt_path)
        _update_wandb_summary(metadata, info)
        print(f"Already uploaded: {info.get('hf_model_ref')} @ {info.get('hf_revision')}")
        return info

    checkpoint_filenames = [str(name) for name in metadata["checkpoint_files"]]
    checkpoint_files = {name: str(bundle_dir / name) for name in checkpoint_filenames}
    for name, path in checkpoint_files.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing staged file {name}: {path}")

    selected_filename = str(metadata["selected_checkpoint_filename"])
    expected_sha256 = str(metadata["selected_checkpoint_sha256"])
    actual_sha256 = _sha256(Path(checkpoint_files[selected_filename]))
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Checksum mismatch for {selected_filename}: expected {expected_sha256}, got {actual_sha256}"
        )

    existing_info = _recover_existing_remote_package(
        args=args,
        metadata=metadata,
        selected_filename=selected_filename,
        expected_sha256=actual_sha256,
    )
    if existing_info is not None and not args.force:
        existing_info["bundle_dir"] = str(bundle_dir)
        _write_json(receipt_path, existing_info)
        _update_wandb_summary(metadata, existing_info)
        print(
            "Recovered existing upload: "
            f"{existing_info.get('hf_model_ref')} @ {existing_info.get('hf_revision')}"
        )
        return existing_info

    info = save_checkpoint_package(
        checkpoint_backend="hf",
        hf_entity=args.hf_entity,
        hf_repo_prefix=args.hf_repo_prefix,
        hf_private=args.hf_private,
        problem_id=str(metadata["problem_id"]),
        algo=str(metadata["algo"]),
        seed=int(metadata["seed"]),
        checkpoint_files=checkpoint_files,
        run_config=run_config,
        metadata=metadata,
        primary_files=[str(name) for name in metadata["primary_files"]],
        extra_path_parts=selected_extra_path_parts(
            str(metadata["release"]),
            str(metadata["package_label"]) if metadata.get("package_label") else None,
        ),
        upload_run_copy=False,
    )
    info.update(
        {
            "bundle_dir": str(bundle_dir),
            "selected_checkpoint_sha256": actual_sha256,
        }
    )
    _write_json(receipt_path, info)
    _update_wandb_summary(metadata, info)
    print(f"Uploaded: {info.get('hf_model_ref')} @ {info.get('hf_revision')}")
    return info


def _recover_existing_remote_package(
    *,
    args: Args,
    metadata: dict[str, Any],
    selected_filename: str,
    expected_sha256: str,
) -> dict[str, Any] | None:
    """Recover an identical remote package or reject a conflicting release path."""
    repo_id = build_hf_repo_id(args.hf_entity, args.hf_repo_prefix, str(metadata["algo"]))
    package_path = build_hf_package_path(
        str(metadata["problem_id"]),
        int(metadata["seed"]),
        selected_extra_path_parts(
            str(metadata["release"]),
            str(metadata["package_label"]) if metadata.get("package_label") else None,
        ),
    )
    metadata_filename = f"{package_path}/metadata.json"
    try:
        remote_metadata_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=metadata_filename,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None

    remote_metadata = _read_json(Path(remote_metadata_path))
    remote_expected_sha256 = str(remote_metadata.get("selected_checkpoint_sha256", ""))
    if remote_expected_sha256 != expected_sha256:
        raise FileExistsError(
            f"HF release path already exists with different content: {repo_id}/{package_path}. "
            "Use a new release label instead of overwriting it."
        )

    remote_checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=f"{package_path}/{selected_filename}",
    )
    remote_actual_sha256 = _sha256(Path(remote_checkpoint_path))
    if remote_actual_sha256 != expected_sha256:
        raise ValueError(
            f"Remote checkpoint checksum mismatch at {repo_id}/{package_path}/{selected_filename}: "
            f"expected {expected_sha256}, got {remote_actual_sha256}"
        )

    revision = HfApi().repo_info(repo_id=repo_id, repo_type="model").sha
    return {
        "checkpoint_backend": "hf",
        "hf_repo_id": repo_id,
        "hf_package_path": package_path,
        "hf_model_ref": f"hf://{repo_id}/{package_path}",
        "hf_run_package_path": None,
        "hf_revision": revision,
        "hf_run_revision": None,
        "selected_checkpoint_sha256": expected_sha256,
        "recovered_existing_upload": True,
    }


def _update_wandb_summary(metadata: dict[str, Any], info: dict[str, Any]) -> None:
    _update_one_wandb_run(metadata, info)
    training_wandb = metadata.get("training_wandb")
    if isinstance(training_wandb, dict):
        _update_one_wandb_run(training_wandb, info)


def _update_one_wandb_run(run_metadata: dict[str, Any], info: dict[str, Any]) -> None:
    entity = run_metadata.get("wandb_entity")
    project = run_metadata.get("wandb_project")
    run_id = run_metadata.get("wandb_run_id")
    if not all((entity, project, run_id)):
        return
    run = wandb.Api().run(f"{entity}/{project}/{run_id}")
    run.summary["checkpoint_backend"] = "hf"
    run.summary["hf_repo_id"] = info.get("hf_repo_id")
    run.summary["hf_package_path"] = info.get("hf_package_path")
    run.summary["hf_model_ref"] = info.get("hf_model_ref")
    run.summary["hf_revision"] = info.get("hf_revision")
    run.summary["selection/selected_checkpoint_sha256"] = info.get("selected_checkpoint_sha256")
    run.update()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    publish_staged_bundle(tyro.cli(Args))
