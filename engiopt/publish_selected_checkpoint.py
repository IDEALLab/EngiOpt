"""Publish a staged selected-checkpoint bundle to HF and update its W&B run."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import tyro
import wandb

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
