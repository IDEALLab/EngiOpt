"""Helpers for archiving and restoring top-k checkpoint directories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Literal

import torch as th

from engiopt.checkpoint_store import resolve_named_checkpoint
from engiopt.checkpoint_store import save_checkpoint_package

CheckpointSource = Literal["auto", "local", "hf"]
CheckpointBackend = Literal["hf", "none"]
CheckpointArchiveMode = Literal["eval", "full"]


@dataclass(frozen=True)
class TopKBundleSpec:
    """Description of a top-k checkpoint bundle."""

    model_id: str
    problem_id: str
    seed: int
    checkpoint_dir: Path
    top_k: int = 5
    package_label: str | None = None
    include_final: bool = True
    include_discriminator: bool = True
    archive_mode: CheckpointArchiveMode = "eval"


def infer_checkpoint_package_label(model_id: str, run_config: dict[str, Any] | None = None) -> str | None:
    """Infer a stable optional package label from model-specific config."""
    if not run_config:
        return None
    if model_id == "flow_matching_2d_cond":
        method = run_config.get("method")
        steps = run_config.get("integration_steps")
        if method is not None and steps is not None:
            return f"{method}_{steps}"
    if model_id == "diffusion_2d_cond":
        num_timesteps = run_config.get("num_timesteps")
        if num_timesteps is not None:
            return f"timesteps_{num_timesteps}"
    if model_id == "cgan_cnn_2d":
        activation = run_config.get("generator_output_activation")
        if activation is not None:
            return f"activation_{activation}"
    return None


def topk_extra_path_parts(package_label: str | None = None) -> list[str]:
    """Return the HF package path parts used for top-k checkpoint bundles."""
    parts = ["topk"]
    if package_label:
        parts.append(package_label)
    return parts


def restore_topk_checkpoint_dir(
    *,
    model_id: str,
    problem_id: str,
    seed: int,
    checkpoint_source: CheckpointSource,
    checkpoint_dir: str | Path | None,
    hf_entity: str,
    hf_repo_prefix: str,
    wandb_project: str,
    wandb_entity: str | None,
    package_label: str | None = None,
) -> tuple[Path, str]:
    """Resolve a top-k checkpoint directory, preferring local files when available."""
    local_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_source in {"auto", "local"} and local_dir is not None and (local_dir / "validation_metrics.json").exists():
        return local_dir, "local_top_k_checkpoint"
    if checkpoint_source == "local":
        if local_dir is None:
            raise FileNotFoundError("checkpoint_source='local' requires --checkpoint-dir")
        raise FileNotFoundError(f"Local top-k checkpoint directory is missing validation_metrics.json: {local_dir}")
    if checkpoint_source not in {"auto", "hf"}:
        raise ValueError(f"Unsupported checkpoint source: {checkpoint_source}")

    resolved = resolve_named_checkpoint(
        model_source="hf",
        problem_id=problem_id,
        algo=model_id,
        seed=seed,
        hf_entity=hf_entity,
        hf_repo_prefix=hf_repo_prefix,
        required_files=["validation_metrics.json"],
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_artifact_names={},
        extra_path_parts=topk_extra_path_parts(package_label),
    )
    return Path(resolved.root_dir), "hf_top_k_checkpoint"


def archive_topk_checkpoint_bundle(
    *,
    spec: TopKBundleSpec,
    checkpoint_backend: CheckpointBackend,
    hf_entity: str,
    hf_repo_prefix: str,
    hf_private: bool,
    run_config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Archive a local top-k checkpoint directory to the configured backend."""
    if checkpoint_backend == "none":
        return {"checkpoint_backend": "none"}

    checkpoint_files, inferred_config = collect_topk_checkpoint_files(spec)
    with tempfile.TemporaryDirectory(prefix="engiopt_topk_archive_") as tmpdir:
        upload_files = (
            _materialize_eval_checkpoint_files(checkpoint_files, spec.model_id, Path(tmpdir))
            if spec.archive_mode == "eval"
            else checkpoint_files
        )
        merged_config = dict(inferred_config)
        if run_config:
            merged_config.update(run_config)
        inferred_package_label = infer_checkpoint_package_label(spec.model_id, merged_config)
        package_label = spec.package_label or inferred_package_label

        bundle_metadata = {
            "bundle_type": "top_k_checkpoints",
            "checkpoint_archive_mode": spec.archive_mode,
            "model_id": spec.model_id,
            "problem_id": spec.problem_id,
            "seed": spec.seed,
            "top_k": spec.top_k,
            "package_label": package_label,
            "inferred_package_label": inferred_package_label,
        }
        if metadata:
            bundle_metadata.update(metadata)

        return save_checkpoint_package(
            checkpoint_backend=checkpoint_backend,
            hf_entity=hf_entity,
            hf_repo_prefix=hf_repo_prefix,
            hf_private=hf_private,
            problem_id=spec.problem_id,
            algo=spec.model_id,
            seed=spec.seed,
            checkpoint_files=upload_files,
            run_config=merged_config,
            metadata=bundle_metadata,
            primary_files=["validation_metrics.json"],
            extra_path_parts=topk_extra_path_parts(package_label),
            upload_run_copy=False,
        )


def collect_topk_checkpoint_files(spec: TopKBundleSpec) -> tuple[dict[str, str], dict[str, Any]]:
    """Collect validation metrics and the checkpoint files needed for top-k evaluation."""
    checkpoint_dir = spec.checkpoint_dir
    metrics_path = checkpoint_dir / "validation_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Validation metrics file not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    files: dict[str, str] = {"validation_metrics.json": str(metrics_path)}
    run_config: dict[str, Any] = {}
    for item in sorted(summary.get("top_k_epochs", []), key=lambda row: float(row["metric_value"]))[: spec.top_k]:
        epoch = int(item["epoch"]) + 1
        for filename in _checkpoint_filenames(spec.model_id, epoch, include_discriminator=spec.include_discriminator):
            _add_existing_file(files, checkpoint_dir, filename, required=True)

    if spec.include_final:
        for filename in _final_checkpoint_filenames(spec.model_id, include_discriminator=spec.include_discriminator):
            _add_existing_file(files, checkpoint_dir, filename, required=False)

    if len(files) == 1:
        raise ValueError(f"No top-k checkpoint files found in {checkpoint_dir}")

    for file_path in files.values():
        if file_path.endswith(".pth"):
            run_config = _load_checkpoint_args(Path(file_path))
            if run_config:
                break

    return files, run_config


def _checkpoint_filenames(model_id: str, epoch: int, *, include_discriminator: bool) -> list[str]:
    if model_id == "cgan_cnn_2d":
        filenames = [f"generator_epoch_{epoch:04d}.pth"]
        if include_discriminator:
            filenames.append(f"discriminator_epoch_{epoch:04d}.pth")
        return filenames
    return [f"epoch_{epoch:04d}.pth"]


def _final_checkpoint_filenames(model_id: str, *, include_discriminator: bool) -> list[str]:
    if model_id == "cgan_cnn_2d":
        filenames = ["final_generator.pth", "best_generator.pth"]
        if include_discriminator:
            filenames.extend(["final_discriminator.pth", "best_discriminator.pth"])
        return filenames
    return ["final_model.pth", "model.pth"]


def _add_existing_file(files: dict[str, str], checkpoint_dir: Path, filename: str, *, required: bool) -> None:
    path = checkpoint_dir / filename
    if path.exists():
        files[filename] = str(path)
    elif required:
        raise FileNotFoundError(f"Checkpoint not found: {path}")


def _load_checkpoint_args(checkpoint_path: Path) -> dict[str, Any]:
    try:
        checkpoint = th.load(checkpoint_path, map_location="cpu")
    except Exception:
        return {}
    args = checkpoint.get("args", {})
    return dict(args) if isinstance(args, dict) else {}


def _materialize_eval_checkpoint_files(
    checkpoint_files: dict[str, str],
    model_id: str,
    output_dir: Path,
) -> dict[str, str]:
    """Create eval-only checkpoint copies while preserving non-checkpoint files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized: dict[str, str] = {}
    for filename, source in checkpoint_files.items():
        source_path = Path(source)
        if source_path.suffix != ".pth":
            materialized[filename] = str(source_path)
            continue

        checkpoint = th.load(source_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Expected checkpoint dict in {source_path}")

        target_path = output_dir / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)
        th.save(_eval_only_checkpoint(checkpoint, model_id), target_path)
        materialized[filename] = str(target_path)
    return materialized


def _eval_only_checkpoint(checkpoint: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Return the subset of a checkpoint required for deterministic evaluation."""
    keep_by_model = {
        "flow_matching_2d_cond": {
            "args",
            "design_max",
            "design_min",
            "design_shape",
            "encoder_hid_dim",
            "epoch",
            "loss",
            "model",
            "model_config",
            "training_wandb",
        },
        "diffusion_2d_cond": {
            "args",
            "batches_done",
            "design_max",
            "design_min",
            "diffusion_sample_max",
            "diffusion_sample_min",
            "epoch",
            "loss",
            "model",
            "model_config",
            "training_wandb",
        },
        "cgan_cnn_2d": {
            "args",
            "batches_done",
            "discriminator",
            "epoch",
            "generator",
            "loss",
            "training_wandb",
        },
    }
    keys = keep_by_model.get(model_id)
    if keys is None:
        raise ValueError(f"Unsupported model_id for eval-only archive: {model_id}")
    return {key: checkpoint[key] for key in keys if key in checkpoint}
