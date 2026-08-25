"""Archive the checkpoint selected by the post-training validation tournament."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Literal

import torch as th
import wandb

from engiopt.checkpoint_store import active_wandb_run_metadata
from engiopt.checkpoint_store import CheckpointBackend
from engiopt.checkpoint_store import save_checkpoint_package
from engiopt.topk_checkpoint_bundle import _eval_only_checkpoint
from engiopt.topk_checkpoint_bundle import infer_checkpoint_package_label


@dataclass(frozen=True)
class SelectedCheckpointSpec:
    """Files and protocol metadata for one selected checkpoint."""

    model_id: str
    problem_id: str
    seed: int
    checkpoint_path: Path
    validation_metrics_path: Path
    selection_rows: list[dict[str, Any]]
    selected_rank: int
    selected_epoch: int
    top_k: int
    selection_batch_size: int
    selection_seed: int
    test_seed: int
    test_generation_seed: int
    release: str
    package_label: str | None = None


SelectedCheckpointBackend = Literal["hf", "local", "none"]


def selected_extra_path_parts(release: str, package_label: str | None) -> list[str]:
    """Return the stable HF path components for a selected release checkpoint."""
    parts = ["selected", release]
    if package_label:
        parts.append(package_label)
    return parts


def archive_selected_checkpoint_bundle(
    *,
    spec: SelectedCheckpointSpec,
    checkpoint_backend: CheckpointBackend,
    hf_entity: str,
    hf_repo_prefix: str,
    hf_private: bool,
    test_metrics: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish one eval-ready checkpoint and the evidence used to select it."""
    if checkpoint_backend == "none":
        return {"checkpoint_backend": "none"}
    with tempfile.TemporaryDirectory(prefix="engiopt_selected_checkpoint_") as tmpdir:
        prepared = _materialize_selected_checkpoint_bundle(
            spec=spec,
            stage_dir=Path(tmpdir),
            checkpoint_backend=checkpoint_backend,
            test_metrics=test_metrics,
            metadata=metadata,
        )

        info = save_checkpoint_package(
            checkpoint_backend=checkpoint_backend,
            hf_entity=hf_entity,
            hf_repo_prefix=hf_repo_prefix,
            hf_private=hf_private,
            problem_id=spec.problem_id,
            algo=spec.model_id,
            seed=spec.seed,
            checkpoint_files=prepared["checkpoint_files"],
            run_config=prepared["run_config"],
            metadata=prepared["metadata"],
            primary_files=prepared["primary_files"],
            extra_path_parts=prepared["extra_path_parts"],
            upload_run_copy=False,
        )

    info.update(
        {
            "selected_checkpoint_epoch": spec.selected_epoch,
            "selected_checkpoint_rank": spec.selected_rank,
            "selected_checkpoint_sha256": prepared["checkpoint_sha256"],
        }
    )
    if wandb.run is not None:
        wandb.run.summary["selection/selected_checkpoint_epoch"] = spec.selected_epoch
        wandb.run.summary["selection/selected_checkpoint_rank"] = spec.selected_rank
        wandb.run.summary["selection/selected_checkpoint_sha256"] = prepared["checkpoint_sha256"]
    return info


def stage_selected_checkpoint_bundle(
    *,
    spec: SelectedCheckpointSpec,
    staging_root: str | Path,
    test_metrics: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an upload-ready package to persistent storage without contacting HF."""
    checkpoint = _load_checkpoint(spec.checkpoint_path)
    run_config = _checkpoint_run_config(checkpoint)
    package_label = spec.package_label or infer_checkpoint_package_label(spec.model_id, run_config)
    stage_dir = Path(staging_root).joinpath(
        spec.model_id,
        spec.problem_id,
        *selected_extra_path_parts(spec.release, package_label),
        f"seed_{spec.seed}",
    )
    receipt_path = stage_dir / "upload_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(
            f"Selected checkpoint release is already published at {stage_dir}; use a new release label for a rerun"
        )
    stage_dir.mkdir(parents=True, exist_ok=True)
    prepared = _materialize_selected_checkpoint_bundle(
        spec=spec,
        stage_dir=stage_dir,
        checkpoint_backend="local",
        test_metrics=test_metrics,
        metadata=metadata,
        checkpoint=checkpoint,
    )
    _write_json(stage_dir / "run_config.json", prepared["run_config"])
    _write_json(stage_dir / "metadata.json", prepared["metadata"])

    info = {
        "checkpoint_backend": "local",
        "selected_checkpoint_staging_dir": str(stage_dir),
        "selected_checkpoint_epoch": spec.selected_epoch,
        "selected_checkpoint_rank": spec.selected_rank,
        "selected_checkpoint_sha256": prepared["checkpoint_sha256"],
    }
    if wandb.run is not None:
        for key, value in info.items():
            wandb.run.summary[key] = value
    return info


def _materialize_selected_checkpoint_bundle(
    *,
    spec: SelectedCheckpointSpec,
    stage_dir: Path,
    checkpoint_backend: str,
    test_metrics: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = checkpoint or _load_checkpoint(spec.checkpoint_path)
    run_config = _checkpoint_run_config(checkpoint)
    package_label = spec.package_label or infer_checkpoint_package_label(spec.model_id, run_config)
    normalized_rows = _normalize_selection_rows(spec)
    selected_filename = _selected_checkpoint_filename(spec.model_id)

    selected_path = stage_dir / selected_filename
    eval_checkpoint = _eval_only_checkpoint(checkpoint, spec.model_id)
    if spec.model_id == "cgan_cnn_2d":
        eval_checkpoint.pop("discriminator", None)
    th.save(eval_checkpoint, selected_path)
    shutil.copy2(spec.validation_metrics_path, stage_dir / "validation_metrics.json")

    selection_payload = {
        "protocol": {
            "shortlist_metric": "validation_mmd",
            "selection_metric": "validation_cog",
            "selection_statistic": "median",
            "top_k": spec.top_k,
            "selection_batch_size": spec.selection_batch_size,
            "selection_split": "val",
            "selection_seed": spec.selection_seed,
            "test_split": "test",
            "test_seed": spec.test_seed,
            "test_generation_seed": spec.test_generation_seed,
        },
        "selected_rank": spec.selected_rank,
        "selected_epoch": spec.selected_epoch,
        "candidates": normalized_rows,
    }
    _write_json(stage_dir / "selection_results.json", selection_payload)
    _write_selection_csv(stage_dir / "selection_results.csv", normalized_rows)

    checkpoint_sha256 = _sha256(selected_path)
    primary_files = [selected_filename]
    checkpoint_filenames = [
        selected_filename,
        "validation_metrics.json",
        "selection_results.json",
        "selection_results.csv",
    ]
    bundle_metadata = {
        "bundle_type": "selected_checkpoint",
        "checkpoint_archive_mode": "eval",
        "problem_id": spec.problem_id,
        "algo": spec.model_id,
        "seed": spec.seed,
        "checkpoint_backend": checkpoint_backend,
        "checkpoint_files": checkpoint_filenames,
        "primary_files": primary_files,
        "release": spec.release,
        "package_label": package_label,
        "selected_checkpoint_source_filename": spec.checkpoint_path.name,
        "selected_checkpoint_filename": selected_filename,
        "selected_checkpoint_epoch": spec.selected_epoch,
        "selected_checkpoint_rank": spec.selected_rank,
        "selected_checkpoint_sha256": checkpoint_sha256,
        "selected_checkpoint_size_bytes": selected_path.stat().st_size,
        "selection_top_k": spec.top_k,
        "selection_batch_size": spec.selection_batch_size,
        "selection_seed": spec.selection_seed,
        "test_seed": spec.test_seed,
        "test_generation_seed": spec.test_generation_seed,
        "engiopt_git_commit": os.environ.get("ENGIOPT_GIT_COMMIT"),
        "engibench_git_commit": os.environ.get("ENGIBENCH_GIT_COMMIT"),
        "training_wandb": checkpoint.get("training_wandb", {}),
        "test_metrics": _public_metrics(test_metrics or {}),
    }
    bundle_metadata.update(active_wandb_run_metadata())
    if metadata:
        bundle_metadata.update(metadata)

    return {
        "checkpoint_files": {name: str(stage_dir / name) for name in checkpoint_filenames},
        "run_config": run_config,
        "metadata": bundle_metadata,
        "primary_files": primary_files,
        "extra_path_parts": selected_extra_path_parts(spec.release, package_label),
        "checkpoint_sha256": checkpoint_sha256,
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Selected checkpoint not found: {path}")
    checkpoint = th.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict in {path}")
    return checkpoint


def _checkpoint_run_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    run_config = checkpoint.get("args", {})
    return dict(run_config) if isinstance(run_config, dict) else {}


def _selected_checkpoint_filename(model_id: str) -> str:
    if model_id == "cgan_cnn_2d":
        return "generator.pth"
    if model_id in {"flow_matching_2d_cond", "diffusion_2d_cond"}:
        return "model.pth"
    raise ValueError(f"Unsupported selected checkpoint model: {model_id}")


def _normalize_selection_rows(spec: SelectedCheckpointSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in spec.selection_rows:
        normalized = {
            "rank": int(row["rank"]),
            "epoch": int(row["epoch"]),
            "validation_mmd": float(row["validation_mmd"]),
            "validation_cog": float(row["validation_cog"]),
            "validation_fog": float(row["validation_fog"]),
            "validation_eval_mmd": float(row["validation_eval_mmd"]),
            "validation_generation_seed": int(row["validation_generation_seed"]),
            "checkpoint_filename": Path(str(row["checkpoint_path"])).name,
            "selected": int(row["rank"]) == spec.selected_rank,
        }
        rows.append(normalized)
    if sum(bool(row["selected"]) for row in rows) != 1:
        raise ValueError("Selection evidence must identify exactly one selected checkpoint")
    return rows


def _write_selection_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "epoch",
        "validation_mmd",
        "validation_cog",
        "validation_fog",
        "validation_eval_mmd",
        "validation_generation_seed",
        "checkpoint_filename",
        "selected",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    excluded = {"checkpoint_dir", "checkpoint_path"}
    return {key: value for key, value in metrics.items() if key not in excluded}
