"""Shared checkpoint save/load helpers for EngiOpt model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Literal

from huggingface_hub import HfApi
from huggingface_hub import snapshot_download

import wandb

CheckpointBackend = Literal["hf", "none"]
ModelSource = Literal["auto", "hf", "wandb", "local"]


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """Resolved checkpoint package with local files and serialized config."""

    source: Literal["hf", "wandb", "local"]
    root_dir: str
    files: dict[str, str]
    run_config: dict[str, Any]
    metadata: dict[str, Any]


def build_hf_repo_id(hf_entity: str, hf_repo_prefix: str, algo: str) -> str:
    """Return the canonical HF repo id for an EngiOpt model family."""
    repo_suffix = algo.replace("_", "-")
    return f"{hf_entity}/{hf_repo_prefix}-{repo_suffix}"


def build_hf_package_path(problem_id: str, seed: int, extra_parts: list[str] | None = None) -> str:
    """Return the canonical package path inside an HF repo."""
    parts = [_sanitize_path_component(problem_id)]
    if extra_parts:
        parts.extend(_sanitize_path_component(part) for part in extra_parts)
    parts.append(f"seed_{seed}")
    return "/".join(parts)


def build_hf_run_package_path(package_path: str, wandb_run_id: str | None) -> str | None:
    """Return an immutable run-specific package path when a W&B run id is available."""
    if not wandb_run_id:
        return None
    return f"{package_path}/run_{_sanitize_path_component(wandb_run_id)}"


def save_checkpoint_package(  # noqa: PLR0913
    *,
    checkpoint_backend: CheckpointBackend,
    hf_entity: str,
    hf_repo_prefix: str,
    hf_private: bool,
    problem_id: str,
    algo: str,
    seed: int,
    checkpoint_files: dict[str, str],
    run_config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    primary_files: list[str] | None = None,
    extra_path_parts: list[str] | None = None,
    upload_run_copy: bool = True,
) -> dict[str, Any]:
    """Save a checkpoint package to HuggingFace.

    W&B is no longer a checkpoint storage backend; the active W&B run still
    receives a summary pointing at the HF package for traceability.
    """
    info: dict[str, Any] = {
        "checkpoint_backend": checkpoint_backend,
        "hf_repo_id": None,
        "hf_package_path": None,
        "hf_run_package_path": None,
        "hf_revision": None,
        "hf_run_revision": None,
    }
    metadata_payload = _build_metadata(
        problem_id=problem_id,
        algo=algo,
        seed=seed,
        checkpoint_backend=checkpoint_backend,
        checkpoint_files=checkpoint_files,
        primary_files=primary_files,
        metadata=metadata,
    )
    metadata_payload.update(_build_wandb_run_metadata())

    if checkpoint_backend == "hf":
        repo_id = build_hf_repo_id(hf_entity, hf_repo_prefix, algo)
        package_path = build_hf_package_path(problem_id, seed, extra_path_parts)
        info["hf_repo_id"] = repo_id
        info["hf_package_path"] = package_path
        metadata_payload["hf_repo_id"] = repo_id
        metadata_payload["hf_package_path"] = package_path

        run_package_path = (
            build_hf_run_package_path(package_path, metadata_payload.get("wandb_run_id"))
            if upload_run_copy
            else None
        )
        info["hf_run_package_path"] = run_package_path
        if run_package_path is not None:
            metadata_payload["hf_run_package_path"] = run_package_path
            run_revision = _upload_package_to_hf(
                repo_id=repo_id,
                hf_private=hf_private,
                package_path=run_package_path,
                checkpoint_files=checkpoint_files,
                run_config=run_config,
                metadata=metadata_payload,
                algo=algo,
            )
            info["hf_run_revision"] = run_revision
            metadata_payload["hf_run_revision"] = run_revision

        revision = _upload_package_to_hf(
            repo_id=repo_id,
            hf_private=hf_private,
            package_path=package_path,
            checkpoint_files=checkpoint_files,
            run_config=run_config,
            metadata=metadata_payload,
            algo=algo,
        )
        info["hf_revision"] = revision
        metadata_payload["hf_revision"] = revision

    if wandb.run is not None:
        _log_checkpoint_summary_to_wandb(metadata_payload, info)

    return info


def resolve_named_checkpoint(  # noqa: PLR0913
    *,
    model_source: ModelSource,
    problem_id: str,
    algo: str,
    seed: int,
    hf_entity: str,
    hf_repo_prefix: str,
    required_files: list[str],
    wandb_project: str,
    wandb_entity: str | None,
    wandb_artifact_names: dict[str, str],
    wandb_config_artifact_name: str | None = None,
    wandb_artifact_alias: str | None = None,
    local_model_dir: str | None = None,
    extra_path_parts: list[str] | None = None,
) -> ResolvedCheckpoint:
    """Resolve a checkpoint package by the standard EngiOpt problem/algo/seed naming.

    ``wandb_artifact_alias`` overrides the default ``f"seed_{seed}"`` alias used when
    falling back to legacy W&B artifacts. Callers with custom alias schemes (e.g.
    ``f"seed_{seed}_rec{r}_perf{p}"``) pass it here so the read-fallback resolves
    historical artifacts that pre-date the HF cutover.
    """
    alias = wandb_artifact_alias or f"seed_{seed}"
    errors: list[str] = []
    if model_source in {"auto", "hf"}:
        try:
            return _resolve_hf_package(
                repo_id=build_hf_repo_id(hf_entity, hf_repo_prefix, algo),
                package_path=build_hf_package_path(problem_id, seed, extra_path_parts),
                required_files=required_files,
            )
        except Exception as exc:
            if model_source == "hf":
                raise
            errors.append(f"hf: {exc}")

    if model_source in {"auto", "wandb"}:
        try:
            return _resolve_wandb_package(
                _required_files=required_files,
                artifact_names=wandb_artifact_names,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                alias=alias,
                config_artifact_name=wandb_config_artifact_name,
            )
        except Exception as exc:
            if model_source == "wandb":
                raise
            errors.append(f"wandb: {exc}")

    if local_model_dir is not None and model_source in {"auto", "local"}:
        return _resolve_local_package(local_model_dir, required_files)

    attempted = ", ".join(errors) if errors else "no backends attempted"
    raise FileNotFoundError(f"Unable to resolve checkpoint for {algo}/{problem_id}/seed_{seed}: {attempted}")


def resolve_checkpoint_reference(
    *,
    model_source: ModelSource,
    model_ref: str,
    required_files: list[str] | None = None,
    active_wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> ResolvedCheckpoint:
    """Resolve a checkpoint package from an explicit HF/W&B/local reference."""
    inferred_source = model_source
    normalized_ref = model_ref
    if model_source == "auto":
        if model_ref.startswith("hf://"):
            inferred_source = "hf"
        elif model_ref.startswith("wandb://"):
            inferred_source = "wandb"
        elif os.path.isdir(model_ref):
            inferred_source = "local"
        else:
            inferred_source = "wandb"

    if inferred_source == "hf":
        repo_id, package_path = _parse_hf_reference(model_ref)
        return _resolve_hf_package(repo_id=repo_id, package_path=package_path, required_files=required_files or [])
    if inferred_source == "wandb":
        artifact_path = model_ref.removeprefix("wandb://")
        return _resolve_wandb_reference(
            artifact_path=artifact_path,
            required_files=required_files or [],
            active_wandb_run=active_wandb_run,
        )
    if inferred_source == "local":
        normalized_ref = model_ref.removeprefix("file://")
        return _resolve_local_package(normalized_ref, required_files or [])

    raise ValueError(f"Unsupported model source: {model_source}")


def _upload_package_to_hf(  # noqa: PLR0913
    *,
    repo_id: str,
    hf_private: bool,
    package_path: str,
    checkpoint_files: dict[str, str],
    run_config: dict[str, Any],
    metadata: dict[str, Any],
    algo: str,
) -> str | None:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=hf_private, exist_ok=True)
    _ensure_hf_repo_readme(api, repo_id, algo)

    with tempfile.TemporaryDirectory() as tmpdir:
        stage_dir = Path(tmpdir)
        for package_name, file_path in checkpoint_files.items():
            shutil.copy2(file_path, stage_dir / package_name)
        _write_json(stage_dir / "run_config.json", run_config)
        _write_json(stage_dir / "metadata.json", metadata)

        commit_info = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=tmpdir,
            path_in_repo=package_path,
            commit_message=f"Upload checkpoint for {algo} {package_path}",
        )
    return getattr(commit_info, "oid", None)


def _resolve_hf_package(*, repo_id: str, package_path: str, required_files: list[str]) -> ResolvedCheckpoint:
    repo_snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{package_path}/*"],
    )
    root_dir = os.path.join(repo_snapshot, package_path)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"HF package path not found: {repo_id}/{package_path}")
    return _load_package_from_directory(root_dir=root_dir, required_files=required_files, source="hf")


def _resolve_local_package(local_model_dir: str, required_files: list[str]) -> ResolvedCheckpoint:
    if not os.path.isdir(local_model_dir):
        raise FileNotFoundError(f"Local checkpoint directory not found: {local_model_dir}")
    return _load_package_from_directory(root_dir=local_model_dir, required_files=required_files, source="local")


def _resolve_wandb_package(
    *,
    _required_files: list[str],
    artifact_names: dict[str, str],
    wandb_project: str,
    wandb_entity: str | None,
    alias: str,
    config_artifact_name: str | None,
) -> ResolvedCheckpoint:
    api = wandb.Api()
    files: dict[str, str] = {}
    for file_name, artifact_name in artifact_names.items():
        artifact_path = _build_wandb_artifact_path(
            artifact_name=artifact_name,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            alias=alias,
        )
        artifact = api.artifact(artifact_path, type="model")
        artifact_dir = artifact.download()
        files[file_name] = os.path.join(artifact_dir, file_name)

    config_artifact = config_artifact_name or next(iter(artifact_names.values()))
    config_artifact_path = _build_wandb_artifact_path(
        artifact_name=config_artifact,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        alias=alias,
    )
    artifact = api.artifact(config_artifact_path, type="model")
    run = artifact.logged_by()
    if run is None or not hasattr(run, "config"):
        raise ValueError("Failed to retrieve W&B run config from artifact")

    return ResolvedCheckpoint(
        source="wandb",
        root_dir=os.path.dirname(next(iter(files.values()))),
        files=files,
        run_config=dict(run.config),
        metadata={
            "source": "wandb",
            "artifact_paths": {
                file_name: _build_wandb_artifact_path(
                    artifact_name=artifact_name,
                    wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                    alias=alias,
                )
                for file_name, artifact_name in artifact_names.items()
            },
        },
    )


def _resolve_wandb_reference(
    *,
    artifact_path: str,
    required_files: list[str],
    active_wandb_run: wandb.sdk.wandb_run.Run | None,
) -> ResolvedCheckpoint:
    artifact = (
        active_wandb_run.use_artifact(artifact_path, type="model")
        if active_wandb_run is not None
        else wandb.Api().artifact(artifact_path, type="model")
    )
    artifact_dir = artifact.download()
    files = _discover_reference_files(artifact_dir, required_files)
    run = artifact.logged_by()
    if run is None or not hasattr(run, "config"):
        raise ValueError("Failed to retrieve W&B run config from artifact reference")
    return ResolvedCheckpoint(
        source="wandb",
        root_dir=artifact_dir,
        files=files,
        run_config=dict(run.config),
        metadata={"source": "wandb", "artifact_path": artifact_path},
    )


def _load_package_from_directory(
    *, root_dir: str, required_files: list[str], source: Literal["hf", "local"]
) -> ResolvedCheckpoint:
    run_config_path = os.path.join(root_dir, "run_config.json")
    metadata_path = os.path.join(root_dir, "metadata.json")
    if not os.path.exists(run_config_path):
        raise FileNotFoundError(f"Missing run_config.json in {root_dir}")
    run_config = _read_json(run_config_path)
    metadata = _read_json(metadata_path) if os.path.exists(metadata_path) else {}
    package_files = required_files or _discover_package_files(root_dir, metadata)
    files = {file_name: os.path.join(root_dir, file_name) for file_name in package_files}
    for file_name, file_path in files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing checkpoint file {file_name} in {root_dir}")
    return ResolvedCheckpoint(source=source, root_dir=root_dir, files=files, run_config=run_config, metadata=metadata)


def _discover_reference_files(root_dir: str, required_files: list[str]) -> dict[str, str]:
    if required_files:
        files = {file_name: os.path.join(root_dir, file_name) for file_name in required_files}
        for file_name, file_path in files.items():
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing checkpoint file {file_name} in {root_dir}")
        return files

    discovered = [entry for entry in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, entry))]
    if len(discovered) != 1:
        raise ValueError(f"Expected exactly one file in {root_dir}, found {discovered}")
    file_name = discovered[0]
    return {file_name: os.path.join(root_dir, file_name)}


def _discover_package_files(root_dir: str, metadata: dict[str, Any]) -> list[str]:
    primary_files = metadata.get("primary_files")
    if isinstance(primary_files, list) and primary_files:
        return [str(file_name) for file_name in primary_files]

    excluded_names = {"metadata.json", "run_config.json"}
    discovered = [
        entry
        for entry in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, entry)) and entry not in excluded_names
    ]
    if not discovered:
        raise FileNotFoundError(f"No checkpoint files found in {root_dir}")
    return sorted(discovered)


def _parse_hf_reference(model_ref: str) -> tuple[str, str]:
    normalized = model_ref.removeprefix("hf://")
    first_sep = normalized.find("/")
    second_sep = normalized.find("/", first_sep + 1)
    if first_sep == -1 or second_sep == -1:
        raise ValueError(f"HF model references must look like hf://<entity>/<repo>/<package_path>, got {model_ref}")
    repo_id = normalized[:second_sep]
    package_path = normalized[second_sep + 1 :]
    return repo_id, package_path


def _build_wandb_artifact_path(
    *,
    artifact_name: str,
    wandb_project: str,
    wandb_entity: str | None,
    alias: str,
) -> str:
    project_path = f"{wandb_entity}/{wandb_project}" if wandb_entity is not None else wandb_project
    return f"{project_path}/{artifact_name}:{alias}"


def _build_metadata(  # noqa: PLR0913
    *,
    problem_id: str,
    algo: str,
    seed: int,
    checkpoint_backend: CheckpointBackend,
    checkpoint_files: dict[str, str],
    primary_files: list[str] | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = dict(metadata or {})
    payload.update(
        {
            "problem_id": problem_id,
            "algo": algo,
            "seed": seed,
            "checkpoint_backend": checkpoint_backend,
            "checkpoint_files": sorted(checkpoint_files),
            "primary_files": primary_files or sorted(checkpoint_files),
        }
    )
    return payload


def _build_wandb_run_metadata() -> dict[str, Any]:
    """Return W&B run identity fields for checkpoint metadata when tracking is active."""
    if wandb.run is None:
        return {}

    run = wandb.run
    entity = getattr(run, "entity", None)
    project = getattr(run, "project", None)
    run_id = getattr(run, "id", None)
    run_url = None
    if entity and project and run_id:
        run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"

    return {
        "wandb_entity": entity,
        "wandb_project": project,
        "wandb_run_id": run_id,
        "wandb_run_url": run_url,
    }


def _ensure_hf_repo_readme(api: HfApi, repo_id: str, algo: str) -> None:
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    if "README.md" in files:
        return
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".md") as tmpfile:
        tmpfile.write(_repo_readme_text(algo))
        tmp_path = tmpfile.name
    try:
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            commit_message=f"Add README for {algo} checkpoint repo",
        )
    finally:
        os.unlink(tmp_path)


def _repo_readme_text(algo: str) -> str:
    return (
        f"# EngiOpt {algo}\n\n"
        "This repository stores EngiOpt checkpoint packages for one model family.\n\n"
        "Each checkpoint package contains model weight files together with `run_config.json` "
        "and `metadata.json` so evaluation can run without depending on W&B run config state.\n"
    )


def _log_checkpoint_summary_to_wandb(metadata: dict[str, Any], info: dict[str, Any]) -> None:
    if wandb.run is None:
        return
    wandb.summary["checkpoint_backend"] = info["checkpoint_backend"]
    if info["hf_repo_id"] is not None:
        wandb.summary["hf_repo_id"] = info["hf_repo_id"]
        wandb.summary["hf_package_path"] = info["hf_package_path"]
        wandb.summary["hf_revision"] = info["hf_revision"]
    if info["hf_run_package_path"] is not None:
        wandb.summary["hf_run_package_path"] = info["hf_run_package_path"]
        wandb.summary["hf_run_revision"] = info["hf_run_revision"]
    wandb.summary["checkpoint_primary_files"] = metadata["primary_files"]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _sanitize_path_component(value: str) -> str:
    return value.replace(os.sep, "_").replace(" ", "_")
