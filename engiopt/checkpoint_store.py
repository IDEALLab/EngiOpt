"""Shared checkpoint save/load helpers for EngiOpt model artifacts.

HuggingFace is the single home for anything that has to be *reloaded or
compared*: checkpoints, their run configs, and their evaluation metrics.
Weights & Biases, when enabled, hosts the things you only ever *look at* --
loss curves, sample images, and a pointer back to the HF package.

Each run is filed under `{problem_id}/cfg_{fingerprint}/seed_{seed}`, so every
hyperparameter setting is independently addressable, plus the canonical
`{problem_id}/seed_{seed}` when the run used the training script's defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Literal

from huggingface_hub import HfApi
from huggingface_hub import snapshot_download

import wandb

METRICS_FILE = "metrics.json"
"""Evaluation scores stored alongside the weights they describe."""

CheckpointBackend = Literal["hf", "none"]
ModelSource = Literal["auto", "hf", "local"]


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """Resolved checkpoint package with local files and serialized config."""

    source: Literal["hf", "local"]
    root_dir: str
    files: dict[str, str]
    run_config: dict[str, Any]
    metadata: dict[str, Any]
    revision: str | None = None
    """Repo commit the package was downloaded at; None for local packages."""
    content_hash: str | None = None
    """Hash of the weight files themselves, identifying these exact weights.

    Two runs of the same configuration and seed produce different weights, and
    the repo path cannot tell them apart. This can, so a leaderboard row is
    traceable to the model that produced it.
    """


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


def save_checkpoint_package(
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
    config_fingerprint: str | None = None,
    is_default_config: bool = True,
    condition_keys: list[str] | tuple[str, ...] | None = None,
    condition_stats: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Save a checkpoint package to HuggingFace.

    A run is filed under up to three paths in its model repo:

    - `{problem_id}/cfg_{fingerprint}/seed_{seed}` -- this exact set of
      hyperparameters. Every configuration in a sweep gets its own, so runs
      never overwrite each other.
    - `{problem_id}/seed_{seed}` -- the canonical location that a plain
      `from_pretrained(problem, seed=...)` reads. Written **only** when the run
      used the training script's default hyperparameters, so a sweep cannot
      redefine what the bare model name means.
    Pass `config_fingerprint` and `is_default_config` together via
    `engiopt.core.checkpoint_identity(args)`. Omitting them keeps the older
    behaviour of writing only the canonical path.

    Each uploaded package carries its own `metadata.json` describing *that*
    package, so a `metadata.json` never points at a path this run did not write.

    Pass `condition_keys` (from `engiopt.transforms.condition_keys`) so the
    checkpoint records the condition schema it was trained under; loading then
    rebuilds the network for exactly those columns. A model that also *rescales*
    its conditions passes `condition_stats={"mean": [...], "std": [...]}` so the
    evaluator can reproduce the same scale, instead of refitting it on the far
    smaller evaluation sample.

    W&B is no longer a checkpoint storage backend; the active W&B run still
    receives a summary pointing at the HF package for traceability.
    """
    info: dict[str, Any] = {
        "checkpoint_backend": checkpoint_backend,
        "hf_repo_id": None,
        "hf_package_path": None,
        "hf_config_package_path": None,
        "hf_revision": None,
        "hf_config_revision": None,
    }
    base_metadata = _build_metadata(
        problem_id=problem_id,
        algo=algo,
        seed=seed,
        checkpoint_backend=checkpoint_backend,
        checkpoint_files=checkpoint_files,
        primary_files=primary_files,
        metadata=metadata,
        condition_keys=condition_keys,
        condition_stats=condition_stats,
    )
    base_metadata.update(_build_wandb_run_metadata())

    if checkpoint_backend == "hf":
        repo_id = build_hf_repo_id(hf_entity, hf_repo_prefix, algo)
        canonical_path = build_hf_package_path(problem_id, seed, extra_path_parts)
        info["hf_repo_id"] = repo_id
        base_metadata["hf_repo_id"] = repo_id
        if config_fingerprint:
            base_metadata["config_fingerprint"] = config_fingerprint

        # Every configuration gets its own addressable location, so a sweep's
        # runs cannot overwrite one another.
        if config_fingerprint:
            config_path = build_hf_package_path(problem_id, seed, [*(extra_path_parts or []), f"cfg_{config_fingerprint}"])
            info["hf_config_package_path"] = config_path
            info["hf_config_revision"] = _upload_package_to_hf(
                repo_id=repo_id,
                hf_private=hf_private,
                package_path=config_path,
                checkpoint_files=checkpoint_files,
                run_config=run_config,
                metadata={**base_metadata, "hf_package_path": config_path},
                algo=algo,
            )

        # The canonical path defines what the bare model name resolves to, so
        # only a default-hyperparameter run may claim it.
        if is_default_config:
            info["hf_package_path"] = canonical_path
            info["hf_revision"] = _upload_package_to_hf(
                repo_id=repo_id,
                hf_private=hf_private,
                package_path=canonical_path,
                checkpoint_files=checkpoint_files,
                run_config=run_config,
                metadata={**base_metadata, "hf_package_path": canonical_path},
                algo=algo,
            )

    if wandb.run is not None:
        _log_checkpoint_summary_to_wandb(base_metadata, info)

    return info


def resolve_named_checkpoint(
    *,
    model_source: ModelSource,
    problem_id: str,
    algo: str,
    seed: int,
    hf_entity: str,
    hf_repo_prefix: str,
    required_files: list[str],
    local_model_dir: str | None = None,
    extra_path_parts: list[str] | None = None,
    revision: str | None = None,
) -> ResolvedCheckpoint:
    """Resolve a checkpoint package by the standard EngiOpt problem/algo/seed naming.

    Args:
        model_source: `auto` tries HuggingFace then any local directory; `hf` or
            `local` restrict it to one.
        problem_id: EngiBench problem the checkpoint was trained on.
        algo: Model family, which selects the HF repo.
        seed: Training seed.
        hf_entity: HF org/user holding the checkpoint repos.
        hf_repo_prefix: Prefix of the per-model-family repo.
        required_files: Files the package must contain.
        local_model_dir: Directory to load from instead of the Hub.
        extra_path_parts: Path components identifying one configuration; see
            `engiopt.core.config_path_parts`.
        revision: Repo commit to read at. Without it the package resolves to
            whatever is on the repo's main branch, so re-uploading to the same
            path changes what an otherwise identical reference means. Pass it
            wherever a result has to stay reproducible -- notably the latent
            metric instrument, whose value defines the column it measures.

    Raises:
        FileNotFoundError: If no backend could supply the package.
    """
    errors: list[str] = []
    if model_source in {"auto", "hf"}:
        try:
            return _resolve_hf_package(
                repo_id=build_hf_repo_id(hf_entity, hf_repo_prefix, algo),
                package_path=build_hf_package_path(problem_id, seed, extra_path_parts),
                required_files=required_files,
                revision=revision,
            )
        except Exception as exc:
            if model_source == "hf":
                raise
            errors.append(f"hf: {exc}")

    if local_model_dir is not None and model_source in {"auto", "local"}:
        return _resolve_local_package(local_model_dir, required_files)

    attempted = ", ".join(errors) if errors else "no backends attempted"
    raise FileNotFoundError(f"Unable to resolve checkpoint for {algo}/{problem_id}/seed_{seed}: {attempted}")


def resolve_checkpoint_reference(
    *,
    model_source: ModelSource,
    model_ref: str,
    required_files: list[str] | None = None,
) -> ResolvedCheckpoint:
    """Resolve a checkpoint package from an explicit `hf://` or local reference.

    Raises:
        ValueError: If the reference cannot be interpreted.
    """
    inferred_source = model_source
    if model_source == "auto":
        inferred_source = "local" if os.path.isdir(model_ref.removeprefix("file://")) else "hf"

    if inferred_source == "hf":
        repo_id, package_path = _parse_hf_reference(model_ref)
        return _resolve_hf_package(repo_id=repo_id, package_path=package_path, required_files=required_files or [])
    if inferred_source == "local":
        return _resolve_local_package(model_ref.removeprefix("file://"), required_files or [])

    raise ValueError(f"Unsupported model source: {model_source}")


def _upload_package_to_hf(
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


def _resolve_hf_package(
    *, repo_id: str, package_path: str, required_files: list[str], revision: str | None = None
) -> ResolvedCheckpoint:
    repo_snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{package_path}/*"],
        revision=revision,
    )
    root_dir = os.path.join(repo_snapshot, package_path)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"HF package path not found: {repo_id}/{package_path}")
    # `snapshot_download` returns `.../snapshots/<commit sha>`, which is the
    # revision the files were actually taken from.
    return _load_package_from_directory(
        root_dir=root_dir, required_files=required_files, source="hf", revision=os.path.basename(repo_snapshot)
    )


def _resolve_local_package(local_model_dir: str, required_files: list[str]) -> ResolvedCheckpoint:
    if not os.path.isdir(local_model_dir):
        raise FileNotFoundError(f"Local checkpoint directory not found: {local_model_dir}")
    return _load_package_from_directory(root_dir=local_model_dir, required_files=required_files, source="local")


def hash_package_contents(root_dir: str) -> str:
    """Content hash of every weight file in a package, identifying these exact weights.

    Hashes what the package *contains* rather than what the model declared it
    needs. A model may load a file it did not list -- VQGAN's conditional
    variant reads `cvqgan.pth`, which is not in its `checkpoint_files` because
    the unconditional variant has none -- and two packages differing only in
    that file generate differently. Hashing the declared list would give them
    the same identity.

    `run_config.json` and `metadata.json` are excluded: they describe the
    package rather than being weights, and `metadata.json` records paths and
    revisions that differ between the two locations one run writes.

    Args:
        root_dir: Local directory holding the package.

    Returns:
        A 16-character hash over the file names and their bytes.
    """
    described = {"run_config.json", "metadata.json"}
    weight_files = sorted(
        entry for entry in os.listdir(root_dir) if entry not in described and os.path.isfile(os.path.join(root_dir, entry))
    )
    hasher = hashlib.sha256()
    for file_name in weight_files:
        hasher.update(file_name.encode())
        with open(os.path.join(root_dir, file_name), "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)
    return hasher.hexdigest()[:16]


def _load_package_from_directory(
    *, root_dir: str, required_files: list[str], source: Literal["hf", "local"], revision: str | None = None
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
    return ResolvedCheckpoint(
        source=source,
        root_dir=root_dir,
        files=files,
        run_config=run_config,
        metadata=metadata,
        revision=revision,
        content_hash=hash_package_contents(root_dir),
    )


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


def _build_metadata(
    *,
    problem_id: str,
    algo: str,
    seed: int,
    checkpoint_backend: CheckpointBackend,
    checkpoint_files: dict[str, str],
    primary_files: list[str] | None,
    metadata: dict[str, Any] | None,
    condition_keys: list[str] | tuple[str, ...] | None = None,
    condition_stats: dict[str, list[float]] | None = None,
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
    if condition_keys is not None:
        payload["condition_keys"] = list(condition_keys)
    if condition_stats is not None:
        payload["condition_stats"] = {key: [float(v) for v in values] for key, values in condition_stats.items()}
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
    """Point the W&B run at the HF package holding its weights.

    W&B stores the media -- loss curves and sample images -- while the weights
    and metrics live on HuggingFace, so each side records where the other is.
    """
    if wandb.run is None:
        return
    wandb.summary["checkpoint_backend"] = info["checkpoint_backend"]
    if info["hf_repo_id"] is not None:
        wandb.summary["hf_repo_id"] = info["hf_repo_id"]
        wandb.summary["hf_package_path"] = info["hf_package_path"]
        wandb.summary["hf_config_package_path"] = info["hf_config_package_path"]
        wandb.summary["hf_revision"] = info["hf_revision"]
        wandb.summary["hf_config_revision"] = info["hf_config_revision"]
    wandb.summary["checkpoint_primary_files"] = metadata["primary_files"]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _sanitize_path_component(value: str) -> str:
    return value.replace(os.sep, "_").replace(" ", "_")


def publish_checkpoint_metrics(
    *,
    hf_entity: str,
    hf_repo_prefix: str,
    problem_id: str,
    algo: str,
    seed: int,
    metrics: dict[str, Any],
    extra_path_parts: list[str] | None = None,
    token: str | None = None,
) -> str:
    """Attach evaluation metrics to a checkpoint package on HuggingFace.

    Metrics live beside the weights they describe, so a checkpoint is
    self-describing: whoever downloads it can see how it scored without
    consulting the leaderboard or a W&B run.

    Args:
        hf_entity: HF org/user holding the checkpoint repos.
        hf_repo_prefix: Prefix of the per-model-family repo.
        problem_id: Problem the checkpoint was evaluated on.
        algo: Model family, selecting the repo.
        seed: Training seed.
        metrics: Scores to record, typically one leaderboard row.
        extra_path_parts: Configuration path components, from
            `engiopt.core.config_path_parts`.
        token: HF token; falls back to the ambient login.

    Returns:
        The path written inside the repo.
    """
    repo_id = build_hf_repo_id(hf_entity, hf_repo_prefix, algo)
    package_path = build_hf_package_path(problem_id, seed, extra_path_parts)
    api = HfApi(token=token)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / METRICS_FILE
        _write_json(local, metrics)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=f"{package_path}/{METRICS_FILE}",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Evaluation metrics for {problem_id}/seed_{seed}",
        )
    return f"{package_path}/{METRICS_FILE}"
