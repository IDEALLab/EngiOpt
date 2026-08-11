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

RUN_CONFIG_FILE = "run_config.json"
"""Hyperparameters the run was launched with, enough to rebuild the network."""

METADATA_FILE = "metadata.json"
"""What this package is: paths, revisions, condition schema, completeness."""

PACKAGE_COMPLETE_FIELD = "package_complete"
"""Metadata flag: whether the run that wrote this package finished.

Multi-stage models (VQGAN trains a condition autoencoder, then a VQGAN, then a
transformer) upload after each stage so a crash does not lose the earlier work.
That means an interrupted run leaves a package carrying a `run_config.json` and
a `metadata.json` -- indistinguishable, from the Hub, from a finished one. This
field is the distinction; single-stage models write `True` and never think about
it.
"""

CONDITION_NORMALIZER_METADATA_FIELD = "condition_normalizer"
"""Metadata field recording the min/max a model scaled its *conditions* by."""

DESIGN_NORMALIZER_METADATA_FIELD = "design_normalizer"
"""Metadata field recording the min/max a model scaled its *designs* by.

Several 1D models normalize designs into `[0, 1]` for training and denormalize
on the way out, with bounds fitted on the training split. Those bounds are part
of the model -- a design decoded against different bounds is a different design
-- but a plain `Normalizer` is not an `nn.Module`, so they never reached the
state dict. Recording them here is what makes such a checkpoint reproducible.
"""

_MAX_LISTED_PACKAGES = 8
"""How many sibling packages an error message names before summarizing.

A sweep publishes hundreds; a wall of them is not more helpful than a handful
plus a count.
"""

DESCRIPTIVE_FILES = frozenset({RUN_CONFIG_FILE, METADATA_FILE, METRICS_FILE})
"""Package files that describe the checkpoint rather than being part of it.

Neither hashed as raw bytes nor served as loadable checkpoint files. `metrics.json`
belongs here for a reason worth stating: it is written *after* evaluation, into
the package it describes, so counting it would let scoring a checkpoint change
that checkpoint's identity.

`metadata.json` is excluded as a *file* but not as *content*: the fields in it
that change what the model computes are folded into the hash separately, by
`identity_metadata`. Hashing the file whole would fail the other way, since it
records the package's own path and revision, which differ between the two
locations one run writes.
"""

ADDRESS_METADATA_FIELDS = frozenset(
    {
        "hf_repo_id",
        "hf_package_path",
        "hf_config_package_path",
        "hf_revision",
        "hf_config_revision",
        "promoted_from",
        "wandb_entity",
        "wandb_project",
        "wandb_run_id",
        "wandb_run_url",
        PACKAGE_COMPLETE_FIELD,
    }
)
"""Metadata fields describing *where a package sits*, not *what it computes*.

Excluded from the content hash. Everything else in `metadata.json` is included,
and that direction is deliberate: a field that changes the model's output but is
missed by the hash lets a behaviourally different checkpoint inherit an old
score, while a field that is merely bookkeeping but gets hashed only causes a
harmless re-evaluation. An allowlist would fail the dangerous way round every
time someone adds a field and forgets to list it.

`package_complete` is here because an incomplete package is now refused at load,
so it can never reach a leaderboard row to be confused with a complete one.
"""


def identity_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """The part of a package's metadata that changes what the model computes.

    Preprocessing state lives here -- `condition_stats`, `condition_normalizer`,
    `design_normalizer`, `condition_keys`. Those are fitted on the training split
    and replayed at load, so two packages with byte-identical weights and
    different normalizer bounds *decode differently* and are different models.
    Leaving them out of the hash would let `--skip-existing` hand one of them the
    other's leaderboard row.
    """
    return {key: value for key, value in metadata.items() if key not in ADDRESS_METADATA_FIELDS}


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
    repo_id: str | None = None
    """HF repo the package came from; None for local packages.

    Recorded so a leaderboard row can name where its weights live. Without it a
    row published by anyone outside the default entity points at nothing
    fetchable, and no third party can re-run the evaluation that produced it.
    """
    package_path: str | None = None
    """Path of the package inside `repo_id`, e.g. `beams2d/cfg_023dd1fb/seed_1`."""

    @property
    def reference(self) -> str | None:
        """A `hf://entity/repo/path` reference that round-trips through `resolve_checkpoint_reference`."""
        if self.repo_id is None or self.package_path is None:
            return None
        return f"hf://{self.repo_id}/{self.package_path}"


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
    condition_normalizer: dict[str, Any] | None = None,
    design_normalizer: dict[str, Any] | None = None,
    package_complete: bool = True,
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

    A model whose preprocessing is a min/max `Normalizer` rather than a mean/std
    rescaling records it through `condition_normalizer` / `design_normalizer`
    (see `engiopt.transforms.normalizer_state`). The principle is the same one:
    whatever a model fitted on the training split travels with the weights, so
    loading never has to re-derive it from whatever dataset happens to be current.

    Multi-stage models pass `package_complete=False` on every upload but the
    last, so an interrupted run leaves a package that says it is unfinished
    rather than one that merely fails to load.

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
        condition_normalizer=condition_normalizer,
        design_normalizer=design_normalizer,
        package_complete=package_complete,
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
        # only a finished default-hyperparameter run may claim it. Otherwise a
        # multi-stage model publishes the bare model name at stage 0 and leaves
        # it unloadable for as long as the remaining stages take -- hours, for
        # VQGAN -- or permanently if the run dies, which 4 of 102 did.
        if is_default_config and package_complete:
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

    Raises:
        FileNotFoundError: If no backend could supply the package.
    """
    errors: list[str] = []
    repo_id = build_hf_repo_id(hf_entity, hf_repo_prefix, algo)
    if model_source in {"auto", "hf"}:
        try:
            return _resolve_hf_package(
                repo_id=repo_id,
                package_path=build_hf_package_path(problem_id, seed, extra_path_parts),
                required_files=required_files,
            )
        except FileNotFoundError as exc:
            hint = _sibling_packages_hint(repo_id, problem_id, seed, extra_path_parts)
            if model_source == "hf":
                raise FileNotFoundError(f"{exc}{hint}") from exc
            errors.append(f"hf: {exc}{hint}")
        except Exception as exc:
            if model_source == "hf":
                raise
            errors.append(f"hf: {exc}")

    if local_model_dir is not None and model_source in {"auto", "local"}:
        return _resolve_local_package(local_model_dir, required_files)

    attempted = ", ".join(errors) if errors else "no backends attempted"
    raise FileNotFoundError(f"Unable to resolve checkpoint for {algo}/{problem_id}/seed_{seed}: {attempted}")


def list_packages(repo_id: str, problem_id: str | None = None) -> list[str]:
    """Every checkpoint package path in a model repo, optionally for one problem.

    A package is a directory holding a `run_config.json`, which is what every
    upload writes and what loading requires.

    Args:
        repo_id: HF model repo, e.g. `IDEALLab/engiopt-cgan-cnn-2d`.
        problem_id: Restrict to one problem's packages.

    Returns:
        Sorted package paths, e.g. `["beams2d/cfg_023dd1fb/seed_42", ...]`.
    """
    api = HfApi()
    prefix = f"{_sanitize_path_component(problem_id)}/" if problem_id else ""
    return sorted(
        entry[: -len(f"/{RUN_CONFIG_FILE}")]
        for entry in api.list_repo_files(repo_id=repo_id, repo_type="model")
        if entry.endswith(f"/{RUN_CONFIG_FILE}") and entry.startswith(prefix)
    )


def _sibling_packages_hint(repo_id: str, problem_id: str, seed: int, extra_path_parts: list[str] | None) -> str:
    """Name the packages that *do* exist when the requested one does not.

    The common failure is asking for the canonical `{problem}/seed_N` when only
    sweep configurations were ever published -- every arm of a sweep varies some
    hyperparameter, so none of them is the default run that claims the canonical
    path. "Not found" alone sends the reader looking for a broken path; listing
    the neighbours shows them what to pass to `--config-fingerprints`, or that
    the canonical run is simply missing and needs training or promoting.
    """
    try:
        available = list_packages(repo_id, problem_id)
    except Exception:  # noqa: BLE001 - a hint must never replace the original error
        return ""
    if not available:
        return f" No packages for {problem_id!r} exist in {repo_id}."
    shown = ", ".join(available[:_MAX_LISTED_PACKAGES]) + (
        f", ... ({len(available)} total)" if len(available) > _MAX_LISTED_PACKAGES else ""
    )
    hint = f" That repo does hold: {shown}."
    if not extra_path_parts:
        hint += (
            f" Nothing claims the canonical {problem_id}/seed_{seed}, which only a run using the training "
            "script's default hyperparameters writes. Pass --config-fingerprints to score one of the above, "
            "or run `python -m engiopt.promote_checkpoint` to make one of them canonical."
        )
    return hint


def promote_to_canonical(
    *,
    hf_entity: str,
    hf_repo_prefix: str,
    problem_id: str,
    algo: str,
    seed: int,
    config_fingerprint: str,
    token: str | None = None,
) -> str:
    """Copy a configuration's package to the canonical `{problem_id}/seed_{seed}` path.

    The canonical path is what a bare model name resolves to, and normally only
    a default-hyperparameter run writes it. A sweep has no such run -- every arm
    varies something -- so a sweep alone leaves the canonical path empty and the
    documented `--seeds 1` command resolving nothing. This promotes one already
    trained arm into that role without retraining it.

    The promoted package keeps its own `config_fingerprint` in metadata, so a
    leaderboard row still records which configuration actually earned the score;
    only the *address* changes.

    Args:
        hf_entity: HF org/user holding the checkpoint repos.
        hf_repo_prefix: Prefix of the per-model-family repo.
        problem_id: Problem the checkpoint was trained on.
        algo: Model family, selecting the repo.
        seed: Training seed; the promoted package keeps it.
        config_fingerprint: Which configuration to promote.
        token: HF token; falls back to the ambient login.

    Returns:
        The canonical path written.

    Raises:
        FileNotFoundError: If the source package is absent or incomplete.
    """
    repo_id = build_hf_repo_id(hf_entity, hf_repo_prefix, algo)
    source_path = build_hf_package_path(problem_id, seed, [f"cfg_{config_fingerprint}"])
    canonical_path = build_hf_package_path(problem_id, seed)

    snapshot = snapshot_download(repo_id=repo_id, repo_type="model", allow_patterns=[f"{source_path}/*"], token=token)
    source_dir = Path(snapshot) / source_path
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Cannot promote {repo_id}/{source_path}: it does not exist.")

    metadata_path = source_dir / METADATA_FILE
    metadata = _read_json(str(metadata_path)) if metadata_path.exists() else {}
    if metadata.get(PACKAGE_COMPLETE_FIELD) is False:
        raise FileNotFoundError(
            f"Refusing to promote {repo_id}/{source_path}: it is marked incomplete, so the canonical "
            "path would resolve to a package that cannot be loaded."
        )

    api = HfApi(token=token)
    with tempfile.TemporaryDirectory() as tmpdir:
        stage_dir = Path(tmpdir)
        for entry in source_dir.iterdir():
            # Metrics describe the score of the package at its old address; the
            # promoted copy has not been evaluated under its new name yet.
            if entry.is_file() and entry.name != METRICS_FILE:
                shutil.copy2(entry, stage_dir / entry.name)
        _write_json(
            stage_dir / METADATA_FILE,
            {**metadata, "hf_package_path": canonical_path, "promoted_from": source_path},
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(stage_dir),
            path_in_repo=canonical_path,
            commit_message=f"Promote cfg_{config_fingerprint} to canonical {canonical_path}",
            # A promotion copies a whole package, so it replaces whatever held
            # the canonical path before. Leaving the previous occupant's files
            # in place would blend two configurations under the name that a bare
            # `--seeds 1` resolves to.
            delete_patterns=["*"],
        )
    return canonical_path


def resolve_checkpoint_reference(
    *,
    model_source: ModelSource,
    model_ref: str,
    required_files: list[str] | None = None,
    revision: str | None = None,
) -> ResolvedCheckpoint:
    """Resolve a checkpoint package from an explicit `hf://` or local reference.

    Args:
        model_source: `auto` infers `local` for an existing directory, else `hf`.
        model_ref: `hf://<entity>/<repo>/<package_path>`, or a local directory.
        required_files: Files the package must contain.
        revision: Repo commit to fetch. Pass the revision a leaderboard row
            recorded to re-read exactly the package it was scored on; without it
            the current head is fetched, which is a different question and can
            be a different model.

    Raises:
        ValueError: If the reference cannot be interpreted.
    """
    inferred_source = model_source
    if model_source == "auto":
        inferred_source = "local" if os.path.isdir(model_ref.removeprefix("file://")) else "hf"

    if inferred_source == "hf":
        repo_id, package_path = _parse_hf_reference(model_ref)
        return _resolve_hf_package(
            repo_id=repo_id, package_path=package_path, required_files=required_files or [], revision=revision
        )
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
            delete_patterns=_stale_file_patterns(metadata),
        )
    return getattr(commit_info, "oid", None)


def _stale_file_patterns(metadata: dict[str, Any]) -> list[str] | None:
    """What to delete from a package path that this upload is not replacing.

    `upload_folder` overwrites the files it carries and leaves everything else
    alone, so re-training a configuration into a path that already held a
    *larger* set of files produces a package mixing two runs' weights. Discovery
    then serves that mixture, and it loads: the filenames are all present, so
    nothing complains.

    Which files are stale depends on whether this upload is the whole package:

    - **Complete** (`package_complete=True`): these files *are* the package, so
      anything else at the path is left over from a previous run and goes. That
      includes `metrics.json`, which described weights that no longer exist.
    - **Incomplete**: a stage of a multi-stage run, deliberately additive.
      VQGAN's second stage uploads `vqgan.pth` and `discriminator.pth` without
      re-uploading the `cvqgan.pth` its first stage wrote, so deleting here
      would destroy the earlier stage's work.
    """
    return ["*"] if metadata.get(PACKAGE_COMPLETE_FIELD, True) else None


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
        root_dir=root_dir,
        required_files=required_files,
        source="hf",
        revision=os.path.basename(repo_snapshot),
        repo_id=repo_id,
        package_path=package_path,
    )


def _resolve_local_package(local_model_dir: str, required_files: list[str]) -> ResolvedCheckpoint:
    if not os.path.isdir(local_model_dir):
        raise FileNotFoundError(f"Local checkpoint directory not found: {local_model_dir}")
    return _load_package_from_directory(root_dir=local_model_dir, required_files=required_files, source="local")


def hash_package_contents(root_dir: str) -> str:
    """Content hash of a package: everything that decides what it generates.

    This is the identity a leaderboard row is traced back to, so the property it
    must have is that *any* change to what the model computes changes the hash.
    Two things decide that, and both are covered:

    **The weight files.** Hashed by what the package *contains* rather than what
    the model declared it needs. A model may load a file it did not list --
    VQGAN's conditional variant reads `cvqgan.pth`, which is not in its
    `checkpoint_files` because the unconditional variant has none -- and two
    packages differing only in that file generate differently. Hashing the
    declared list would give them the same identity.

    **The fitted preprocessing.** Normalizer bounds and condition statistics are
    replayed at load and rescale both what goes into the network and what comes
    out, so they are part of the model even though they never reached a state
    dict. See `identity_metadata` for exactly which fields count.

    Two files are excluded outright:

    - `run_config.json` -- the hyperparameters are already what
      `config_fingerprint` identifies, and it carries operational keys (W&B
      entity, tracking flags) that do not change a single output value.
    - `metrics.json` -- written by `publish_checkpoint_metrics` *after* the
      weights, into this same directory. Including it would make attaching a
      score change the identity of the thing scored: the next `--skip-existing`
      run would see a new hash, re-evaluate, rewrite the metrics, and change the
      hash again -- an evaluation loop with no fixed point.

    Args:
        root_dir: Local directory holding the package.

    Returns:
        A 16-character hash over the weight bytes and the identity metadata.
    """
    weight_files = sorted(
        entry
        for entry in os.listdir(root_dir)
        if entry not in DESCRIPTIVE_FILES and os.path.isfile(os.path.join(root_dir, entry))
    )
    hasher = hashlib.sha256()
    for file_name in weight_files:
        hasher.update(file_name.encode())
        with open(os.path.join(root_dir, file_name), "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)

    metadata_path = os.path.join(root_dir, METADATA_FILE)
    metadata = _read_json(metadata_path) if os.path.exists(metadata_path) else {}
    identity = identity_metadata(metadata)
    if identity:
        hasher.update(json.dumps(identity, sort_keys=True, default=str).encode())
    return hasher.hexdigest()[:16]


def _load_package_from_directory(
    *,
    root_dir: str,
    required_files: list[str],
    source: Literal["hf", "local"],
    revision: str | None = None,
    repo_id: str | None = None,
    package_path: str | None = None,
) -> ResolvedCheckpoint:
    run_config_path = os.path.join(root_dir, RUN_CONFIG_FILE)
    metadata_path = os.path.join(root_dir, METADATA_FILE)
    if not os.path.exists(run_config_path):
        raise FileNotFoundError(f"Missing {RUN_CONFIG_FILE} in {root_dir}")
    run_config = _read_json(run_config_path)
    metadata = _read_json(metadata_path) if os.path.exists(metadata_path) else {}
    # Checked before the files are even looked at. A stage that happens to have
    # written every file the *next* stage's loader asks for is still a package
    # from a run that did not finish, and it must not load just because the
    # filenames line up -- that is precisely the case where a silently wrong
    # model reaches a leaderboard row.
    if metadata.get(PACKAGE_COMPLETE_FIELD) is False:
        raise FileNotFoundError(_incomplete_package_message(root_dir, metadata))
    package_files = required_files or _discover_package_files(root_dir, metadata)
    files = {file_name: os.path.join(root_dir, file_name) for file_name in package_files}
    missing = [file_name for file_name, file_path in files.items() if not os.path.exists(file_path)]
    if missing:
        raise FileNotFoundError(_missing_files_message(root_dir, missing, metadata))
    return ResolvedCheckpoint(
        source=source,
        root_dir=root_dir,
        files=files,
        run_config=run_config,
        metadata=metadata,
        revision=revision,
        content_hash=hash_package_contents(root_dir),
        repo_id=repo_id,
        package_path=package_path,
    )


def _present_files(root_dir: str) -> list[str]:
    """The package's weight files, for error messages that say how far a run got."""
    return sorted(entry for entry in os.listdir(root_dir) if entry not in DESCRIPTIVE_FILES)


def _incomplete_package_message(root_dir: str, metadata: dict[str, Any]) -> str:
    """Explain a package whose run never finished.

    A multi-stage model uploads after each stage, so a run killed partway leaves
    a package that has a `run_config.json` and a `metadata.json` and therefore
    looks like a checkpoint from the outside. `package_complete` is what tells
    the two apart, and saying which stage it stopped at turns "file not found"
    into something the reader can act on.
    """
    present = _present_files(root_dir)
    stage = metadata.get("stage")
    reached = f" It reached the {stage!r} stage." if stage else ""
    return (
        f"Refusing to load {root_dir}: it is marked incomplete, so the training run that wrote it "
        f"did not finish.{reached} The package holds {present or 'no weight files'}. "
        "Re-run training to completion; a partial package cannot be loaded."
    )


def _missing_files_message(root_dir: str, missing: list[str], metadata: dict[str, Any]) -> str:
    """Explain a finished package that is nevertheless missing weights.

    Reached only for packages marked complete -- an unfinished one is refused
    before file discovery, by `_incomplete_package_message` -- so this must not
    blame the run for stopping early. Something else is wrong: the wrong
    `checkpoint_files` for this model, or an upload that lost a file.
    """
    del metadata  # completeness is decided before this point
    present = _present_files(root_dir)
    return f"Missing checkpoint file(s) {missing} in {root_dir}; the package holds {present or 'no weight files'}."


def _discover_package_files(root_dir: str, metadata: dict[str, Any]) -> list[str]:
    primary_files = metadata.get("primary_files")
    if isinstance(primary_files, list) and primary_files:
        return [str(file_name) for file_name in primary_files]

    discovered = [
        entry
        for entry in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, entry)) and entry not in DESCRIPTIVE_FILES
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
    condition_normalizer: dict[str, Any] | None = None,
    design_normalizer: dict[str, Any] | None = None,
    package_complete: bool = True,
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
            PACKAGE_COMPLETE_FIELD: package_complete,
        }
    )
    if condition_keys is not None:
        payload["condition_keys"] = list(condition_keys)
    if condition_stats is not None:
        payload["condition_stats"] = {key: [float(v) for v in values] for key, values in condition_stats.items()}
    if condition_normalizer is not None:
        payload[CONDITION_NORMALIZER_METADATA_FIELD] = condition_normalizer
    if design_normalizer is not None:
        payload[DESIGN_NORMALIZER_METADATA_FIELD] = design_normalizer
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
