#!/usr/bin/env python3
"""Create and audit manifests for the conditional flow-matching experiments."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from datetime import timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

FLOW_CONFIGS = (
    ("euler", 16),
    ("euler", 32),
    ("euler", 48),
    ("midpoint", 8),
    ("midpoint", 16),
    ("midpoint", 24),
    ("rk4", 4),
    ("rk4", 8),
    ("rk4", 12),
)
PROBLEMS = ("beams2d", "heatconduction2d")
BASELINES = (
    ("diffusion_2d_cond", "timesteps_1000"),
    ("cgan_cnn_2d", "activation_sigmoid"),
)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _git_commit(path: str | Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gpu(seed: int, rtx_4090_seeds: set[int]) -> str:
    return "rtx_4090" if seed in rtx_4090_seeds else "rtx_3090"


def _flow_experiments(seeds: tuple[int, ...], rtx_4090_seeds: set[int]) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    n_seeds = len(seeds)
    for config_index, (method, steps) in enumerate(FLOW_CONFIGS):
        package_label = f"{method}_{steps}"
        for problem_index, problem_id in enumerate(PROBLEMS):
            for seed_index, seed in enumerate(seeds):
                train_task_id = config_index * len(PROBLEMS) * n_seeds + problem_index * n_seeds + seed_index
                eval_task_id = (
                    seed_index * len(PROBLEMS) * len(FLOW_CONFIGS) + problem_index * len(FLOW_CONFIGS) + config_index
                )
                experiments.append(
                    {
                        "key": f"flow_matching_2d_cond/{problem_id}/{package_label}/seed_{seed}",
                        "model_id": "flow_matching_2d_cond",
                        "problem_id": problem_id,
                        "seed": seed,
                        "method": method,
                        "steps": steps,
                        "package_label": package_label,
                        "gpu": _gpu(seed, rtx_4090_seeds),
                        "train_task_id": train_task_id,
                        "eval_task_id": eval_task_id,
                        "csv_filename": f"tourn_{problem_id}_{method}_{steps}_s{seed}.csv",
                    }
                )
    return experiments


def _baseline_experiments(seeds: tuple[int, ...], rtx_4090_seeds: set[int], task_offset: int) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    n_seeds = len(seeds)
    for model_index, (model_id, package_label) in enumerate(BASELINES):
        for problem_index, problem_id in enumerate(PROBLEMS):
            for seed_index, seed in enumerate(seeds):
                task_id = task_offset + model_index * len(PROBLEMS) * n_seeds + problem_index * n_seeds + seed_index
                experiments.append(
                    {
                        "key": f"{model_id}/{problem_id}/{package_label}/seed_{seed}",
                        "model_id": model_id,
                        "problem_id": problem_id,
                        "seed": seed,
                        "method": None,
                        "steps": None,
                        "package_label": package_label,
                        "gpu": _gpu(seed, rtx_4090_seeds),
                        "train_task_id": task_id,
                        "eval_task_id": task_id,
                        "csv_filename": f"metrics_{model_id}_na_{problem_id}_seed{seed}.csv",
                    }
                )
    return experiments


def full_experiments(rtx_4090_seeds: set[int]) -> list[dict[str, Any]]:
    """Return the complete 220-experiment campaign mapping."""
    seeds = tuple(range(1, 11))
    flow = _flow_experiments(seeds, rtx_4090_seeds)
    return flow + _baseline_experiments(seeds, rtx_4090_seeds, len(flow))


def smoke_experiments() -> list[dict[str, Any]]:
    """Return the three-experiment public smoke-test mapping."""
    return [
        {
            "key": "flow_matching_2d_cond/beams2d/euler_32/seed_1",
            "model_id": "flow_matching_2d_cond",
            "problem_id": "beams2d",
            "seed": 1,
            "method": "euler",
            "steps": 32,
            "package_label": "euler_32",
            "gpu": "rtx_4090",
            "train_task_id": 0,
            "eval_task_id": 0,
            "csv_filename": "smoke_flow_matching_2d_cond_beams2d_seed1.csv",
        },
        {
            "key": "diffusion_2d_cond/beams2d/timesteps_1000/seed_1",
            "model_id": "diffusion_2d_cond",
            "problem_id": "beams2d",
            "seed": 1,
            "method": None,
            "steps": None,
            "package_label": "timesteps_1000",
            "gpu": "rtx_4090",
            "train_task_id": 1,
            "eval_task_id": 1,
            "csv_filename": "smoke_diffusion_2d_cond_beams2d_seed1.csv",
        },
        {
            "key": "cgan_cnn_2d/beams2d/activation_sigmoid/seed_1",
            "model_id": "cgan_cnn_2d",
            "problem_id": "beams2d",
            "seed": 1,
            "method": None,
            "steps": None,
            "package_label": "activation_sigmoid",
            "gpu": "rtx_4090",
            "train_task_id": 2,
            "eval_task_id": 2,
            "csv_filename": "smoke_cgan_cnn_2d_beams2d_seed1.csv",
        },
    ]


def create_manifest(args: argparse.Namespace) -> None:
    """Create a complete smoke or full-campaign manifest."""
    rtx_4090_seeds = {int(seed) for seed in args.rtx_4090_seeds.split(",") if seed}
    if args.mode == "full":
        if not rtx_4090_seeds.issubset(set(range(1, 11))):
            raise ValueError("RTX 4090 seeds must be a subset of 1..10")
        experiments = full_experiments(rtx_4090_seeds)
        seeds = list(range(1, 11))
    else:
        experiments = smoke_experiments()
        seeds = [1]

    manifest = {
        "schema_version": 1,
        "campaign": args.campaign,
        "mode": args.mode,
        "release": args.release,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "engiopt_commit": args.engiopt_commit,
            "engibench_commit": args.engibench_commit,
        },
        "tracking": {
            "wandb_enabled": args.wandb_enabled,
            "wandb_entity": args.wandb_entity,
            "wandb_project": args.wandb_project,
            "wandb_group": args.wandb_group,
            "checkpoint_publication_enabled": args.checkpoint_publication_enabled,
            "hf_entity": args.hf_entity,
            "hf_repo_prefix": args.hf_repo_prefix,
        },
        "protocol": {
            "problems": ["beams2d"] if args.mode == "smoke" else list(PROBLEMS),
            "training_seeds": seeds,
            "max_epochs": 5 if args.mode == "smoke" else 500,
            "validation_interval_epochs": 1 if args.mode == "smoke" else 10,
            "checkpoint_interval_epochs": 1 if args.mode == "smoke" else 10,
            "min_epoch_for_selection": 1 if args.mode == "smoke" else 80,
            "early_stopping_patience_checks": 10 if args.mode == "smoke" else 25,
            "validation_batch_size": 2 if args.mode == "smoke" else 50,
            "shortlist_metric": "validation_mmd",
            "shortlist_size": 5,
            "selection_metric": "validation_cog",
            "selection_statistic": "median",
            "selection_batch_size": 2 if args.mode == "smoke" else 50,
            "test_batch_size": 2 if args.mode == "smoke" else 50,
            "diffusion_num_timesteps": 1000,
            "cgan_generator_output_activation": "sigmoid",
            "flow_configurations": [{"method": method, "steps": steps} for method, steps in FLOW_CONFIGS],
        },
        "hardware": {
            "rtx_4090_seeds": [1] if args.mode == "smoke" else sorted(rtx_4090_seeds),
            "rtx_3090_seeds": [] if args.mode == "smoke" else sorted(set(range(1, 11)) - rtx_4090_seeds),
            "rtx_4090_concurrency": args.concurrency_4090,
            "rtx_3090_concurrency": args.concurrency_3090,
            "generation_timing_gpu": "rtx_4090",
        },
        "expected": {
            "training_runs": len(experiments),
            "evaluation_runs": len(experiments),
            "selected_bundles": len(experiments),
            "experiments": experiments,
        },
        "jobs": {},
    }
    _write_json(args.output, manifest)
    print(f"Wrote {args.mode} manifest with {len(experiments)} experiments: {args.output}")


def _compress_ids(values: list[int]) -> str:
    if not values:
        return ""
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def task_ids(args: argparse.Namespace) -> None:
    """Print compressed Slurm task IDs for one phase and GPU type."""
    manifest = _read_json(args.manifest)
    key = "train_task_id" if args.phase == "train" else "eval_task_id"
    values = sorted(
        {int(experiment[key]) for experiment in manifest["expected"]["experiments"] if experiment["gpu"] == args.gpu}
    )
    print(_compress_ids(values))


def preflight(args: argparse.Namespace) -> None:
    """Verify source revisions and explicit publication destinations."""
    manifest = _read_json(args.manifest)
    errors: list[str] = []
    for name, path in (("engiopt", args.engiopt_repo), ("engibench", args.engibench_repo)):
        expected = manifest["source"][f"{name}_commit"]
        actual = _git_commit(path)
        if actual != expected:
            errors.append(f"{name} commit mismatch: expected {expected}, got {actual}")
    if manifest["tracking"]["wandb_enabled"]:
        if not manifest["tracking"]["wandb_project"]:
            errors.append("W&B tracking is enabled but wandb_project is empty")
        if not manifest["tracking"]["wandb_entity"]:
            errors.append("W&B tracking is enabled but wandb_entity is empty")
    if manifest["tracking"]["checkpoint_publication_enabled"] and (
        not manifest["tracking"]["hf_entity"] or not manifest["tracking"]["hf_repo_prefix"]
    ):
        errors.append("Checkpoint publication requires hf_entity and hf_repo_prefix")
    if errors:
        raise SystemExit("Preflight failed:\n- " + "\n- ".join(errors))
    print("Reproduction preflight passed.")


def record_jobs(args: argparse.Namespace) -> None:
    """Record submitted Slurm job IDs without changing the protocol."""
    manifest = _read_json(args.manifest)
    for item in args.job:
        name, separator, job_id = item.partition("=")
        if not separator or not name or not job_id:
            raise ValueError(f"Invalid --job value: {item}; expected name=job_id")
        manifest.setdefault("jobs", {})[name] = job_id
    manifest["jobs_recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(args.manifest, manifest)


def _selected_bundle(staging_root: Path, experiment: dict[str, Any], release: str) -> Path:
    return staging_root.joinpath(
        experiment["model_id"],
        experiment["problem_id"],
        "selected",
        release,
        experiment["package_label"],
        f"seed_{experiment['seed']}",
    )


def _audit_selected_bundle(bundle: Path, *, require_published: bool) -> tuple[list[str], Counter[str]]:
    errors: list[str] = []
    counts: Counter[str] = Counter()
    metadata_path = bundle / "metadata.json"
    if not metadata_path.exists():
        return [f"Missing selected bundle metadata: {metadata_path}"], counts

    counts["selected_bundles"] += 1
    metadata = _read_json(metadata_path)
    checkpoint_name = str(metadata.get("selected_checkpoint_filename", ""))
    checkpoint_path = bundle / checkpoint_name
    if not checkpoint_name or not checkpoint_path.exists():
        errors.append(f"Missing selected checkpoint in {bundle}")
    elif metadata.get("selected_checkpoint_sha256") != _sha256(checkpoint_path):
        errors.append(f"Checkpoint checksum mismatch: {checkpoint_path}")

    evidence_files = (
        "run_config.json",
        "validation_metrics.json",
        "selection_results.json",
        "selection_results.csv",
    )
    errors.extend(
        f"Missing selection evidence: {bundle / evidence}"
        for evidence in evidence_files
        if not (bundle / evidence).exists()
    )
    if require_published:
        receipt = bundle / "upload_receipt.json"
        if receipt.exists():
            counts["published_bundles"] += 1
        else:
            errors.append(f"Missing upload receipt: {receipt}")
    return errors, counts


def _timing_csv(timing_csv_dir: Path, experiment: dict[str, Any]) -> Path:
    method_suffix = experiment["package_label"] if experiment["model_id"] == "flow_matching_2d_cond" else "na"
    return timing_csv_dir / (
        f"generation_timing_{experiment['model_id']}_{method_suffix}_"
        f"{experiment['problem_id']}_seed{experiment['seed']}.csv"
    )


def _audit_experiment_outputs(
    experiment: dict[str, Any],
    csv_dir: Path,
    timing_csv_dir: Path | None,
) -> tuple[list[str], Counter[str]]:
    errors: list[str] = []
    counts: Counter[str] = Counter()
    csv_path = csv_dir / experiment["csv_filename"]
    if csv_path.exists():
        counts["evaluation_csv"] += 1
    else:
        errors.append(f"Missing evaluation CSV: {csv_path}")

    if timing_csv_dir is not None:
        timing_path = _timing_csv(timing_csv_dir, experiment)
        if timing_path.exists():
            counts["generation_timing_csv"] += 1
        else:
            errors.append(f"Missing generation timing CSV: {timing_path}")
    return errors, counts


def _audit_qualitative(qualitative_root: Path) -> tuple[list[str], Counter[str]]:
    errors: list[str] = []
    counts: Counter[str] = Counter()
    for problem_id, seed in (("beams2d", 8), ("heatconduction2d", 5)):
        bundle = qualitative_root / f"{problem_id}_seed{seed}"
        if bundle.is_dir() and any(bundle.iterdir()):
            counts["qualitative_bundles"] += 1
        else:
            errors.append(f"Missing qualitative bundle: {bundle}")
    return errors, counts


def audit_local(args: argparse.Namespace) -> None:
    """Verify local selected bundles, evidence, checksums, and CSV files."""
    manifest = _read_json(args.manifest)
    staging_root = Path(args.staging_root)
    csv_dir = Path(args.csv_dir)
    timing_csv_dir = Path(args.timing_csv_dir) if args.timing_csv_dir else None
    qualitative_root = Path(args.qualitative_root) if args.qualitative_root else None
    if args.require_timing and timing_csv_dir is None:
        raise ValueError("--require-timing requires --timing-csv-dir")
    if args.require_qualitative and qualitative_root is None:
        raise ValueError("--require-qualitative requires --qualitative-root")

    errors: list[str] = []
    counts: Counter[str] = Counter()
    for experiment in manifest["expected"]["experiments"]:
        bundle = _selected_bundle(staging_root, experiment, manifest["release"])
        bundle_errors, bundle_counts = _audit_selected_bundle(
            bundle,
            require_published=args.require_published,
        )
        errors.extend(bundle_errors)
        counts.update(bundle_counts)

        output_errors, output_counts = _audit_experiment_outputs(
            experiment,
            csv_dir,
            timing_csv_dir if args.require_timing else None,
        )
        errors.extend(output_errors)
        counts.update(output_counts)

    if args.require_qualitative and qualitative_root is not None:
        qualitative_errors, qualitative_counts = _audit_qualitative(qualitative_root)
        errors.extend(qualitative_errors)
        counts.update(qualitative_counts)

    result = {
        "ok": not errors,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": dict(counts),
        "errors": errors,
    }
    _write_json(args.output, result)
    if errors:
        raise SystemExit(f"Audit failed with {len(errors)} error(s); see {args.output}")
    print(f"Audit passed: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="Create a smoke or full campaign manifest")
    manifest.add_argument("--mode", choices=("smoke", "full"), required=True)
    manifest.add_argument("--campaign", default="conditional-flow-matching-reproduction-v1")
    manifest.add_argument("--release", default="v1")
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--engiopt-commit", required=True)
    manifest.add_argument("--engibench-commit", required=True)
    manifest.add_argument("--rtx-4090-seeds", default="1,2,3")
    manifest.add_argument("--concurrency-4090", type=int, default=6)
    manifest.add_argument("--concurrency-3090", type=int, default=20)
    manifest.add_argument("--wandb-enabled", action="store_true")
    manifest.add_argument("--wandb-entity", default="")
    manifest.add_argument("--wandb-project", default="")
    manifest.add_argument("--wandb-group", default="")
    manifest.add_argument("--checkpoint-publication-enabled", action="store_true")
    manifest.add_argument("--hf-entity", default="")
    manifest.add_argument("--hf-repo-prefix", default="")
    manifest.set_defaults(func=create_manifest)

    ids = subparsers.add_parser("task-ids", help="Print compressed task IDs for one phase and GPU")
    ids.add_argument("--manifest", required=True)
    ids.add_argument("--phase", choices=("train", "eval"), required=True)
    ids.add_argument("--gpu", choices=("rtx_4090", "rtx_3090"), required=True)
    ids.set_defaults(func=task_ids)

    check = subparsers.add_parser("preflight", help="Verify source commits and publication settings")
    check.add_argument("--manifest", required=True)
    check.add_argument("--engiopt-repo", required=True)
    check.add_argument("--engibench-repo", required=True)
    check.set_defaults(func=preflight)

    jobs = subparsers.add_parser("record-jobs", help="Record submitted Slurm job IDs")
    jobs.add_argument("--manifest", required=True)
    jobs.add_argument("--job", action="append", default=[])
    jobs.set_defaults(func=record_jobs)

    audit = subparsers.add_parser("audit-local", help="Verify selected bundles and evaluation CSV files")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--staging-root", required=True)
    audit.add_argument("--csv-dir", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--require-published", action="store_true")
    audit.add_argument("--require-timing", action="store_true")
    audit.add_argument("--timing-csv-dir")
    audit.add_argument("--require-qualitative", action="store_true")
    audit.add_argument("--qualitative-root")
    audit.set_defaults(func=audit_local)
    return parser


def main() -> None:
    """Run the requested campaign-manifest command."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
