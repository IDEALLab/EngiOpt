"""Evaluation for the CGAN 2D w/ CNN."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import time
from typing import Any, Literal

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import torch as th
import tyro
import wandb

from engiopt import metrics
from engiopt.cgan_cnn_2d.cgan_cnn_2d import Generator
from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.reporting import build_display_name
from engiopt.reporting import write_metrics_csv
from engiopt.topk_checkpoint_bundle import restore_topk_checkpoint_dir


@dataclasses.dataclass
class Args:
    """Command-line arguments for a single-seed cGAN CNN 2D evaluation."""

    problem_id: str = "beams2d"
    """Problem identifier."""
    seed: int = 1
    """Random seed to run."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name."""
    hf_entity: str = "IDEALLab"
    """HF org/user where checkpoint packages are stored."""
    hf_repo_prefix: str = "engiopt"
    """HF repo prefix used for model-family repositories."""
    track: bool = True
    """Log evaluation metrics and metadata to W&B."""
    run_name: str | None = None
    """Optional W&B run name override."""
    wandb_job_type: str = "evaluation"
    """W&B job type for evaluation runs."""
    n_samples: int = 50
    """Number of generated samples per seed."""
    sigma: float = 10.0
    """Kernel bandwidth for MMD and DPP metrics."""
    output_csv: str = "cgan_cnn_2d_{problem_id}_metrics.csv"
    """Output CSV path template; may include {problem_id}."""
    append_output: bool = True
    """Append to an existing CSV. Use --no-append-output to overwrite instead."""
    checkpoint_path: str | None = None
    """Optional local generator checkpoint path. Preferred over WandB artifacts when set."""
    checkpoint_dir: str | None = None
    """Optional directory containing validation_metrics.json and generator epoch checkpoints for top-k post-training selection."""
    checkpoint_source: Literal["local", "auto", "hf"] = "local"
    """Source for top-k checkpoints. 'local' preserves legacy behavior; 'auto' falls back to HF."""
    checkpoint_package_label: str | None = None
    """Optional HF package label for top-k checkpoint bundles."""
    select_best_of_top_k: bool = True
    """If True and checkpoint_dir is set, rank the shortlisted checkpoints on validation COG/FOG before final test evaluation."""
    top_k: int = 5
    """Number of validation-MMD shortlisted checkpoints to inspect during selection."""
    selection_batch_size: int = 50
    """Number of validation samples used for the COG/FOG selection pass."""
    selection_seed_offset: int = 123
    """Offset added to the base seed when sampling the validation set for selection."""
    device: str = "auto"
    """Device selection for local smoke runs and evaluation."""
    clip_min: float = 1e-3
    """Minimum value used when clipping generated designs."""
    clip_max: float = 1.0
    """Maximum value used when clipping generated designs."""
    validation_log_precision: int = 10
    """Decimal precision when formatting validation/test numeric values in image captions."""


@dataclasses.dataclass
class EvaluationContext:
    """Compact bundle of parameters needed to evaluate one checkpoint."""

    problem: Any
    device: th.device
    args: Args
    generation_seed: int
    checkpoint_source: str


def build_design_grid(designs_np: np.ndarray, rows: int = 5, cols: int = 5) -> np.ndarray:
    """Build a simple tiled image grid from a batch of 2D designs."""
    if designs_np.ndim == 4 and designs_np.shape[1] == 1:
        designs_np = designs_np[:, 0]

    if designs_np.ndim != 3:
        raise ValueError(f"Expected designs with shape (n, h, w), got {designs_np.shape}")

    n_tiles = rows * cols
    n_samples, h, w = designs_np.shape
    n_use = min(n_tiles, n_samples)

    grid = np.zeros((rows * h, cols * w), dtype=np.float32)
    for idx in range(n_use):
        r = idx // cols
        c = idx % cols
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = designs_np[idx]

    return grid


def select_device(device_arg: str) -> th.device:
    """Return the best available torch device."""
    if device_arg != "auto":
        if device_arg == "mps" and not th.backends.mps.is_available():
            raise ValueError("MPS device requested but not available")
        if device_arg == "cuda" and not th.cuda.is_available():
            raise ValueError("CUDA device requested but not available")
        return th.device(device_arg)
    if th.backends.mps.is_available():
        return th.device("mps")
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_artifact_checkpoint(
    problem_id: str,
    seed: int,
    wandb_project: str,
    wandb_entity: str | None,
    device: th.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a model checkpoint and config from a WandB artifact."""
    if wandb_entity is not None:
        artifact_path = f"{wandb_entity}/{wandb_project}/{problem_id}_cgan_cnn_2d_generator:seed_{seed}"
    else:
        artifact_path = f"{wandb_project}/{problem_id}_cgan_cnn_2d_generator:seed_{seed}"

    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")

    class RunRetrievalError(ValueError):
        def __init__(self):
            super().__init__("Failed to retrieve the run")

    run = artifact.logged_by()
    if run is None or not hasattr(run, "config"):
        raise RunRetrievalError

    artifact_dir = artifact.download()
    ckpt = th.load(os.path.join(artifact_dir, "generator.pth"), map_location=device)
    return ckpt, dict(run.config)


def load_top_k_candidates(checkpoint_dir: Path, top_k: int) -> list[dict[str, Any]]:
    """Load the shortlisted generator checkpoints saved by the training run."""
    metrics_path = checkpoint_dir / "validation_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Validation metrics file not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    top_k_epochs = summary.get("top_k_epochs", [])
    if not top_k_epochs:
        raise ValueError(f"No top-k checkpoints found in {metrics_path}")

    candidates: list[dict[str, Any]] = []
    for item in sorted(top_k_epochs, key=lambda row: float(row["metric_value"]))[:top_k]:
        epoch = int(item["epoch"])
        checkpoint_path = checkpoint_dir / f"generator_epoch_{epoch + 1:04d}.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        candidates.append(
            {
                "epoch": epoch,
                "metric_value": float(item["metric_value"]),
                "checkpoint_path": checkpoint_path,
            }
        )
    return candidates


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    conditions_tensor: th.Tensor,
    sampled_conditions,
    sampled_designs_np: np.ndarray,
    context: EvaluationContext,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate a single cGAN generator checkpoint on a fixed condition sample."""
    phase_start = time.perf_counter()
    ckpt = th.load(str(checkpoint_path), map_location=context.device)
    run_config: dict[str, Any] = dict(ckpt.get("args", {}))

    model = Generator(
        latent_dim=int(run_config["latent_dim"]),
        n_conds=len(context.problem.conditions_keys),
        design_shape=context.problem.design_space.shape,
        generator_output_activation=run_config.get("generator_output_activation", "tanh"),
    )
    model.load_state_dict(ckpt["generator"])
    model.eval()
    model.to(context.device)

    th.manual_seed(context.generation_seed)
    n_gen = conditions_tensor.shape[0]
    z = th.randn((n_gen, int(run_config["latent_dim"]), 1, 1), device=context.device, dtype=th.float)

    generation_start = time.perf_counter()
    gen_designs = model(z, conditions_tensor)
    generation_runtime_sec = time.perf_counter() - generation_start
    generation_samples_per_sec = n_gen / generation_runtime_sec if generation_runtime_sec > 0 else float("nan")

    gen_designs_np = gen_designs.detach().cpu().numpy().reshape(n_gen, *context.problem.design_space.shape)
    gen_designs_np = np.clip(gen_designs_np, context.args.clip_min, context.args.clip_max)

    metrics_start = time.perf_counter()
    metrics_dict = metrics.metrics(
        context.problem,
        gen_designs_np,
        sampled_designs_np,
        sampled_conditions,
        sigma=context.args.sigma,
        gen_time=generation_runtime_sec,
        gen_speed=generation_samples_per_sec,
    )
    metrics_runtime_sec = time.perf_counter() - metrics_start
    evaluation_runtime_sec = time.perf_counter() - phase_start

    metrics_dict.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_source": context.checkpoint_source,
            "generation_runtime_sec": generation_runtime_sec,
            "metrics_runtime_sec": metrics_runtime_sec,
            "evaluation_runtime_sec": evaluation_runtime_sec,
            "generation_samples_per_sec": generation_samples_per_sec,
            "latent_dim": int(run_config["latent_dim"]),
        }
    )
    return metrics_dict, gen_designs_np


if __name__ == "__main__":
    args = tyro.cli(Args)
    eval_start = time.perf_counter()

    seed = args.seed
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=seed)

    # Reproducibility
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = True

    device = select_device(args.device)

    out_path = args.output_csv.format(problem_id=args.problem_id)
    final_generated_designs_np: np.ndarray | None = None
    final_reference_designs_np: np.ndarray | None = None
    selection_validation_rows: list[dict[str, Any]] = []

    if args.checkpoint_path is not None:
        conditions_tensor, sampled_conditions, sampled_designs_np, _ = sample_conditions(
            problem=problem,
            n_samples=args.n_samples,
            device=device,
            seed=seed,
        )
        conditions_tensor = conditions_tensor.unsqueeze(-1).unsqueeze(-1)

        metrics_dict, final_generated_designs_np = evaluate_checkpoint(
            checkpoint_path=args.checkpoint_path,
            conditions_tensor=conditions_tensor,
            sampled_conditions=sampled_conditions,
            sampled_designs_np=sampled_designs_np,
            context=EvaluationContext(
                problem=problem,
                device=device,
                args=args,
                generation_seed=seed + 2000,
                checkpoint_source="local_checkpoint",
            ),
        )
        metrics_dict.update(
            {
                "seed": seed,
                "problem_id": args.problem_id,
                "model_id": "cgan_cnn_2d",
                "n_samples": args.n_samples,
                "sigma": args.sigma,
            }
        )
        final_reference_designs_np = sampled_designs_np
        write_metrics_csv([metrics_dict], out_path, append_output=args.append_output)
        checkpoint_source = "local_checkpoint"

    elif args.select_best_of_top_k and (args.checkpoint_dir is not None or args.checkpoint_source in {"auto", "hf"}):
        checkpoint_dir, topk_checkpoint_source = restore_topk_checkpoint_dir(
            model_id="cgan_cnn_2d",
            problem_id=args.problem_id,
            seed=args.seed,
            checkpoint_source=args.checkpoint_source,
            checkpoint_dir=args.checkpoint_dir,
            hf_entity=args.hf_entity,
            hf_repo_prefix=args.hf_repo_prefix,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            package_label=args.checkpoint_package_label,
        )
        candidates = load_top_k_candidates(checkpoint_dir, args.top_k)

        val_conditions_tensor, val_sampled_conditions, val_sampled_designs_np, _ = sample_conditions(
            problem=problem,
            n_samples=args.selection_batch_size,
            device=device,
            seed=seed + args.selection_seed_offset,
            split="val",
        )
        val_conditions_tensor = val_conditions_tensor.unsqueeze(-1).unsqueeze(-1)

        best_candidate: dict[str, Any] | None = None
        best_candidate_val_cog = float("inf")

        for rank, candidate in enumerate(candidates, 1):
            val_metrics, _ = evaluate_checkpoint(
                checkpoint_path=candidate["checkpoint_path"],
                conditions_tensor=val_conditions_tensor,
                sampled_conditions=val_sampled_conditions,
                sampled_designs_np=val_sampled_designs_np,
                context=EvaluationContext(
                    problem=problem,
                    device=device,
                    args=args,
                    generation_seed=seed + 1000 + rank,
                    checkpoint_source=topk_checkpoint_source,
                ),
            )
            selection_validation_rows.append(
                {
                    "rank": rank,
                    "epoch": int(candidate["epoch"] + 1),
                    "checkpoint_path": str(candidate["checkpoint_path"]),
                    "validation_mmd": float(candidate["metric_value"]),
                    "validation_cog": float(val_metrics["cog"]),
                    "validation_fog": float(val_metrics["fog"]),
                    "validation_eval_mmd": float(val_metrics["mmd"]),
                }
            )
            if float(val_metrics["cog"]) < best_candidate_val_cog:
                best_candidate_val_cog = float(val_metrics["cog"])
                best_candidate = candidate

        if best_candidate is None:
            raise RuntimeError("No candidate selected from top-k checkpoints")

        test_conditions_tensor, test_sampled_conditions, test_sampled_designs_np, _ = sample_conditions(
            problem=problem,
            n_samples=args.n_samples,
            device=device,
            seed=seed,
            split="test",
        )
        test_conditions_tensor = test_conditions_tensor.unsqueeze(-1).unsqueeze(-1)

        selected_rank = next(
            idx for idx, row in enumerate(selection_validation_rows, 1) if row["epoch"] == int(best_candidate["epoch"] + 1)
        )

        metrics_dict, final_generated_designs_np = evaluate_checkpoint(
            checkpoint_path=best_candidate["checkpoint_path"],
            conditions_tensor=test_conditions_tensor,
            sampled_conditions=test_sampled_conditions,
            sampled_designs_np=test_sampled_designs_np,
            context=EvaluationContext(
                problem=problem,
                device=device,
                args=args,
                generation_seed=seed + 2000,
                checkpoint_source=topk_checkpoint_source,
            ),
        )
        metrics_dict.update(
            {
                "seed": seed,
                "problem_id": args.problem_id,
                "model_id": "cgan_cnn_2d",
                "phase": "test_selected",
                "checkpoint_dir": str(checkpoint_dir),
                "selection_mode": "top_k_val_cog_select_then_test",
                "selection_top_k": args.top_k,
                "selection_batch_size": args.selection_batch_size,
                "selection_rank": selected_rank,
                "selection_candidate_epoch": best_candidate["epoch"] + 1,
                "selection_candidate_mmd": best_candidate["metric_value"],
                "selection_candidate_validation_cog": best_candidate_val_cog,
            }
        )
        metrics_dict["display_name"] = build_display_name(metrics_dict)
        write_metrics_csv([metrics_dict], out_path, append_output=args.append_output)
        checkpoint_source = topk_checkpoint_source
        final_reference_designs_np = test_sampled_designs_np

    else:
        conditions_tensor, sampled_conditions, sampled_designs_np, _ = sample_conditions(
            problem=problem,
            n_samples=args.n_samples,
            device=device,
            seed=seed,
        )
        conditions_tensor = conditions_tensor.unsqueeze(-1).unsqueeze(-1)

        ckpt, run_config = load_artifact_checkpoint(
            problem_id=args.problem_id,
            seed=seed,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            device=device,
        )
        checkpoint_source = "wandb_artifact"

        model = Generator(
            latent_dim=int(run_config["latent_dim"]),
            n_conds=len(problem.conditions_keys),
            design_shape=problem.design_space.shape,
            generator_output_activation=run_config.get("generator_output_activation", "tanh"),
        )
        model.load_state_dict(ckpt["generator"])
        model.eval()
        model.to(device)

        z = th.randn((args.n_samples, int(run_config["latent_dim"]), 1, 1), device=device, dtype=th.float)

        generation_start = time.perf_counter()
        gen_designs = model(z, conditions_tensor)
        generation_runtime_sec = time.perf_counter() - generation_start
        gen_designs_np = gen_designs.detach().cpu().numpy()
        gen_designs_np = gen_designs_np.reshape(args.n_samples, *problem.design_space.shape)
        generation_samples_per_sec = args.n_samples / generation_runtime_sec if generation_runtime_sec > 0 else float("nan")

        gen_designs_np = np.clip(gen_designs_np, args.clip_min, args.clip_max)

        metrics_start = time.perf_counter()
        metrics_dict = metrics.metrics(
            problem,
            gen_designs_np,
            sampled_designs_np,
            sampled_conditions,
            sigma=args.sigma,
            gen_time=generation_runtime_sec,
            gen_speed=generation_samples_per_sec,
        )
        metrics_runtime_sec = time.perf_counter() - metrics_start
        evaluation_runtime_sec = time.perf_counter() - eval_start
        generation_samples_per_sec = args.n_samples / generation_runtime_sec if generation_runtime_sec > 0 else float("nan")

        metrics_dict.update(
            {
                "seed": seed,
                "problem_id": args.problem_id,
                "model_id": "cgan_cnn_2d",
                "n_samples": args.n_samples,
                "sigma": args.sigma,
                "latent_dim": int(run_config["latent_dim"]),
                "checkpoint_source": checkpoint_source,
                "generation_runtime_sec": generation_runtime_sec,
                "metrics_runtime_sec": metrics_runtime_sec,
                "evaluation_runtime_sec": evaluation_runtime_sec,
                "generation_samples_per_sec": generation_samples_per_sec,
            }
        )
        final_generated_designs_np = gen_designs_np
        final_reference_designs_np = sampled_designs_np
        write_metrics_csv([metrics_dict], out_path, append_output=args.append_output)

    generation_runtime_sec = float(metrics_dict["generation_runtime_sec"])
    metrics_runtime_sec = float(metrics_dict["metrics_runtime_sec"])
    evaluation_runtime_sec = float(metrics_dict["evaluation_runtime_sec"])
    generation_samples_per_sec = float(metrics_dict["generation_samples_per_sec"])

    if args.track:
        run_name = args.run_name or f"{args.problem_id}__cgan_cnn_2d__eval__seed{seed}__{int(time.time())}"
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type=args.wandb_job_type,
            name=run_name,
            config={**vars(args), "model_id": "cgan_cnn_2d", "checkpoint_source": checkpoint_source},
        )
        if run is None:
            raise RuntimeError("Failed to initialize Weights & Biases run")
        if selection_validation_rows:
            run.log(
                {
                    "selection/validation_table": wandb.Table(
                        columns=["rank", "epoch", "validation_mmd", "validation_cog", "validation_fog", "validation_eval_mmd", "checkpoint_path"],
                        data=[
                            [
                                row["rank"],
                                row["epoch"],
                                row["validation_mmd"],
                                row["validation_cog"],
                                row["validation_fog"],
                                row["validation_eval_mmd"],
                                row["checkpoint_path"],
                            ]
                            for row in selection_validation_rows
                        ],
                    )
                }
            )
        run.log(
            {
                "cog": metrics_dict["cog"],
                "fog": metrics_dict["fog"],
                "iog_raw_objective": metrics_dict["iog"],
                "mmd": metrics_dict["mmd"],
                "dpp": metrics_dict["dpp"],
                "viol": metrics_dict["viol"],
                "bin_gap": metrics_dict.get("binarization", 0.0),
                "connectivity": metrics_dict.get("connectivity", 0.0),
                "generation_runtime_sec": generation_runtime_sec,
                "metrics_runtime_sec": metrics_runtime_sec,
                "evaluation_runtime_sec": evaluation_runtime_sec,
                "generation_samples_per_sec": generation_samples_per_sec,
                "eval/cog": metrics_dict["cog"],
                "eval/fog": metrics_dict["fog"],
                "eval/iog_raw_objective": metrics_dict["iog"],
                "eval/mmd": metrics_dict["mmd"],
                "eval/dpp": metrics_dict["dpp"],
                "eval/viol": metrics_dict["viol"],
                "eval/bin_gap": metrics_dict.get("binarization", 0.0),
                "eval/connectivity": metrics_dict.get("connectivity", 0.0),
                "eval/runtime/generation_sec": generation_runtime_sec,
                "eval/runtime/metrics_sec": metrics_runtime_sec,
                "eval/runtime/total_sec": evaluation_runtime_sec,
                "eval/runtime/generation_samples_per_sec": generation_samples_per_sec,
            }
        )
        if final_generated_designs_np is not None and final_reference_designs_np is not None:
            gen_grid = build_design_grid(final_generated_designs_np)
            ref_grid = build_design_grid(final_reference_designs_np)
            final_display_name = build_display_name(metrics_dict) if metrics_dict else "selected_candidate"
            run.log(
                {
                    f"eval/designs_generated_grid/{final_display_name}": wandb.Image(
                        gen_grid,
                        caption=f"Generated designs for {final_display_name}",
                    ),
                    f"eval/designs_reference_grid/{final_display_name}": wandb.Image(
                        ref_grid,
                        caption=f"Reference test designs for {final_display_name}",
                    ),
                }
            )
            run.log(
                {
                    "selection/final_checkpoint_epoch": metrics_dict.get("selection_candidate_epoch", -1),
                    "selection/final_validation_cog": metrics_dict.get("selection_candidate_validation_cog", float("nan")),
                }
            )
        run.finish()

    print(
        f"Seed {seed} done; wrote metrics to {out_path} "
        f"(gen={generation_runtime_sec:.2f}s, total={evaluation_runtime_sec:.2f}s)"
    )
