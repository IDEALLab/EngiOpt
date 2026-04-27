"""Evaluation for the Diffusion 2d_cond w/ seed looping and CSV saving."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import time
from typing import Any

from diffusers import UNet2DConditionModel
from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import torch as th
import tyro
import wandb

from engiopt import metrics
from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.diffusion_2d_cond.diffusion_2d_cond import beta_schedule
from engiopt.diffusion_2d_cond.diffusion_2d_cond import DiffusionSampler
from engiopt.reporting import write_metrics_csv


@dataclasses.dataclass
class Args:
    """Command-line arguments for a single-seed Diffusion 2D Conditional evaluation."""

    problem_id: str = "beams2d"
    """Problem identifier."""
    seed: int = 1
    """Random seed to run."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name."""
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
    output_csv: str = "diffusion_2d_cond_{problem_id}_metrics.csv"
    """Output CSV path template; may include {problem_id}."""
    append_output: bool = True
    """Append to an existing CSV. Use --no-append-output to overwrite instead."""
    checkpoint_path: str | None = None
    """Optional local checkpoint path. Preferred over WandB artifacts when set."""
    checkpoint_dir: str | None = None
    """Optional directory containing validation_metrics.json and epoch checkpoints for top-k post-training selection."""
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
        artifact_path = f"{wandb_entity}/{wandb_project}/{problem_id}_diffusion_2d_cond_model:seed_{seed}"
    else:
        artifact_path = f"{wandb_project}/{problem_id}_diffusion_2d_cond_model:seed_{seed}"

    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")

    class RunRetrievalError(ValueError):
        def __init__(self):
            super().__init__("Failed to retrieve the run")

    run = artifact.logged_by()
    if run is None or not hasattr(run, "config"):
        raise RunRetrievalError

    artifact_dir = artifact.download()
    ckpt = th.load(os.path.join(artifact_dir, "model.pth"), map_location=device)
    return ckpt, dict(run.config)


def load_top_k_candidates(checkpoint_dir: Path, top_k: int) -> list[dict[str, Any]]:
    """Load the shortlisted checkpoints saved by the training run."""
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
        checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1:04d}.pth"
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
    """Evaluate a single diffusion checkpoint on a fixed condition sample."""
    phase_start = time.perf_counter()
    ckpt = th.load(str(checkpoint_path), map_location=context.device)
    run_config: dict[str, Any] = dict(ckpt.get("args", {}))

    model = UNet2DConditionModel(
        sample_size=context.problem.design_space.shape,
        in_channels=1,
        out_channels=1,
        cross_attention_dim=64,
        block_out_channels=(32, 64, 128, 256),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        layers_per_block=int(ckpt.get("model_config", {}).get("layers_per_block", run_config["layers_per_block"])),
        transformer_layers_per_block=1,
        encoder_hid_dim=len(context.problem.conditions_keys),
        only_cross_attention=True,
    ).to(context.device)

    options = {
        "cosine": ckpt.get("model_config", {}).get("noise_schedule", run_config["noise_schedule"]) == "cosine",
        "exp_biasing": ckpt.get("model_config", {}).get("noise_schedule", run_config["noise_schedule"]) == "exp",
        "exp_bias_factor": 1,
    }
    num_timesteps = int(ckpt.get("model_config", {}).get("num_timesteps", run_config["num_timesteps"]))
    betas = beta_schedule(
        t=num_timesteps,
        start=1e-4,
        end=0.02,
        scale=1.0,
        options=options,
    )
    ddm_sampler = DiffusionSampler(num_timesteps, betas)

    model.load_state_dict(ckpt["model"])
    model.eval()

    th.manual_seed(context.generation_seed)
    generation_start = time.perf_counter()
    design_shape: tuple = context.problem.design_space.shape
    n_gen = conditions_tensor.shape[0]
    gen_designs = th.randn((n_gen, 1, *design_shape), device=context.device)
    for i in reversed(range(num_timesteps)):
        t = th.full((n_gen,), i, device=context.device, dtype=th.long)
        gen_designs = ddm_sampler.sample_timestep(model, gen_designs, t, conditions_tensor)
    generation_runtime_sec = time.perf_counter() - generation_start
    generation_samples_per_sec = n_gen / generation_runtime_sec if generation_runtime_sec > 0 else float("nan")

    gen_designs = gen_designs.squeeze(1)
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
            "num_timesteps": num_timesteps,
            "layers_per_block": int(ckpt.get("model_config", {}).get("layers_per_block", run_config["layers_per_block"])),
            "noise_schedule": ckpt.get("model_config", {}).get("noise_schedule", run_config["noise_schedule"]),
        }
    )
    return metrics_dict, gen_designs_np


if __name__ == "__main__":
    args = tyro.cli(Args)
    eval_start = time.perf_counter()

    seed = args.seed
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=seed)

    # Seeding for reproducibility
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = True

    # Select device
    device = select_device(args.device)

    out_path = args.output_csv.format(problem_id=args.problem_id)
    final_generated_designs_np: np.ndarray | None = None
    final_reference_designs_np: np.ndarray | None = None

    if args.checkpoint_path is not None:
        conditions_tensor, sampled_conditions, sampled_designs_np, _ = sample_conditions(
            problem=problem,
            n_samples=args.n_samples,
            device=device,
            seed=seed,
        )
        conditions_tensor = conditions_tensor.unsqueeze(1)

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
                "model_id": "diffusion_2d_cond",
                "n_samples": args.n_samples,
                "sigma": args.sigma,
            }
        )
        final_reference_designs_np = sampled_designs_np
        write_metrics_csv([metrics_dict], out_path, append_output=args.append_output)
        checkpoint_source = "local_checkpoint"

    elif args.checkpoint_dir is not None and args.select_best_of_top_k:
        checkpoint_dir = Path(args.checkpoint_dir)
        candidates = load_top_k_candidates(checkpoint_dir, args.top_k)

        selection_conditions_tensor, selection_sampled_conditions, selection_sampled_designs_np, _ = sample_conditions(
            problem=problem,
            n_samples=args.selection_batch_size,
            device=device,
            seed=seed + args.selection_seed_offset,
            split="val",
        )
        selection_conditions_tensor = selection_conditions_tensor.unsqueeze(1)

        candidate_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_metrics, _ = evaluate_checkpoint(
                checkpoint_path=candidate["checkpoint_path"],
                conditions_tensor=selection_conditions_tensor,
                sampled_conditions=selection_sampled_conditions,
                sampled_designs_np=selection_sampled_designs_np,
                context=EvaluationContext(
                    problem=problem,
                    device=device,
                    args=args,
                    generation_seed=seed + 1000,
                    checkpoint_source="local_checkpoint",
                ),
            )
            candidate_metrics.update(
                {
                    "seed": seed,
                    "problem_id": args.problem_id,
                    "model_id": "diffusion_2d_cond",
                    "phase": "validation_selection",
                    "selection_candidate_epoch": candidate["epoch"] + 1,
                    "selection_candidate_mmd": candidate["metric_value"],
                }
            )
            candidate_rows.append(candidate_metrics)

        candidate_rows.sort(key=lambda row: (row["cog"], row["fog"]))
        selected_candidate = candidate_rows[0]
        selected_checkpoint_path = selected_candidate["checkpoint_path"]

        print("Selection results (sorted by validation COG, then FOG):")
        for rank, row in enumerate(candidate_rows, 1):
            print(
                f"  {rank}. epoch {int(row['selection_candidate_epoch'])}: cog={row['cog']:.6f}, "
                f"fog={row['fog']:.6f}, mmd={row['selection_candidate_mmd']:.6f}"
            )
        print(f"Selected checkpoint: {selected_checkpoint_path}")

        test_conditions_tensor, test_sampled_conditions, test_sampled_designs_np, _ = sample_conditions(
            problem=problem,
            n_samples=args.n_samples,
            device=device,
            seed=seed,
            split="test",
        )
        test_conditions_tensor = test_conditions_tensor.unsqueeze(1)

        metrics_dict, final_generated_designs_np = evaluate_checkpoint(
            checkpoint_path=selected_checkpoint_path,
            conditions_tensor=test_conditions_tensor,
            sampled_conditions=test_sampled_conditions,
            sampled_designs_np=test_sampled_designs_np,
            context=EvaluationContext(
                problem=problem,
                device=device,
                args=args,
                generation_seed=seed + 2000,
                checkpoint_source="selected_top_k_checkpoint",
            ),
        )
        metrics_dict.update(
            {
                "seed": seed,
                "problem_id": args.problem_id,
                "model_id": "diffusion_2d_cond",
                "n_samples": args.n_samples,
                "sigma": args.sigma,
                "checkpoint_dir": str(checkpoint_dir),
                "selection_mode": "top_k_cog_fog",
                "selection_top_k": args.top_k,
                "selection_candidate_count": len(candidate_rows),
                "selected_validation_epoch": int(selected_candidate["selection_candidate_epoch"]),
                "selected_validation_cog": float(selected_candidate["cog"]),
                "selected_validation_fog": float(selected_candidate["fog"]),
                "selected_validation_mmd": float(selected_candidate["selection_candidate_mmd"]),
            }
        )
        final_reference_designs_np = test_sampled_designs_np
        write_metrics_csv([metrics_dict], out_path, append_output=args.append_output)
        checkpoint_source = "selected_top_k_checkpoint"

    else:
        conditions_tensor, sampled_conditions, sampled_designs_np, _ = sample_conditions(
            problem=problem,
            n_samples=args.n_samples,
            device=device,
            seed=seed,
        )
        conditions_tensor = conditions_tensor.unsqueeze(1)

        ckpt, run_config = load_artifact_checkpoint(
            problem_id=args.problem_id,
            seed=seed,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            device=device,
        )
        checkpoint_source = "wandb_artifact"

        model = UNet2DConditionModel(
            sample_size=problem.design_space.shape,
            in_channels=1,
            out_channels=1,
            cross_attention_dim=64,
            block_out_channels=(32, 64, 128, 256),
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            layers_per_block=int(ckpt.get("model_config", {}).get("layers_per_block", run_config["layers_per_block"])),
            transformer_layers_per_block=1,
            encoder_hid_dim=len(problem.conditions_keys),
            only_cross_attention=True,
        ).to(device)

        options = {
            "cosine": ckpt.get("model_config", {}).get("noise_schedule", run_config["noise_schedule"]) == "cosine",
            "exp_biasing": ckpt.get("model_config", {}).get("noise_schedule", run_config["noise_schedule"]) == "exp",
            "exp_bias_factor": 1,
        }
        num_timesteps = int(ckpt.get("model_config", {}).get("num_timesteps", run_config["num_timesteps"]))
        betas = beta_schedule(
            t=num_timesteps,
            start=1e-4,
            end=0.02,
            scale=1.0,
            options=options,
        )
        ddm_sampler = DiffusionSampler(num_timesteps, betas)

        model.load_state_dict(ckpt["model"])
        model.eval()

        design_shape: tuple = problem.design_space.shape
        generation_start = time.perf_counter()
        gen_designs = th.randn((args.n_samples, 1, *design_shape), device=device)
        for i in reversed(range(num_timesteps)):
            t = th.full((args.n_samples,), i, device=device, dtype=th.long)
            gen_designs = ddm_sampler.sample_timestep(model, gen_designs, t, conditions_tensor)
        generation_runtime_sec = time.perf_counter() - generation_start
        generation_samples_per_sec = args.n_samples / generation_runtime_sec if generation_runtime_sec > 0 else float("nan")

        gen_designs = gen_designs.squeeze(1)
        gen_designs_np = gen_designs.detach().cpu().numpy().reshape(args.n_samples, *problem.design_space.shape)
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
                "model_id": "diffusion_2d_cond",
                "n_samples": args.n_samples,
                "sigma": args.sigma,
                "num_timesteps": num_timesteps,
                "layers_per_block": int(ckpt.get("model_config", {}).get("layers_per_block", run_config["layers_per_block"])),
                "noise_schedule": ckpt.get("model_config", {}).get("noise_schedule", run_config["noise_schedule"]),
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
        run_name = args.run_name or f"{args.problem_id}__diffusion_2d_cond__eval__seed{seed}__{int(time.time())}"
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type=args.wandb_job_type,
            name=run_name,
            config={**vars(args), "model_id": "diffusion_2d_cond", "checkpoint_source": checkpoint_source},
        )
        if run is None:
            raise RuntimeError("Failed to initialize Weights & Biases run")
        if args.checkpoint_dir is not None and args.select_best_of_top_k and args.checkpoint_path is None:
            run.log(
                {
                    "selection/validation_table": wandb.Table(
                        columns=["rank", "epoch", "mmd", "cog", "fog", "checkpoint_path"],
                        data=[
                            [
                                rank,
                                int(row["selection_candidate_epoch"]),
                                float(row["selection_candidate_mmd"]),
                                float(row["cog"]),
                                float(row["fog"]),
                                str(row["checkpoint_path"]),
                            ]
                            for rank, row in enumerate(candidate_rows, 1)
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
            run.log(
                {
                    "eval/designs_generated_grid": wandb.Image(
                        gen_grid,
                        caption="Generated designs (same batch used for final table metrics)",
                    ),
                    "eval/designs_reference_grid": wandb.Image(
                        ref_grid,
                        caption="Reference test designs (same batch used for final table metrics)",
                    ),
                }
            )
        run.finish()

    print(
        f"Seed {seed} done; wrote metrics to {out_path} "
        f"(gen={generation_runtime_sec:.2f}s, total={evaluation_runtime_sec:.2f}s)"
    )
