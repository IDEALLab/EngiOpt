"""Evaluation for the CGAN 2D w/ CNN."""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Any

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import torch as th
import tyro

from engiopt import metrics
from engiopt.cgan_cnn_2d.cgan_cnn_2d import Generator
from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.reporting import write_metrics_csv
import wandb


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
    device: str = "auto"
    """Device selection for local smoke runs and evaluation."""
    clip_min: float = 1e-3
    """Minimum value used when clipping generated designs."""
    clip_max: float = 1.0
    """Maximum value used when clipping generated designs."""


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

    ### Set up testing conditions ###
    conditions_tensor, sampled_conditions, sampled_designs_np, _ = sample_conditions(
        problem=problem, n_samples=args.n_samples, device=device, seed=seed
    )

    # Reshape to match the expected input shape for the model
    conditions_tensor = conditions_tensor.unsqueeze(-1).unsqueeze(-1)

    ### Set Up Generator ###

    # Restores the pytorch model from wandb
    if args.checkpoint_path is not None:
        ckpt = th.load(args.checkpoint_path, map_location=device)
        run_config: dict[str, Any] = dict(ckpt.get("args", {}))
        checkpoint_source = "local_checkpoint"
    else:
        if args.wandb_entity is not None:
            artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_cgan_cnn_2d_generator:seed_{seed}"
        else:
            artifact_path = f"{args.wandb_project}/{args.problem_id}_cgan_cnn_2d_generator:seed_{seed}"

        api = wandb.Api()
        artifact = api.artifact(artifact_path, type="model")

        class RunRetrievalError(ValueError):
            def __init__(self):
                super().__init__("Failed to retrieve the run")

        run = artifact.logged_by()
        if run is None or not hasattr(run, "config"):
            raise RunRetrievalError
        artifact_dir = artifact.download()

        ckpt_path = os.path.join(artifact_dir, "generator.pth")
        ckpt = th.load(ckpt_path, map_location=device)
        run_config = dict(run.config)
        checkpoint_source = "wandb_artifact"
    model = Generator(
        latent_dim=int(run_config["latent_dim"]),
        n_conds=len(problem.conditions_keys),
        design_shape=problem.design_space.shape,
    )
    model.load_state_dict(ckpt["generator"])
    model.eval()  # Set to evaluation mode
    model.to(device)

    # Sample noise as generator input
    z = th.randn((args.n_samples, int(run_config["latent_dim"]), 1, 1), device=device, dtype=th.float)

    # Generate a batch of designs
    generation_start = time.perf_counter()
    gen_designs = model(z, conditions_tensor)
    generation_runtime_sec = time.perf_counter() - generation_start
    gen_designs_np = gen_designs.detach().cpu().numpy()
    gen_designs_np = gen_designs_np.reshape(args.n_samples, *problem.design_space.shape)

    # Clip to boundaries for running THIS IS PROBLEM DEPENDENT
    gen_designs_np = np.clip(gen_designs_np, args.clip_min, args.clip_max)

    # Compute metrics
    metrics_start = time.perf_counter()
    metrics_dict = metrics.metrics(
        problem,
        gen_designs_np,
        sampled_designs_np,
        sampled_conditions,
        sigma=args.sigma,
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

    # Append result row to CSV
    out_path = args.output_csv.format(problem_id=args.problem_id)
    write_metrics_csv([metrics_dict], out_path, append_output=args.append_output)

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
        run.log(
            {
                "cog": metrics_dict["cog"],
                "fog": metrics_dict["fog"],
                "iog_raw_objective": metrics_dict["iog"],
                "mmd": metrics_dict["mmd"],
                "dpp": metrics_dict["dpp"],
                "viol": metrics_dict["viol"],
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
                "eval/runtime/generation_sec": generation_runtime_sec,
                "eval/runtime/metrics_sec": metrics_runtime_sec,
                "eval/runtime/total_sec": evaluation_runtime_sec,
                "eval/runtime/generation_samples_per_sec": generation_samples_per_sec,
            }
        )
        run.finish()

    print(
        f"Seed {seed} done; wrote metrics to {out_path} "
        f"(gen={generation_runtime_sec:.2f}s, total={evaluation_runtime_sec:.2f}s)"
    )
