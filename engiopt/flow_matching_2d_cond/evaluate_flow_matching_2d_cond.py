"""Evaluation for the conditional 2D flow-matching baseline."""

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
from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.flow_matching_2d_cond.core import build_model
from engiopt.flow_matching_2d_cond.core import generate_samples
from engiopt.flow_matching_2d_cond.core import load_local_checkpoint
from engiopt.reporting import write_metrics_csv
import wandb


@dataclasses.dataclass
class Args:
    """Command-line arguments for a single-seed flow-matching evaluation."""

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
    output_csv: str = "flow_matching_2d_cond_{problem_id}_metrics.csv"
    """Output CSV path template; may include {problem_id}."""
    append_output: bool = True
    """Append to an existing CSV. Use --no-append-output to overwrite instead."""
    checkpoint_path: str | None = None
    """Optional local checkpoint path. Preferred over WandB artifacts when set."""
    device: str = "auto"
    """Device selection for local smoke runs and evaluation."""
    integration_steps: int | None = None
    """Optional override for the number of Euler integration steps."""
    clip_min: float = 1e-3
    """Minimum value used when clipping generated designs."""
    clip_max: float = 1.0
    """Maximum value used when clipping generated designs."""
    method: str = "euler"
    """Integration method: euler, midpoint, heun, rk4, rk45, dpm."""
    atol: float = 1e-3
    """Absolute tolerance for adaptive solvers (RK45)."""
    rtol: float = 1e-3
    """Relative tolerance for adaptive solvers (RK45)."""

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


def checkpoint_config(checkpoint: dict[str, Any], key: str, default: Any) -> Any:
    """Read a configuration value from a checkpoint with a fallback."""
    return checkpoint.get("model_config", {}).get(key, checkpoint.get("args", {}).get(key, default))


def load_artifact_checkpoint(
    problem_id: str,
    seed: int,
    wandb_project: str,
    wandb_entity: str | None,
    device: th.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a model checkpoint and config from a WandB artifact."""
    if wandb_entity is not None:
        artifact_path = f"{wandb_entity}/{wandb_project}/{problem_id}_flow_matching_2d_cond_model:seed_{seed}"
    else:
        artifact_path = f"{wandb_project}/{problem_id}_flow_matching_2d_cond_model:seed_{seed}"

    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")

    class RunRetrievalError(ValueError):
        def __init__(self):
            super().__init__("Failed to retrieve the run")

    run = artifact.logged_by()
    if run is None or not hasattr(run, "config"):
        raise RunRetrievalError

    artifact_dir = artifact.download()
    checkpoint = load_local_checkpoint(os.path.join(artifact_dir, "model.pth"), device)
    return checkpoint, dict(run.config)


if __name__ == "__main__":
    args = tyro.cli(Args)
    eval_start = time.perf_counter()

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    device = select_device(args.device)

    conditions_tensor, sampled_conditions, sampled_designs_np, _ = sample_conditions(
        problem=problem,
        n_samples=args.n_samples,
        device=device,
        seed=args.seed,
    )
    conditions_tensor = conditions_tensor.unsqueeze(1)

    if args.checkpoint_path is not None:
        checkpoint = load_local_checkpoint(args.checkpoint_path, device)
        run_config: dict[str, Any] = dict(checkpoint.get("args", {}))
        checkpoint_source = "local_checkpoint"
    else:
        checkpoint, run_config = load_artifact_checkpoint(
            problem_id=args.problem_id,
            seed=args.seed,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            device=device,
        )
        checkpoint_source = "wandb_artifact"

    layers_per_block = int(checkpoint_config(checkpoint, "layers_per_block", run_config.get("layers_per_block", 2)))
    num_train_timesteps = int(
        checkpoint_config(checkpoint, "num_train_timesteps", run_config.get("num_train_timesteps", 1000))
    )
    integration_steps = args.integration_steps
    if integration_steps is None:
        integration_steps = int(checkpoint_config(checkpoint, "integration_steps", run_config.get("integration_steps", 50)))

    model = build_model(
        design_shape=problem.design_space.shape,
        encoder_hid_dim=len(problem.conditions_keys),
        layers_per_block=layers_per_block,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    generation_start = time.perf_counter()
    gen_designs = generate_samples(
	model=model,
        design_shape=problem.design_space.shape,
        encoder_hidden_states=conditions_tensor,
        integration_steps=integration_steps,
        num_train_timesteps=num_train_timesteps,
        device=device,
        atol=args.atol,      # Add this
        rtol=args.rtol,
	method=args.method
    )
    generation_runtime_sec = time.perf_counter() - generation_start
    gen_designs = gen_designs.squeeze(1)
    gen_designs_np = gen_designs.detach().cpu().numpy().reshape(args.n_samples, *problem.design_space.shape)
    gen_designs_np = np.clip(gen_designs_np, args.clip_min, args.clip_max)
    generation_samples_per_sec = args.n_samples / generation_runtime_sec if generation_runtime_sec > 0 else float("nan")

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
            "seed": args.seed,
            "problem_id": args.problem_id,
            "model_id": "flow_matching_2d_cond",
            "n_samples": args.n_samples,
            "sigma": args.sigma,
            "integration_steps": integration_steps,
            "num_train_timesteps": num_train_timesteps,
            "layers_per_block": layers_per_block,
            "checkpoint_source": checkpoint_source,
            "generation_runtime_sec": generation_runtime_sec,
            "metrics_runtime_sec": metrics_runtime_sec,
            "evaluation_runtime_sec": evaluation_runtime_sec,
            "generation_samples_per_sec": generation_samples_per_sec,
        }
    )

    out_path = args.output_csv.format(problem_id=args.problem_id)
    write_metrics_csv([metrics_dict], out_path, append_output=args.append_output)

    if args.track:
        run_name = args.run_name or f"{args.problem_id}__flow_matching_2d_cond__eval__seed{args.seed}__{int(time.time())}"
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type=args.wandb_job_type,
            name=run_name,
            config={
                **vars(args),
                "model_id": "flow_matching_2d_cond",
                "checkpoint_source": checkpoint_source,
                "resolved_integration_steps": integration_steps,
                "resolved_num_train_timesteps": num_train_timesteps,
                "resolved_layers_per_block": layers_per_block,
            },
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
                "bin_gap": metrics_dict["bin_gap"],
                "connectivity": metrics_dict["connectivity"],
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
                "eval/bin_gap": metrics_dict["bin_gap"],
                "eval/connectivity": metrics_dict["connectivity"],
                "eval/runtime/generation_sec": generation_runtime_sec,
                "eval/runtime/metrics_sec": metrics_runtime_sec,
                "eval/runtime/total_sec": evaluation_runtime_sec,
                "eval/runtime/generation_samples_per_sec": generation_samples_per_sec,
            }
        )
        run.finish()

    print(
        f"Seed {args.seed} done; wrote metrics to {out_path} "
        f"(gen={generation_runtime_sec:.2f}s, total={evaluation_runtime_sec:.2f}s)"
    )
