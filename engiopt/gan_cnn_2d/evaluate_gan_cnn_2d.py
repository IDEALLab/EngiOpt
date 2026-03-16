"""Evaluation for the CGAN 2D w/ CNN."""

from __future__ import annotations

import dataclasses
import os
from typing import Any

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import torch as th
import tyro

from engiopt import metrics
from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.gan_cnn_2d.gan_cnn_2d import Generator
from engiopt.reporting import write_metrics_csv
import wandb


@dataclasses.dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "beams2d"
    """Problem identifier (e.g. beams2d)."""
    seed: int = 1
    """Random seed to run."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name (if any)."""
    n_samples: int = 50
    """Number of generated samples per seed."""
    sigma: float = 10.0
    """Kernel bandwidth for MMD and DPP metrics."""
    output_csv: str = "gan_cnn_2d_{problem_id}_metrics.csv"
    """Output CSV path template; may include {problem_id}."""
    append_output: bool = True
    """Append to an existing CSV. Use --no-append-output to overwrite instead."""
    checkpoint_path: str | None = None
    """Optional local generator checkpoint path. Preferred over WandB artifacts when set."""


if __name__ == "__main__":
    args = tyro.cli(Args)

    seed = args.seed
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=seed)

    # Seeding
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = True

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    ### Set up testing conditions ###
    conditions_tensor, sampled_conditions, sampled_designs_np, selected_indices = sample_conditions(
        problem=problem, n_samples=args.n_samples, device=device, seed=seed
    )

    # Reshape to match the expected input shape for the model
    conditions_tensor = conditions_tensor.unsqueeze(-1).unsqueeze(-1)

    ### Set Up Generator ###

    # Restores the pytorch model from wandb
    if args.checkpoint_path is not None:
        ckpt = th.load(args.checkpoint_path, map_location=th.device(device))
        run_config: dict[str, Any] = dict(ckpt.get("args", {}))
    else:
        if args.wandb_entity is not None:
            artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_gan_cnn_2d_generator:seed_{seed}"
        else:
            artifact_path = f"{args.wandb_project}/{args.problem_id}_gan_cnn_2d_generator:seed_{seed}"

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
        ckpt = th.load(ckpt_path, map_location=th.device(device))
        run_config = dict(run.config)
    model = Generator(latent_dim=int(run_config["latent_dim"]), design_shape=problem.design_space.shape)
    model.load_state_dict(ckpt["generator"])
    model.eval()  # Set to evaluation mode
    model.to(device)

    # Sample noise as generator input
    z = th.randn((args.n_samples, int(run_config["latent_dim"]), 1, 1), device=device, dtype=th.float)

    # Generate a batch of designs
    gen_designs = model(z)
    gen_designs_np = gen_designs.detach().cpu().numpy()
    gen_designs_np = gen_designs_np.reshape(args.n_samples, *problem.design_space.shape)

    # Clip to boundaries for running THIS IS PROBLEM DEPENDENT
    gen_designs_np = np.clip(gen_designs_np, 1e-3, 1)

    # Compute metrics
    metrics_dict = metrics.metrics(
        problem,
        gen_designs_np,
        sampled_designs_np,
        sampled_conditions,
        sigma=args.sigma,
    )

    # Add metadata to metrics
    metrics_dict.update(
        {
            "seed": seed,
            "problem_id": args.problem_id,
            "model_id": "gan_cnn_2d",
            "n_samples": args.n_samples,
            "sigma": args.sigma,
        }
    )

    # Append result row to CSV
    out_path = args.output_csv.format(problem_id=args.problem_id)
    write_metrics_csv([metrics_dict], out_path, append_output=args.append_output)

    print(f"Seed {seed} done; wrote metrics to {out_path}")
