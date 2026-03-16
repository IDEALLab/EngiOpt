"""Evaluation for the GAN 1D."""

from __future__ import annotations

import dataclasses
import os
from typing import Any

from engibench.utils.all_problems import BUILTIN_PROBLEMS
from gymnasium import spaces
import numpy as np
import torch as th
import tyro

from engiopt import metrics
from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.gan_1d.gan_1d import Generator
from engiopt.gan_1d.gan_1d import prepare_data
from engiopt.reporting import write_metrics_csv
import wandb


@dataclasses.dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "airfoil"
    """Problem identifier."""
    seed: int = 1
    """Random seed to run."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name."""
    n_samples: int = 50
    """Number of generated samples per seed."""
    sigma: float = 10.0
    """Kernel bandwidth for MMD and DPP metrics."""
    output_csv: str = "gan_1d_{problem_id}_metrics.csv"
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

    # Seeding for reproducibility
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = True

    # Select device
    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    if isinstance(problem.design_space, spaces.Box):
        design_shape = problem.design_space.shape
    else:
        dummy_design, _ = problem.random_design()
        design_shape = spaces.flatten(problem.design_space, dummy_design).shape

    ### Set up testing conditions ###
    conditions_tensor, sampled_conditions, sampled_designs_np, _ = sample_conditions(
        problem=problem,
        n_samples=args.n_samples,
        device=device,
        seed=seed,
    )

    ### Set Up Generator ###
    if args.checkpoint_path is not None:
        ckpt = th.load(args.checkpoint_path, map_location=device)
        run_config: dict[str, Any] = dict(ckpt.get("args", {}))
    else:
        if args.wandb_entity is not None:
            artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_gan_1d_generator:seed_{seed}"
        else:
            artifact_path = f"{args.wandb_project}/{args.problem_id}_gan_1d_generator:seed_{seed}"

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

    _, design_normalizer = prepare_data(problem, device)

    model = Generator(
        latent_dim=int(run_config["latent_dim"]), design_shape=design_shape, design_normalizer=design_normalizer
    ).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()

    # Sample noise and generate designs
    z = th.randn((args.n_samples, int(run_config["latent_dim"])), device=device)
    gen_designs = model(z)
    gen_designs_np = gen_designs.detach().cpu().numpy()

    fail_ratio = metrics.simulate_failure_ratio(
        problem=problem,
        gen_designs=gen_designs_np,
        sampled_conditions=sampled_conditions,
    )

    # Append result row to CSV
    results_dict = {
        "problem_id": args.problem_id,
        "model_id": "gan_1d",
        "seed": seed,
        "n_samples": args.n_samples,
        "fail_ratio": fail_ratio,
    }
    out_path = args.output_csv.format(problem_id=args.problem_id)
    write_metrics_csv([results_dict], out_path, append_output=args.append_output)

    print(f"Seed {seed} done; wrote metrics to {out_path}")
