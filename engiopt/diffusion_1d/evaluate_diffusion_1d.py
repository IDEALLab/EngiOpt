"""Evaluation for the Diffusion 1D."""

from __future__ import annotations

import dataclasses
import os
from typing import Any

from denoising_diffusion_pytorch import GaussianDiffusion1D
from denoising_diffusion_pytorch import Unet1D
from engibench.utils.all_problems import BUILTIN_PROBLEMS
from gymnasium import spaces
import numpy as np
import torch as th
import tyro

from engiopt import metrics
from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.diffusion_1d.diffusion_1d import prepare_data
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
    n_samples: int = 10
    """Number of generated samples per seed."""
    sigma: float = 10.0
    """Kernel bandwidth for MMD and DPP metrics."""
    output_csv: str = "diffusion_1d_{problem_id}_metrics.csv"
    """Output CSV path template; may include {problem_id}."""
    append_output: bool = True
    """Append to an existing CSV. Use --no-append-output to overwrite instead."""
    checkpoint_path: str | None = None
    """Optional local checkpoint path. Preferred over WandB artifacts when set."""


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
        flattened = spaces.flatten(problem.design_space, dummy_design)
        design_shape = np.array(flattened).shape

    # Add padding for the UNet (1D requires the input to be divisible by 8)
    padding_size = (8 - design_shape[0] % 8) % 8  # Only pad if needed
    padded_size = design_shape[0] + padding_size
    if padding_size > 0:
        print(f"Padding design from {design_shape[0]} to {padded_size} dimensions")
    design_shape = (padded_size,)

    ### Set up testing conditions ###
    conditions_tensor, sampled_conditions, sampled_designs_np, _ = sample_conditions(
        problem=problem,
        n_samples=args.n_samples,
        device=device,
        seed=seed,
    )

    ### Load Diffusion Model ###
    if args.checkpoint_path is not None:
        ckpt = th.load(args.checkpoint_path, map_location=device)
        run_config: dict[str, Any] = dict(ckpt.get("args", {}))
    else:
        if args.wandb_entity is not None:
            artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_diffusion_1d_model:seed_{seed}"
        else:
            artifact_path = f"{args.wandb_project}/{args.problem_id}_diffusion_1d_model:seed_{seed}"

        api = wandb.Api()
        artifact = api.artifact(artifact_path, type="model")

        class RunRetrievalError(ValueError):
            def __init__(self):
                super().__init__("Failed to retrieve the run")

        run = artifact.logged_by()
        if run is None or not hasattr(run, "config"):
            raise RunRetrievalError

        artifact_dir = artifact.download()
        ckpt_path = os.path.join(artifact_dir, "model.pth")
        ckpt = th.load(ckpt_path, map_location=device)
        run_config = dict(run.config)

    _, design_normalizer = prepare_data(problem, padding_size, device)

    model = Unet1D(
        dim=int(run_config["unet_dim"]),  # Used for the sinusoidal positional embeddings
        channels=int(run_config["n_channels"]),  # Number of channels in the input
    ).to(device)

    diffusion = GaussianDiffusion1D(
        model,
        seq_length=np.prod(design_shape),
        auto_normalize=True,
    ).to(device)

    diffusion.load_state_dict(ckpt["model"])

    # Sample noise and generate designs
    gen_designs = diffusion.sample(args.n_samples).squeeze(1)
    gen_designs = design_normalizer.denormalize(gen_designs)
    gen_designs_np = gen_designs.detach().cpu().numpy()
    if padding_size > 0:
        gen_designs_np = gen_designs_np[:, :-padding_size]

    fail_ratio = metrics.simulate_failure_ratio(
        problem=problem,
        gen_designs=gen_designs_np,
        sampled_conditions=sampled_conditions,
    )

    # Append result row to CSV
    results_dict = {
        "problem_id": args.problem_id,
        "model_id": "diffusion_1d",
        "seed": seed,
        "n_samples": args.n_samples,
        "fail_ratio": fail_ratio,
    }
    out_path = args.output_csv.format(problem_id=args.problem_id)
    write_metrics_csv([results_dict], out_path, append_output=args.append_output)

    print(f"Seed {seed} done; wrote metrics to {out_path}")
