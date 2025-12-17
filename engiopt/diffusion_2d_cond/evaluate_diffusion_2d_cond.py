"""Evaluation for the Diffusion 2d_cond w/ seed looping and CSV saving."""

from __future__ import annotations

import dataclasses
import os

from diffusers import UNet2DConditionModel
from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd
import torch as th
import tyro

from engiopt import metrics
from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.diffusion_2d_cond.diffusion_2d_cond import beta_schedule
from engiopt.diffusion_2d_cond.diffusion_2d_cond import DiffusionSampler
import wandb


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
    n_samples: int = 50
    """Number of generated samples per seed."""
    sigma: float = 10.0
    """Kernel bandwidth for MMD and DPP metrics."""
    output_csv: str = "diffusion_2d_cond_{problem_id}_metrics.csv"
    """Output CSV path template; may include {problem_id}."""


if __name__ == "__main__":
    args = tyro.cli(Args)

    seed = args.seed
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=seed)

    # Seeding for reproducibility
    th.manual_seed(seed)
    rng = np.random.default_rng(seed)
    th.backends.cudnn.deterministic = True

    # Select device
    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    ### Set up testing conditions ###
    conditions_tensor, sampled_conditions, sampled_designs_np, _ = sample_conditions(
        problem=problem,
        n_samples=args.n_samples,
        device=device,
        seed=seed,
    )
    # Add channel dim
    conditions_tensor = conditions_tensor.unsqueeze(1)

    ### Set Up Diffusion Model ###
    if args.wandb_entity is not None:
        artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_diffusion_2d_cond_model:seed_{seed}"
    else:
        artifact_path = f"{args.wandb_project}/{args.problem_id}_diffusion_2d_cond_model:seed_{seed}"

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

    # Build UNet
    model = UNet2DConditionModel(
        sample_size=problem.design_space.shape,
        in_channels=1,
        out_channels=1,
        cross_attention_dim=64,
        block_out_channels=(32, 64, 128, 256),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        layers_per_block=run.config["layers_per_block"],
        transformer_layers_per_block=1,
        encoder_hid_dim=len(problem.conditions_keys),
        only_cross_attention=True,
    ).to(device)

    # Noise schedule
    options = {
        "cosine": run.config["noise_schedule"] == "cosine",
        "exp_biasing": run.config["noise_schedule"] == "exp",
        "exp_bias_factor": 1,
    }
    betas = beta_schedule(
        t=run.config["num_timesteps"],
        start=1e-4,
        end=0.02,
        scale=1.0,
        options=options,
    )
    ddm_sampler = DiffusionSampler(run.config["num_timesteps"], betas)

    model.load_state_dict(ckpt["model"])
    model.eval()

    # Generate and reshape
    design_shape: tuple = problem.design_space.shape
    gen_designs = th.randn((args.n_samples, 1, *design_shape), device=device)
    assert run.config["num_timesteps"] is not None
    for i in reversed(range(run.config["num_timesteps"])):
        t = th.full((args.n_samples,), i, device=device, dtype=th.long)
        gen_designs = ddm_sampler.sample_timestep(model, gen_designs, t, conditions_tensor)

    gen_designs = gen_designs.squeeze(1)
    gen_designs_np = gen_designs.detach().cpu().numpy().reshape(args.n_samples, *problem.design_space.shape)
    gen_designs_np = np.clip(gen_designs_np, 1e-3, 1.0)

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
            "model_id": "diffusion_2d_cond",
            "n_samples": args.n_samples,
            "sigma": args.sigma,
        }
    )

    # Append result row to CSV
    metrics_df = pd.DataFrame([metrics_dict])
    out_path = args.output_csv.format(problem_id=args.problem_id)
    write_header = not os.path.exists(out_path)
    metrics_df.to_csv(out_path, mode="a", header=write_header, index=False)

    print(f"Seed {seed} done; appended to {out_path}")
