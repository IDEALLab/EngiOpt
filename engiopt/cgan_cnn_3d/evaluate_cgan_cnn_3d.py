"""Evaluation for the CGAN 3D w/ CNN."""

from __future__ import annotations

import dataclasses
import os

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd
import torch as th
import tyro

from engiopt import metrics
from engiopt.cgan_cnn_3d.cgan_cnn_3d import Generator3D
from engiopt.dataset_sample_conditions import sample_conditions
import wandb


@dataclasses.dataclass
class Args:
    """Command-line arguments for a single-seed cGAN CNN 3D evaluation."""

    problem_id: str = "heatconduction3d"
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
    output_csv: str = "cgan_cnn_3d_{problem_id}_metrics.csv"
    """Output CSV path template; may include {problem_id}."""


if __name__ == "__main__":
    args = tyro.cli(Args)

    seed = args.seed
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=seed)

    # Reproducibility
    th.manual_seed(seed)
    rng = np.random.default_rng(seed)
    th.backends.cudnn.deterministic = True

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    ### Set up testing conditions ###
    conditions_tensor, sampled_conditions, sampled_designs_np, _ = sample_conditions(
        problem=problem, n_samples=args.n_samples, device=device, seed=seed
    )

    # Reshape to match the expected input shape for the model
    conditions_tensor = conditions_tensor.unsqueeze(-1).unsqueeze(-1)
    conditions_tensor = conditions_tensor.view(args.n_samples, len(problem.conditions_keys), 1, 1, 1)

    ### Set Up Generator ###

    # Restores the pytorch model from wandb
    if args.wandb_entity is not None:
        artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_cgan_cnn_3d_generator_3d:seed_{seed}"
    else:
        artifact_path = f"{args.wandb_project}/{args.problem_id}_cgan_cnn_3d_generator_3d:seed_{seed}"

    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")

    class RunRetrievalError(ValueError):
        def __init__(self):
            super().__init__("Failed to retrieve the run")

    run = artifact.logged_by()
    if run is None or not hasattr(run, "config"):
        raise RunRetrievalError
    artifact_dir = artifact.download()

    ckpt_path = os.path.join(artifact_dir, "generator_3d.pth")
    ckpt = th.load(ckpt_path, map_location=th.device(device))
    # Safer debug output
    for key in ckpt:
        print("Checkpoint key:", key)
    model = Generator3D(
        latent_dim=run.config["latent_dim"], n_conds=len(problem.conditions), design_shape=problem.design_space.shape
    )
    model.load_state_dict(ckpt["generator"])
    model.eval()  # Set to evaluation mode
    model.to(device)

    # Sample noise as generator input
    z = th.randn((args.n_samples, run.config["latent_dim"], 1, 1, 1), device=device, dtype=th.float)

    # Generate a batch of designs
    gen_designs = model(z, conditions_tensor)
    print("gen_designs.shape:", gen_designs.shape)
    gen_designs_np = gen_designs.squeeze(1).detach().cpu().numpy()
    # Removal of Padding
    crop_start = (64 - 51) // 2  # 6
    crop_end = crop_start + 51  # 6 + 51 = 57

    gen_designs_np = gen_designs_np[:, crop_start:crop_end, crop_start:crop_end, crop_start:crop_end]

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

    metrics_dict.update(
        {
            "seed": seed,
            "problem_id": args.problem_id,
            "model_id": "cgan_cnn_3d",
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
