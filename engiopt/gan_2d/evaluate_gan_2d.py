"""Evaluation for a single seed of GAN 2D."""

from __future__ import annotations

import dataclasses
import os

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd
import torch as th
import tyro

from engiopt import metrics
from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.gan_2d.gan_2d import Generator
import wandb


@dataclasses.dataclass
class Args:
    """Command-line arguments for a single-seed GAN 2D evaluation."""

    problem_id: str = "beams2d"
    """Problem identifier (e.g. beams2d)."""
    seed: int = 1
    """Random seed to run."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name."""
    n_samples: int = 100
    """Number of generated samples per seed."""
    sigma: float = 1.0
    """Kernel bandwidth for MMD and DPP metrics."""
    output_csv: str = "gan_2d_{problem_id}_metrics.csv"
    """Output CSV path template; may include {problem_id}."""


if __name__ == "__main__":
    args = tyro.cli(Args)

    seed = args.seed
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=seed)

    # Reproducibility
    th.manual_seed(seed)
    np.random.seed(seed)
    th.backends.cudnn.deterministic = True

    # Device selection
    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    # Sample conditions & designs
    _, sampled_conditions, sampled_designs_np, _ = sample_conditions(
        problem=problem,
        n_samples=args.n_samples,
        device=device,
        seed=seed,
    )

    # Load GAN 2D generator artifact from WandB
    if args.wandb_entity:
        artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_gan_2d_generator:seed_{seed}"
    else:
        artifact_path = f"{args.wandb_project}/{args.problem_id}_gan_2d_generator:seed_{seed}"

    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")

    class RunRetrievalError(ValueError):
        def __init__(self):
            super().__init__("Failed to retrieve the run")

    run = artifact.logged_by()
    if run is None or not hasattr(run, "config"):
        raise RunRetrievalError

    artifact_dir = artifact.download()
    ckpt = th.load(
        os.path.join(artifact_dir, "generator.pth"),
        map_location=device,
    )

    model = Generator(
        latent_dim=run.config["latent_dim"],
        design_shape=problem.design_space.shape,
    ).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()

    # Generate designs
    z = th.randn((args.n_samples, run.config["latent_dim"]), device=device)
    gen_designs = model(z)
    gen_designs_np = gen_designs.detach().cpu().numpy()
    gen_designs_np = np.clip(gen_designs_np, 1e-3, 1.0)

    # Compute metrics
    metrics_dict = metrics.metrics(
        problem,
        gen_designs_np,
        sampled_designs_np,
        sampled_conditions,
        sigma=args.sigma,
    )
    metrics_dict["seed"] = seed

    # Append result row to CSV
    df = pd.DataFrame([metrics_dict])
    out_path = args.output_csv.format(problem_id=args.problem_id)
    write_header = not os.path.exists(out_path)
    df.to_csv(out_path, mode="a", header=write_header, index=False)

    print(f"Seed {seed} done; appended to {out_path}")
