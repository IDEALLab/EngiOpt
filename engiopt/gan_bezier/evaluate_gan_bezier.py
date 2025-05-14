"""Evaluation for the GAN Bezier."""

from __future__ import annotations

import dataclasses
import os

from engibench.utils.all_problems import BUILTIN_PROBLEMS
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch as th
import tyro

from engiopt import metrics
from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.gan_bezier.gan_bezier import Generator
from engiopt.gan_bezier.gan_bezier import prepare_data
from engiopt.transforms import flatten_dict_factory
import wandb

_EPS = 1e-7


@dataclasses.dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "airfoil"
    """Problem identifier."""
    seed: int = 1
    """Random seed to run."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str = "engibench"
    """Wandb entity name."""
    n_samples: int = 100
    """Number of generated samples per seed."""
    sigma: float = 10.0
    """Kernel bandwidth for MMD and DPP metrics."""
    output_csv: str = "gan_bezier_{problem_id}_metrics.csv"
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

    coords_space: spaces.Box = problem.design_space["coords"]

    ### Set up testing conditions ###
    conditions_tensor, sampled_conditions, sampled_designs_np, _ = sample_conditions(
        problem=problem,
        n_samples=args.n_samples,
        device=device,
        seed=seed,
    )

    ### Set Up Generator ###
    if args.wandb_entity is not None:
        artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_gan_bezier_generator:seed_{seed}"
    else:
        artifact_path = f"{args.wandb_project}/{args.problem_id}_gan_bezier_generator:seed_{seed}"

    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")

    class RunRetrievalError(ValueError):
        def __init__(self):
            super().__init__("Failed to retrieve the run")

    run = artifact.logged_by()
    if run is None or not hasattr(run, "config"):
        raise RunRetrievalError

    artifact_dir = artifact.download()
    ckpt_path = os.path.join(artifact_dir, "bezier_generator.pth")
    ckpt = th.load(ckpt_path, map_location=device)

    _, design_scalars_normalizer, _ = prepare_data(problem, args.n_samples, device)

    model = Generator(
        latent_dim=run.config["latent_dim"],
        noise_dim=run.config["noise_dim"],
        n_control_points=run.config["bezier_control_pts"],
        n_data_points=coords_space.shape[1],
        design_scalars_normalizer=design_scalars_normalizer,
        eps=_EPS,
        scalar_features=1,
    ).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()

    # Sample noise and generate designs
    bounds = (0.0, 1.0)  # Bounds for angle of attack
    c = (bounds[1] - bounds[0]) * th.rand(args.n_samples, run.config["latent_dim"], device=device) + bounds[0]
    z = 0.5 * th.randn(args.n_samples, run.config["noise_dim"], device=device)
    gen_designs, _, _, _, _, alphas = model(c, z)

    gen_designs_np = gen_designs.detach().cpu().numpy()
    alphas_np = alphas.detach().cpu().numpy()

    # Reshape as dict
    gen_designs_dict = []
    for idx, design in enumerate(gen_designs_np):
        d = {"coords": design, "angle_of_attack": alphas_np[idx][0]}
        gen_designs_dict.append(d)

    # Flatten dict for metrics
    transform = flatten_dict_factory(problem, device)
    transformed_gen_designs = transform(gen_designs_dict)
    transformed_gen_designs = transformed_gen_designs.cpu().numpy()

    fail_ratio = metrics.simulate_failure_ratio(
        problem=problem,
        gen_designs=transformed_gen_designs,
        sampled_conditions=sampled_conditions,
    )

    # Append result row to CSV
    results_dict = {
        "problem_id": args.problem_id,
        "model_id": "gan_bezier",
        "seed": seed,
        "n_samples": args.n_samples,
        "fail_ratio": fail_ratio,
    }
    metrics_df = pd.DataFrame(results_dict, index=[0])
    out_path = args.output_csv.format(problem_id=args.problem_id)
    write_header = not os.path.exists(out_path)
    metrics_df.to_csv(out_path, mode="a", header=write_header, index=False)

    print(f"Seed {seed} done; appended to {out_path}")
