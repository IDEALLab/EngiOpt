"""Evaluation for the GAN 1D."""

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
from engiopt.checkpoint_store import ModelSource
from engiopt.checkpoint_store import resolve_named_checkpoint
from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.gan_1d.gan_1d import Generator
from engiopt.gan_1d.gan_1d import prepare_data


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
    model_source: ModelSource = "auto"
    """Where to load the checkpoint package from."""
    hf_entity: str = "IDEALLab"
    """HF org/user where checkpoints are stored."""
    hf_repo_prefix: str = "engiopt"
    """HF repo prefix used for model-family repositories."""
    local_model_dir: str | None = None
    """Optional local checkpoint package directory."""
    n_samples: int = 50
    """Number of generated samples per seed."""
    sigma: float = 10.0
    """Kernel bandwidth for MMD and DPP metrics."""
    output_csv: str = "gan_1d_{problem_id}_metrics.csv"
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
    resolved = resolve_named_checkpoint(
        model_source=args.model_source,
        problem_id=args.problem_id,
        algo="gan_1d",
        seed=seed,
        hf_entity=args.hf_entity,
        hf_repo_prefix=args.hf_repo_prefix,
        required_files=["generator.pth"],
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_artifact_names={"generator.pth": f"{args.problem_id}_gan_1d_generator"},
        local_model_dir=args.local_model_dir,
    )
    run_config = resolved.run_config

    ckpt_path = resolved.files["generator.pth"]
    ckpt = th.load(ckpt_path, map_location=device)

    _, design_normalizer = prepare_data(problem, device)

    model = Generator(
        latent_dim=run_config["latent_dim"], design_shape=design_shape, design_normalizer=design_normalizer
    ).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()

    # Sample noise and generate designs
    z = th.randn((args.n_samples, run_config["latent_dim"]), device=device)
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
    metrics_df = pd.DataFrame(results_dict, index=[0])
    out_path = args.output_csv.format(problem_id=args.problem_id)
    write_header = not os.path.exists(out_path)
    metrics_df.to_csv(out_path, mode="a", header=write_header, index=False)

    print(f"Seed {seed} done; appended to {out_path}")
