"""Evaluation of the PixelCNN++."""
from dataclasses import dataclass
import os

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd
import torch as th
import tyro
import wandb

from engiopt import metrics
from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.pixel_cnn_pp_2d.pixel_cnn_pp_2d import PixelCNNpp
from engiopt.pixel_cnn_pp_2d.pixel_cnn_pp_2d import sample_from_discretized_mix_logistic


@dataclass
class Args:
    """Command-line arguments for a single-seed PixelCNN++ 2D evaluation."""

    problem_id: str = "beams2d"
    """Problem identifier."""
    seed: int = 1
    """Random seed to run."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str = "jstehlin-eth-z-rich" #| None = None
    """Wandb entity name."""
    n_samples: int = 50
    """Number of generated samples per seed."""
    sigma: float = 10.0
    """Kernel bandwidth for MMD and DPP metrics."""
    output_csv: str = "pixel_cnn_pp_2d_{problem_id}_metrics.csv"
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
    conditions_tensor, sampled_conditions, sampled_designs_np, selected_indices = sample_conditions(
        problem=problem,
        n_samples=args.n_samples,
        device=device,
        seed=seed,
    )

    # --------------------------------------------------------
    # adapt to PixelCNN++ input shape requirements
    conditions_tensor = conditions_tensor.unsqueeze(-1).unsqueeze(-1)
    # print(f"Conditions tensor shape: {conditions_tensor.shape}")
    # print(conditions_tensor)
    # print(f"Sampled conditions shape: {sampled_conditions.shape}")
    # print(sampled_conditions)
    # print(f"Sampled designs shape: {sampled_designs_np.shape}")
    # print(sampled_designs_np)
    # print(f"Selected indices: {selected_indices}")
    design_shape = (problem.design_space.shape[0], problem.design_space.shape[1])

    ### Set Up PixelCNN++ Model ###
    if args.wandb_entity is not None:
        artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_pixel_cnn_pp_2d_model:seed_{seed}"
    else:
        artifact_path = f"{args.wandb_project}/{args.problem_id}_pixel_cnn_pp_2d_model:seed_{seed}"

    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")

    class RunRetrievalError(ValueError):
        def __init__(self):
            super().__init__("Failed to retrieve the run")

    run = artifact.logged_by()
    if run is None or not hasattr(run, "config"):
        raise RunRetrievalError

    artifact_dir = artifact.download()
    ckpt_path = os.path.join(artifact_dir, "model_epoch400.pth") # change model.pth if necessary
    ckpt = th.load(ckpt_path, map_location=device) # or th.device(device)


    # Build PixelCNN++ Model
    model = PixelCNNpp(
        nr_resnet=run.config["nr_resnet"],
        nr_filters=run.config["nr_filters"],
        nr_logistic_mix=run.config["nr_logistic_mix"],
        resnet_nonlinearity=run.config["resnet_nonlinearity"],
        dropout_p=run.config["dropout_p"],
        input_channels=1,
        nr_conditions=conditions_tensor.shape[1]
    )

    model.load_state_dict(ckpt["model"])
    model.eval()  # Set to evaluation mode
    model.to(device)

    # input
    gen_designs = th.zeros((args.n_samples, 1, *design_shape), device=device)

    # Generate a batch of designs
    # Autoregressive pixel sampling: iterate spatial positions and condition on
    # previously sampled pixels and the desired_conds.
    for i in range(design_shape[0]):
        for j in range(design_shape[1]):
            out = model(gen_designs, conditions_tensor)
            out_sample = sample_from_discretized_mix_logistic(out, run.config["nr_logistic_mix"])
            # `out_sample` has shape [B, 1, H, W]; copy the sampled value for (i,j)
            gen_designs[:, :, i, j] = out_sample.data[:, :, i, j] #out_sample[:, :, i, j] # out_sample.data[:, :, i, j]


    gen_designs_np = gen_designs.detach().cpu().numpy()
    gen_designs_np = gen_designs_np.reshape(args.n_samples, *problem.design_space.shape)
    # Clip to boundaries for running THIS IS PROBLEM DEPENDENT
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
            "model_id": "pixel_cnn_pp_2d",
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
