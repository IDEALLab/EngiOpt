"""Evaluation for the Diffusion 2d_cond w/ seed looping and CSV saving."""

from __future__ import annotations

import dataclasses
import itertools
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
    sigma: float | None = None
    """Kernel bandwidth for MMD and DPP metrics. If None, uses median heuristic from reference data."""
    output_csv: str = "diffusion_2d_cond_{problem_id}_metrics.csv"
    """Output CSV path template; may include {problem_id}."""

    # LV metrics (optional) - provide seed and at least one threshold list to enable
    lvae_seed: int | None = None
    """LVAE seed. If provided along with thresholds, computes LV-MMD and LV-DPP."""
    lvae_rec_thresholds: tuple[float, ...] | None = None
    """LVAE reconstruction NMSE thresholds (evaluated for all combinations with perf thresholds)."""
    lvae_perf_thresholds: tuple[float, ...] | None = None
    """LVAE performance NMSE thresholds (evaluated for all combinations with rec thresholds)."""

    # WandB logging
    log_to_wandb: bool = False
    """Log evaluation metrics to the original WandB training run."""


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
        }
    )

    # Compute LV metrics for all (rec, perf) threshold combinations
    if args.lvae_seed is not None and args.lvae_rec_thresholds is not None and args.lvae_perf_thresholds is not None:
        from engiopt.vanilla_lvae.utils import encode_designs
        from engiopt.vanilla_lvae.utils import load_lvae_encoder

        metrics_dict["lvae_seed"] = args.lvae_seed

        for rec_thresh, perf_thresh in itertools.product(args.lvae_rec_thresholds, args.lvae_perf_thresholds):
            suffix = f"_rec{rec_thresh}_perf{perf_thresh}"
            print(f"Loading LVAE (seed={args.lvae_seed}, rec={rec_thresh}, perf={perf_thresh})...")
            try:
                encoder, lvae_config = load_lvae_encoder(
                    problem_id=args.problem_id,
                    seed=args.lvae_seed,
                    rec_threshold=rec_thresh,
                    perf_threshold=perf_thresh,
                    wandb_project=args.wandb_project,
                    wandb_entity=args.wandb_entity,
                    device=device,
                )

                # Encode designs to latent space
                z_gen = encode_designs(encoder, gen_designs_np, device)
                z_data = encode_designs(encoder, sampled_designs_np, device)
                n_active = int((np.var(z_data, axis=0) > 1e-8).sum())

                # Without importance weighting (sigma from reference data)
                lv_sigma = metrics.compute_median_sigma(z_data)
                lv_mmd_val = metrics.mmd(z_gen, z_data, sigma=lv_sigma, importance_weighted=False)
                lv_dpp_val = metrics.dpp_diversity(z_gen, sigma=lv_sigma, importance_weighted=False)

                # With importance weighting (sigma from weighted reference data)
                z_data_w = metrics.apply_importance_weights(z_data, z_data)
                lv_sigma_iw = metrics.compute_median_sigma(z_data_w)
                lv_mmd_iw_val = metrics.mmd(z_gen, z_data, sigma=lv_sigma_iw, importance_weighted=True)
                lv_dpp_iw_val = metrics.dpp_diversity(z_gen, sigma=lv_sigma_iw, importance_weighted=True)

                metrics_dict[f"lv_mmd{suffix}"] = lv_mmd_val
                metrics_dict[f"lv_dpp{suffix}"] = lv_dpp_val
                metrics_dict[f"lv_sigma{suffix}"] = lv_sigma
                metrics_dict[f"lv_mmd_iw{suffix}"] = lv_mmd_iw_val
                metrics_dict[f"lv_dpp_iw{suffix}"] = lv_dpp_iw_val
                metrics_dict[f"lv_sigma_iw{suffix}"] = lv_sigma_iw
                metrics_dict[f"lvae_n_active_dims{suffix}"] = n_active
                print(f"  sigma={lv_sigma:.4f}, LV-MMD: {lv_mmd_val:.6f}, LV-DPP: {lv_dpp_val:.6e}")
                print(f"  sigma_iw={lv_sigma_iw:.4f}, LV-MMD(iw): {lv_mmd_iw_val:.6f}, LV-DPP(iw): {lv_dpp_iw_val:.6e}")
                print(f"  Active dims: {n_active}")
            except Exception as e:
                print(f"  Failed for rec={rec_thresh}, perf={perf_thresh}: {e}")

    # Append result row to CSV
    metrics_df = pd.DataFrame([metrics_dict])
    out_path = args.output_csv.format(problem_id=args.problem_id)
    write_header = not os.path.exists(out_path)
    metrics_df.to_csv(out_path, mode="a", header=write_header, index=False)

    # Log to WandB training run
    if args.log_to_wandb and run is not None:
        # Always log base metrics
        run.summary["eval/mmd"] = metrics_dict.get("mmd")
        run.summary["eval/dpp"] = metrics_dict.get("dpp")
        run.summary["eval/iog"] = metrics_dict.get("iog")
        run.summary["eval/cog"] = metrics_dict.get("cog")
        run.summary["eval/fog"] = metrics_dict.get("fog")
        run.summary["eval/mmd_sigma"] = metrics_dict.get("mmd_sigma")

        # Log per-combination LV metrics
        if args.lvae_seed is not None and args.lvae_rec_thresholds is not None and args.lvae_perf_thresholds is not None:
            for rec_thresh, perf_thresh in itertools.product(args.lvae_rec_thresholds, args.lvae_perf_thresholds):
                suffix = f"_rec{rec_thresh}_perf{perf_thresh}"
                prefix = f"eval/lv_rec{rec_thresh}_perf{perf_thresh}_lvae{args.lvae_seed}"
                run.summary[f"{prefix}/lv_mmd"] = metrics_dict.get(f"lv_mmd{suffix}")
                run.summary[f"{prefix}/lv_dpp"] = metrics_dict.get(f"lv_dpp{suffix}")
                run.summary[f"{prefix}/lv_sigma"] = metrics_dict.get(f"lv_sigma{suffix}")
                run.summary[f"{prefix}/lv_mmd_iw"] = metrics_dict.get(f"lv_mmd_iw{suffix}")
                run.summary[f"{prefix}/lv_dpp_iw"] = metrics_dict.get(f"lv_dpp_iw{suffix}")
                run.summary[f"{prefix}/lv_sigma_iw"] = metrics_dict.get(f"lv_sigma_iw{suffix}")
                run.summary[f"{prefix}/n_active_dims"] = metrics_dict.get(f"lvae_n_active_dims{suffix}")

        run.summary.update()
        print(f"  Logged metrics to WandB run: {run.name}")

    print(f"Seed {seed} done; appended to {out_path}")
