"""Evaluation for the CGAN 2D w/ CNN."""

from __future__ import annotations

import dataclasses
import itertools
import os

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd
import torch as th
import tyro

from engiopt import metrics
from engiopt.cgan_cnn_2d.cgan_cnn_2d import Generator
from engiopt.dataset_sample_conditions import sample_conditions
import wandb


@dataclasses.dataclass
class Args:
    """Command-line arguments for a single-seed cGAN CNN 2D evaluation."""

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
    output_csv: str = "cgan_cnn_2d_{problem_id}_metrics.csv"
    """Output CSV path template; may include {problem_id}."""

    # LV metrics (optional) - provide seed and at least one threshold list to enable
    lvae_seed: int | None = None
    """LVAE seed. If provided along with thresholds, computes LV-MMD and LV-DPP."""
    lvae_rec_thresholds: str = ""
    """Comma-separated LVAE reconstruction NMSE thresholds (e.g. '0.0005,0.001,0.005')."""
    lvae_perf_thresholds: str = ""
    """Comma-separated LVAE performance NMSE thresholds (e.g. '0.001,1000')."""

    # WandB logging
    log_to_wandb: bool = False
    """Log evaluation metrics to the original WandB training run."""


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Parse comma-separated threshold strings into lists of floats
    rec_thresholds = [float(x) for x in args.lvae_rec_thresholds.split(",") if x.strip()] if args.lvae_rec_thresholds else []
    perf_thresholds = [float(x) for x in args.lvae_perf_thresholds.split(",") if x.strip()] if args.lvae_perf_thresholds else []

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

    ### Set Up Generator ###

    # Restores the pytorch model from wandb
    if args.wandb_entity is not None:
        artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_cgan_cnn_2d_generator:seed_{seed}"
    else:
        artifact_path = f"{args.wandb_project}/{args.problem_id}_cgan_cnn_2d_generator:seed_{seed}"

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
    model = Generator(
        latent_dim=run.config["latent_dim"], n_conds=len(problem.conditions_keys), design_shape=problem.design_space.shape
    )
    model.load_state_dict(ckpt["generator"])
    model.eval()  # Set to evaluation mode
    model.to(device)

    # Sample noise as generator input
    z = th.randn((args.n_samples, run.config["latent_dim"], 1, 1), device=device, dtype=th.float)

    # Generate a batch of designs
    gen_designs = model(z, conditions_tensor)
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

    metrics_dict.update(
        {
            "seed": seed,
            "problem_id": args.problem_id,
            "model_id": "cgan_cnn_2d",
            "n_samples": args.n_samples,
        }
    )

    # Compute LV metrics for all (rec, perf) threshold combinations
    if args.lvae_seed is not None and rec_thresholds and perf_thresholds:
        from sklearn.decomposition import PCA

        from engiopt.vanilla_lvae.utils import encode_designs
        from engiopt.vanilla_lvae.utils import load_lvae_encoder

        metrics_dict["lvae_seed"] = args.lvae_seed

        for rec_thresh, perf_thresh in itertools.product(rec_thresholds, perf_thresholds):
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

                # Without importance weighting (sigma from reference data for MMD, self for DPP)
                lv_sigma = metrics.compute_median_sigma(z_data)
                lv_mmd_val = metrics.mmd(z_gen, z_data, sigma=lv_sigma, importance_weighted=False)
                lv_dpp_val = metrics.dpp_diversity(z_gen, sigma=lv_sigma, importance_weighted=False)

                # With importance weighting
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

                # PCA baseline: project to same dimensionality as LVAE active dims
                n_pca = min(n_active, args.n_samples - 1)  # PCA needs n_components < n_samples
                if n_pca > 0:
                    flat_gen = gen_designs_np.reshape(gen_designs_np.shape[0], -1)
                    flat_data = sampled_designs_np.reshape(sampled_designs_np.shape[0], -1)
                    pca = PCA(n_components=n_pca)
                    pca.fit(flat_data)
                    pca_gen = pca.transform(flat_gen)
                    pca_data = pca.transform(flat_data)

                    pca_sigma = metrics.compute_median_sigma(pca_data)
                    pca_mmd_val = metrics.mmd(pca_gen, pca_data, sigma=pca_sigma, importance_weighted=False)
                    pca_dpp_val = metrics.dpp_diversity(pca_gen, sigma=pca_sigma, importance_weighted=False)

                    pca_data_w = metrics.apply_importance_weights(pca_data, pca_data)
                    pca_sigma_iw = metrics.compute_median_sigma(pca_data_w)
                    pca_mmd_iw_val = metrics.mmd(pca_gen, pca_data, sigma=pca_sigma_iw, importance_weighted=True)
                    pca_dpp_iw_val = metrics.dpp_diversity(pca_gen, sigma=pca_sigma_iw, importance_weighted=True)

                    metrics_dict[f"pca_mmd{suffix}"] = pca_mmd_val
                    metrics_dict[f"pca_dpp{suffix}"] = pca_dpp_val
                    metrics_dict[f"pca_sigma{suffix}"] = pca_sigma
                    metrics_dict[f"pca_mmd_iw{suffix}"] = pca_mmd_iw_val
                    metrics_dict[f"pca_dpp_iw{suffix}"] = pca_dpp_iw_val
                    metrics_dict[f"pca_sigma_iw{suffix}"] = pca_sigma_iw
                    print(f"  PCA({n_pca}): MMD={pca_mmd_val:.6f}, DPP={pca_dpp_val:.6e}")

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
        if args.lvae_seed is not None and rec_thresholds and perf_thresholds:
            for rec_thresh, perf_thresh in itertools.product(rec_thresholds, perf_thresholds):
                suffix = f"_rec{rec_thresh}_perf{perf_thresh}"
                prefix = f"eval/lv_rec{rec_thresh}_perf{perf_thresh}_lvae{args.lvae_seed}"
                run.summary[f"{prefix}/lv_mmd"] = metrics_dict.get(f"lv_mmd{suffix}")
                run.summary[f"{prefix}/lv_dpp"] = metrics_dict.get(f"lv_dpp{suffix}")
                run.summary[f"{prefix}/lv_sigma"] = metrics_dict.get(f"lv_sigma{suffix}")
                run.summary[f"{prefix}/lv_mmd_iw"] = metrics_dict.get(f"lv_mmd_iw{suffix}")
                run.summary[f"{prefix}/lv_dpp_iw"] = metrics_dict.get(f"lv_dpp_iw{suffix}")
                run.summary[f"{prefix}/lv_sigma_iw"] = metrics_dict.get(f"lv_sigma_iw{suffix}")
                run.summary[f"{prefix}/n_active_dims"] = metrics_dict.get(f"lvae_n_active_dims{suffix}")
                run.summary[f"{prefix}/pca_mmd"] = metrics_dict.get(f"pca_mmd{suffix}")
                run.summary[f"{prefix}/pca_dpp"] = metrics_dict.get(f"pca_dpp{suffix}")
                run.summary[f"{prefix}/pca_sigma"] = metrics_dict.get(f"pca_sigma{suffix}")
                run.summary[f"{prefix}/pca_mmd_iw"] = metrics_dict.get(f"pca_mmd_iw{suffix}")
                run.summary[f"{prefix}/pca_dpp_iw"] = metrics_dict.get(f"pca_dpp_iw{suffix}")
                run.summary[f"{prefix}/pca_sigma_iw"] = metrics_dict.get(f"pca_sigma_iw{suffix}")

        run.summary.update()
        print(f"  Logged metrics to WandB run: {run.name}")

    print(f"Seed {seed} done; appended to {out_path}")
