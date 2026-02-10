"""Evaluation for the VQGAN."""

from __future__ import annotations

import dataclasses
import itertools
import os

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd
import torch as th
import tyro
import wandb

from engiopt import metrics
from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.transforms import drop_constant
from engiopt.transforms import normalize
from engiopt.transforms import resize_to
from engiopt.vqgan.vqgan import VQGAN
from engiopt.vqgan.vqgan import VQGANTransformer


@dataclasses.dataclass
class Args:
    """Command-line arguments for a single-seed VQGAN 2D evaluation."""

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
    output_csv: str = "vqgan_{problem_id}_metrics.csv"
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

    ### Set Up Transformer ###

    # Restores the pytorch model from wandb
    if args.wandb_entity is not None:
        artifact_path_0 = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_vqgan_cvqgan:seed_{seed}"
        artifact_path_1 = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_vqgan_vqgan:seed_{seed}"
        artifact_path_2 = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_vqgan_transformer:seed_{seed}"
    else:
        artifact_path_0 = f"{args.wandb_project}/{args.problem_id}_vqgan_cvqgan:seed_{seed}"
        artifact_path_1 = f"{args.wandb_project}/{args.problem_id}_vqgan_vqgan:seed_{seed}"
        artifact_path_2 = f"{args.wandb_project}/{args.problem_id}_vqgan_transformer:seed_{seed}"

    api = wandb.Api()
    artifact_0 = api.artifact(artifact_path_0, type="model")
    artifact_1 = api.artifact(artifact_path_1, type="model")
    artifact_2 = api.artifact(artifact_path_2, type="model")

    class RunRetrievalError(ValueError):
        def __init__(self):
            super().__init__("Failed to retrieve the run")

    run = artifact_2.logged_by()
    if run is None or not hasattr(run, "config"):
        raise RunRetrievalError
    artifact_dir_0 = artifact_0.download()
    artifact_dir_1 = artifact_1.download()
    artifact_dir_2 = artifact_2.download()

    ckpt_path_0 = os.path.join(artifact_dir_0, "cvqgan.pth")
    ckpt_path_1 = os.path.join(artifact_dir_1, "vqgan.pth")
    ckpt_path_2 = os.path.join(artifact_dir_2, "transformer.pth")
    ckpt_0 = th.load(ckpt_path_0, map_location=th.device(device), weights_only=False)
    ckpt_1 = th.load(ckpt_path_1, map_location=th.device(device), weights_only=False)
    ckpt_2 = th.load(ckpt_path_2, map_location=th.device(device), weights_only=False)

    vqgan = VQGAN(
        device=device,
        is_c=False,
        encoder_channels=run.config["encoder_channels"],
        encoder_start_resolution=run.config["image_size"],
        encoder_attn_resolutions=run.config["encoder_attn_resolutions"],
        encoder_num_res_blocks=run.config["encoder_num_res_blocks"],
        decoder_channels=run.config["decoder_channels"],
        decoder_start_resolution=run.config["latent_size"],
        decoder_attn_resolutions=run.config["decoder_attn_resolutions"],
        decoder_num_res_blocks=run.config["decoder_num_res_blocks"],
        image_channels=run.config["image_channels"],
        latent_dim=run.config["latent_dim"],
        num_codebook_vectors=run.config["num_codebook_vectors"],
    )
    vqgan.load_state_dict(ckpt_1["vqgan"])
    vqgan.eval()  # Set to evaluation mode
    vqgan.to(device)

    cvqgan = VQGAN(
        device=device,
        is_c=True,
        cond_feature_map_dim=run.config["cond_feature_map_dim"],
        cond_dim=run.config["cond_dim"],
        cond_hidden_dim=run.config["cond_hidden_dim"],
        cond_latent_dim=run.config["cond_latent_dim"],
        cond_codebook_vectors=run.config["cond_codebook_vectors"],
    )
    cvqgan.load_state_dict(ckpt_0["cvqgan"])
    cvqgan.eval()  # Set to evaluation mode
    cvqgan.to(device)

    model = VQGANTransformer(
        conditional=run.config["conditional"],
        vqgan=vqgan,
        cvqgan=cvqgan,
        image_size=run.config["image_size"],
        decoder_channels=run.config["decoder_channels"],
        cond_feature_map_dim=run.config["cond_feature_map_dim"],
        num_codebook_vectors=run.config["num_codebook_vectors"],
        n_layer=run.config["n_layer"],
        n_head=run.config["n_head"],
        n_embd=run.config["n_embd"],
        dropout=run.config["dropout"],
    )
    model.load_state_dict(ckpt_2["transformer"])
    model.eval()  # Set to evaluation mode
    model.to(device)

    ### Set up testing conditions ###
    _, sampled_conditions, sampled_designs_np, _ = sample_conditions(
        problem=problem, n_samples=args.n_samples, device=device, seed=seed
    )

    # Clean up conditions based on model training settings and convert back to tensor
    sampled_conditions_new = sampled_conditions.select(range(len(sampled_conditions)))
    conditions = sampled_conditions_new.column_names

    # Drop constant condition columns if enabled
    if run.config["drop_constant_conditions"]:
        sampled_conditions_new, conditions = drop_constant(sampled_conditions_new, sampled_conditions_new.column_names)

    # Normalize condition columns if enabled
    if run.config["normalize_conditions"]:
        sampled_conditions_new, mean, std = normalize(sampled_conditions_new, conditions)

    # Convert to tensor
    conditions_tensor = th.stack([th.as_tensor(sampled_conditions_new[c][:]).float() for c in conditions], dim=1).to(device)

    # Set the start-of-sequence tokens for the transformer using the CVQGAN to discretize the conditions if enabled
    if run.config["conditional"]:
        c = model.encode_to_z(x=conditions_tensor, is_c=True)[1]
    else:
        c = th.ones(args.n_samples, 1, dtype=th.int64, device=device) * model.sos_token

    # Generate a batch of designs
    latent_designs = model.sample(
        x=th.empty(args.n_samples, 0, dtype=th.int64, device=device), c=c, steps=(run.config["latent_size"] ** 2)
    )
    gen_designs = resize_to(
        data=model.z_to_image(latent_designs), h=problem.design_space.shape[0], w=problem.design_space.shape[1]
    )
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
            "model_id": "vqgan",
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
