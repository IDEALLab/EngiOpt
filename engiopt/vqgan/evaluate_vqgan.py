"""Evaluation for the VQGAN."""

from __future__ import annotations

import dataclasses
import os

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd
import torch as th
import tyro

from engiopt import metrics
from engiopt.checkpoint_store import ModelSource
from engiopt.checkpoint_store import resolve_named_checkpoint
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
    model_source: ModelSource = "auto"
    """Where to load the checkpoint package from."""
    hf_entity: str = "IDEALLab"
    """HF organization or user for checkpoint storage."""
    hf_repo_prefix: str = "engiopt"
    """HF repo prefix used to build per-family model repos."""
    local_model_dir: str | None = None
    """Optional local checkpoint package directory."""
    n_samples: int = 50
    """Number of generated samples per seed."""
    sigma: float = 10.0
    """Kernel bandwidth for MMD and DPP metrics."""
    output_csv: str = "vqgan_{problem_id}_metrics.csv"
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

    ### Set Up Transformer ###

    resolved = resolve_named_checkpoint(
        model_source=args.model_source,
        problem_id=args.problem_id,
        algo="vqgan",
        seed=seed,
        hf_entity=args.hf_entity,
        hf_repo_prefix=args.hf_repo_prefix,
        required_files=["vqgan.pth", "transformer.pth"],
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_artifact_names={
            "vqgan.pth": f"{args.problem_id}_vqgan_vqgan",
            "transformer.pth": f"{args.problem_id}_vqgan_transformer",
        },
        wandb_config_artifact_name=f"{args.problem_id}_vqgan_transformer",
        local_model_dir=args.local_model_dir,
    )
    run_config = resolved.run_config

    ckpt_path_cvqgan = os.path.join(resolved.root_dir, "cvqgan.pth")
    if run_config["conditional"] and not os.path.exists(ckpt_path_cvqgan):
        cvqgan_resolved = resolve_named_checkpoint(
            model_source="wandb" if args.model_source == "wandb" else "auto",
            problem_id=args.problem_id,
            algo="vqgan",
            seed=seed,
            hf_entity=args.hf_entity,
            hf_repo_prefix=args.hf_repo_prefix,
            required_files=["cvqgan.pth"],
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_artifact_names={"cvqgan.pth": f"{args.problem_id}_vqgan_cvqgan"},
            wandb_config_artifact_name=f"{args.problem_id}_vqgan_transformer",
            local_model_dir=args.local_model_dir,
        )
        ckpt_path_cvqgan = cvqgan_resolved.files["cvqgan.pth"]

    ckpt_path_vqgan = resolved.files["vqgan.pth"]
    ckpt_path_transformer = resolved.files["transformer.pth"]
    ckpt_cvqgan = None
    if os.path.exists(ckpt_path_cvqgan):
        ckpt_cvqgan = th.load(ckpt_path_cvqgan, map_location=th.device(device), weights_only=False)
    elif run_config["conditional"]:
        raise FileNotFoundError("Conditional VQGAN evaluation requires cvqgan.pth, but no checkpoint was found.")
    ckpt_vqgan = th.load(ckpt_path_vqgan, map_location=th.device(device), weights_only=False)
    ckpt_transformer = th.load(ckpt_path_transformer, map_location=th.device(device), weights_only=False)

    vqgan = VQGAN(
        device=device,
        is_c=False,
        encoder_channels=run_config["encoder_channels"],
        encoder_start_resolution=run_config["image_size"],
        encoder_attn_resolutions=run_config["encoder_attn_resolutions"],
        encoder_num_res_blocks=run_config["encoder_num_res_blocks"],
        decoder_channels=run_config["decoder_channels"],
        decoder_start_resolution=run_config["latent_size"],
        decoder_attn_resolutions=run_config["decoder_attn_resolutions"],
        decoder_num_res_blocks=run_config["decoder_num_res_blocks"],
        image_channels=run_config["image_channels"],
        latent_dim=run_config["latent_dim"],
        num_codebook_vectors=run_config["num_codebook_vectors"],
    )
    vqgan.load_state_dict(ckpt_vqgan["vqgan"])
    vqgan.eval()  # Set to evaluation mode
    vqgan.to(device)

    cvqgan = VQGAN(
        device=device,
        is_c=True,
        cond_feature_map_dim=run_config["cond_feature_map_dim"],
        cond_dim=run_config["cond_dim"],
        cond_hidden_dim=run_config["cond_hidden_dim"],
        cond_latent_dim=run_config["cond_latent_dim"],
        cond_codebook_vectors=run_config["cond_codebook_vectors"],
    )
    if ckpt_cvqgan is not None:
        cvqgan.load_state_dict(ckpt_cvqgan["cvqgan"])
    cvqgan.eval()  # Set to evaluation mode
    cvqgan.to(device)

    model = VQGANTransformer(
        conditional=run_config["conditional"],
        vqgan=vqgan,
        cvqgan=cvqgan,
        image_size=run_config["image_size"],
        decoder_channels=run_config["decoder_channels"],
        cond_feature_map_dim=run_config["cond_feature_map_dim"],
        num_codebook_vectors=run_config["num_codebook_vectors"],
        n_layer=run_config["n_layer"],
        n_head=run_config["n_head"],
        n_embd=run_config["n_embd"],
        dropout=run_config["dropout"],
    )
    model.load_state_dict(ckpt_transformer["transformer"])
    model.eval()  # Set to evaluation mode
    model.to(device)

    ### Set up testing conditions ###
    _, sampled_conditions, sampled_designs_np, _sampled_designs_tensor = sample_conditions(
        problem=problem, n_samples=args.n_samples, device=device, seed=seed
    )

    # Clean up conditions based on model training settings and convert back to tensor
    sampled_conditions_new = sampled_conditions.select(range(len(sampled_conditions)))
    conditions = sampled_conditions_new.column_names

    # Drop constant condition columns if enabled
    if run_config["drop_constant_conditions"]:
        sampled_conditions_new, conditions = drop_constant(sampled_conditions_new, sampled_conditions_new.column_names)

    # Normalize condition columns if enabled
    if run_config["normalize_conditions"]:
        sampled_conditions_new, mean, std = normalize(sampled_conditions_new, conditions)

    # Convert to tensor
    conditions_tensor = th.stack([th.as_tensor(sampled_conditions_new[c][:]).float() for c in conditions], dim=1).to(device)

    # Set the start-of-sequence tokens for the transformer using the CVQGAN to discretize the conditions if enabled
    if run_config["conditional"]:
        c = model.encode_to_z(x=conditions_tensor, is_c=True)[1]
    else:
        c = th.ones(args.n_samples, 1, dtype=th.int64, device=device) * model.sos_token

    # Generate a batch of designs
    latent_designs = model.sample(
        x=th.empty(args.n_samples, 0, dtype=th.int64, device=device), c=c, steps=(run_config["latent_size"] ** 2)
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
            "sigma": args.sigma,
        }
    )

    # Append result row to CSV
    metrics_df = pd.DataFrame([metrics_dict])
    out_path = args.output_csv.format(problem_id=args.problem_id)
    write_header = not os.path.exists(out_path)
    metrics_df.to_csv(out_path, mode="a", header=write_header, index=False)

    print(f"Seed {seed} done; appended to {out_path}")
