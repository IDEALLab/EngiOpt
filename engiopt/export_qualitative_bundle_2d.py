"""Export fixed qualitative comparison bundles for the 2D benchmark problems."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import time
from typing import Any

from diffusers import UNet2DConditionModel
from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import tyro
import wandb

from engiopt.cgan_cnn_2d.cgan_cnn_2d import Generator
from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.diffusion_2d_cond.diffusion_2d_cond import beta_schedule
from engiopt.diffusion_2d_cond.diffusion_2d_cond import DiffusionSampler
from engiopt.flow_matching_2d_cond.core import build_model
from engiopt.flow_matching_2d_cond.core import generate_samples
from engiopt.flow_matching_2d_cond.core import load_local_checkpoint
from engiopt.flow_matching_2d_cond.evaluate_flow_matching_2d_cond import checkpoint_config
from engiopt.flow_matching_2d_cond.evaluate_flow_matching_2d_cond import load_artifact_checkpoint
from engiopt.flow_matching_2d_cond.evaluate_flow_matching_2d_cond import select_device


@dataclasses.dataclass
class Args:
    """Arguments for exporting qualitative image bundles."""

    problems: str = "beams2d,photonics2d,heatconduction2d"
    """Comma-separated problem identifiers."""
    seed: int = 1
    """Seed used for checkpoint selection and random state."""
    condition_seed: int | None = None
    """Seed used when sampling test conditions. Defaults to seed."""
    sample_index: int = 0
    """Index within sampled conditions to export."""
    n_samples: int = 8
    """Number of sampled conditions to draw before selecting sample_index."""
    wandb_project: str = "engiopt"
    """W&B project name used for artifact lookup."""
    wandb_entity: str | None = None
    """W&B entity name used for artifact lookup."""
    flow_checkpoint_path: str | None = None
    """Optional local flow-matching checkpoint path pattern; supports {problem_id} and {seed}."""
    diffusion_checkpoint_path: str | None = None
    """Optional local diffusion checkpoint path pattern; supports {problem_id} and {seed}."""
    cgan_checkpoint_path: str | None = None
    """Optional local cGAN checkpoint path pattern; supports {problem_id} and {seed}."""
    flow_integration_steps: int | None = None
    """Optional override for flow-matching integration steps."""
    device: str = "auto"
    """Device selection for qualitative generation."""
    clip_min: float = 1e-3
    """Minimum value used when clipping generated designs."""
    clip_max: float = 1.0
    """Maximum value used when clipping generated designs."""
    output_dir: str = "qualitative_bundle"
    """Output root directory."""
    track: bool = False
    """Log output image files to W&B."""
    run_name: str | None = None
    """Optional W&B run name override."""


def parse_problem_list(value: str) -> list[str]:
    """Parse a comma-separated problem list."""
    problems = [token.strip() for token in value.split(",") if token.strip()]
    if not problems:
        raise ValueError("No problems were provided")
    return problems


def resolve_optional_pattern(pattern: str | None, problem_id: str, seed: int) -> str | None:
    """Render optional path patterns with problem and seed placeholders."""
    if pattern is None:
        return None
    return pattern.format(problem_id=problem_id, seed=seed)


def load_diffusion_checkpoint(
    problem_id: str,
    seed: int,
    wandb_project: str,
    wandb_entity: str | None,
    device: th.device,
    checkpoint_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load diffusion checkpoint and config from local path or W&B artifact."""
    if checkpoint_path is not None:
        checkpoint = th.load(checkpoint_path, map_location=device)
        run_config: dict[str, Any] = dict(checkpoint.get("args", {}))
        return checkpoint, run_config

    if wandb_entity is not None:
        artifact_path = f"{wandb_entity}/{wandb_project}/{problem_id}_diffusion_2d_cond_model:seed_{seed}"
    else:
        artifact_path = f"{wandb_project}/{problem_id}_diffusion_2d_cond_model:seed_{seed}"

    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")
    run = artifact.logged_by()
    if run is None or not hasattr(run, "config"):
        raise ValueError("Failed to retrieve diffusion run config")

    artifact_dir = artifact.download()
    checkpoint = th.load(os.path.join(artifact_dir, "model.pth"), map_location=device)
    return checkpoint, dict(run.config)


def load_cgan_checkpoint(
    problem_id: str,
    seed: int,
    wandb_project: str,
    wandb_entity: str | None,
    device: th.device,
    checkpoint_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load cGAN checkpoint and config from local path or W&B artifact."""
    if checkpoint_path is not None:
        checkpoint = th.load(checkpoint_path, map_location=device)
        run_config: dict[str, Any] = dict(checkpoint.get("args", {}))
        return checkpoint, run_config

    if wandb_entity is not None:
        artifact_path = f"{wandb_entity}/{wandb_project}/{problem_id}_cgan_cnn_2d_generator:seed_{seed}"
    else:
        artifact_path = f"{wandb_project}/{problem_id}_cgan_cnn_2d_generator:seed_{seed}"

    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")
    run = artifact.logged_by()
    if run is None or not hasattr(run, "config"):
        raise ValueError("Failed to retrieve cGAN run config")

    artifact_dir = artifact.download()
    checkpoint = th.load(os.path.join(artifact_dir, "generator.pth"), map_location=device)
    return checkpoint, dict(run.config)


def save_grayscale_image(image: np.ndarray, output_path: Path) -> None:
    """Save a single 2D design image in grayscale."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, image, cmap="gray", vmin=0.0, vmax=1.0)


def to_jsonable_dict(values: dict[str, Any]) -> dict[str, Any]:
    """Convert condition dictionaries into JSON-safe Python scalars."""
    converted: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, np.generic):
            converted[key] = value.item()
        else:
            converted[key] = value
    return converted


if __name__ == "__main__":
    args = tyro.cli(Args)
    start_time = time.perf_counter()
    problems = parse_problem_list(args.problems)
    device = select_device(args.device)
    condition_seed = args.seed if args.condition_seed is None else args.condition_seed

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "seed": args.seed,
        "condition_seed": condition_seed,
        "sample_index": args.sample_index,
        "n_samples": args.n_samples,
        "problems": {},
    }

    for problem_id in problems:
        if problem_id not in BUILTIN_PROBLEMS:
            raise ValueError(f"Unknown problem '{problem_id}'")

        problem = BUILTIN_PROBLEMS[problem_id]()
        problem.reset(seed=args.seed)
        design_shape = problem.design_space.shape

        conditions_tensor, sampled_conditions, sampled_designs_np, sampled_indices = sample_conditions(
            problem=problem,
            n_samples=args.n_samples,
            device=device,
            seed=condition_seed,
        )
        sample_idx = max(0, min(args.sample_index, args.n_samples - 1))

        # Flow matching generation
        flow_ckpt_path = resolve_optional_pattern(args.flow_checkpoint_path, problem_id, args.seed)
        if flow_ckpt_path is not None:
            flow_checkpoint = load_local_checkpoint(flow_ckpt_path, device)
            flow_run_config: dict[str, Any] = dict(flow_checkpoint.get("args", {}))
        else:
            flow_checkpoint, flow_run_config = load_artifact_checkpoint(
                problem_id=problem_id,
                seed=args.seed,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                device=device,
            )
        flow_layers_per_block = int(
            checkpoint_config(flow_checkpoint, "layers_per_block", flow_run_config.get("layers_per_block", 2))
        )
        flow_num_train_timesteps = int(
            checkpoint_config(flow_checkpoint, "num_train_timesteps", flow_run_config.get("num_train_timesteps", 1000))
        )
        flow_integration_steps = args.flow_integration_steps
        if flow_integration_steps is None:
            flow_integration_steps = int(
                checkpoint_config(flow_checkpoint, "integration_steps", flow_run_config.get("integration_steps", 50))
            )
        flow_model = build_model(
            design_shape=design_shape,
            encoder_hid_dim=len(problem.conditions_keys),
            layers_per_block=flow_layers_per_block,
        ).to(device)
        flow_model.load_state_dict(flow_checkpoint["model"])
        flow_model.eval()
        flow_conditions = conditions_tensor.unsqueeze(1)
        flow_designs = generate_samples(
            model=flow_model,
            design_shape=design_shape,
            encoder_hidden_states=flow_conditions,
            integration_steps=flow_integration_steps,
            num_train_timesteps=flow_num_train_timesteps,
            device=device,
        ).squeeze(1)
        flow_np = flow_designs.detach().cpu().numpy().reshape(args.n_samples, *design_shape)
        flow_np = np.clip(flow_np, args.clip_min, args.clip_max)

        # Diffusion generation
        diffusion_ckpt_path = resolve_optional_pattern(args.diffusion_checkpoint_path, problem_id, args.seed)
        diffusion_ckpt, diffusion_config = load_diffusion_checkpoint(
            problem_id=problem_id,
            seed=args.seed,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            device=device,
            checkpoint_path=diffusion_ckpt_path,
        )
        diffusion_layers_per_block = int(
            diffusion_ckpt.get("model_config", {}).get("layers_per_block", diffusion_config["layers_per_block"])
        )
        diffusion_timesteps = int(diffusion_ckpt.get("model_config", {}).get("num_timesteps", diffusion_config["num_timesteps"]))
        diffusion_noise_schedule = diffusion_ckpt.get("model_config", {}).get("noise_schedule", diffusion_config["noise_schedule"])
        diffusion_model = UNet2DConditionModel(
            sample_size=design_shape,
            in_channels=1,
            out_channels=1,
            cross_attention_dim=64,
            block_out_channels=(32, 64, 128, 256),
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            layers_per_block=diffusion_layers_per_block,
            transformer_layers_per_block=1,
            encoder_hid_dim=len(problem.conditions_keys),
            only_cross_attention=True,
        ).to(device)
        diffusion_options = {
            "cosine": diffusion_noise_schedule == "cosine",
            "exp_biasing": diffusion_noise_schedule == "exp",
            "exp_bias_factor": 1,
        }
        diffusion_betas = beta_schedule(
            t=diffusion_timesteps,
            start=1e-4,
            end=0.02,
            scale=1.0,
            options=diffusion_options,
        )
        diffusion_sampler = DiffusionSampler(diffusion_timesteps, diffusion_betas)
        diffusion_model.load_state_dict(diffusion_ckpt["model"])
        diffusion_model.eval()
        diffusion_designs = th.randn((args.n_samples, 1, *design_shape), device=device)
        diffusion_conditions = conditions_tensor.unsqueeze(1)
        for timestep in reversed(range(diffusion_timesteps)):
            t = th.full((args.n_samples,), timestep, device=device, dtype=th.long)
            diffusion_designs = diffusion_sampler.sample_timestep(diffusion_model, diffusion_designs, t, diffusion_conditions)
        diffusion_np = diffusion_designs.squeeze(1).detach().cpu().numpy().reshape(args.n_samples, *design_shape)
        diffusion_np = np.clip(diffusion_np, args.clip_min, args.clip_max)

        # cGAN generation
        cgan_ckpt_path = resolve_optional_pattern(args.cgan_checkpoint_path, problem_id, args.seed)
        cgan_ckpt, cgan_config = load_cgan_checkpoint(
            problem_id=problem_id,
            seed=args.seed,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            device=device,
            checkpoint_path=cgan_ckpt_path,
        )
        cgan_model = Generator(
            latent_dim=int(cgan_config["latent_dim"]),
            n_conds=len(problem.conditions_keys),
            design_shape=design_shape,
        ).to(device)
        cgan_model.load_state_dict(cgan_ckpt["generator"])
        cgan_model.eval()
        cgan_conditions = conditions_tensor.unsqueeze(-1).unsqueeze(-1)
        noise = th.randn((args.n_samples, int(cgan_config["latent_dim"]), 1, 1), device=device, dtype=th.float)
        cgan_designs = cgan_model(noise, cgan_conditions)
        cgan_np = cgan_designs.detach().cpu().numpy().reshape(args.n_samples, *design_shape)
        cgan_np = np.clip(cgan_np, args.clip_min, args.clip_max)

        problem_output_dir = output_root / problem_id
        reference = np.clip(sampled_designs_np[sample_idx], args.clip_min, args.clip_max)
        flow_image = flow_np[sample_idx]
        diffusion_image = diffusion_np[sample_idx]
        cgan_image = cgan_np[sample_idx]

        save_grayscale_image(reference, problem_output_dir / "reference.png")
        save_grayscale_image(flow_image, problem_output_dir / "flow_matching_2d_cond.png")
        save_grayscale_image(diffusion_image, problem_output_dir / "diffusion_2d_cond.png")
        save_grayscale_image(cgan_image, problem_output_dir / "cgan_cnn_2d.png")

        metadata["problems"][problem_id] = {
            "selected_dataset_index": int(sampled_indices[sample_idx]),
            "sample_index": int(sample_idx),
            "conditions": to_jsonable_dict(sampled_conditions[sample_idx]),
            "flow_integration_steps": flow_integration_steps,
            "flow_num_train_timesteps": flow_num_train_timesteps,
            "diffusion_timesteps": diffusion_timesteps,
            "diffusion_noise_schedule": diffusion_noise_schedule,
            "cgan_latent_dim": int(cgan_config["latent_dim"]),
        }

    metadata_path = output_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    if args.track:
        run_name = args.run_name or f"qualitative_bundle_2d__seed{args.seed}__{int(time.time())}"
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type="qualitative-export",
            name=run_name,
            config={**vars(args), "resolved_problems": problems},
        )
        if run is None:
            raise RuntimeError("Failed to initialize Weights & Biases run")
        for problem_id in problems:
            problem_dir = output_root / problem_id
            run.log(
                {
                    f"qualitative/{problem_id}/reference": wandb.Image(str(problem_dir / "reference.png")),
                    f"qualitative/{problem_id}/flow_matching": wandb.Image(
                        str(problem_dir / "flow_matching_2d_cond.png")
                    ),
                    f"qualitative/{problem_id}/diffusion": wandb.Image(str(problem_dir / "diffusion_2d_cond.png")),
                    f"qualitative/{problem_id}/cgan": wandb.Image(str(problem_dir / "cgan_cnn_2d.png")),
                }
            )
        artifact = wandb.Artifact(f"qualitative_bundle_seed{args.seed}", type="qualitative-bundle")
        artifact.add_dir(str(output_root))
        run.log_artifact(artifact)
        run.finish()

    print(f"Wrote qualitative bundle to {output_root} in {time.perf_counter() - start_time:.2f}s")
