"""Export fixed qualitative comparison bundles for the 2D benchmark problems."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import time
from typing import Any

from diffusers import UNet2DConditionModel
from engiopt.diffusion_2d_cond.diffusion_2d_cond import denormalize_designs_from_diffusion_range
from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
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
from engiopt.flow_matching_2d_cond.evaluate_flow_matching_2d_cond import build_design_grid
from engiopt.flow_matching_2d_cond.evaluate_flow_matching_2d_cond import load_top_k_candidates
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
    n_samples: int = 3
    """Number of sampled conditions to draw before building the 3x1 raster."""
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
    checkpoint_dir: str | None = None
    """Optional directory with epoch checkpoints and validation_metrics.json for top-k export."""
    select_best_of_top_k: bool = True
    """If True and checkpoint_dir is set, export all shortlisted top-k checkpoints."""
    top_k: int = 5
    """Number of shortlisted checkpoints to export when select_best_of_top_k is enabled."""
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
    image_dpi: int = 300
    """DPI metadata written to exported qualitative PNGs."""
    image_min_pixels: int = 600
    """Minimum shorter-side pixel count for exported qualitative PNGs."""
    track: bool = False
    """Log output image files to W&B."""
    run_name: str | None = None
    """Optional W&B run name override."""
    flow_method: str = "euler"
    """Integration method used for flow-matching: euler, midpoint, rk4, etc."""


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


def save_grayscale_image(image: np.ndarray, output_path: Path, *, dpi: int = 300, min_pixels: int = 600) -> None:
    """Save a single 2D design image in EngiBench Reds_r colormap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_colormap_png(image, output_path, dpi=dpi, min_pixels=min_pixels)


def save_colormap_png(image: np.ndarray, output_path: Path, *, dpi: int = 300, min_pixels: int = 600) -> None:
    """Save a design array as a publication-resolution PNG without changing values."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D image array, got shape {array.shape}")
    normalized = np.clip(array, 0.0, 1.0)
    rgba = plt.get_cmap("viridis")(normalized, bytes=True)
    pil_image = Image.fromarray(rgba, mode="RGBA")
    shorter_side = min(pil_image.size)
    if min_pixels > 0 and shorter_side < min_pixels:
        scale = int(np.ceil(min_pixels / shorter_side))
        new_size = (pil_image.size[0] * scale, pil_image.size[1] * scale)
        pil_image = pil_image.resize(new_size, resample=Image.Resampling.NEAREST)
    pil_image.save(output_path, dpi=(dpi, dpi))


def to_jsonable_dict(values: dict[str, Any]) -> dict[str, Any]:
    """Convert condition dictionaries into JSON-safe Python scalars."""
    converted: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, np.generic):
            converted[key] = value.item()
        else:
            converted[key] = value
    return converted


def save_design_raster(designs_np: np.ndarray, output_path: Path, *, dpi: int = 300, min_pixels: int = 600) -> None:
    """Save a tiled raster of designs using 3x1 layout (3 rows, 1 col for compact visualization)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = build_design_grid(designs_np, rows=3, cols=1)
    save_colormap_png(grid, output_path, dpi=dpi, min_pixels=min_pixels)


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
        "checkpoint_dir": args.checkpoint_dir,
        "select_best_of_top_k": args.select_best_of_top_k,
        "top_k": args.top_k,
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
        problem_output_dir = output_root / problem_id

        flow_conditions = conditions_tensor.unsqueeze(1)
        flow_top_k_metadata: list[dict[str, Any]] = []
        flow_integration_steps: int | None = None
        flow_num_train_timesteps: int | None = None
        if args.checkpoint_dir is not None and args.select_best_of_top_k:
            checkpoint_dir = Path(args.checkpoint_dir)
            candidates = load_top_k_candidates(checkpoint_dir, args.top_k)
            flow_np: np.ndarray | None = None
            flow_rank_1_np: np.ndarray | None = None
            for rank, candidate in enumerate(candidates, 1):
                flow_checkpoint = load_local_checkpoint(str(candidate["checkpoint_path"]), device)
                flow_run_config_top_k: dict[str, Any] = dict(flow_checkpoint.get("args", {}))
                flow_layers_per_block = int(
                    checkpoint_config(flow_checkpoint, "layers_per_block", flow_run_config_top_k.get("layers_per_block", 2))
                )
                flow_num_train_timesteps = int(
                    checkpoint_config(
                        flow_checkpoint, "num_train_timesteps", flow_run_config_top_k.get("num_train_timesteps", 1000)
                    )
                )
                flow_integration_steps = args.flow_integration_steps
                if flow_integration_steps is None:
                    flow_integration_steps = int(
                        checkpoint_config(flow_checkpoint, "integration_steps", flow_run_config_top_k.get("integration_steps", 50))
                    )
                flow_model = build_model(
                    design_shape=design_shape,
                    encoder_hid_dim=len(problem.conditions_keys),
                    layers_per_block=flow_layers_per_block,
                ).to(device)
                flow_model.load_state_dict(flow_checkpoint["model"])
                flow_model.eval()
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
                save_design_raster(
                    flow_np,
                    problem_output_dir / f"rank_{rank}.png",
                    dpi=args.image_dpi,
                    min_pixels=args.image_min_pixels,
                )
                save_design_raster(
                    flow_np,
                    problem_output_dir / f"flow_matching_2d_cond_rank_{rank}.png",
                    dpi=args.image_dpi,
                    min_pixels=args.image_min_pixels,
                )
                if rank == 1:
                    flow_rank_1_np = flow_np
                flow_top_k_metadata.append(
                    {
                        "rank": rank,
                        "epoch": int(candidate["epoch"] + 1),
                        "validation_mmd": float(candidate["metric_value"]),
                        "checkpoint_path": str(candidate["checkpoint_path"]),
                        "flow_integration_steps": int(flow_integration_steps),
                        "flow_num_train_timesteps": int(flow_num_train_timesteps),
                    }
                )
            assert flow_rank_1_np is not None
            flow_integration_steps = int(flow_top_k_metadata[0]["flow_integration_steps"])
            flow_num_train_timesteps = int(flow_top_k_metadata[0]["flow_num_train_timesteps"])
            flow_np = flow_rank_1_np
            save_design_raster(
                flow_np,
                problem_output_dir / "flow_matching_2d_cond.png",
                dpi=args.image_dpi,
                min_pixels=args.image_min_pixels,
            )
        else:
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
            th.manual_seed(args.seed + 2000)
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
        th.manual_seed(args.seed + 2000)
        diffusion_designs = th.randn((args.n_samples, 1, *design_shape), device=device)
        diffusion_conditions = conditions_tensor.unsqueeze(1)
        for timestep in reversed(range(diffusion_timesteps)):
            t = th.full((args.n_samples,), timestep, device=device, dtype=th.long)
            diffusion_designs = diffusion_sampler.sample_timestep(diffusion_model, diffusion_designs, t, diffusion_conditions)
        diffusion_designs = diffusion_designs.squeeze(1)
        if "design_min" in diffusion_ckpt and "design_max" in diffusion_ckpt:
            diffusion_designs = denormalize_designs_from_diffusion_range(
                diffusion_designs,
                diffusion_ckpt["design_min"].to(device),
                diffusion_ckpt["design_max"].to(device),
            )
        diffusion_np = diffusion_designs.detach().cpu().numpy().reshape(args.n_samples, *design_shape)
        if "design_min" in diffusion_ckpt and "design_max" in diffusion_ckpt:
            diffusion_np = np.clip(
                diffusion_np,
                diffusion_ckpt["design_min"].detach().cpu().numpy(),
                diffusion_ckpt["design_max"].detach().cpu().numpy(),
            )
        else:
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
            generator_output_activation=cgan_config.get("generator_output_activation", "tanh"),
        ).to(device)
        cgan_model.load_state_dict(cgan_ckpt["generator"])
        cgan_model.eval()
        cgan_conditions = conditions_tensor.unsqueeze(-1).unsqueeze(-1)
        th.manual_seed(args.seed + 2000)
        noise = th.randn((args.n_samples, int(cgan_config["latent_dim"]), 1, 1), device=device, dtype=th.float)
        cgan_designs = cgan_model(noise, cgan_conditions)
        cgan_np = cgan_designs.detach().cpu().numpy().reshape(args.n_samples, *design_shape)
        cgan_np = np.clip(cgan_np, args.clip_min, args.clip_max)

        reference = np.clip(sampled_designs_np[sample_idx], args.clip_min, args.clip_max)
        flow_image = flow_np[sample_idx]
        diffusion_image = diffusion_np[sample_idx]
        cgan_image = cgan_np[sample_idx]

        save_grayscale_image(
            reference,
            problem_output_dir / "reference.png",
            dpi=args.image_dpi,
            min_pixels=args.image_min_pixels,
        )
        save_design_raster(
            sampled_designs_np,
            problem_output_dir / "reference_raster.png",
            dpi=args.image_dpi,
            min_pixels=args.image_min_pixels,
        )
        save_grayscale_image(
            flow_image,
            problem_output_dir / "flow_matching_2d_cond.png",
            dpi=args.image_dpi,
            min_pixels=args.image_min_pixels,
        )
        save_design_raster(
            flow_np,
            problem_output_dir / "flow_matching_2d_cond_raster.png",
            dpi=args.image_dpi,
            min_pixels=args.image_min_pixels,
        )
        save_grayscale_image(
            diffusion_image,
            problem_output_dir / "diffusion_2d_cond.png",
            dpi=args.image_dpi,
            min_pixels=args.image_min_pixels,
        )
        save_design_raster(
            diffusion_np,
            problem_output_dir / "diffusion_2d_cond_raster.png",
            dpi=args.image_dpi,
            min_pixels=args.image_min_pixels,
        )
        save_grayscale_image(
            cgan_image,
            problem_output_dir / "cgan_cnn_2d.png",
            dpi=args.image_dpi,
            min_pixels=args.image_min_pixels,
        )
        save_design_raster(
            cgan_np,
            problem_output_dir / "cgan_cnn_2d_raster.png",
            dpi=args.image_dpi,
            min_pixels=args.image_min_pixels,
        )

        problem_metadata = metadata["problems"].setdefault(problem_id, {})
        problem_metadata.update(
            {
                "selected_dataset_index": int(sampled_indices[sample_idx]),
                "sample_index": int(sample_idx),
                "conditions": to_jsonable_dict(sampled_conditions[sample_idx]),
                "flow_integration_steps": flow_integration_steps,
                "flow_num_train_timesteps": flow_num_train_timesteps,
                "diffusion_timesteps": diffusion_timesteps,
                "diffusion_noise_schedule": diffusion_noise_schedule,
                "cgan_latent_dim": int(cgan_config["latent_dim"]),
            }
        )
        if flow_top_k_metadata:
            problem_metadata["flow_selection_mode"] = "top_k"
            problem_metadata["flow_selection_top_k"] = args.top_k
            problem_metadata["flow_selected_checkpoints"] = flow_top_k_metadata

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
            flow_steps = metadata.get("problems", {}).get(problem_id, {}).get("flow_integration_steps", 50)
            log_dict = {
                f"qualitative/{problem_id}/reference": wandb.Image(str(problem_dir / "reference.png")),
                f"qualitative/{problem_id}/reference_raster": wandb.Image(str(problem_dir / "reference_raster.png")),
                f"qualitative/{problem_id}/flow_matching_2d_cond": wandb.Image(
                    str(problem_dir / "flow_matching_2d_cond.png")
                ),
                f"qualitative/{problem_id}/flow_matching_2d_cond_raster_{args.flow_method}_steps{flow_steps}": wandb.Image(
                    str(problem_dir / "flow_matching_2d_cond_raster.png")
                ),
                f"qualitative/{problem_id}/diffusion_2d_cond": wandb.Image(str(problem_dir / "diffusion_2d_cond.png")),
                f"qualitative/{problem_id}/diffusion_2d_cond_raster": wandb.Image(
                    str(problem_dir / "diffusion_2d_cond_raster.png")
                ),
                f"qualitative/{problem_id}/cgan_cnn_2d": wandb.Image(str(problem_dir / "cgan_cnn_2d.png")),
                f"qualitative/{problem_id}/cgan_cnn_2d_raster": wandb.Image(
                    str(problem_dir / "cgan_cnn_2d_raster.png")
                ),
            }
            for rank in range(1, 6):
                rank_test_path = problem_dir / f"flow_matching_2d_cond_rank_{rank}.png"
                if rank_test_path.exists():
                    log_dict[f"qualitative/{problem_id}/flow_matching_2d_cond_rank_{rank}_{args.flow_method}_steps{flow_steps}"] = wandb.Image(str(rank_test_path))
            run.log(log_dict)
        artifact = wandb.Artifact(f"qualitative_bundle_seed{args.seed}", type="qualitative-bundle")
        artifact.add_dir(str(output_root))
        run.log_artifact(artifact)
        run.finish()

    print(f"Wrote qualitative bundle to {output_root} in {time.perf_counter() - start_time:.2f}s")
