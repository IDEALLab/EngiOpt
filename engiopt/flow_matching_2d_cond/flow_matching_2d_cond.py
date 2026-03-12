"""Conditional 2D flow matching baseline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
import time
from typing import Literal

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import tqdm
import tyro

from engiopt.flow_matching_2d_cond.core import args_to_dict
from engiopt.flow_matching_2d_cond.core import build_model
from engiopt.flow_matching_2d_cond.core import compute_flow_matching_loss
from engiopt.flow_matching_2d_cond.core import generate_samples
import wandb


@dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "beams2d"
    """Problem identifier."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this algorithm."""

    # Tracking
    track: bool = True
    """Track the experiment with wandb."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name."""
    seed: int = 1
    """Random seed."""
    save_model: bool = False
    """Saves the model to disk."""
    checkpoint_path: str = "model.pth"
    """Local checkpoint path used when save_model is enabled."""
    checkpoint_interval_epochs: int = 0
    """Save a local checkpoint every N epochs. Disabled when set to 0."""
    checkpoint_dir: str = "checkpoints"
    """Directory for periodic local checkpoints."""
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    """Device selection for local smoke runs and training."""

    # Algorithm specific
    n_epochs: int = 200
    """Number of epochs of training."""
    batch_size: int = 32
    """Size of the batches."""
    lr: float = 4e-4
    """Learning rate."""
    b1: float = 0.9
    """Decay of first order momentum of gradient."""
    b2: float = 0.999
    """Decay of second order momentum of gradient."""
    n_cpu: int = 8
    """Number of CPU threads to use."""
    sample_interval: int = 400
    """Interval between image samples."""
    layers_per_block: int = 2
    """Layers per UNet block."""
    num_train_timesteps: int = 1000
    """Embedding range used for continuous flow-matching time values."""
    integration_steps: int = 50
    """Number of Euler steps used for sampling."""
    time_sampling: Literal["uniform"] = "uniform"
    """Continuous time sampling strategy."""
    max_train_batches: int | None = None
    """Optional cap on batches processed across the full run for smoke tests."""
    clip_min: float = 1e-3
    """Minimum value used when clipping generated designs for previews."""
    clip_max: float = 1.0
    """Maximum value used when clipping generated designs for previews."""


def select_device(device_arg: Literal["auto", "cpu", "mps", "cuda"]) -> th.device:
    """Return the best available torch device."""
    if device_arg != "auto":
        if device_arg == "mps" and not th.backends.mps.is_available():
            raise ValueError("MPS device requested but not available")
        if device_arg == "cuda" and not th.cuda.is_available():
            raise ValueError("CUDA device requested but not available")
        return th.device(device_arg)
    if th.backends.mps.is_available():
        return th.device("mps")
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def normalize_designs(designs: th.Tensor) -> tuple[th.Tensor, float, float]:
    """Normalize designs into [0, 1] for stable flow training."""
    design_min = float(designs.min().item())
    design_max = float(designs.max().item())
    scale = max(design_max - design_min, 1e-8)
    normalized = (designs - design_min) / scale
    return normalized, design_min, design_max


def sample_preview_conditions(conds_min: th.Tensor, conds_max: th.Tensor, n_designs: int, device: th.device) -> th.Tensor:
    """Create a simple sweep across the condition range for preview logging."""
    steps = th.linspace(0, 1, n_designs, device=device).view(n_designs, 1, 1)
    return conds_min + steps * (conds_max - conds_min)


def save_design_grid(
    designs: th.Tensor,
    hidden_states: th.Tensor,
    problem,
    img_fname: str,
    clip_min: float,
    clip_max: float,
):
    """Save a grid of sampled designs with condition labels."""
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    axes = axes.flatten()

    clipped_designs = designs.detach().cpu().numpy().clip(clip_min, clip_max)
    hidden_states_cpu = hidden_states.detach().cpu()

    for idx, design in enumerate(clipped_designs):
        axes[idx].imshow(design[0])
        condition_text = "\n ".join(
            f"{condition}: {hidden_states_cpu[idx, 0, cond_idx]:.2f}"
            for cond_idx, condition in enumerate(problem.conditions_keys)
        )
        axes[idx].title.set_text(condition_text)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

    plt.tight_layout()
    plt.savefig(img_fname)
    plt.close(fig)


if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), save_code=True, name=run_name)

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    th.set_num_threads(args.n_cpu)

    device = select_device(args.device)
    design_shape = problem.design_space.shape
    encoder_hid_dim = len(problem.conditions_keys)

    os.makedirs("images", exist_ok=True)
    if args.checkpoint_interval_epochs > 0:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    model = build_model(
        design_shape=design_shape,
        encoder_hid_dim=encoder_hid_dim,
        layers_per_block=args.layers_per_block,
    )
    model.to(device)

    training_split = problem.dataset.with_format("torch", device=device)["train"]
    clean_designs = th.zeros(len(training_split), design_shape[0], design_shape[1], device=device)
    for idx in range(len(training_split)):
        clean_designs[idx] = training_split[idx]["optimal_design"][:].reshape(design_shape[0], design_shape[1])

    clean_designs, design_min, design_max = normalize_designs(clean_designs)
    training_ds = th.utils.data.TensorDataset(
        clean_designs.flatten(1),
        *[training_split[key][:] for key in problem.conditions_keys],
    )
    cond_tensors = th.stack(training_ds.tensors[1 : len(problem.conditions_keys) + 1])
    conds_min = cond_tensors.amin(dim=tuple(range(1, cond_tensors.ndim))).view(1, 1, -1)
    conds_max = cond_tensors.amax(dim=tuple(range(1, cond_tensors.ndim))).view(1, 1, -1)

    dataloader = th.utils.data.DataLoader(training_ds, batch_size=args.batch_size, shuffle=True)
    optimizer = th.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    last_loss: float | None = None
    last_epoch = 0
    last_batch = 0
    batches_processed = 0
    stop_training = False

    run_start_time = time.time()
    for epoch in tqdm.trange(args.n_epochs):
        epoch_start_time = time.time()
        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()
            optimizer.zero_grad()

            designs = batch[0].reshape(-1, 1, design_shape[0], design_shape[1]).to(device)
            conditions = th.stack(batch[1:], dim=1).reshape(-1, 1, encoder_hid_dim).to(device)

            if args.time_sampling != "uniform":
                raise ValueError(f"Unsupported time sampling strategy: {args.time_sampling}")

            loss, sampled_time = compute_flow_matching_loss(
                model=model,
                clean_designs=designs,
                encoder_hidden_states=conditions,
                num_train_timesteps=args.num_train_timesteps,
            )
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
            last_epoch = epoch
            last_batch = epoch * len(dataloader) + batch_idx
            batches_processed += 1

            if args.track:
                batches_done = epoch * len(dataloader) + batch_idx
                wandb.log(
                    {
                        "loss": loss.item(),
                        "epoch": epoch,
                        "batch": batches_done,
                        "sampled_time_mean": float(sampled_time.mean().item()),
                    }
                )

                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {batch_idx}/{len(dataloader)}] "
                    f"[loss: {loss.item():.6f}] [{time.time() - batch_start_time:.2f} sec]"
                )

                should_preview = args.sample_interval > 0 and batches_done > 0 and batches_done % args.sample_interval == 0
                if should_preview:
                    preview_conditions = sample_preview_conditions(conds_min, conds_max, 25, device)
                    was_training = model.training
                    model.eval()
                    with th.no_grad():
                        preview_designs = generate_samples(
                            model=model,
                            design_shape=design_shape,
                            encoder_hidden_states=preview_conditions,
                            integration_steps=args.integration_steps,
                            num_train_timesteps=args.num_train_timesteps,
                            device=device,
                        )
                    if was_training:
                        model.train()
                    img_fname = f"images/{batches_done}.png"
                    save_design_grid(
                        designs=preview_designs,
                        hidden_states=preview_conditions,
                        problem=problem,
                        img_fname=img_fname,
                        clip_min=args.clip_min,
                        clip_max=args.clip_max,
                    )
                    wandb.log({"designs": wandb.Image(img_fname)})

            if args.max_train_batches is not None and batches_processed >= args.max_train_batches:
                stop_training = True
                break

        if stop_training:
            break

        if args.track:
            wandb.log(
                {
                    "epoch_runtime_sec": time.time() - epoch_start_time,
                    "cumulative_runtime_sec": time.time() - run_start_time,
                    "epoch_completed": epoch,
                }
            )

        should_save_periodic = (
            args.checkpoint_interval_epochs > 0
            and (epoch + 1) % args.checkpoint_interval_epochs == 0
            and last_loss is not None
        )
        if should_save_periodic:
            periodic_path = Path(args.checkpoint_dir) / f"epoch_{epoch + 1:04d}.pth"
            th.save(
                {
                    "epoch": last_epoch,
                    "batch": last_batch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": last_loss,
                    "args": args_to_dict(args),
                    "model_config": {
                        "layers_per_block": args.layers_per_block,
                        "num_train_timesteps": args.num_train_timesteps,
                        "integration_steps": args.integration_steps,
                    },
                    "design_shape": design_shape,
                    "encoder_hid_dim": encoder_hid_dim,
                    "design_min": design_min,
                    "design_max": design_max,
                },
                periodic_path,
            )

    if args.save_model and last_loss is not None:
        checkpoint = {
            "epoch": last_epoch,
            "batch": last_batch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": last_loss,
            "args": args_to_dict(args),
            "model_config": {
                "layers_per_block": args.layers_per_block,
                "num_train_timesteps": args.num_train_timesteps,
                "integration_steps": args.integration_steps,
            },
            "design_shape": design_shape,
            "encoder_hid_dim": encoder_hid_dim,
            "design_min": design_min,
            "design_max": design_max,
        }
        th.save(checkpoint, args.checkpoint_path)
        if args.track:
            artifact_model = wandb.Artifact(f"{args.problem_id}_{args.algo}_model", type="model")
            artifact_model.add_file(args.checkpoint_path, name="model.pth")
            wandb.log_artifact(artifact_model, aliases=[f"seed_{args.seed}"])

    if args.track:
        wandb.finish()
