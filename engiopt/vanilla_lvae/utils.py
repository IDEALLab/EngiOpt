"""Utilities for loading trained LVAE models from WandB.

Provides functions to load LVAE encoders and full models for evaluation
and visualization purposes.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import numpy.typing as npt
import torch as th
from torch import nn

from engiopt.vanilla_lvae.components import Encoder2D
from engiopt.vanilla_lvae.components import SNMLPPredictor
from engiopt.vanilla_lvae.components import TrueSNDecoder2D
import wandb


@dataclass
class LVAEConfig:
    """Configuration extracted from LVAE checkpoint."""

    latent_dim: int
    perf_dim: int
    resize_dimensions: tuple[int, int]
    design_shape: tuple[int, int]
    decoder_lipschitz_scale: float
    predictor_lipschitz_scale: float
    predictor_hidden_dims: tuple[int, ...]
    conditional_predictor: bool
    nmse_threshold_rec: float
    nmse_threshold_perf: float


class PrunedEncoder(nn.Module):
    """Wraps a raw encoder and applies the LVAE pruning mask.

    Pruned dimensions are clamped to their frozen values (the latent mean
    at the time of pruning), matching LeastVolumeAE_DynamicPruning.encode().
    """

    def __init__(self, encoder: nn.Module, pruning_mask: th.Tensor, frozen_z: th.Tensor) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("_p", pruning_mask)
        self.register_buffer("_z", frozen_z)

    def forward(self, x: th.Tensor) -> th.Tensor:
        z = self.encoder(x)
        z[:, self._p] = self._z[self._p].to(z.device)
        return z


def load_lvae_encoder(
    problem_id: str,
    seed: int,
    rec_threshold: float,
    perf_threshold: float,
    wandb_project: str = "lv_mmd",
    wandb_entity: str | None = None,
    device: th.device | str = "cpu",
) -> tuple[nn.Module, LVAEConfig]:
    """Load a trained LVAE encoder from WandB.

    Args:
        problem_id: Problem identifier (e.g., "beams2d").
        seed: Random seed used during training.
        rec_threshold: Reconstruction NMSE threshold used in training.
        perf_threshold: Performance NMSE threshold used in training.
        wandb_project: WandB project name.
        wandb_entity: WandB entity name (None for default).
        device: Device to load model onto.

    Returns:
        Tuple of (encoder module, LVAEConfig dataclass).

    Raises:
        ValueError: If artifact or run config cannot be retrieved.
    """
    from engiopt.vanilla_lvae.components import Encoder2D

    # Build artifact path
    artifact_name = f"{problem_id}_constrained_vanilla_plvae_2d"
    alias = f"seed_{seed}_rec{rec_threshold}_perf{perf_threshold}"

    if wandb_entity is not None:
        artifact_path = f"{wandb_entity}/{wandb_project}/{artifact_name}:{alias}"
    else:
        artifact_path = f"{wandb_project}/{artifact_name}:{alias}"

    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")

    run = artifact.logged_by()
    if run is None or not hasattr(run, "config"):
        raise ValueError(f"Failed to retrieve run config for artifact: {artifact_path}")

    config = run.config

    # Extract config into dataclass
    perf_dim_raw = config.get("perf_dim", config["latent_dim"])
    perf_dim = config["latent_dim"] if perf_dim_raw == -1 else perf_dim_raw

    lvae_config = LVAEConfig(
        latent_dim=config["latent_dim"],
        perf_dim=perf_dim,
        resize_dimensions=tuple(config["resize_dimensions"]),
        design_shape=tuple(config.get("design_shape", (100, 100))),
        decoder_lipschitz_scale=config.get("decoder_lipschitz_scale", 1.0),
        predictor_lipschitz_scale=config.get("predictor_lipschitz_scale", 1.0),
        predictor_hidden_dims=tuple(config.get("predictor_hidden_dims", (256, 128))),
        conditional_predictor=config.get("conditional_predictor", False),
        nmse_threshold_rec=config["nmse_threshold_rec"],
        nmse_threshold_perf=config["nmse_threshold_perf"],
    )

    # Download and load checkpoint
    artifact_dir = artifact.download()
    ckpt_path = os.path.join(artifact_dir, "constrained_vanilla_plvae.pth")
    ckpt = th.load(ckpt_path, map_location=device, weights_only=False)

    # Reconstruct encoder
    raw_encoder = Encoder2D(
        latent_dim=lvae_config.latent_dim,
        design_shape=lvae_config.design_shape,
        resize_dimensions=lvae_config.resize_dimensions,
    )
    raw_encoder.load_state_dict(ckpt["encoder"])

    # Wrap with pruning mask if available in checkpoint
    if "pruning_mask" in ckpt and "pruning_frozen_z" in ckpt:
        encoder = PrunedEncoder(raw_encoder, ckpt["pruning_mask"], ckpt["pruning_frozen_z"])
    else:
        encoder = raw_encoder

    encoder.eval()
    encoder.to(device)

    return encoder, lvae_config


def load_full_lvae(
    problem_id: str,
    seed: int,
    rec_threshold: float,
    perf_threshold: float,
    wandb_project: str = "engiopt",
    wandb_entity: str | None = None,
    device: th.device | str = "cpu",
) -> tuple[nn.Module, nn.Module, nn.Module, LVAEConfig]:
    """Load full LVAE (encoder, decoder, predictor) from WandB.

    Args:
        problem_id: Problem identifier (e.g., "beams2d").
        seed: Random seed used during training.
        rec_threshold: Reconstruction NMSE threshold used in training.
        perf_threshold: Performance NMSE threshold used in training.
        wandb_project: WandB project name.
        wandb_entity: WandB entity name (None for default).
        device: Device to load model onto.

    Returns:
        Tuple of (encoder, decoder, predictor, LVAEConfig).
    """
    # Build artifact path
    artifact_name = f"{problem_id}_constrained_vanilla_plvae_2d"
    alias = f"seed_{seed}_rec{rec_threshold}_perf{perf_threshold}"

    if wandb_entity is not None:
        artifact_path = f"{wandb_entity}/{wandb_project}/{artifact_name}:{alias}"
    else:
        artifact_path = f"{wandb_project}/{artifact_name}:{alias}"

    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")

    run = artifact.logged_by()
    if run is None:
        raise ValueError(f"Cannot retrieve run for {artifact_path}")

    config = run.config
    artifact_dir = artifact.download()
    ckpt = th.load(os.path.join(artifact_dir, "constrained_vanilla_plvae.pth"), map_location=device, weights_only=False)

    # Extract config
    design_shape = tuple(config.get("design_shape", (100, 100)))
    resize_dimensions = tuple(config["resize_dimensions"])
    latent_dim = config["latent_dim"]
    perf_dim_raw = config.get("perf_dim", latent_dim)
    perf_dim = latent_dim if perf_dim_raw == -1 else perf_dim_raw
    n_conds = len(BUILTIN_PROBLEMS[problem_id]().conditions_keys)

    lvae_config = LVAEConfig(
        latent_dim=latent_dim,
        perf_dim=perf_dim,
        resize_dimensions=resize_dimensions,
        design_shape=design_shape,
        decoder_lipschitz_scale=config.get("decoder_lipschitz_scale", 1.0),
        predictor_lipschitz_scale=config.get("predictor_lipschitz_scale", 1.0),
        predictor_hidden_dims=tuple(config.get("predictor_hidden_dims", (256, 128))),
        conditional_predictor=config.get("conditional_predictor", False),
        nmse_threshold_rec=config["nmse_threshold_rec"],
        nmse_threshold_perf=config["nmse_threshold_perf"],
    )

    # Reconstruct models
    raw_encoder = Encoder2D(latent_dim, design_shape, resize_dimensions)
    raw_encoder.load_state_dict(ckpt["encoder"])

    # Wrap with pruning mask if available in checkpoint
    if "pruning_mask" in ckpt and "pruning_frozen_z" in ckpt:
        encoder = PrunedEncoder(raw_encoder, ckpt["pruning_mask"], ckpt["pruning_frozen_z"])
    else:
        encoder = raw_encoder

    encoder.eval().to(device)

    decoder = TrueSNDecoder2D(
        latent_dim,
        design_shape,
        lipschitz_scale=config.get("decoder_lipschitz_scale", 1.0),
    )
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval().to(device)

    predictor_input_dim = perf_dim + (n_conds if config.get("conditional_predictor", False) else 0)
    predictor = SNMLPPredictor(
        input_dim=predictor_input_dim,
        output_dim=1,
        hidden_dims=tuple(config.get("predictor_hidden_dims", (256, 128))),
        lipschitz_scale=config.get("predictor_lipschitz_scale", 1.0),
    )
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval().to(device)

    return encoder, decoder, predictor, lvae_config


def encode_designs(
    encoder: nn.Module,
    designs: npt.NDArray,
    device: th.device | str,
    batch_size: int = 256,
) -> npt.NDArray:
    """Encode designs to latent codes using an LVAE encoder.

    Args:
        encoder: Trained LVAE encoder.
        designs: Designs of shape (N, H, W) or (N, 1, H, W).
        device: Device for encoding.
        batch_size: Batch size for encoding.

    Returns:
        Latent codes of shape (N, latent_dim).
    """
    encoder.eval()
    device = th.device(device) if isinstance(device, str) else device

    designs_t = th.from_numpy(designs).float()

    if designs_t.ndim == 3:
        designs_t = designs_t.unsqueeze(1)

    codes = []
    with th.no_grad():
        for i in range(0, len(designs_t), batch_size):
            batch = designs_t[i : i + batch_size].to(device)
            z = encoder(batch)
            codes.append(z.cpu().numpy())

    return np.concatenate(codes, axis=0)


__all__ = [
    "LVAEConfig",
    "PrunedEncoder",
    "encode_designs",
    "load_full_lvae",
    "load_lvae_encoder",
]
