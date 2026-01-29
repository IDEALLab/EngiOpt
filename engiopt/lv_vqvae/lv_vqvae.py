"""Least-Volume Vector Quantized Variational Autoencoder (LV-VQVAE).

Based on https://github.com/dome272/VQGAN-pytorch with an "Online Clustered Codebook" for better codebook usage from https://github.com/lyndonzheng/CVQ-VAE/blob/main/quantise.py

This implementation is composed of two primary Stages:
    - Stage 1 is a VQVAE: an autoencoder (AE) with a discrete latent space represented by a codebook.
      We additionally apply a least-volume (LV) objective with dynamic pruning to reduce the effective latent
      dimensionality n_z over the course of training.
    - Stage 2 is a generative model (a transformer in this case) trained on the discrete latent tokens from Stage 1.

The transformer uses nanoGPT (https://github.com/karpathy/nanoGPT) instead of minGPT (https://github.com/karpathy/minGPT) as in the original implementation.

For Stage 2, we take the indices of the codebook vectors and flatten them into a 1D sequence, treating them as training tokens.
The transformer is then trained to autoregressively predict each token in the sequence, after which it is reshaped back to the original 2D latent space and passed through the Stage 1 decoder to generate an image.
To make Stage 2 conditional, we train a separate VQVAE on the conditions only (CVQVAE) and replace the start-of-sequence tokens of the transformer with the CVQVAE latent tokens.

Notes for this version:
    - Check noqa's to remove later and implement the needed features
    - Need to reimplement features such as pruning and masking, including masking in the codebook and transformer stage, as well as extensive wandb metrics to validate
    - Implemented as of 20 January: spherical norm for codebook, commitment loss term beta 0.25 --> 0.01 for codebook, 1-Lipschitz decoder via new blocks and activation function (scaled tanh) + removal of non-Lipschitz blocks (i.e., self-attention)
    - Add back in ASAP: codebook perplexity, usage fraction/amount for codes and dims, number of active (non-pruned) codes and dims, etc.
"""


from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import os
import random
import time
from typing import Any
import warnings

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as f
from torch.nn.utils.parametrizations import spectral_norm
import tqdm
import tyro
import wandb

from engiopt.lv_vqvae.utils import Codebook
from engiopt.lv_vqvae.utils import DownSampleBlock
from engiopt.lv_vqvae.utils import GPT
from engiopt.lv_vqvae.utils import GPTConfig
from engiopt.lv_vqvae.utils import GroupNorm
from engiopt.lv_vqvae.utils import GroupSort
from engiopt.lv_vqvae.utils import LinearCombo
from engiopt.lv_vqvae.utils import loss_vol
from engiopt.lv_vqvae.utils import NonLocalBlock
from engiopt.lv_vqvae.utils import ResidualBlock
from engiopt.lv_vqvae.utils import ScaledTanh
from engiopt.lv_vqvae.utils import Swish
from engiopt.lv_vqvae.utils import token_stats_from_indices
from engiopt.lv_vqvae.utils import TrueSNResidualBlock
from engiopt.lv_vqvae.utils import TrueSNUpsample
from engiopt.lvae_core import polynomial_schedule
from engiopt.lvae_core import PruningPolicy
from engiopt.transforms import drop_constant
from engiopt.transforms import normalize
from engiopt.transforms import resize_to


@dataclass
class Args:
    """Command-line arguments for LV-VQVAE."""

    problem_id: str = "beams2d"
    """Problem identifier for 2D engineering design."""
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
    save_model: bool = True
    """Saves the model to disk."""

    # Algorithm-specific: General
    conditional: bool = True
    """whether the model is conditional or not"""
    normalize_conditions: bool = True
    """whether to normalize the condition columns to zero mean and unit std"""
    drop_constant_conditions: bool = True
    """whether to drop constant condition columns (i.e., overhang_constraint in beams2d)"""
    image_size: int = 128
    """desired size of the square image input to the model"""

    # Algorithm-specific: Stage 1 Conditional AE or "CVQVAE" if the model is specified as conditional
    cond_dim: int = 3
    """dimensionality of the condition space"""
    cond_hidden_dim: int = 256
    """hidden dimension of the CVQVAE MLP"""
    cond_latent_dim: int = 4
    """individual code dimension for CVQVAE"""
    cond_codebook_vectors: int = 64
    """number of vectors in the CVQVAE codebook"""
    cond_feature_map_dim: int = 4
    """feature map dimension for the CVQVAE encoder output"""
    batch_size_cvqvae: int = 16
    """size of the batches for CVQVAE"""
    n_epochs_cvqvae: int = 1000
    """number of epochs of CVQVAE training"""
    cond_lr: float = 2e-4
    """learning rate for CVQVAE"""

    # Algorithm-specific: Stage 1 (VQVAE)
    n_epochs_vqvae: int = 100
    """number of epochs of training"""
    batch_size_vqvae: int = 16
    """size of the batches for Stage 1"""
    lr_vqvae: float = 4e-4
    """learning rate for Stage 1"""
    beta: float = 0.25
    """beta hyperparameter for the codebook commitment loss"""
    b1: float = 0.5
    """decay of first order momentum of gradient"""
    b2: float = 0.9
    """decay of first order momentum of gradient"""
    n_cpu: int = 8
    """number of cpu threads to use during batch generation"""
    latent_dim: int = 256
    """dimensionality of the latent space"""
    num_codebook_vectors: int = 1024
    """number of vectors in the codebook"""
    rec_loss_factor: float = 1.0
    """weighting factor for the reconstruction loss"""
    encoder_channels: tuple[int, ...] = (64, 64, 128, 128, 256)
    """tuple of channel sizes for each encoder layer"""
    encoder_attn_resolutions: tuple[int, ...] = (16,)
    """tuple of resolutions at which to apply attention in the encoder"""
    encoder_num_res_blocks: int = 2
    """number of residual blocks per encoder layer"""
    decoder_channels: tuple[int, ...] = (256, 128, 128, 64)
    """tuple of channel sizes for each decoder layer"""
    decoder_attn_resolutions: tuple[int, ...] = (16,)
    """tuple of resolutions at which to apply attention in the decoder"""
    decoder_num_res_blocks: int = 3
    """number of residual blocks per decoder layer"""
    sample_interval_vqvae: int = 100
    """interval between Stage 1 image samples"""

    # LV + dynamic pruning (n_z). NOTE: All LV params share the same names with the LVAE code, except the prefix "lv_" has been added to them.
    lv_start_epoch: int = 10
    """epoch to start applying LV loss (keep LV loss weight = 0 before this)"""
    lv_pruning_epoch: int = 20
    """epoch to start pruning latent dimensions (after LV has had time to shape the space)"""
    lv_w_max: float = 0.01
    """maximum weight for the LV loss after ramp-up"""
    lv_ramp_epochs: int = 10
    """number of epochs to linearly ramp LV loss weight from 0 to lv_w_max"""
    lv_min_active_dims: int = 16
    """minimum number of latent dimensions allowed to remain active (pruning will not go below this)"""
    lv_max_prune_per_epoch: int | None = None
    """maximum number of latent dimensions to prune per epoch; None for no limit"""
    lv_val_mae_target: float = 0.03
    """validation MAE reconstruction target used as the pruning guardrail"""
    lv_pruning_strategy: str = "plummet"
    """strategy name (plummet, pca_cdf, lognorm, probabilistic)"""
    lv_pruning_params: dict[str, Any] | None = field(default_factory=lambda: {"threshold": 0.02, "beta": 0.9})
    """least volume pruning parameters, default for plummet strategy with threshold 0.02 and beta 0.9"""
    lv_eta: float = 1e-4
    """smoothing parameter for volume loss"""
    lv_cooldown_epochs: int = 0
    """epochs to wait between pruning events (default: 0 = no cooldown)"""
    lv_k_consecutive: int = 1
    """consecutive epochs below threshold required (default: 1 = immediate)"""
    lv_recon_tol: float = float("inf")
    """relative tolerance to best validation recon (default: inf = no constraint)"""

    # Codebook pruning (|Z|)
    # cb_prune_start_epoch: int = 20  # noqa: ERA001
    # """epoch to start pruning codebook entries (tokens)"""  # noqa: ERA001
    # cb_min_active_codes: int = 32  # noqa: ERA001
    # """minimum number of codebook entries allowed to remain active"""
    # cb_cooldown_epochs: int = 1  # noqa: ERA001
    # """number of epochs to wait after a codebook prune event before pruning again"""
    # cb_val_mae_target: float = 0.03  # noqa: ERA001
    # """validation MAE reconstruction target used as the codebook pruning guardrail"""
    #  lv_codebook_freeze_epochs: int = 2  # noqa: ERA001
    #  """freeze online codebook reinitialization for this many epochs after a prune event"""

    # Algorithm-specific: Stage 2 (Transformer)
    n_epochs_transformer: int = 100
    """number of epochs of training"""
    early_stopping: bool = True
    """whether to use early stopping for the transformer; if True requires args.track to be True"""
    early_stopping_patience: int = 3
    """number of epochs with no improvement after which training will be stopped"""
    early_stopping_delta: float = 1e-3
    """minimum change in the monitored quantity to qualify as an improvement"""
    batch_size_transformer: int = 16
    """size of the batches for Stage 2"""
    lr_transformer: float = 6e-4
    """learning rate for Stage 2"""
    n_layer: int = 12
    """number of layers in the transformer"""
    n_head: int = 12
    """number of attention heads"""
    n_embd: int = 768
    """transformer embedding dimension"""
    dropout: float = 0.3
    """dropout rate in the transformer"""
    sample_interval_transformer: int = 100
    """interval between Stage 2 image samples"""


class Encoder(nn.Module):
    """Encoder module for Stage 1 (VQVAE).

    # Simplified architecture: image -> conv -> [resblock -> attn? -> downsample]* -> norm -> swish -> final conv -> latent image
    Where `?` indicates a block that is only included at certain resolutions and `*` indicates a block that is repeated.
    """

    def __init__(  # noqa: PLR0913
        self,
        encoder_channels: tuple[int, ...],
        encoder_start_resolution: int,
        encoder_attn_resolutions: tuple[int, ...],
        encoder_num_res_blocks: int,
        image_channels: int,
        latent_dim: int,
    ):
        super().__init__()
        channels = encoder_channels
        resolution = encoder_start_resolution
        layers = [nn.Conv2d(image_channels, channels[0], kernel_size=3, stride=1, padding=1)]
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for _ in range(encoder_num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in encoder_attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != len(channels) - 2:
                layers.append(DownSampleBlock(channels[i + 1]))
                resolution //= 2
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], latent_dim, kernel_size=3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)


class CondEncoder(nn.Module):
    """Simpler MLP-based encoder for the CVQVAE if enabled."""

    def __init__(self, cond_feature_map_dim: int, cond_dim: int, cond_hidden_dim: int, cond_latent_dim: int):
        super().__init__()
        self.c_feature_map_dim = cond_feature_map_dim
        self.model = nn.Sequential(
            LinearCombo(cond_dim, cond_hidden_dim),
            LinearCombo(cond_hidden_dim, cond_hidden_dim),
            nn.Linear(cond_hidden_dim, cond_latent_dim * cond_feature_map_dim**2),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        encoded = self.model(x)
        s = encoded.shape
        return encoded.view(s[0], s[1] // self.c_feature_map_dim**2, self.c_feature_map_dim, self.c_feature_map_dim)


class TrueSNDecoder(nn.Module):
    """A VQVAE decoder that is provably 1-Lipschitz. Mirrors the architecture of the encoder but without attention layers.

    Lipschitz Guarantees:
    - All convolutions use spectral_norm (sigma_max <= 1)
    - GroupSort activation is exactly 1-Lipschitz
    - Residual connections scaled by 1/sqrt(2)
    - ConvTranspose upsampling is spectral-normalized
    - Attention removed (as it is generally not 1-Lipschitz)

    Args:
        decoder_channels: Channel progression (e.g., [512, 256, 128])
        decoder_start_resolution: Resolution of the latent code
        decoder_num_res_blocks: ResBlocks per resolution level
        image_channels: Output channels (usually 3)
        latent_dim: Input latent dimension
        sn_n_power_iterations: Precision of spectral norm (default 1). Increase for tighter bound.
    """
    def __init__(  # noqa: PLR0913
        self,
        decoder_channels: tuple[int, ...],
        decoder_start_resolution: int,
        decoder_num_res_blocks: int,
        image_channels: int,
        latent_dim: int,
        sn_n_power_iterations: int = 1
    ):
        super().__init__()
        self.sn_iters = sn_n_power_iterations

        in_channels = decoder_channels[0]
        resolution = decoder_start_resolution
        layers = []

        # 1. Initial Projection
        layers.append(
            spectral_norm(
                nn.Conv2d(latent_dim, in_channels, kernel_size=3, padding=1),
                n_power_iterations=self.sn_iters
            )
        )

        # 2. Main Decoder Loop
        for i in range(len(decoder_channels)):
            out_channels = decoder_channels[i]

            # Upsample (skip first block to match VQVAE logic)
            if i != 0:
                layers.append(TrueSNUpsample(in_channels, n_power_iterations=self.sn_iters))
                resolution *= 2

            # Residual Blocks
            for _ in range(decoder_num_res_blocks):
                layers.append(
                    TrueSNResidualBlock(
                        in_channels,
                        out_channels,
                        n_power_iterations=self.sn_iters
                    )
                )
                in_channels = out_channels

            # Attention blocks removed

        # 3. Output Projection
        layers.append(GroupSort(group_size=2))
        layers.append(
            spectral_norm(
                nn.Conv2d(in_channels, image_channels, kernel_size=3, padding=1),
                n_power_iterations=self.sn_iters
            )
        )

        # 4. Final Activation (Scaled Tanh)
        layers.append(ScaledTanh(scale=0.99))

        self.model = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)

    def get_spectral_norms(self):
        """Debugging tool: Returns the actual spectral sigma of all SN layers.

        Use this to verify that no layer exceeds 1.0 during training.
        """
        norms = {}
        for name, module in self.named_modules():
            # Check for Conv2d or ConvTranspose2d with spectral norm attributes
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):  # noqa: SIM102
                if hasattr(module, "weight_u"):
                    u = module.weight_u
                    v = module.weight_v
                    w = module.weight_orig
                    # Estimate sigma = u^T * W * v
                    # (Note: PyTorch stores flattened versions in u/v)
                    w_mat = w.view(w.size(0), -1)
                    sigma = th.dot(u, th.mv(w_mat, v))
                    norms[name] = sigma.item()
        return norms


class TrueSNCondDecoder(nn.Module):
    """1-Lipschitz Decoder for N(0,1) conditional data."""

    def __init__(
        self,
        cond_latent_dim: int,
        cond_dim: int,
        cond_hidden_dim: int,
        cond_feature_map_dim: int,
        sn_n_power_iterations: int = 1
    ):
        super().__init__()

        input_dim = cond_latent_dim * (cond_feature_map_dim**2)

        self.model = nn.Sequential(
            # Layer 1: Linear -> GroupSort
            spectral_norm(
                nn.Linear(input_dim, cond_hidden_dim),
                n_power_iterations=sn_n_power_iterations
            ),
            GroupSort(group_size=2),

            # Layer 2: Linear -> GroupSort
            spectral_norm(
                nn.Linear(cond_hidden_dim, cond_hidden_dim),
                n_power_iterations=sn_n_power_iterations
            ),
            GroupSort(group_size=2),

            # Layer 3: Output Projection -> Identity
            # SNLinear is already 1-Lipschitz.
            # No activation needed for N(0,1) targets.
            spectral_norm(
                nn.Linear(cond_hidden_dim, cond_dim),
                n_power_iterations=sn_n_power_iterations
            ),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        # Flatten input: (B, C, H, W) -> (B, C*H*W)
        return self.model(x.contiguous().view(len(x), -1))


class VQVAE(nn.Module):
    """VQVAE model for Stage 1.

    Can be configured as a CVQVAE if desired.

    Parameters:
        device (th.device): torch device to use

        **CVQVAE params**
        is_c (bool): If True, use CVQVAE architecture (MLP-based encoder/decoder).
        cond_feature_map_dim (int): Feature map dimension for the CVQVAE encoder output.
        cond_dim (int): Number of input features for the CVQVAE encoder.
        cond_hidden_dim (int): Hidden dimension of the CVQVAE MLP.
        cond_latent_dim (int): Individual code dimension for CVQVAE.
        cond_codebook_vectors (int): Number of codebook vectors for CVQVAE.

        **VQVAE params**
        encoder_channels (tuple[int, ...]): Tuple of channel sizes for each encoder layer.
        encoder_start_resolution (int): Starting resolution for the encoder.
        encoder_attn_resolutions (tuple[int, ...]): Tuple of resolutions at which to apply attention in the encoder.
        encoder_num_res_blocks (int): Number of residual blocks per encoder layer.
        decoder_channels (tuple[int, ...]): Tuple of channel sizes for each decoder layer.
        decoder_start_resolution (int): Starting resolution for the decoder.
        decoder_num_res_blocks (int): Number of residual blocks per decoder layer.
        image_channels (int): Number of channels in the input/output image.
        latent_dim (int): Dimensionality of the latent space.
        num_codebook_vectors (int): Number of codebook vectors.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        device: th.device,
        # CVQVAE parameters
        is_c: bool = False,
        cond_feature_map_dim: int = 4,
        cond_dim: int = 3,
        cond_hidden_dim: int = 256,
        cond_latent_dim: int = 4,
        cond_codebook_vectors: int = 64,
        # VQVAE + Codebook parameters
        encoder_channels: tuple[int, ...] = (64, 64, 128, 128, 256),
        encoder_start_resolution: int = 128,
        encoder_attn_resolutions: tuple[int, ...] = (16,),
        encoder_num_res_blocks: int = 2,
        decoder_channels: tuple[int, ...] = (256, 128, 128, 64),
        decoder_start_resolution: int = 16,
        decoder_num_res_blocks: int = 3,
        image_channels: int = 1,
        latent_dim: int = 16,
        num_codebook_vectors: int = 256,
    ):
        super().__init__()
        if is_c:
            self.encoder = CondEncoder(cond_feature_map_dim, cond_dim, cond_hidden_dim, cond_latent_dim).to(device=device)

            self.decoder = TrueSNCondDecoder(cond_latent_dim, cond_dim, cond_hidden_dim, cond_feature_map_dim).to(device=device)

            self.quant_conv = nn.Conv2d(cond_latent_dim, cond_latent_dim, kernel_size=1).to(device=device)
            self.post_quant_conv = nn.Conv2d(cond_latent_dim, cond_latent_dim, kernel_size=1).to(device=device)
        else:
            self.encoder = Encoder(
                encoder_channels,
                encoder_start_resolution,
                encoder_attn_resolutions,
                encoder_num_res_blocks,
                image_channels,
                latent_dim,
            ).to(device=device)

            self.decoder = TrueSNDecoder(
                decoder_channels,
                decoder_start_resolution,
                # decoder_attn_resolutions,
                decoder_num_res_blocks,
                image_channels,
                latent_dim,
            ).to(device=device)

            self.quant_conv = nn.Conv2d(latent_dim, latent_dim, kernel_size=1).to(device=device)
            self.post_quant_conv = nn.Conv2d(latent_dim, latent_dim, kernel_size=1).to(device=device)

        self.codebook = Codebook(
            num_codebook_vectors=cond_codebook_vectors if is_c else num_codebook_vectors,
            latent_dim=cond_latent_dim if is_c else latent_dim,
        ).to(device=device)

    def forward(
        self,
        designs: th.Tensor,
        *,
        return_latents: bool = False,
        active_mask: th.Tensor | None = None,
        frozen_mean: th.Tensor | None = None
    ):
        """Full VQVAE forward pass."""
        encoded = self.encoder(designs)
        quant_encoded = self.quant_conv(encoded)
        quant_encoded = self.apply_pruning(quant_encoded, active_mask, frozen_mean)
        quant, indices, q_loss, _, _ = self.codebook(quant_encoded)
        post_quant = self.post_quant_conv(quant)
        decoded = self.decoder(post_quant)
        if return_latents:
            return decoded, indices, q_loss, quant_encoded
        return decoded, indices, q_loss

    def encode(
        self,
        designs: th.Tensor,
        active_mask: th.Tensor | None = None,
        frozen_mean: th.Tensor | None = None
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Encode image batch into quantized latent representation."""
        encoded = self.encoder(designs)
        quant_encoded = self.quant_conv(encoded)
        quant_encoded = self.apply_pruning(quant_encoded, active_mask, frozen_mean)
        z_q, indices, loss, min_encodings, perplexity = self.codebook(quant_encoded)
        return z_q, indices, loss, min_encodings, perplexity

    def decode(
        self,
        z: th.Tensor,
        active_mask: th.Tensor | None = None,
        frozen_mean: th.Tensor | None = None
    ) -> th.Tensor:
        """Decode quantized latent representation back to image space."""
        z = self.apply_pruning(z, active_mask, frozen_mean)
        return self.decoder(self.post_quant_conv(z))

    def apply_pruning(self, z, active_mask, frozen_mean):
        if active_mask is None or frozen_mean is None:
            return z
        pruned = ~active_mask
        if not pruned.any():
            return z
        z = z.clone()
        z[:, pruned, :, :] = frozen_mean[pruned].view(1, -1, 1, 1)
        return z

    @th.no_grad()
    def prune_step(  # noqa: PLR0913
        self,
        min_active_dims: int,
        recon_tol: float,
        k_consecutive: int,
        max_prune_per_epoch: int,
        cooldown_epochs: int,
        zmean: th.Tensor,
        zstd: th.Tensor,
        frozen_mean: th.Tensor,
        frozen_std: th.Tensor,
        policy: PruningPolicy,
        active_mask: th.Tensor,
        below_counts: th.Tensor,
        epoch: int,
        next_prune_epoch: int,
        best_val_recon: float,
        val_recon: float | None = None
    ) -> (th.Tensor, th.Tensor, th.Tensor, th.Tensor, int) | None:
        """Core pruning step.

        Applies safeguards + policy to decide whether to prune dims this epoch.
        """
        dim = int(active_mask.sum().item())

        #  --- Safeguards ---
        if dim <= min_active_dims:  #  stop if we go below min dims
            return active_mask, frozen_mean, frozen_std, below_counts, next_prune_epoch
        if epoch < next_prune_epoch:  #  respect cooldown
            return active_mask, frozen_mean, frozen_std, below_counts, next_prune_epoch
        if (
            val_recon is not None
            and best_val_recon < float("inf")
            and (val_recon - best_val_recon) / best_val_recon > recon_tol
        ):
            #  Do not prune if recon is already worse than best_val by > tol
            return active_mask, frozen_mean, frozen_std, below_counts, next_prune_epoch

        #  --- Candidate selection from policy ---
        #  Only consider active (unpruned) dimensions for pruning policy
        z_std_active = zstd[active_mask]
        cand_active = policy(z_std_active).to(zstd.device)

        #  Map back to full dimension space
        cand = th.zeros_like(active_mask, dtype=th.bool)
        cand[active_mask] = cand_active

        #  --- Debounce with consecutive evidence ---
        #  Keep a counter of how many epochs each dim has been marked as candidate
        #  If a dim is not a candidate in the current epoch, reset its count to 0.
        if k_consecutive <= 1:
            stable = cand
            below_counts.zero_()
        else:
            #  Count how many consecutive epochs each dim is a candidate
            below_counts = (below_counts + cand.long()) * cand.long()
            stable = below_counts >= k_consecutive

        #  Filter to *active* dims only (don not re-prune pruned ones)
        candidates = th.where(stable & active_mask)[0]
        if len(candidates) == 0:
            return active_mask, frozen_mean, frozen_std, below_counts, next_prune_epoch

        #  --- Cap how many dims we prune at once ---
        prune_idx = candidates if max_prune_per_epoch is None else candidates[:max_prune_per_epoch]

        # Freeze std BEFORE marking as pruned (capture current variance for volume loss)
        frozen_std[prune_idx] = zstd[prune_idx].clone()

        # --- Commit pruning ---
        active_mask[prune_idx] = False  # mark as pruned
        frozen_mean[prune_idx] = zmean[prune_idx]
        next_prune_epoch = epoch + cooldown_epochs  # set next allowed prune epoch

        return active_mask, frozen_mean, frozen_std, below_counts, next_prune_epoch


    @th.no_grad()
    def update_moving_mean(
        self,
        z: th.Tensor,
        zstd: th.Tensor | None,
        zmean: th.Tensor | None,
        beta: float
    ) -> (th.Tensor, th.Tensor):
        """Update exponential moving average of latent statistics."""
        if zstd is None or zmean is None:
            zstd = z.std(dim=(0,2,3))  # Compute per-channel (per-latent-dim) std
            zmean = z.mean(dim=(0,2,3))  # Compute per-channel (per-latent-dim) mean
        else:
            zstd = th.lerp(zstd, z.std(dim=(0,2,3)), 1 - beta)
            zmean = th.lerp(zmean, z.mean(dim=(0,2,3)), 1 - beta)

        return zstd, zmean


class VQVAETransformer(nn.Module):
    """Wrapper for LV-VQVAE Stage 2: Transformer.

    Generative component trained on the Stage 1 discrete latent space.

    Parameters:
        conditional (bool): If True, use CVQVAE for conditioning.
        vqvae (VQVAE): Pretrained VQVAE model for primary image encoding/decoding.
        cvqvae (VQVAE): Pretrained CVQVAE model for conditional encoding (if conditional=True).
        image_size (int): Input image size (assumed square).
        decoder_channels (tuple[int, ...]): Decoder channels from the VQVAE model.
        cond_feature_map_dim (int): Feature map dimension from the CVQVAE encoder (if conditional=True).
        num_codebook_vectors (int): Number of codebook vectors from the VQVAE model.
        n_layer (int): Number of Transformer layers.
        n_head (int): Number of attention heads in the Transformer.
        n_embd (int): Embedding dimension in the Transformer.
        dropout (float): Dropout rate in the Transformer.
        bias (bool): If True, use bias terms in the Transformer layers.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        conditional: bool = True,
        vqvae: VQVAE,
        cvqvae: VQVAE,
        image_size: int,
        decoder_channels: tuple[int, ...],
        cond_feature_map_dim: int,
        num_codebook_vectors: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: int,
        bias: bool = True,
    ):
        super().__init__()
        self.sos_token = 0
        self.vqvae = vqvae
        self.cvqvae = cvqvae

        #  block_size is automatically set to the combined sequence length of the VQVAE and CVQVAE
        block_size = (image_size // (2 ** (len(decoder_channels) - 1))) ** 2
        if conditional:
            block_size += cond_feature_map_dim**2

        #  Create config object for NanoGPT
        transformer_config = GPTConfig(
            vocab_size=num_codebook_vectors,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,  #  Add dropout parameter (default in nanoGPT)
            bias=bias,  #  Add bias parameter (default in nanoGPT)
        )
        self.transformer = GPT(transformer_config)
        self.conditional = conditional
        self.sidelen = image_size // (2 ** (len(decoder_channels) - 1))  #  Note: assumes square image

    @th.no_grad()
    def encode_to_z(self, *, x: th.Tensor, is_c: bool = False) -> tuple[th.Tensor, th.Tensor]:
        """Encode images to quantized latent vectors (z) and their indices."""
        if is_c:  #  For the conditional tokens, use the CVQVAE encoder
            quant_z, indices, _, _, _ = self.cvqvae.encode(x)
        else:
            quant_z, indices, _, _, _ = self.vqvae.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @th.no_grad()
    def z_to_image(self, indices: th.Tensor) -> th.Tensor:
        """Convert quantized latent indices back to image space."""
        ix_to_vectors = self.vqvae.codebook.embedding(indices).reshape(indices.shape[0], self.sidelen, self.sidelen, -1)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        return self.vqvae.decode(ix_to_vectors)

    def forward(self, x: th.Tensor, c: th.Tensor, pkeep: float = 1.0) -> tuple[th.Tensor, th.Tensor]:
        """Forward pass through the Transformer. Returns logits and targets for loss computation."""
        _, indices = self.encode_to_z(x=x)

        # Replace the start token with the encoded conditional input if using CVQVAE
        if self.conditional:
            _, sos_tokens = self.encode_to_z(x=c, is_c=True)
        else:
            sos_tokens = th.ones(x.shape[0], 1) * self.sos_token
            sos_tokens = sos_tokens.long().to(x.device)

        if pkeep < 1.0:
            mask = th.bernoulli(pkeep * th.ones(indices.shape, device=indices.device))
            mask = mask.round().to(dtype=th.int64)
            random_indices = th.randint_like(indices, self.transformer.config.vocab_size)
            new_indices = mask * indices + (1 - mask) * random_indices
        else:
            new_indices = indices

        new_indices = th.cat((sos_tokens, new_indices), dim=1)

        target = indices

        # NanoGPT forward doesn't use embeddings parameter, but takes targets
        # We're ignoring the loss returned by NanoGPT
        logits, _ = self.transformer(new_indices[:, :-1], None)
        logits = logits[:, -indices.shape[1] :]  # Always predict the last 256 tokens

        return logits, target

    def top_k_logits(self, logits: th.Tensor, k: int) -> th.Tensor:
        """Zero out all logits that are not in the top-k."""
        v, _ = th.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out


    @th.no_grad()
    def sample(
        self, x: th.Tensor, c: th.Tensor, steps: int, temperature: float = 1.0, top_k: int | None = None
    ) -> th.Tensor:
        """Autoregressively sample from the model given initial context x and conditional c."""
        x = th.cat((c, x), dim=1)

        # Keep the original sampling logic for compatibility
        for _ in range(steps):
            logits, _ = self.transformer(x, None)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                # Determine the actual vocabulary size for this batch
                # Count non-negative infinity values in the logits
                n_tokens = th.sum(th.isfinite(logits), dim=-1).min().item()

                # Use the minimum of top_k and the actual number of tokens
                effective_top_k = min(top_k, n_tokens)

                # Apply top_k with the effective value
                if effective_top_k > 0:  # Ensure we have at least one token to sample
                    logits = self.top_k_logits(logits, effective_top_k)
                else:
                    # Fallback if all logits are -inf (shouldn't happen, but just in case)
                    warnings.warn("Warning: No finite logits found for sampling", stacklevel=2)
                    # Make all logits equal (uniform distribution)
                    logits = th.zeros_like(logits)

            probs = f.softmax(logits, dim=-1)
            ix = th.multinomial(probs, num_samples=1)  # Use multinomial sampling for variety and to mitigate image collapse
            x = th.cat((x, ix), dim=1)

        return x[:, c.shape[1] :]

    @th.no_grad()
    def log_images(self, x: th.Tensor, c: th.Tensor, top_k: int | None = None) -> tuple[dict[str, th.Tensor], th.Tensor]:
        """Generate reconstructions and samples from the model for logging."""
        log = {}

        _, indices = self.encode_to_z(x=x)
        # Replace the start token with the encoded conditional input if using CVQVAE
        if self.conditional:
            _, sos_tokens = self.encode_to_z(x=c, is_c=True)
        else:
            sos_tokens = th.ones(x.shape[0], 1) * self.sos_token
            sos_tokens = sos_tokens.long().to(x.device)

        start_indices = indices[:, : indices.shape[1] // 2]
        sample_indices = self.sample(
            start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1], top_k=top_k
        )
        half_sample = self.z_to_image(sample_indices)

        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1], top_k=top_k)
        full_sample = self.z_to_image(sample_indices)

        x_rec = self.z_to_image(indices)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = half_sample
        log["full_sample"] = full_sample

        return log, th.concat((x, x_rec, half_sample, full_sample))


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Seeding
    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)
    design_shape = problem.design_space.shape

    # Configure data loader (keep on CPU for preprocessing)
    training_ds = problem.dataset.with_format("torch")["train"]
    len_dataset = len(training_ds)

    # Add in the upsampled optimal design column and remove the original optimal design column
    training_ds = training_ds.map(
        lambda batch: {
            "optimal_upsampled": resize_to(data=batch["optimal_design"][:], h=args.image_size, w=args.image_size)
            .cpu()
            .numpy()
        },
        batched=True,
    )
    training_ds = training_ds.remove_columns("optimal_design")

    # Now we assume the dataset is of shape (N, C, H, W) and work from there
    image_channels = training_ds["optimal_upsampled"][:].shape[1]
    latent_size = args.image_size // (2 ** (len(args.encoder_channels) - 2))

    conditions = problem.conditions_keys
    # Optionally drop condition columns that are constant like overhang_constraint in beams2d
    if args.drop_constant_conditions:
        training_ds, conditions = drop_constant(training_ds, conditions)

    # Optionally normalize condition columns
    if args.normalize_conditions:
        training_ds, mean, std = normalize(training_ds, conditions)

    n_conds = len(conditions)
    args.cond_dim = n_conds
    condition_tensors = [training_ds[key][:] for key in conditions]

    # Move to device only here
    th_training_ds = th.utils.data.TensorDataset(
        th.as_tensor(training_ds["optimal_upsampled"][:]).to(device),
        *[th.as_tensor(training_ds[key][:]).to(device) for key in conditions],
    )
    dataloader_cvqvae = th.utils.data.DataLoader(
        th_training_ds,
        batch_size=args.batch_size_cvqvae,
        shuffle=True,
    )
    dataloader_vqvae = th.utils.data.DataLoader(
        th_training_ds,
        batch_size=args.batch_size_vqvae,
        shuffle=True,
    )
    dataloader_transformer = th.utils.data.DataLoader(
        th_training_ds,
        batch_size=args.batch_size_transformer,
        shuffle=True,
    )

    # Create a validation dataloader (used optionally for transformer early stopping)
    val_ds = problem.dataset.with_format("torch")["val"]
    val_ds = val_ds.map(
        lambda batch: {
            "optimal_upsampled": resize_to(data=batch["optimal_design"][:], h=args.image_size, w=args.image_size)
            .cpu()
            .numpy()
        },
        batched=True,
    )
    val_ds = val_ds.remove_columns("optimal_design")

    # Optionally drop condition columns that are constant like overhang_constraint in beams2d
    if args.drop_constant_conditions:
        to_drop = [c for c in problem.conditions_keys if c not in conditions]
        if to_drop:
            val_ds = val_ds.remove_columns(to_drop)

    # If enabled, normalize using training mean/std (computed above)
    if args.normalize_conditions:
        val_ds = val_ds.map(
            lambda batch: {
                c: ((th.as_tensor(batch[c][:]).float() - mean[i]) / std[i]).numpy() for i, c in enumerate(conditions)
            },
            batched=True,
        )

    # Move to device only here
    th_val_ds = th.utils.data.TensorDataset(
        th.as_tensor(val_ds["optimal_upsampled"][:]).to(device),
        *[th.as_tensor(val_ds[key][:]).to(device) for key in conditions],
    )
    dataloader_val = th.utils.data.DataLoader(
        th_val_ds,
        batch_size=args.batch_size_transformer,
        shuffle=False,
    )

    # For logging a fixed set of designs in Stage 1
    n_logged_designs = 25
    fixed_indices = random.sample(range(len_dataset), n_logged_designs)
    log_subset = th.utils.data.Subset(th_training_ds, fixed_indices)
    log_dataloader = th.utils.data.DataLoader(
        log_subset,
        batch_size=n_logged_designs,
        shuffle=False,
    )

    # Logging
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            save_code=True,
            name=run_name,
            dir="./logs/wandb",
        )

        #  Base VQ-related metrics
        wandb.define_metric("cvqvae_step", summary="max")
        wandb.define_metric("cvqvae_loss", step_metric="cvqvae_step")
        wandb.define_metric("epoch_cvqvae", step_metric="cvqvae_step")
        wandb.define_metric("vqvae_step", summary="max")
        wandb.define_metric("vqvae_rec_loss", step_metric="vqvae_step")
        wandb.define_metric("vqvae_val_mae", step_metric="vqvae_step")
        wandb.define_metric("vqvae_q_loss", step_metric="vqvae_step")
        wandb.define_metric("vqvae_loss", step_metric="vqvae_step")
        wandb.define_metric("epoch_vqvae", step_metric="vqvae_step")
        wandb.define_metric("transformer_step", summary="max")
        wandb.define_metric("transformer_loss", step_metric="transformer_step")
        wandb.define_metric("epoch_transformer", step_metric="transformer_step")
        if args.early_stopping:
            wandb.define_metric("transformer_val_loss", step_metric="transformer_step")
        wandb.config["image_channels"] = image_channels
        wandb.config["latent_size"] = latent_size

        #  LV-VQVAE-specific metrics
        wandb.define_metric("vqvae_lv_loss", step_metric="vqvae_step")
        wandb.define_metric("vqvae_lv_weight", step_metric="vqvae_step")
        wandb.define_metric("vqvae_token_perplexity", step_metric="vqvae_step")
        wandb.define_metric("vqvae_token_perplexity_frac", step_metric="vqvae_step")
        wandb.define_metric("vqvae_token_usage_frac", step_metric="vqvae_step")
        wandb.define_metric("transformer_logits_entropy", step_metric="transformer_step")

    vqvae = VQVAE(
        device=device,
        is_c=False,
        encoder_channels=args.encoder_channels,
        encoder_start_resolution=args.image_size,
        encoder_attn_resolutions=args.encoder_attn_resolutions,
        encoder_num_res_blocks=args.encoder_num_res_blocks,
        decoder_channels=args.decoder_channels,
        decoder_start_resolution=latent_size,
        decoder_num_res_blocks=args.decoder_num_res_blocks,
        image_channels=image_channels,
        latent_dim=args.latent_dim,
        num_codebook_vectors=args.num_codebook_vectors,
    ).to(device=device)

    cvqvae = VQVAE(
        device=device,
        is_c=True,
        cond_feature_map_dim=args.cond_feature_map_dim,
        cond_dim=args.cond_dim,
        cond_hidden_dim=args.cond_hidden_dim,
        cond_latent_dim=args.cond_latent_dim,
        cond_codebook_vectors=args.cond_codebook_vectors,
    ).to(device=device)

    transformer = VQVAETransformer(
        conditional=args.conditional,
        vqvae=vqvae,
        cvqvae=cvqvae,
        image_size=args.image_size,
        decoder_channels=args.decoder_channels,
        cond_feature_map_dim=args.cond_feature_map_dim,
        num_codebook_vectors=args.num_codebook_vectors,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device=device)

    # CVQVAE Stage 0 optimizer
    opt_cvq = th.optim.Adam(
        list(cvqvae.encoder.parameters())
        + list(cvqvae.decoder.parameters())
        + list(cvqvae.codebook.parameters())
        + list(cvqvae.quant_conv.parameters())
        + list(cvqvae.post_quant_conv.parameters()),
        lr=args.cond_lr,
        eps=1e-08,
        betas=(args.b1, args.b2),
    )

    # VQVAE Stage 1 optimizer
    opt_vq = th.optim.Adam(
        list(vqvae.encoder.parameters())
        + list(vqvae.decoder.parameters())
        + list(vqvae.codebook.parameters())
        + list(vqvae.quant_conv.parameters())
        + list(vqvae.post_quant_conv.parameters()),
        lr=args.lr_vqvae,
        eps=1e-08,
        betas=(args.b1, args.b2),
    )

    # Transformer Stage 2 optimizer
    decay, no_decay = set(), set()
    whitelist_weight_modules = (nn.Linear,)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

    for mn, m in transformer.transformer.named_modules():
        for pn, _ in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn

            if pn.endswith("bias"):
                no_decay.add(fpn)

            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)

            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    no_decay.add("pos_emb")

    param_dict = dict(transformer.transformer.named_parameters())
    decay = {pn for pn in decay if pn in param_dict}
    no_decay = {pn for pn in no_decay if pn in param_dict}

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": 0.01},
        {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
    ]

    opt_transformer = th.optim.AdamW(optim_groups, lr=args.lr_transformer, betas=(0.9, 0.95))

    @th.no_grad()
    def sample_designs_vqvae(n_designs: int) -> list[th.Tensor]:
        """Sample reconstructions from trained Stage 1 (VQVAE)."""
        vqvae.eval()

        designs, *_ = next(iter(log_dataloader))
        designs = designs[:n_designs].to(device)
        reconstructions, _, _ = vqvae(designs)

        vqvae.train()
        return reconstructions

    @th.no_grad()
    def sample_designs_transformer(n_designs: int) -> tuple[th.Tensor, th.Tensor]:
        """Sample generated designs from trained Stage 2."""
        transformer.eval()

        # Create condition grid
        all_conditions = th.stack(condition_tensors, dim=1)
        linspaces = [
            th.linspace(all_conditions[:, i].min(), all_conditions[:, i].max(), n_designs, device=device)
            for i in range(all_conditions.shape[1])
        ]
        desired_conds = th.stack(linspaces, dim=1)

        if args.conditional:
            c = transformer.encode_to_z(x=desired_conds, is_c=True)[1]
        else:
            c = th.ones(n_designs, 1, dtype=th.int64, device=device) * transformer.sos_token

        latent_imgs = transformer.sample(
            x=th.empty(n_designs, 0, dtype=th.int64, device=device), c=c, steps=(latent_size**2)
        )
        gen_imgs = transformer.z_to_image(latent_imgs)

        transformer.train()
        return desired_conds, gen_imgs

    # ---------------------------
    #  Stage 0: Training CVQVAE
    # ---------------------------
    if args.conditional:
        print("Stage 0: Training CVQVAE")
        cvqvae.train()
        for epoch in tqdm.trange(args.n_epochs_cvqvae):
            for i, data in enumerate(dataloader_cvqvae):
                # THIS IS PROBLEM DEPENDENT
                conds = th.stack((data[1:]), dim=1).to(dtype=th.float32, device=device)
                decoded_images, codebook_indices, q_loss = cvqvae(conds)

                opt_cvq.zero_grad()
                rec_loss = th.abs(conds - decoded_images).mean()
                cvq_loss = rec_loss + q_loss
                cvq_loss.backward()
                opt_cvq.step()

                # ----------
                #  Logging
                # ----------
                if args.track:
                    batches_done = epoch * len(dataloader_cvqvae) + i
                    wandb.log(
                        {
                            "cvqvae_step": batches_done,
                            "cvqvae_loss": cvq_loss.item(),
                            "epoch_cvqvae": epoch,
                        }
                    )
                    print(
                        f"[Epoch {epoch}/{args.n_epochs_cvqvae}] [Batch {i}/{len(dataloader_cvqvae)}] [CVQ loss: {cvq_loss.item()}]"
                    )

                    # --------------
                    #  Save model
                    # --------------
                    if args.save_model and epoch == args.n_epochs_cvqvae - 1 and i == len(dataloader_cvqvae) - 1:
                        ckpt_cvq = {
                            "epoch": epoch,
                            "batches_done": batches_done,
                            "cvqvae": cvqvae.state_dict(),
                            "optimizer_cvqvae": opt_cvq.state_dict(),
                            "loss": cvq_loss.item(),
                        }

                        th.save(ckpt_cvq, "lv_cvqvae.pth")
                        artifact_lv_cvq = wandb.Artifact(f"{args.problem_id}_{args.algo}_lv_cvqvae", type="model")
                        artifact_lv_cvq.add_file("lv_cvqvae.pth")
                        wandb.log_artifact(artifact_lv_cvq, aliases=[f"seed_{args.seed}"])

        # Freeze CVQVAE for later use in Stage 2 Transformer
        for p in cvqvae.parameters():
            p.requires_grad_(requires_grad=False)
        cvqvae.eval()

    # --------------------------
    #  Stage 1: Training VQVAE
    # --------------------------
    print("Stage 1: Training VQVAE")
    vqvae.train()
    codebook_freeze = 0

    lv_policy = PruningPolicy(args.lv_pruning_strategy, args.lv_pruning_params)  # We assume a plummet strategy here with the associated baseline parameters.
    lv_w_schedule = polynomial_schedule(args.lv_w_max, args.lv_ramp_epochs, 1, 0, args.lv_start_epoch)  # The value of 1 specifies a linear schedule while the value of 0 specifies the starting LV weight
    zstd = None
    zmean = None
    next_prune_epoch = args.lv_pruning_epoch
    best_val_mae = float("inf")
    active_mask = th.ones(args.latent_dim, dtype=th.bool, device=device)
    below_counts = th.zeros(args.latent_dim, dtype=th.long, device=device)
    frozen_mean = th.zeros(args.latent_dim, dtype=th.float32, device=device)
    frozen_std = th.ones(args.latent_dim, dtype=th.float32, device=device)

    for epoch in tqdm.trange(args.n_epochs_vqvae):
        for i, data in enumerate(dataloader_vqvae):
            # THIS IS PROBLEM DEPENDENT
            designs = data[0].to(dtype=th.float32, device=device)
            decoded_images, codebook_indices, q_loss, z = vqvae(
                designs, return_latents=True, active_mask=active_mask, frozen_mean=frozen_mean
            )

            # Update EMA values of z_std and z_mean for pruning at end of epoch
            zstd, zmean = vqvae.update_moving_mean(z, zstd, zmean, args.lv_pruning_params["beta"])

            # LV loss + weight
            lv_weight = lv_w_schedule(epoch)
            lv_loss = th.tensor(0.0, device=device)
            if epoch >= args.lv_start_epoch and lv_weight > 0.0:
                lv_loss = loss_vol(z, active_mask, frozen_std, eta=float(args.lv_eta))

            rec_loss = th.abs(designs - decoded_images).mean()
            vq_loss = args.rec_loss_factor * rec_loss + q_loss  # + lv_w * lv_loss

            opt_vq.zero_grad()
            vq_loss.backward()
            opt_vq.step()

            # ----------
            #  Logging
            # ----------
            if args.track:
                batches_done = epoch * len(dataloader_vqvae) + i
                with th.no_grad():
                    tstats = token_stats_from_indices(
                        codebook_indices,
                        vocab_size=args.num_codebook_vectors,
                    )
                log_vq = {
                    "vqvae_step": batches_done,
                    "epoch_vqvae": epoch,
                    "vqvae_loss": vq_loss.item(),
                    "vqvae_rec_loss": rec_loss.item(),
                    "vqvae_q_loss": q_loss.item(),
                    "vqvae_lv_loss": float(lv_loss.item()),
                    "vqvae_lv_weight": float(lv_weight),
                    "vqvae_token_perplexity": tstats["token_perplexity"],
                    "vqvae_token_perplexity_frac": tstats["token_perplexity_frac"],
                    "vqvae_token_usage_frac": tstats["token_usage_frac"],
                }
                print(
                    f"[Epoch {epoch}/{args.n_epochs_vqvae}] [Batch {i}/{len(dataloader_vqvae)}] [VQ loss: {vq_loss.item()}]"
                )

                # This saves a grid image of 25 generated designs every sample_interval
                if batches_done % args.sample_interval_vqvae == 0:
                    # Extract 25 designs
                    designs = resize_to(
                        data=sample_designs_vqvae(n_designs=n_logged_designs), h=design_shape[0], w=design_shape[1]
                    )
                    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

                    # Flatten axes for easy indexing
                    axes = axes.flatten()

                    # Plot each tensor as a scatter plot
                    for j, tensor in enumerate(designs):
                        img = tensor.cpu().numpy().reshape(design_shape[0], design_shape[1])  # Extract x and y coordinates
                        axes[j].imshow(img)  # Scatter plot
                        axes[j].title.set_text(f"Reconstruction {j + 1}")  # Set title
                        axes[j].set_xticks([])  # Hide x ticks
                        axes[j].set_yticks([])  # Hide y ticks

                    plt.tight_layout()
                    log_vq["designs_vqvae"] = wandb.Image(fig)
                    plt.close(fig)

                wandb.log(log_vq)

                # --------------
                #  Save models
                # --------------
                if args.save_model and epoch == args.n_epochs_vqvae - 1 and i == len(dataloader_vqvae) - 1:
                    ckpt_vq = {
                        "epoch": epoch,
                        "batches_done": batches_done,
                        "vqvae": vqvae.state_dict(),
                        "optimizer_vqvae": opt_vq.state_dict(),
                        "loss": vq_loss.item(),
                        "lv_state": {
                            "active_mask": active_mask.detach().cpu(),
                            "frozen_mean": frozen_mean.detach().cpu(),
                            "frozen_std": frozen_std.detach().cpu(),
                            "zmean_ema": None if zmean is None else zmean.detach().cpu(),
                            "zstd_ema": None if zstd is None else zstd.detach().cpu(),
                            "below_counts": below_counts.detach().cpu(),
                            "next_prune_epoch": int(next_prune_epoch),
                            "best_val_mae": float(best_val_mae),
                        },
                    }

                    th.save(ckpt_vq, "lv_vqvae.pth")
                    artifact_lv_vq = wandb.Artifact(f"{args.problem_id}_{args.algo}_lv_vqvae", type="model")
                    artifact_lv_vq.add_file("lv_vqvae.pth")
                    wandb.log_artifact(artifact_lv_vq, aliases=[f"seed_{args.seed}"])

        # End-of-epoch: held-out val MAE
        vqvae.eval()
        maes = []
        with th.no_grad():
            for val_data in dataloader_val:
                val_designs = val_data[0].to(dtype=th.float32, device=device)
                val_recon, _, _ = vqvae(val_designs, active_mask=active_mask, frozen_mean=frozen_mean)
                maes.append(th.abs(val_designs - val_recon).mean().item())
        val_mae = sum(maes) / max(1, len(maes))
        best_val_mae = min(best_val_mae, val_mae)

        if args.track:
            batches_done = epoch * len(dataloader_vqvae) + i
            wandb.log(
                {
                    "vqvae_step": batches_done,
                    "epoch_vqvae": epoch,
                    "vqvae_val_mae": val_mae,
                }
            )

        # End-of-epoch: pruning step
        active_mask, frozen_mean, frozen_std, below_counts, next_prune_epoch = vqvae.prune_step(
            args.lv_min_active_dims,
            args.lv_recon_tol,
            args.lv_k_consecutive,
            args.lv_max_prune_per_epoch,
            args.lv_cooldown_epochs,
            zmean,
            zstd,
            frozen_mean,
            frozen_std,
            lv_policy,
            active_mask,
            below_counts,
            epoch,
            next_prune_epoch,
            best_val_mae,
            val_mae
        )

        vqvae.train()

    # Freeze VQVAE for later use in Stage 2 Transformer
    for p in vqvae.parameters():
        p.requires_grad_(requires_grad=False)
    vqvae.eval()

    #  TODO: Persist the final active mask into Stage 2 so tokenization/decoding stays consistent

    # --------------------------------
    #  Stage 2: Training Transformer
    # --------------------------------
    print("Stage 2: Training Transformer")
    transformer.train()

    # If early stopping enabled, initialize necessary variables
    if args.early_stopping:
        best_val = float("inf")
        best_ckpt_tr: dict | None = None
        patience_counter = 0
        patience = args.early_stopping_patience

    for epoch in tqdm.trange(args.n_epochs_transformer):
        for i, data in enumerate(dataloader_transformer):
            # THIS IS PROBLEM DEPENDENT
            designs = data[0].to(dtype=th.float32, device=device)
            conds = th.stack((data[1:]), dim=1).to(dtype=th.float32, device=device)

            opt_transformer.zero_grad()
            logits, targets = transformer(designs, conds)
            loss = f.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            with th.no_grad():
                probs = f.softmax(logits, dim=-1)
                transformer_logits_entropy = float(
                    (-(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)).mean().item()
                )
            loss.backward()
            opt_transformer.step()

            # ----------
            #  Logging
            # ----------
            if args.track:
                batches_done = epoch * len(dataloader_transformer) + i
                wandb.log({"transformer_loss": loss.item(), "transformer_step": batches_done})
                wandb.log({"epoch_transformer": epoch, "transformer_step": batches_done})
                wandb.log(
                    {
                        "transformer_loss": loss.item(),
                        "transformer_logits_entropy": transformer_logits_entropy,
                        "transformer_step": batches_done,
                    }
                )
                wandb.log({"epoch_transformer": epoch, "transformer_step": batches_done})
                print(
                    f"[Epoch {epoch}/{args.n_epochs_transformer}] [Batch {i}/{len(dataloader_transformer)}] [Transformer loss: {loss.item()}]"
                )

                # This saves a grid image of 25 generated designs every sample_interval
                if batches_done % args.sample_interval_transformer == 0:
                    # Extract 25 designs
                    desired_conds, designs = sample_designs_transformer(n_designs=n_logged_designs)
                    if args.normalize_conditions:
                        desired_conds = (desired_conds.cpu() * std) + mean
                    designs = resize_to(data=designs, h=design_shape[0], w=design_shape[1])
                    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

                    # Flatten axes for easy indexing
                    axes = axes.flatten()

                    # Plot each tensor as a scatter plot
                    for j, tensor in enumerate(designs):
                        img = tensor.cpu().numpy().reshape(design_shape[0], design_shape[1])  # Extract x and y coordinates
                        dc = desired_conds[j].cpu()
                        axes[j].imshow(img)  # Scatter plot
                        title = [(conditions[i][0], f"{dc[i]:.2f}") for i in range(n_conds)]
                        title_string = "\n ".join(f"{condition}: {value}" for condition, value in title)
                        axes[j].title.set_text(title_string)  # Set title
                        axes[j].set_xticks([])  # Hide x ticks
                        axes[j].set_yticks([])  # Hide y ticks

                    plt.tight_layout()
                    wandb.log({"designs_transformer": wandb.Image(fig), "transformer_step": batches_done})
                    plt.close(fig)

        # Early stopping based on held-out validation loss
        if args.track and args.early_stopping:
            transformer.eval()
            val_losses = []
            with th.no_grad():
                for val_data in dataloader_val:
                    val_designs = val_data[0].to(dtype=th.float32, device=device)
                    val_conds = th.stack((val_data[1:]), dim=1).to(dtype=th.float32, device=device)
                    val_logits, val_targets = transformer(val_designs, val_conds)
                    val_loss = f.cross_entropy(val_logits.reshape(-1, val_logits.size(-1)), val_targets.reshape(-1))
                    val_losses.append(val_loss.item())
            val_loss = sum(val_losses) / len(val_losses)
            wandb.log({"transformer_val_loss": val_loss, "transformer_step": batches_done})

            if val_loss < best_val - args.early_stopping_delta:
                best_val = val_loss
                patience_counter = 0

                # Cache best model in memory; upload to W&B once at the end.
                if args.save_model:
                    best_ckpt_tr = {
                        "epoch": epoch,
                        "batches_done": batches_done,
                        "transformer": transformer.state_dict(),
                        "optimizer_transformer": opt_transformer.state_dict(),
                        "loss": loss.item(),
                        "val_loss": val_loss,
                    }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch} | best val loss: {best_val:.6f}")
                    break
            transformer.train()

    # --------------
    #  Save model
    # --------------
    if args.track and args.save_model:
        if args.early_stopping:
            ckpt_tr = best_ckpt_tr if best_ckpt_tr is not None else {
                "epoch": epoch,
                "batches_done": batches_done,
                "transformer": transformer.state_dict(),
                "optimizer_transformer": opt_transformer.state_dict(),
                "loss": loss.item(),
                "val_loss": float("nan"),
            }
        else:
            ckpt_tr = {
                "epoch": epoch,
                "batches_done": batches_done,
                "transformer": transformer.state_dict(),
                "optimizer_transformer": opt_transformer.state_dict(),
                "loss": loss.item(),
            }

        th.save(ckpt_tr, "transformer.pth")
        artifact_tr = wandb.Artifact(f"{args.problem_id}_{args.algo}_transformer", type="model")
        artifact_tr.add_file("transformer.pth")
        wandb.log_artifact(artifact_tr, aliases=[f"seed_{args.seed}"])

    wandb.finish()
