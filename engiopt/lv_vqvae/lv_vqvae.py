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
    - Discriminator/adversarial training is removed.
    - Perceptual loss is removed.
    - LV + dynamic pruning is used to reduce n_z.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import time
import warnings

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as f
import tqdm
import tyro
import wandb

from engiopt.lv_vqvae.utils import Codebook
from engiopt.lv_vqvae.utils import DownSampleBlock
from engiopt.lv_vqvae.utils import GPT
from engiopt.lv_vqvae.utils import GPTConfig
from engiopt.lv_vqvae.utils import GroupNorm
from engiopt.lv_vqvae.utils import LinearCombo
from engiopt.lv_vqvae.utils import NonLocalBlock
from engiopt.lv_vqvae.utils import ResidualBlock
from engiopt.lv_vqvae.utils import Swish
from engiopt.lv_vqvae.utils import UpSampleBlock
from engiopt.transforms import drop_constant
from engiopt.transforms import normalize
from engiopt.transforms import resize_to


def _entropy_from_probs(p: th.Tensor) -> th.Tensor:
    """Shannon entropy (nats) of a probability vector."""
    p = p.clamp_min(1e-12)
    return -(p * p.log()).sum()


def _token_stats_from_indices(
    indices: th.Tensor,
    *,
    vocab_size: int,
    active_codes: th.Tensor | None = None,
) -> dict[str, float]:
    """Compute basic token-usage stats from a flat index tensor."""
    idx = indices.view(-1).to(dtype=th.long)
    counts = th.bincount(idx, minlength=vocab_size).float()
    if active_codes is not None:
        a = active_codes.to(device=counts.device, dtype=th.bool)
        counts = counts[a]
        active_n = int(a.sum().item())
    else:
        active_n = vocab_size

    total = counts.sum().clamp_min(1.0)
    probs = counts / total
    entropy = _entropy_from_probs(probs)
    perplexity = float(entropy.exp().item())
    unique = int((counts > 0).sum().item())
    usage_frac = float(unique / max(1, active_n))
    return {
        "token_entropy": float(entropy.item()),
        "token_perplexity": perplexity,
        "unique_tokens": float(unique),
        "token_usage_frac": usage_frac,
    }

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

    # LV + dynamic pruning (n_z)
    lv_start_epoch: int = 5
    """epoch to start applying LV loss (keep LV loss weight = 0 before this)"""
    lv_prune_start_epoch: int = 10
    """epoch to start pruning latent dimensions (after LV has had time to shape the space)"""
    lv_w_max: float = 0.01
    """maximum weight for the LV loss after ramp-up"""
    lv_ramp_epochs: int = 10
    """number of epochs to linearly ramp LV loss weight from 0 to lv_w_max"""
    lv_eta: float = 1.0
    """scaling inside LV loss (multiplies the log-volume term)"""
    lv_ema_beta: float = 0.9
    """EMA momentum for tracking per-dimension std used by pruning decisions"""
    lv_min_active_dims: int = 1
    """minimum number of latent dimensions allowed to remain active (pruning will not go below this)"""
    lv_max_prune_per_epoch: int = 32
    """maximum number of latent dimensions to prune per epoch"""
    lv_cooldown_epochs: int = 1
    """number of epochs to wait after a prune event before pruning again"""
    lv_k_consecutive: int = 1
    """require this many consecutive 'safe' epochs (val MAE below threshold) before pruning"""
    lv_val_mae_target: float = 0.1
    """validation MAE reconstruction target used as the pruning guardrail"""
    lv_val_mae_slack: float = 0.005
    """safety margin below lv_val_mae_target required before allowing pruning"""

    # Codebook pruning (|Z|)
    cb_prune_start_epoch: int = 10
    """epoch to start pruning codebook entries (tokens)"""
    cb_min_active_codes: int = 32
    """minimum number of codebook entries allowed to remain active"""
    cb_max_prune_per_epoch: int = 128
    """maximum number of codebook entries to prune per epoch"""
    cb_cooldown_epochs: int = 1
    """number of epochs to wait after a codebook prune event before pruning again"""
    cb_k_consecutive: int = 1
    """require this many consecutive 'safe' epochs before pruning codebook entries"""
    cb_val_mae_target: float = 0.1
    """validation MAE reconstruction target used as the codebook pruning guardrail"""
    cb_val_mae_slack: float = 0.005
    """safety margin below cb_val_mae_target required before allowing codebook pruning"""
    cb_ema_beta: float = 0.9
    """EMA momentum for tracking per-token usage used by codebook pruning decisions"""

    lv_codebook_freeze_epochs: int = 2
    """freeze online codebook reinitialization for this many epochs after a prune event"""

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


class Decoder(nn.Module):
    """Decoder module for Stage 1 (VQVAE)."""

    def __init__(  # noqa: PLR0913
        self,
        decoder_channels: tuple[int, ...],
        decoder_start_resolution: int,
        decoder_attn_resolutions: tuple[int, ...],
        decoder_num_res_blocks: int,
        image_channels: int,
        latent_dim: int,
    ):
        super().__init__()
        in_channels = decoder_channels[0]
        resolution = decoder_start_resolution
        layers = [
            nn.Conv2d(latent_dim, in_channels, kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels, in_channels),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels),
        ]

        for i in range(len(decoder_channels)):
            out_channels = decoder_channels[i]
            for _ in range(decoder_num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in decoder_attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))

            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, image_channels, kernel_size=3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)


class CondDecoder(nn.Module):
    """Simpler MLP-based decoder for the CVQVAE if enabled."""

    def __init__(self, cond_latent_dim: int, cond_dim: int, cond_hidden_dim: int, cond_feature_map_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            LinearCombo(cond_latent_dim * cond_feature_map_dim**2, cond_hidden_dim),
            LinearCombo(cond_hidden_dim, cond_hidden_dim),
            nn.Linear(cond_hidden_dim, cond_dim),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x.contiguous().view(len(x), -1))


class VQVAE(nn.Module):
    """VQVAE model for Stage 1.

    Can be configured as a CVQVAE if desired.
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
        decoder_attn_resolutions: tuple[int, ...] = (16,),
        decoder_num_res_blocks: int = 3,
        image_channels: int = 1,
        latent_dim: int = 16,
        num_codebook_vectors: int = 256,
    ):
        super().__init__()
        if is_c:
            self.encoder = CondEncoder(cond_feature_map_dim, cond_dim, cond_hidden_dim, cond_latent_dim).to(device=device)

            self.decoder = CondDecoder(cond_latent_dim, cond_dim, cond_hidden_dim, cond_feature_map_dim).to(device=device)

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

            self.decoder = Decoder(
                decoder_channels,
                decoder_start_resolution,
                decoder_attn_resolutions,
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
        active_mask: th.Tensor | None = None,
        return_latents: bool = False,
    ):
        """Full VQVAE forward pass."""
        encoded = self.encoder(designs)
        quant_encoded = self.quant_conv(encoded)
        if active_mask is not None:
            quant_encoded = quant_encoded * active_mask.view(1, -1, 1, 1)
        quant, indices, q_loss, _, _ = self.codebook(quant_encoded, active_mask=active_mask)
        if active_mask is not None:
            quant = quant * active_mask.view(1, -1, 1, 1)
        post_quant = self.post_quant_conv(quant)
        decoded = self.decoder(post_quant)
        if return_latents:
            return decoded, indices, q_loss, quant_encoded
        return decoded, indices, q_loss

    def encode(
        self,
        designs: th.Tensor,
        *,
        active_mask: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Encode image batch into quantized latent representation."""
        encoded = self.encoder(designs)
        quant_encoded = self.quant_conv(encoded)
        if active_mask is not None:
            quant_encoded = quant_encoded * active_mask.view(1, -1, 1, 1)
        z_q, indices, loss, min_encodings, perplexity = self.codebook(quant_encoded, active_mask=active_mask)
        if active_mask is not None:
            z_q = z_q * active_mask.view(1, -1, 1, 1)
        return z_q, indices, loss, min_encodings, perplexity

    def decode(self, z: th.Tensor) -> th.Tensor:
        """Decode quantized latent representation back to image space."""
        return self.decoder(self.post_quant_conv(z))


class VQVAETransformer(nn.Module):
    """Wrapper for Stage 2: Transformer.

    Generative component trained on the Stage 1 discrete latent space.
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
        cond_codebook_vectors: int,
        num_codebook_vectors: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: int,
        bias: bool = True,
    ):
        super().__init__()
        self.vqvae = vqvae
        self.cvqvae = cvqvae

        # Disjoint token namespaces:
        #   - conditional tokens: [0, cond_codebook_vectors)
        #   - image tokens:       [cond_codebook_vectors, cond_codebook_vectors + num_codebook_vectors)
        #   - SOS token:          cond_codebook_vectors + num_codebook_vectors
        self.cond_vocab = cond_codebook_vectors if conditional else 0
        self.image_vocab = num_codebook_vectors
        self.image_token_offset = self.cond_vocab
        self.sos_token = self.cond_vocab + self.image_vocab
        self.vqvae_active_mask: th.Tensor | None = None

        #  block_size is automatically set to the combined sequence length of the VQVAE and CVQVAE
        block_size = (image_size // (2 ** (len(decoder_channels) - 1))) ** 2
        if conditional:
            block_size += cond_feature_map_dim**2

        #  Create config object for NanoGPT
        transformer_config = GPTConfig(
            vocab_size=(num_codebook_vectors + (cond_codebook_vectors if conditional else 0) + 1),
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
            quant_z, indices, _, _, _ = self.vqvae.encode(x, active_mask=self.vqvae_active_mask)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @th.no_grad()
    def z_to_image(self, indices: th.Tensor) -> th.Tensor:
        """Convert quantized latent indices back to image space."""
        indices = indices - self.image_token_offset
        ix_to_vectors = self.vqvae.codebook.embedding(indices).reshape(indices.shape[0], self.sidelen, self.sidelen, -1)
        if self.vqvae_active_mask is not None:
            ix_to_vectors = ix_to_vectors * self.vqvae_active_mask.view(1, 1, 1, -1)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        return self.vqvae.decode(ix_to_vectors)

    def forward(self, x: th.Tensor, c: th.Tensor, pkeep: float = 1.0) -> tuple[th.Tensor, th.Tensor]:
        """Forward pass through the Transformer. Returns logits and targets for loss computation."""
        _, indices = self.encode_to_z(x=x)
        indices = indices + self.image_token_offset

        # Replace the start token with the encoded conditional input if using CVQVAE
        if self.conditional:
            _, sos_tokens = self.encode_to_z(x=c, is_c=True)
        else:
            sos_tokens = th.ones(x.shape[0], 1) * self.sos_token
            sos_tokens = sos_tokens.long().to(x.device)

        if pkeep < 1.0:
            mask = th.bernoulli(pkeep * th.ones(indices.shape, device=indices.device))
            mask = mask.round().to(dtype=th.int64)
            random_indices = th.randint(
                low=self.image_token_offset,
                high=self.image_token_offset + self.image_vocab,
                size=indices.shape,
                device=indices.device,
                dtype=indices.dtype,
            )
            new_indices = mask * indices + (1 - mask) * random_indices
        else:
            new_indices = indices

        new_indices = th.cat((sos_tokens, new_indices), dim=1)

        target = indices

        # NanoGPT forward doesn't use embeddings parameter, but takes targets
        # We're ignoring the loss returned by NanoGPT
        logits, _ = self.transformer(new_indices[:, :-1], None)
        logits = logits[:, -indices.shape[1] :]  # Always predict the last 256 tokens
        logits = self._apply_image_token_constraints(logits)

        return logits, target

    def top_k_logits(self, logits: th.Tensor, k: int) -> th.Tensor:
        """Zero out all logits that are not in the top-k."""
        v, _ = th.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out


    def _apply_image_token_constraints(self, logits: th.Tensor) -> th.Tensor:
        """Mask logits so only *active image tokens* are ever produced.

        This prevents:
          - sampling conditional-token IDs at image positions
          - sampling the SOS token at image positions
          - sampling deactivated (pruned) image codebook entries
        """
        vocab = logits.size(-1)
        mask = th.zeros(vocab, dtype=th.bool, device=logits.device)
        start = self.image_token_offset
        end = self.image_token_offset + self.image_vocab
        active = self.vqvae.codebook.active_codes.to(device=logits.device) if hasattr(self.vqvae.codebook, "active_codes") else th.ones(self.image_vocab, dtype=th.bool, device=logits.device)
        mask[start:end] = active
        return logits.masked_fill(~mask.view(1, 1, -1), -float("inf"))

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
            logits = self._apply_image_token_constraints(logits.unsqueeze(1)).squeeze(1)

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

            # In the VQGAN paper we use multinomial sampling (top_k=None, greedy=False)
            ix = th.multinomial(probs, num_samples=1)

            x = th.cat((x, ix), dim=1)

        return x[:, c.shape[1] :]

    @th.no_grad()
    def log_images(self, x: th.Tensor, c: th.Tensor, top_k: int | None = None) -> tuple[dict[str, th.Tensor], th.Tensor]:
        """Generate reconstructions and samples from the model for logging."""
        log = {}

        _, indices = self.encode_to_z(x=x)
        indices = indices + self.image_token_offset
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

    # Create a validation dataloader (used for LV pruning and optionally transformer early stopping)
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
        wandb.define_metric("cvqvae_step", summary="max")
        wandb.define_metric("cvqvae_loss", step_metric="cvqvae_step")
        wandb.define_metric("epoch_cvqvae", step_metric="cvqvae_step")
        wandb.define_metric("vqvae_step", summary="max")

        wandb.define_metric("vqvae_rec_loss", step_metric="vqvae_step")
        wandb.define_metric("vqvae_q_loss", step_metric="vqvae_step")
        wandb.define_metric("vqvae_lv_loss", step_metric="vqvae_step")
        wandb.define_metric("vqvae_lv_w", step_metric="vqvae_step")
        wandb.define_metric("vqvae_val_mae", step_metric="vqvae_step")
        wandb.define_metric("lv_active_dims", step_metric="vqvae_step")
        wandb.define_metric("lv_pruned_this_epoch", step_metric="vqvae_step")
        wandb.define_metric("lv_ema_std_active_mean", step_metric="vqvae_step")
        wandb.define_metric("lv_ema_std_active_min", step_metric="vqvae_step")
        wandb.define_metric("lv_ema_std_active_max", step_metric="vqvae_step")
        wandb.define_metric("vqvae_z_abs_mean", step_metric="vqvae_step")
        wandb.define_metric("vqvae_z_l2_rms", step_metric="vqvae_step")
        wandb.define_metric("vqvae_z_std", step_metric="vqvae_step")
        wandb.define_metric("vqvae_token_entropy", step_metric="vqvae_step")
        wandb.define_metric("vqvae_token_perplexity", step_metric="vqvae_step")
        wandb.define_metric("vqvae_unique_tokens", step_metric="vqvae_step")
        wandb.define_metric("vqvae_token_usage_frac", step_metric="vqvae_step")
        wandb.define_metric("codebook_active_codes", step_metric="vqvae_step")
        wandb.define_metric("codebook_pruned_this_epoch", step_metric="vqvae_step")
        wandb.define_metric("codebook_ema_entropy", step_metric="vqvae_step")
        wandb.define_metric("codebook_ema_perplexity", step_metric="vqvae_step")

        wandb.define_metric("vqvae_loss", step_metric="vqvae_step")
        wandb.define_metric("epoch_vqvae", step_metric="vqvae_step")
        wandb.define_metric("transformer_step", summary="max")

        wandb.define_metric("transformer_target_entropy", step_metric="transformer_step")
        wandb.define_metric("transformer_target_perplexity", step_metric="transformer_step")
        wandb.define_metric("transformer_target_unique_tokens", step_metric="transformer_step")
        wandb.define_metric("transformer_target_usage_frac", step_metric="transformer_step")
        wandb.define_metric("transformer_logits_entropy", step_metric="transformer_step")

        wandb.define_metric("transformer_loss", step_metric="transformer_step")
        wandb.define_metric("epoch_transformer", step_metric="transformer_step")
        if args.early_stopping:
            wandb.define_metric("transformer_val_loss", step_metric="transformer_step")
        wandb.config["image_channels"] = image_channels
        wandb.config["latent_size"] = latent_size

    from engiopt.lv_vqvae.utils import CodebookPruner
    from engiopt.lv_vqvae.utils import LatentDimPruner
    lv_pruner = LatentDimPruner(
        latent_dim=args.latent_dim,
        start_epoch_lv=args.lv_start_epoch,
        start_epoch_prune=args.lv_prune_start_epoch,
        min_active_dims=args.lv_min_active_dims,
        max_prune_per_epoch=args.lv_max_prune_per_epoch,
        cooldown_epochs=args.lv_cooldown_epochs,
        k_consecutive=args.lv_k_consecutive,
        val_mae_target=args.lv_val_mae_target,
        val_mae_slack=args.lv_val_mae_slack,
        ema_beta=args.lv_ema_beta,
        eta=args.lv_eta,
    ).to(device)

    cb_pruner = CodebookPruner(
        num_codebook_vectors=args.num_codebook_vectors,
        start_epoch_prune=args.cb_prune_start_epoch,
        min_active_codes=args.cb_min_active_codes,
        max_prune_per_epoch=args.cb_max_prune_per_epoch,
        cooldown_epochs=args.cb_cooldown_epochs,
        k_consecutive=args.cb_k_consecutive,
        val_mae_target=args.cb_val_mae_target,
        val_mae_slack=args.cb_val_mae_slack,
        ema_beta=args.cb_ema_beta,
    ).to(device)


    vqvae = VQVAE(
        device=device,
        is_c=False,
        encoder_channels=args.encoder_channels,
        encoder_start_resolution=args.image_size,
        encoder_attn_resolutions=args.encoder_attn_resolutions,
        encoder_num_res_blocks=args.encoder_num_res_blocks,
        decoder_channels=args.decoder_channels,
        decoder_start_resolution=latent_size,
        decoder_attn_resolutions=args.decoder_attn_resolutions,
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
        cond_codebook_vectors=args.cond_codebook_vectors,
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
        reconstructions, _, _ = vqvae(designs, active_mask=lv_pruner.active_mask_float(device))

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
                    wandb.log({"cvqvae_loss": cvq_loss.item(), "cvqvae_step": batches_done})
                    wandb.log({"epoch_cvqvae": epoch, "cvqvae_step": batches_done})
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
    for epoch in tqdm.trange(args.n_epochs_vqvae):
        if codebook_freeze > 0:
            codebook_freeze -= 1
        if hasattr(vqvae.codebook, "reinit_enabled"):
            vqvae.codebook.reinit_enabled = (codebook_freeze == 0)

        for i, data in enumerate(dataloader_vqvae):
            # THIS IS PROBLEM DEPENDENT
            designs = data[0].to(dtype=th.float32, device=device)
            mask = lv_pruner.active_mask_float(device) if (epoch >= args.lv_start_epoch) else None
            decoded_images, codebook_indices, q_loss, quant_encoded = vqvae(
                designs, active_mask=mask, return_latents=True
            )

            lv_pruner.update_stats(quant_encoded.detach())
            cb_pruner.update_stats(codebook_indices.detach())

            rec_loss = th.abs(designs - decoded_images).mean()

            # LV weight schedule: 0 until epoch 5, then ramp
            lv_w = 0.0
            lv_loss = th.tensor(0.0, device=device)
            if epoch >= args.lv_start_epoch:
                t = min(1.0, (epoch - args.lv_start_epoch + 1) / max(1, args.lv_ramp_epochs))
                lv_w = args.lv_w_max * t
                lv_loss = lv_pruner.lv_loss(quant_encoded)

            vq_loss = args.rec_loss_factor * rec_loss + q_loss + lv_w * lv_loss

            # Stats for tracking (computed on the continuous latents used for LV/pruning)
            with th.no_grad():
                z_abs_mean = float(quant_encoded.abs().mean().item())
                z_l2_rms = float(quant_encoded.pow(2).mean().sqrt().item())
                z_std = float(quant_encoded.std().item())

                active_codes = getattr(vqvae.codebook, "active_codes", None)
                token_stats = _token_stats_from_indices(
                    codebook_indices,
                    vocab_size=vqvae.codebook.num_embed,
                    active_codes=active_codes,
                )

                # EMA token-usage stats from the codebook pruner (already a probability distribution)
                cb_p = cb_pruner.ema_usage
                cb_entropy = float(_entropy_from_probs(cb_p).item())
                cb_perplexity = float(th.exp(_entropy_from_probs(cb_p)).item())

                # LV stats on EMA std
                active = lv_pruner.active.to(device=lv_pruner.ema_std.device)
                if int(active.sum().item()) > 0:
                    std_active = lv_pruner.ema_std[active]
                    lv_std_mean = float(std_active.mean().item())
                    lv_std_min = float(std_active.min().item())
                    lv_std_max = float(std_active.max().item())
                else:
                    lv_std_mean = lv_std_min = lv_std_max = 0.0

                codebook_active_n = int(vqvae.codebook.active_codes.sum().item())

            opt_vq.zero_grad()
            vq_loss.backward()
            opt_vq.step()

            # ----------
            #  Logging
            # ----------
            if args.track:
                batches_done = epoch * len(dataloader_vqvae) + i
                wandb.log(
                    {
                        "vqvae_loss": vq_loss.item(),
                        "vqvae_rec_loss": rec_loss.item(),
                        "vqvae_q_loss": q_loss.item(),
                        "vqvae_lv_loss": float(lv_loss.item()),
                        "vqvae_lv_w": float(lv_w),
                        "lv_active_dims": int(lv_pruner.n_active),
                        "lv_ema_std_active_mean": lv_std_mean,
                        "lv_ema_std_active_min": lv_std_min,
                        "lv_ema_std_active_max": lv_std_max,
                        "vqvae_z_abs_mean": z_abs_mean,
                        "vqvae_z_l2_rms": z_l2_rms,
                        "vqvae_z_std": z_std,
                        "vqvae_token_entropy": token_stats["token_entropy"],
                        "vqvae_token_perplexity": token_stats["token_perplexity"],
                        "vqvae_unique_tokens": token_stats["unique_tokens"],
                        "vqvae_token_usage_frac": token_stats["token_usage_frac"],
                        "codebook_active_codes": codebook_active_n,
                        "codebook_ema_entropy": cb_entropy,
                        "codebook_ema_perplexity": cb_perplexity,
                        "vqvae_step": batches_done,
                        "epoch_vqvae": epoch,
                    },
                    step=batches_done,
                )
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
                    wandb.log({"designs_vqvae": wandb.Image(fig), "vqvae_step": batches_done})
                    plt.close(fig)

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
                            "lv_active_mask": lv_pruner.active_mask_float(th.device("cpu")).cpu(),
                            "codebook_active_codes": vqvae.codebook.active_codes.detach().cpu(),
                    }

                    th.save(ckpt_vq, "lv_vqvae.pth")
                    artifact_lv_vq = wandb.Artifact(f"{args.problem_id}_{args.algo}_lv_vqvae", type="model")
                    artifact_lv_vq.add_file("lv_vqvae.pth")
                    wandb.log_artifact(artifact_lv_vq, aliases=[f"seed_{args.seed}"])

        # End-of-epoch: held-out val MAE + pruning (Stage 2-style block)
        vqvae.eval()
        maes = []
        with th.no_grad():
            for val_data in dataloader_val:
                val_designs = val_data[0].to(dtype=th.float32, device=device)
                val_mask = lv_pruner.active_mask_float(device) if (epoch >= args.lv_start_epoch) else None
                val_recon, _, _ = vqvae(val_designs, active_mask=val_mask)
                maes.append(th.abs(val_designs - val_recon).mean().item())
        val_mae = sum(maes) / max(1, len(maes))

        pruned = lv_pruner.maybe_prune(
            epoch=epoch,
            val_mae=val_mae,
            codebook=vqvae.codebook.embedding,
        )

        pruned_codes = cb_pruner.maybe_prune(
            epoch=epoch,
            val_mae=val_mae,
            codebook=vqvae.codebook,
        )

        if pruned:
            codebook_freeze = max(codebook_freeze, args.lv_codebook_freeze_epochs)

        if args.track:
            batches_done = epoch * len(dataloader_vqvae) + i
            wandb.log(
                {
                    "vqvae_val_mae": val_mae,
                    "lv_active_dims": lv_pruner.n_active,
                    "lv_pruned_this_epoch": int(pruned),
                    "lv_codebook_reinit_enabled": int(getattr(vqvae.codebook, "reinit_enabled", True)),
                    "vqvae_step": batches_done,
                    "codebook_active_codes": int(vqvae.codebook.active_codes.sum().item()),
                    "codebook_pruned_this_epoch": int(pruned_codes),
                    "codebook_ema_entropy": float(_entropy_from_probs(cb_pruner.ema_usage).item()),
                    "codebook_ema_perplexity": float(th.exp(_entropy_from_probs(cb_pruner.ema_usage)).item()),
}
            )

        vqvae.train()

    # Freeze VQVAE for later use in Stage 2 Transformer
    for p in vqvae.parameters():
        p.requires_grad_(requires_grad=False)
    vqvae.eval()

    # Persist the final active mask into Stage 2 so tokenization/decoding stays consistent
    transformer.vqvae_active_mask = lv_pruner.active_mask_float(device)

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

            # Token-usage / uncertainty stats for tracking
            with th.no_grad():
                targets_img = (targets - transformer.image_token_offset).clamp_min(0)
                tstats = _token_stats_from_indices(
                    targets_img,
                    vocab_size=transformer.image_vocab,
                    active_codes=getattr(vqvae.codebook, "active_codes", None),
                )
                probs = f.softmax(logits, dim=-1)
                start = transformer.image_token_offset
                end = transformer.image_token_offset + transformer.image_vocab
                p_img = probs[..., start:end]
                logits_entropy = float((-(p_img.clamp_min(1e-12) * p_img.clamp_min(1e-12).log()).sum(dim=-1)).mean().item())
            loss.backward()
            opt_transformer.step()

            # ----------
            #  Logging
            # ----------
            if args.track:
                batches_done = epoch * len(dataloader_transformer) + i
                wandb.log(
                    {
                        "transformer_loss": loss.item(),
                        "transformer_target_entropy": tstats["token_entropy"],
                        "transformer_target_perplexity": tstats["token_perplexity"],
                        "transformer_target_unique_tokens": tstats["unique_tokens"],
                        "transformer_target_usage_frac": tstats["token_usage_frac"],
                        "transformer_logits_entropy": logits_entropy,
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

                # Cache best model in memory; we'll upload to W&B once at the end.
                if args.save_model:
                    best_ckpt_tr = {
                        "epoch": epoch,
                        "batches_done": batches_done,
                        "transformer": transformer.state_dict(),
                        "optimizer_transformer": opt_transformer.state_dict(),
                        "loss": loss.item(),
                        "val_loss": val_loss,
                        "lv_active_mask": lv_pruner.active_mask_float(th.device("cpu")).cpu(),
                        "codebook_active_codes": vqvae.codebook.active_codes.detach().cpu(),
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
                "lv_active_mask": lv_pruner.active_mask_float(th.device("cpu")).cpu(),
                "codebook_active_codes": vqvae.codebook.active_codes.detach().cpu(),
            }
        else:
            ckpt_tr = {
                "epoch": epoch,
                "batches_done": batches_done,
                "transformer": transformer.state_dict(),
                "optimizer_transformer": opt_transformer.state_dict(),
                "loss": loss.item(),
                "lv_active_mask": lv_pruner.active_mask_float(th.device("cpu")).cpu(),
                "codebook_active_codes": vqvae.codebook.active_codes.detach().cpu(),
            }
            th.save(ckpt_tr, "transformer.pth")

        artifact_tr = wandb.Artifact(f"{args.problem_id}_{args.algo}_transformer", type="model")
        artifact_tr.add_file("transformer.pth")
        wandb.log_artifact(artifact_tr, aliases=[f"seed_{args.seed}"])

    wandb.finish()
