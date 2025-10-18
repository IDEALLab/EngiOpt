"""Vector Quantized Generative Adversarial Network (VQGAN).

Based on https://github.com/dome272/VQGAN-pyth with an "Online Clustered Codebook" for better codebook usage from https://github.com/lyndonzheng/CVQ-VAE/blob/main/quantise.py

VQGAN is composed of two primary Stages:
    - Stage 1 is similar to an autoencoder (AE) but with a discrete latent space represented by a codebook.
    - Stage 2 is a generative model (a transformer in this case) trained on the latent space of Stage 1.

The transformer now uses nanoGPT (https://github.com/karpathy/nanoGPT) instead of minGPT (https://github.com/karpathy/minGPT) as in the original implementation.

For Stage 2, we take the indices of the codebook vectors and flatten them into a 1D sequence, treating them as training tokens.
The transformer is then trained to autoregressively predict each token in the sequence, after which it is reshaped back to the original 2D latent space and passed through the decoder of Stage 1 to generate an image.
To make VQGAN conditional, we train a separate VQGAN on the conditions only (CVQGAN) and replace the start-of-sequence tokens of the transformer with the CVQGAN latent tokens.

We have updated the transformer architecture, converted VQGAN from a two-stage to a single-stage approach, added several new arguments, switched to wandb for logging, added greyscale support to the perceptual loss, and more.
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

from engiopt.transforms import drop_constant
from engiopt.transforms import normalize
from engiopt.transforms import resize_to
from engiopt.vqgan.utils import Codebook
from engiopt.vqgan.utils import Discriminator
from engiopt.vqgan.utils import DownSampleBlock
from engiopt.vqgan.utils import GPT
from engiopt.vqgan.utils import GPTConfig
from engiopt.vqgan.utils import GreyscaleLPIPS
from engiopt.vqgan.utils import GroupNorm
from engiopt.vqgan.utils import LinearCombo
from engiopt.vqgan.utils import NonLocalBlock
from engiopt.vqgan.utils import ResidualBlock
from engiopt.vqgan.utils import Swish
from engiopt.vqgan.utils import UpSampleBlock


@dataclass
class Args:
    """Command-line arguments for VQGAN."""

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

    # Algorithm-specific: Stage 1 Conditional AE or "CVQGAN" if the model is specified as conditional
    # Note that a Discriminator is not used for CVQGAN, as it is generally a much simpler model.
    cond_dim: int = 3
    """dimensionality of the condition space"""
    cond_hidden_dim: int = 256
    """hidden dimension of the CVQGAN MLP"""
    cond_latent_dim: int = 4
    """individual code dimension for CVQGAN"""
    cond_codebook_vectors: int = 64
    """number of vectors in the CVQGAN codebook"""
    cond_feature_map_dim: int = 4
    """feature map dimension for the CVQGAN encoder output"""
    batch_size_cvqgan: int = 16
    """size of the batches for CVQGAN"""
    n_epochs_cvqgan: int = 1000  # Default: 1000
    """number of epochs of CVQGAN training"""
    cond_lr: float = 2e-4  # Default: 2e-4
    """learning rate for CVQGAN"""
    latent_size: int = 16
    """size of the latent feature map (automatically determined later)"""
    image_channels: int = 1
    """number of channels in the input image (automatically determined later)"""

    # Algorithm-specific: Stage 1 (AE)
    # From original implementation: assume image_channels=1, use greyscale LPIPS only, use_Online=True, determine image_size automatically, calculate decoder_start_resolution automatically
    n_epochs_vqgan: int = 100  # Default: 100
    """number of epochs of training"""
    batch_size_vqgan: int = 16
    """size of the batches for Stage 1"""
    lr_vqgan: float = 5e-5  # Default: 2e-4
    """learning rate for Stage 1"""
    beta: float = 0.25
    """beta hyperparameter for the codebook commitment loss"""
    b1: float = 0.5
    """decay of first order momentum of gradient"""
    b2: float = 0.9
    """decay of first order momentum of gradient"""
    n_cpu: int = 8
    """number of cpu threads to use during batch generation"""
    latent_dim: int = 16
    """dimensionality of the latent space"""
    num_codebook_vectors: int = 256
    """number of vectors in the codebook"""
    disc_start: int = 0
    """epoch to start discriminator training"""
    disc_factor: float = 0.1
    """weighting factor for the adversarial loss from the discriminator"""
    rec_loss_factor: float = 1.0
    """weighting factor for the reconstruction loss"""
    perceptual_loss_factor: float = 0.1
    """weighting factor for the perceptual loss"""
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
    sample_interval_vqgan: int = 100
    """interval between Stage 1 image samples"""

    # Algorithm-specific: Stage 2 (Transformer)
    # From original implementation: assume pkeep=1.0, sos_token=0, bias=True
    n_epochs_transformer: int = 100  # Default: 100
    """number of epochs of training"""
    early_stopping: bool = True
    """whether to use early stopping for the transformer; if True requires args.track to be True"""
    early_stopping_patience: int = 3
    """number of epochs with no improvement after which training will be stopped"""
    early_stopping_delta: float = 1e-3
    """minimum change in the monitored quantity to qualify as an improvement"""
    batch_size_transformer: int = 16
    """size of the batches for Stage 2"""
    lr_transformer: float = 6e-4  # Default: 6e-4
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
    """Encoder module for VQGAN Stage 1.

    # Simplified architecture: image -> conv -> [resblock -> attn? -> downsample]* -> norm -> swish -> final conv -> latent image
    Where `?` indicates a block that is only included at certain resolutions and `*` indicates a block that is repeated.

    Consists of a series of convolutional, residual, and attention blocks arranged using the provided arguments.
    The number of downsample blocks is determined by the length of the encoder channels tuple minus two.
    For example, if encoder_channels=(128, 128, 128, 128) and the starting resolution is 128, the encoder will downsample the input image twice, from 128x128 to 32x32.

    Parameters:
        encoder_channels (tuple[int, ...]): tuple of channel sizes for each encoder layer
        encoder_start_resolution (int): starting resolution for the encoder
        encoder_attn_resolutions (tuple[int, ...]): tuple of resolutions at which to apply attention in the encoder
        encoder_num_res_blocks (int): number of residual blocks per encoder layer
        image_channels (int): number of channels in the input image
        latent_dim (int): dimensionality of the latent space
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
    """Simpler MLP-based encoder for the CVQGAN if enabled.

    Parameters:
        cond_feature_map_dim (int): feature map dimension for the CVQGAN encoder output
        cond_dim (int): number of input features
        cond_hidden_dim (int): hidden dimension of the CVQGAN MLP
        cond_latent_dim (int): individual code dimension for CVQGAN
    """

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
    """Decoder module for VQGAN Stage 1.

    Simplified architecture: latent image -> conv -> [resblock -> attn? -> upsample]* -> norm -> swish -> final conv -> image
    Where `?` indicates a block that is only included at certain resolutions and `*` indicates a block that is repeated.

    Consists of a series of convolutional, residual, and attention blocks arranged using the provided arguments.
    The number of upsample blocks is determined by the length of the decoder channels tuple minus one.
    For example, if decoder_channels=(128, 128, 128) and the starting resolution is 32, the decoder will upsample the input image twice, from 32x32 to 128x128.

    Parameters:
        decoder_channels (tuple[int, ...]): tuple of channel sizes for each decoder layer
        decoder_start_resolution (int): starting resolution for the decoder
        decoder_attn_resolutions (tuple[int, ...]): tuple of resolutions at which to apply attention in the decoder
        decoder_num_res_blocks (int): number of residual blocks per decoder layer
        image_channels (int): number of channels in the output image
        latent_dim (int): dimensionality of the latent space
    """

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
    """Simpler MLP-based decoder for the CVQGAN if enabled.

    Parameters:
        cond_feature_map_dim (int): feature map dimension for the CVQGAN encoder output
        cond_dim (int): number of input features
        cond_hidden_dim (int): hidden dimension of the CVQGAN MLP
        cond_latent_dim (int): individual code dimension for CVQGAN
    """

    def __init__(self, cond_latent_dim: int, cond_dim: int, cond_hidden_dim: int, cond_feature_map_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            LinearCombo(cond_latent_dim * cond_feature_map_dim**2, cond_hidden_dim),
            LinearCombo(cond_hidden_dim, cond_hidden_dim),
            nn.Linear(cond_hidden_dim, cond_dim),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x.contiguous().view(len(x), -1))


class VQGAN(nn.Module):
    """VQGAN model for Stage 1.

    Can be configured as a CVQGAN if desired.

    Parameters:
        device (th.device): torch device to use

        **CVQGAN params**
        is_c (bool): If True, use CVQGAN architecture (MLP-based encoder/decoder).
        cond_feature_map_dim (int): Feature map dimension for the CVQGAN encoder output.
        cond_dim (int): Number of input features for the CVQGAN encoder.
        cond_hidden_dim (int): Hidden dimension of the CVQGAN MLP.
        cond_latent_dim (int): Individual code dimension for CVQGAN.
        cond_codebook_vectors (int): Number of codebook vectors for CVQGAN.

        **VQGAN params**
        encoder_channels (tuple[int, ...]): Tuple of channel sizes for each encoder layer.
        encoder_start_resolution (int): Starting resolution for the encoder.
        encoder_attn_resolutions (tuple[int, ...]): Tuple of resolutions at which to apply attention in the encoder.
        encoder_num_res_blocks (int): Number of residual blocks per encoder layer.
        decoder_channels (tuple[int, ...]): Tuple of channel sizes for each decoder layer.
        decoder_start_resolution (int): Starting resolution for the decoder.
        decoder_attn_resolutions (tuple[int, ...]): Tuple of resolutions at which to apply attention in the decoder.
        decoder_num_res_blocks (int): Number of residual blocks per decoder layer.
        image_channels (int): Number of channels in the input/output image.
        latent_dim (int): Dimensionality of the latent space.
        num_codebook_vectors (int): Number of codebook vectors.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        device: th.device,
        # CVQGAN parameters
        is_c: bool = False,
        cond_feature_map_dim: int = 4,
        cond_dim: int = 3,
        cond_hidden_dim: int = 256,
        cond_latent_dim: int = 4,
        cond_codebook_vectors: int = 64,
        # VQGAN + Codebook parameters
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

    def forward(self, designs: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Full VQGAN forward pass."""
        encoded = self.encoder(designs)
        quant_encoded = self.quant_conv(encoded)
        quant, indices, q_loss, _, _ = self.codebook(quant_encoded)
        post_quant = self.post_quant_conv(quant)
        decoded = self.decoder(post_quant)
        return decoded, indices, q_loss

    def encode(self, designs: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Encode image batch into quantized latent representation."""
        encoded = self.encoder(designs)
        quant_encoded = self.quant_conv(encoded)
        return self.codebook(quant_encoded)

    def decode(self, z: th.Tensor) -> th.Tensor:
        """Decode quantized latent representation back to image space."""
        return self.decoder(self.post_quant_conv(z))

    def calculate_lambda(self, perceptual_loss: th.Tensor, gan_loss: th.Tensor) -> th.Tensor:
        """Compute balancing factor λ between discriminator loss and the remaining loss terms."""
        last_layer = self.decoder.model[-1]
        last_weight = last_layer.weight
        grad_perc = th.autograd.grad(perceptual_loss, last_weight, retain_graph=True)[0]
        grad_gan = th.autograd.grad(gan_loss, last_weight, retain_graph=True)[0]
        lamb = th.norm(grad_perc) / (th.norm(grad_gan) + 1e-4)
        return 0.8 * th.clamp(lamb, 0.0, 1e4).detach()

    @staticmethod
    def adopt_weight(disc_factor: float, i: int, threshold: int, value: float = 0.0) -> float:
        """Adopt weight scheduling: zero out `disc_factor` before threshold."""
        return value if i < threshold else disc_factor


class VQGANTransformer(nn.Module):
    """Wrapper for VQGAN Stage 2: Transformer.

    Generative component of VQGAN trained on the Stage 1 discrete latent space.

    Parameters:
        conditional (bool): If True, use CVQGAN for conditioning.
        vqgan (VQGAN): Pretrained VQGAN model for primary image encoding/decoding.
        cvqgan (VQGAN): Pretrained CVQGAN model for conditional encoding (if conditional=True).
        image_size (int): Input image size (assumed square).
        decoder_channels (tuple[int, ...]): Decoder channels from the VQGAN model.
        cond_feature_map_dim (int): Feature map dimension from the CVQGAN encoder (if conditional=True).
        num_codebook_vectors (int): Number of codebook vectors from the VQGAN model.
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
        vqgan: VQGAN,
        cvqgan: VQGAN,
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
        self.vqgan = vqgan
        self.cvqgan = cvqgan

        #  block_size is automatically set to the combined sequence length of the VQGAN and CVQGAN
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
        if is_c:  #  For the conditional tokens, use the CVQGAN encoder
            quant_z, indices, _, _, _ = self.cvqgan.encode(x)
        else:
            quant_z, indices, _, _, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @th.no_grad()
    def z_to_image(self, indices: th.Tensor) -> th.Tensor:
        """Convert quantized latent indices back to image space."""
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], self.sidelen, self.sidelen, -1)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        return self.vqgan.decode(ix_to_vectors)

    def forward(self, x: th.Tensor, c: th.Tensor, pkeep: float = 1.0) -> tuple[th.Tensor, th.Tensor]:
        """Forward pass through the Transformer. Returns logits and targets for loss computation."""
        _, indices = self.encode_to_z(x=x)

        # Replace the start token with the encoded conditional input if using CVQGAN
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

            # In the VQGAN paper we use multinomial sampling (top_k=None, greedy=False)
            ix = th.multinomial(probs, num_samples=1)

            x = th.cat((x, ix), dim=1)

        return x[:, c.shape[1] :]

    @th.no_grad()
    def log_images(self, x: th.Tensor, c: th.Tensor, top_k: int | None = None) -> tuple[dict[str, th.Tensor], th.Tensor]:
        """Generate reconstructions and samples from the model for logging."""
        log = {}

        _, indices = self.encode_to_z(x=x)
        # Replace the start token with the encoded conditional input if using CVQGAN
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

    os.makedirs("images/vqgan", exist_ok=True)
    os.makedirs("images/transformer", exist_ok=True)

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
    args.image_channels = training_ds["optimal_upsampled"][:].shape[1]
    args.latent_size = args.image_size // (2 ** (len(args.encoder_channels) - 2))
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
    dataloader_cvqgan = th.utils.data.DataLoader(
        th_training_ds,
        batch_size=args.batch_size_cvqgan,
        shuffle=True,
    )
    dataloader_vqgan = th.utils.data.DataLoader(
        th_training_ds,
        batch_size=args.batch_size_vqgan,
        shuffle=True,
    )
    dataloader_transformer = th.utils.data.DataLoader(
        th_training_ds,
        batch_size=args.batch_size_transformer,
        shuffle=True,
    )

    # If early stopping enabled, create a validation dataloader
    if args.early_stopping:
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
        wandb.define_metric("cvqgan_step", summary="max")
        wandb.define_metric("cvqgan_loss", step_metric="cvqgan_step")
        wandb.define_metric("epoch_cvqgan", step_metric="cvqgan_step")
        wandb.define_metric("vqgan_step", summary="max")
        wandb.define_metric("vqgan_loss", step_metric="vqgan_step")
        wandb.define_metric("discriminator_loss", step_metric="vqgan_step")
        wandb.define_metric("epoch_vqgan", step_metric="vqgan_step")
        wandb.define_metric("transformer_step", summary="max")
        wandb.define_metric("transformer_loss", step_metric="transformer_step")
        wandb.define_metric("epoch_transformer", step_metric="transformer_step")
        if args.early_stopping:
            wandb.define_metric("transformer_val_loss", step_metric="transformer_step")

    vqgan = VQGAN(
        device=device,
        is_c=False,
        encoder_channels=args.encoder_channels,
        encoder_start_resolution=args.image_size,
        encoder_attn_resolutions=args.encoder_attn_resolutions,
        encoder_num_res_blocks=args.encoder_num_res_blocks,
        decoder_channels=args.decoder_channels,
        decoder_start_resolution=args.latent_size,
        decoder_attn_resolutions=args.decoder_attn_resolutions,
        decoder_num_res_blocks=args.decoder_num_res_blocks,
        image_channels=args.image_channels,
        latent_dim=args.latent_dim,
        num_codebook_vectors=args.num_codebook_vectors,
    ).to(device=device)

    discriminator = Discriminator(image_channels=args.image_channels).to(device=device)

    cvqgan = VQGAN(
        device=device,
        is_c=True,
        cond_feature_map_dim=args.cond_feature_map_dim,
        cond_dim=args.cond_dim,
        cond_hidden_dim=args.cond_hidden_dim,
        cond_latent_dim=args.cond_latent_dim,
        cond_codebook_vectors=args.cond_codebook_vectors,
    ).to(device=device)

    transformer = VQGANTransformer(
        conditional=args.conditional,
        vqgan=vqgan,
        cvqgan=cvqgan,
        image_size=args.image_size,
        decoder_channels=args.decoder_channels,
        cond_feature_map_dim=args.cond_feature_map_dim,
        num_codebook_vectors=args.num_codebook_vectors,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device=device)

    # CVQGAN Stage 0 optimizer
    opt_cvq = th.optim.Adam(
        list(cvqgan.encoder.parameters())
        + list(cvqgan.decoder.parameters())
        + list(cvqgan.codebook.parameters())
        + list(cvqgan.quant_conv.parameters())
        + list(cvqgan.post_quant_conv.parameters()),
        lr=args.cond_lr,
        eps=1e-08,
        betas=(args.b1, args.b2),
    )

    # VQGAN Stage 1 optimizer
    opt_vq = th.optim.Adam(
        list(vqgan.encoder.parameters())
        + list(vqgan.decoder.parameters())
        + list(vqgan.codebook.parameters())
        + list(vqgan.quant_conv.parameters())
        + list(vqgan.post_quant_conv.parameters()),
        lr=args.lr_vqgan,
        eps=1e-08,
        betas=(args.b1, args.b2),
    )
    # VQGAN Stage 1 discriminator optimizer
    opt_disc = th.optim.Adam(discriminator.parameters(), lr=args.lr_vqgan, eps=1e-08, betas=(args.b1, args.b2))

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

    perceptual_loss_fcn = GreyscaleLPIPS().eval().to(device)

    @th.no_grad()
    def sample_designs_vqgan(n_designs: int) -> list[th.Tensor]:
        """Sample reconstructions from trained VQGAN Stage 1."""
        vqgan.eval()

        designs, *_ = next(iter(log_dataloader))
        designs = designs[:n_designs].to(device)
        reconstructions, _, _ = vqgan(designs)

        vqgan.train()
        return reconstructions

    @th.no_grad()
    def sample_designs_transformer(n_designs: int) -> tuple[th.Tensor, th.Tensor]:
        """Sample generated designs from trained VQGAN Stage 2."""
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
            x=th.empty(n_designs, 0, dtype=th.int64, device=device), c=c, steps=(args.latent_size**2)
        )
        gen_imgs = transformer.z_to_image(latent_imgs)

        transformer.train()
        return desired_conds, gen_imgs

    # ---------------------------
    #  Stage 0: Training CVQGAN
    # ---------------------------
    if args.conditional:
        print("Stage 0: Training CVQGAN")
        cvqgan.train()
        for epoch in tqdm.trange(args.n_epochs_cvqgan):
            for i, data in enumerate(dataloader_cvqgan):
                # THIS IS PROBLEM DEPENDENT
                conds = th.stack((data[1:]), dim=1).to(dtype=th.float32, device=device)
                decoded_images, codebook_indices, q_loss = cvqgan(conds)

                opt_cvq.zero_grad()
                rec_loss = th.abs(conds - decoded_images).mean()
                cvq_loss = rec_loss + q_loss
                cvq_loss.backward()
                opt_cvq.step()

                # ----------
                #  Logging
                # ----------
                if args.track:
                    batches_done = epoch * len(dataloader_cvqgan) + i
                    wandb.log({"cvqgan_loss": cvq_loss.item(), "cvqgan_step": batches_done})
                    wandb.log({"epoch_cvqgan": epoch, "cvqgan_step": batches_done})
                    print(
                        f"[Epoch {epoch}/{args.n_epochs_cvqgan}] [Batch {i}/{len(dataloader_cvqgan)}] [CVQ loss: {cvq_loss.item()}]"
                    )

                    # --------------
                    #  Save model
                    # --------------
                    if args.save_model and epoch == args.n_epochs_cvqgan - 1 and i == len(dataloader_cvqgan) - 1:
                        ckpt_cvq = {
                            "epoch": epoch,
                            "batches_done": batches_done,
                            "cvqgan": cvqgan.state_dict(),
                            "optimizer_cvqgan": opt_cvq.state_dict(),
                            "loss": cvq_loss.item(),
                        }

                        th.save(ckpt_cvq, "cvqgan.pth")
                        artifact_cvq = wandb.Artifact(f"{args.problem_id}_{args.algo}_cvqgan", type="model")
                        artifact_cvq.add_file("cvqgan.pth")
                        wandb.log_artifact(artifact_cvq, aliases=[f"seed_{args.seed}"])

        # Freeze CVQGAN for later use in Stage 2 Transformer
        for p in cvqgan.parameters():
            p.requires_grad_(requires_grad=False)
        cvqgan.eval()

    # --------------------------
    #  Stage 1: Training VQGAN
    # --------------------------
    print("Stage 1: Training VQGAN")
    vqgan.train()
    discriminator.train()
    for epoch in tqdm.trange(args.n_epochs_vqgan):
        for i, data in enumerate(dataloader_vqgan):
            # THIS IS PROBLEM DEPENDENT
            designs = data[0].to(dtype=th.float32, device=device)
            decoded_images, codebook_indices, q_loss = vqgan(designs)

            disc_real = discriminator(designs)
            disc_fake = discriminator(decoded_images)

            disc_factor = vqgan.adopt_weight(args.disc_factor, epoch, threshold=args.disc_start)

            perceptual_loss = perceptual_loss_fcn(designs, decoded_images)
            rec_loss = th.abs(designs - decoded_images)
            perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
            perceptual_rec_loss = perceptual_rec_loss.mean()
            g_loss = -th.mean(disc_fake)

            lamb = vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
            vq_loss = perceptual_rec_loss + q_loss + disc_factor * lamb * g_loss

            d_loss_real = th.mean(f.relu(1.0 - disc_real))
            d_loss_fake = th.mean(f.relu(1.0 + disc_fake))
            gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

            opt_vq.zero_grad()
            vq_loss.backward(retain_graph=True)

            opt_disc.zero_grad()
            gan_loss.backward()

            opt_vq.step()
            opt_disc.step()

            # ----------
            #  Logging
            # ----------
            if args.track:
                batches_done = epoch * len(dataloader_vqgan) + i
                wandb.log({"vqgan_loss": vq_loss.item(), "vqgan_step": batches_done})
                wandb.log({"discriminator_loss": gan_loss.item(), "vqgan_step": batches_done})
                wandb.log({"epoch_vqgan": epoch, "vqgan_step": batches_done})
                print(
                    f"[Epoch {epoch}/{args.n_epochs_vqgan}] [Batch {i}/{len(dataloader_vqgan)}] [D loss: {gan_loss.item()}] [VQ loss: {vq_loss.item()}]"
                )

                # This saves a grid image of 25 generated designs every sample_interval
                if batches_done % args.sample_interval_vqgan == 0:
                    # Extract 25 designs
                    designs = resize_to(
                        data=sample_designs_vqgan(n_designs=n_logged_designs), h=design_shape[0], w=design_shape[1]
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
                    img_fname = f"images/vqgan/{batches_done}.png"
                    plt.savefig(img_fname)
                    plt.close()
                    wandb.log({"designs_vqgan": wandb.Image(img_fname)})

                # --------------
                #  Save models
                # --------------
                if args.save_model and epoch == args.n_epochs_vqgan - 1 and i == len(dataloader_vqgan) - 1:
                    ckpt_vq = {
                        "epoch": epoch,
                        "batches_done": batches_done,
                        "vqgan": vqgan.state_dict(),
                        "optimizer_vqgan": opt_vq.state_dict(),
                        "loss": vq_loss.item(),
                    }
                    ckpt_disc = {
                        "epoch": epoch,
                        "batches_done": batches_done,
                        "discriminator": discriminator.state_dict(),
                        "optimizer_discriminator": opt_disc.state_dict(),
                        "loss": gan_loss.item(),
                    }

                    th.save(ckpt_vq, "vqgan.pth")
                    th.save(ckpt_disc, "discriminator.pth")
                    artifact_vq = wandb.Artifact(f"{args.problem_id}_{args.algo}_vqgan", type="model")
                    artifact_vq.add_file("vqgan.pth")
                    artifact_disc = wandb.Artifact(f"{args.problem_id}_{args.algo}_discriminator", type="model")
                    artifact_disc.add_file("discriminator.pth")

                    wandb.log_artifact(artifact_vq, aliases=[f"seed_{args.seed}"])
                    wandb.log_artifact(artifact_disc, aliases=[f"seed_{args.seed}"])

    # Freeze VQGAN for later use in Stage 2 Transformer
    for p in vqgan.parameters():
        p.requires_grad_(requires_grad=False)
    vqgan.eval()

    # --------------------------------
    #  Stage 2: Training Transformer
    # --------------------------------
    print("Stage 2: Training Transformer")
    transformer.train()

    # If early stopping enabled, initialize necessary variables
    if args.early_stopping:
        best_val = float("inf")
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
            loss.backward()
            opt_transformer.step()

            # ----------
            #  Logging
            # ----------
            if args.track:
                batches_done = epoch * len(dataloader_transformer) + i
                wandb.log({"transformer_loss": loss.item(), "transformer_step": batches_done})
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
                    img_fname = f"images/transformer/{batches_done}.png"
                    plt.savefig(img_fname)
                    plt.close()
                    wandb.log({"designs_transformer": wandb.Image(img_fname)})

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

                # Save best model (overwrite locally)
                if args.save_model:
                    ckpt_tr = {
                        "epoch": epoch,
                        "batches_done": batches_done,
                        "transformer": transformer.state_dict(),
                        "optimizer_transformer": opt_transformer.state_dict(),
                        "loss": loss.item(),
                        "val_loss": val_loss,
                    }
                    th.save(ckpt_tr, "transformer.pth")
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
        if not args.early_stopping:
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
