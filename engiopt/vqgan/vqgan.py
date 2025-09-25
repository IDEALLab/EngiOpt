# ruff: noqa: F401 # REMOVE THIS LATER
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

from collections import namedtuple
from dataclasses import dataclass
import inspect
import math
import os
import random
import time
from typing import Optional, TYPE_CHECKING
import warnings

from einops import rearrange
from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch as th
from torch import autograd
from torch import nn
from torch.nn import functional as f
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights
import tqdm
from transformers import GPT2LMHeadModel
import tyro
import wandb

from engiopt.metrics import dpp_diversity
from engiopt.metrics import mmd
from engiopt.transforms import resize_to
from engiopt.transforms import upsample_nearest

if TYPE_CHECKING:
    import logging

# URL and checkpoint for LPIPS model
URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}


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
    save_model: bool = False
    """Saves the model to disk."""

    # Algorithm-specific: General
    conditional: bool = True
    """whether the model is conditional or not"""

    # Algorithm-specific: Stage 1 (AE)
    # From original implementation: assume image_channels=1, use greyscale LPIPS only, use_Online=True, determine image_size automatically, calculate decoder_start_resolution automatically
    n_epochs_1: int = 100
    """number of epochs of training"""
    batch_size_1: int = 16
    """size of the batches"""
    lr_1: float = 2e-4
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
    disc_factor: float = 1.0
    """weighting factor for the adversarial loss from the discriminator"""
    rec_loss_factor: float = 1.0
    """weighting factor for the reconstruction loss"""
    perceptual_loss_factor: float = 1.0
    """weighting factor for the perceptual loss"""
    encoder_channels: tuple[int, ...] = (128, 128, 128, 256, 256, 512)
    """tuple of channel sizes for each encoder layer"""
    encoder_attn_resolutions: tuple[int, ...] = (16,)
    """tuple of resolutions at which to apply attention in the encoder"""
    encoder_num_res_blocks: int = 2
    """number of residual blocks per encoder layer"""
    encoder_start_resolution: int = 256
    """starting resolution for the encoder"""
    decoder_channels: tuple[int, ...] = (512, 256, 256, 128, 128)
    """tuple of channel sizes for each decoder layer"""
    decoder_attn_resolutions: tuple[int, ...] = (16,)
    """tuple of resolutions at which to apply attention in the decoder"""
    decoder_num_res_blocks: int = 3
    """number of residual blocks per decoder layer"""
    sample_interval: int = 1600
    """interval between image samples"""

    # Algorithm-specific: Stage 1 (Conditional AE or "CVQGAN" if the model is conditional)
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
    cond_epochs: int = 100
    """number of epochs of CVQGAN training"""
    cond_lr: float = 2e-4
    """learning rate for CVQGAN"""
    cond_sample_interval: int = 1600
    """interval between CVQGAN image samples"""

    # Algorithm-specific: Stage 2 (Transformer)
    # From original implementation: assume pkeep=1.0, sos_token=0, bias=True
    n_epochs_2: int = 100
    """number of epochs of training"""
    batch_size_2: int = 16
    """size of the batches"""
    lr_2: float = 6e-4
    """learning rate for Stage 2"""
    n_layer: int = 12
    """number of layers in the transformer"""
    n_head: int = 12
    """number of attention heads"""
    n_embd: int = 768
    """transformer embedding dimension"""
    dropout: float = 0.3
    """dropout rate in the transformer"""


class Codebook(nn.Module):
    """Improved version over vector quantizer, with the dynamic initialization for the unoptimized "dead" vectors.

    Parameters:
        num_codebook_vectors (int): number of codebook entries
        latent_dim (int): dimensionality of codebook entries
        beta (float): weight for the commitment loss
        decay (float): decay for the moving average of code usage
        distance (str): distance type for looking up the closest code
        anchor (str): anchor sampling methods
        first_batch (bool): if true, the offline version of the model
        contras_loss (bool): if true, use the contras_loss to further improve the performance
        init (bool): if true, the codebook has been initialized
    """
    def __init__(  # noqa: PLR0913
        self, *,
        num_codebook_vectors: int,
        latent_dim: int,
        beta: float = 0.25,
        decay: float = 0.99,
        distance: str = "cos",
        anchor: str = "probrandom",
        first_batch: bool = False,
        contras_loss: bool = False,
        init: bool = False,
    ):
        super().__init__()

        self.num_embed = num_codebook_vectors
        self.embed_dim = latent_dim
        self.beta = beta
        self.decay = decay
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.init = init

        self.pool = FeaturePool(self.num_embed, self.embed_dim)
        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.register_buffer("embed_prob", th.zeros(self.num_embed))

    def forward(self, z: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = z.view(-1, self.embed_dim)

        # clculate the distance
        if self.distance == "l2":
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = - th.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                th.sum(self.embedding.weight ** 2, dim=1) + \
                2 * th.einsum("bd, dn-> bn", z_flattened.detach(), rearrange(self.embedding.weight, "n d-> d n"))
        elif self.distance == "cos":
            # cosine distances from z to embeddings e_j
            normed_z_flattened = f.normalize(z_flattened, dim=1).detach()
            normed_codebook = f.normalize(self.embedding.weight, dim=1)
            d = th.einsum("bd,dn->bn", normed_z_flattened, rearrange(normed_codebook, "n d -> d n"))

        # encoding
        sort_distance, indices = d.sort(dim=1)
        # look up the closest point for the indices
        encoding_indices = indices[:,-1]
        encodings = th.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # quantize and unflatten
        z_q = th.matmul(encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        loss = self.beta * th.mean((z_q.detach()-z)**2) + th.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()
        # count
        avg_probs = th.mean(encodings, dim=0)
        perplexity = th.exp(-th.sum(avg_probs * th.log(avg_probs + 1e-10)))
        min_encodings = encodings

        # online clustered reinitialization for unoptimized points
        if self.training:
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)
            # running average updates
            if self.anchor in ["closest", "random", "probrandom"] and (not self.init):
                # closest sampling
                if self.anchor == "closest":
                    sort_distance, indices = d.sort(dim=0)
                    random_feat = z_flattened.detach()[indices[-1,:]]
                # feature pool based random sampling
                elif self.anchor == "random":
                    random_feat = self.pool.query(z_flattened.detach())
                # probabilitical based random sampling
                elif self.anchor == "probrandom":
                    norm_distance = f.softmax(d.t(), dim=1)
                    prob = th.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                # decay parameter based on the average usage
                decay = th.exp(-(self.embed_prob*self.num_embed*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.embed_dim)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True
            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = d.sort(dim=0)
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0)/self.num_embed)):,:].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0)*1/2),:]
                dis = th.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = f.cross_entropy(dis, th.zeros((dis.size(0),), dtype=th.long, device=dis.device))
                loss +=  contra_loss

        return z_q, encoding_indices, loss, min_encodings, perplexity


class FeaturePool:
    """Implements a feature buffer that stores previously encoded features.

    This buffer enables us to initialize the codebook using a history of generated features rather than the ones produced by the latest encoders.

    Parameters:
        pool_size (int): the size of feature buffer
        dim (int): the dimension of each feature
    """
    def __init__(
        self,
        pool_size: int,
        dim: int = 64
    ):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (th.rand((pool_size, dim)) * 2 - 1) / pool_size

    def query(self, features: th.Tensor) -> th.Tensor:
        """Return features from the pool."""
        self.features = self.features.to(features.device)
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size: # if the batch size is large enough, directly update the whole codebook
                random_feat_id = th.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features
                self.nums_features = num
        elif features.size(0) > int(self.pool_size):
            random_feat_id = th.randint(0, features.size(0), (int(self.pool_size),))
            self.features = features[random_feat_id]
        else:
            random_id = th.randperm(self.pool_size)
            self.features[random_id[:features.size(0)]] = features

        return self.features


class GroupNorm(nn.Module):
    """Group Normalization block to be used in VQGAN Encoder and Decoder.

    Parameters:
        channels (int): number of channels in the input feature map
    """
    def __init__(
        self,
        channels: int
    ):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.gn(x)


class Swish(nn.Module):
    """Swish activation function to be used in VQGAN Encoder and Decoder."""
    def forward(self, x: th.Tensor) -> th.Tensor:
        return x * th.sigmoid(x)


class ResidualBlock(nn.Module):
    """Residual block to be used in VQGAN Encoder and Decoder.

    Parameters:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        return x + self.block(x)


class UpSampleBlock(nn.Module):
    """Up-sampling block to be used in VQGAN Decoder.

    Parameters:
        channels (int): number of channels in the input feature map
    """
    def __init__(
        self,
        channels: int
    ):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = f.interpolate(x, scale_factor=2.0)
        return self.conv(x)


class DownSampleBlock(nn.Module):
    """Down-sampling block to be used in VQGAN Encoder.

    Parameters:
        channels (int): number of channels in the input feature map
    """
    def __init__(
        self,
        channels: int
    ):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x: th.Tensor) -> th.Tensor:
        pad = (0, 1, 0, 1)
        x = f.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class NonLocalBlock(nn.Module):
    """Non-local attention block to be used in VQGAN Encoder and Decoder.

    Parameters:
        channels (int): number of channels in the input feature map
    """
    def __init__(
        self,
        channels: int
    ):
        super().__init__()
        self.in_channels = channels

        self.gn = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x: th.Tensor) -> th.Tensor:
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        attn = th.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = f.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        a = th.bmm(v, attn)
        a = a.reshape(b, c, h, w)

        return x + a


class LinearCombo(nn.Module):
    """Regular fully connected layer combo for the CVQGAN if enabled.

    Parameters:
        in_features (int): number of input features
        out_features (int): number of output features
        alpha (float): negative slope for LeakyReLU
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float = 0.2
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(alpha)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)


class Encoder(nn.Module):
    """Encoder module for VQGAN Stage 1.

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
        layers = [nn.Conv2d(image_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for _ in range(encoder_num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in encoder_attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != len(channels)-2:
                layers.append(DownSampleBlock(channels[i+1]))
                resolution //= 2
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], latent_dim, 3, 1, 1))
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
    def __init__(
        self,
        cond_feature_map_dim: int,
        cond_dim: int,
        cond_hidden_dim: int,
        cond_latent_dim: int
    ):
        super().__init__()
        self.c_feature_map_dim = cond_feature_map_dim
        self.model = nn.Sequential(
            LinearCombo(cond_dim, cond_hidden_dim),
            LinearCombo(cond_hidden_dim, cond_hidden_dim),
            nn.Linear(cond_hidden_dim, cond_latent_dim*cond_feature_map_dim**2)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        encoded = self.model(x)
        s = encoded.shape
        return encoded.view(s[0], s[1]//self.c_feature_map_dim**2, self.c_feature_map_dim, self.c_feature_map_dim)


class Decoder(nn.Module):
    """Decoder module for VQGAN Stage 1.

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
        latent_dim: int
    ):
        super().__init__()
        in_channels = decoder_channels[0]
        resolution = decoder_start_resolution
        layers = [nn.Conv2d(latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels),
                  NonLocalBlock(in_channels),
                  ResidualBlock(in_channels, in_channels)]

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
        layers.append(nn.Conv2d(in_channels, image_channels, 3, 1, 1))
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
    def __init__(
        self,
        cond_latent_dim: int,
        cond_dim: int,
        cond_hidden_dim: int,
        cond_feature_map_dim: int
    ):
        super().__init__()

        self.model = nn.Sequential(
            LinearCombo(cond_latent_dim*cond_feature_map_dim**2, cond_hidden_dim),
            LinearCombo(cond_hidden_dim, cond_hidden_dim),
            nn.Linear(cond_hidden_dim, cond_dim)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x.contiguous().view(len(x), -1))


class Discriminator(nn.Module):
    """PatchGAN-style discriminator.

    Adapted from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538
    This assumes we never use a discriminator for the CVQGAN, since it is generally a much simpler model.

    Parameters:
        num_filters_last: Number of filters in the last conv layer.
        n_layers: Number of convolutional layers.
        image_channels: Number of channels in the input image.
    """

    def __init__(
        self,
        num_filters_last: int = 64,
        n_layers: int = 3,
        image_channels: int = 1
    ):
        super().__init__()

        # Convolutional backbone (PatchGAN)
        layers: list[nn.Module] = [
            nn.Conv2d(image_channels, num_filters_last, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2**i, 8)
            layers += [
                nn.Conv2d(
                    num_filters_last * num_filters_mult_last,
                    num_filters_last * num_filters_mult,
                    kernel_size=4,
                    stride=2 if i < n_layers else 1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        layers.append(
            nn.Conv2d(num_filters_last * num_filters_mult, 1, kernel_size=4, stride=1, padding=1)
        )
        self.model = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._weights_init)


    @staticmethod
    def _weights_init(m: nn.Module) -> None:
        """Custom weight initialization (DCGAN-style)."""
        classname = m.__class__.__name__
        if "Conv" in classname:
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        elif "BatchNorm" in classname:
            nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
            nn.init.constant_(m.bias.data, 0.0)


    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass with optional CVQGAN adapter."""
        return self.model(x)


class ScalingLayer(nn.Module):
    """Channel-wise affine normalization used by LPIPS."""

    def __init__(self):
        super().__init__()
        self.register_buffer("shift", th.tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", th.tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, x: th.Tensor) -> th.Tensor:
        return (x - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """1x1 conv with dropout (per-layer LPIPS linear head)."""

    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)


class VGG16(nn.Module):
    """Torchvision VGG16 feature extractor sliced at LPIPS tap points."""

    def __init__(self):
        super().__init__()
        vgg_feats = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        blocks = [vgg_feats[i] for i in range(30)]
        self.slice1 = nn.Sequential(*blocks[0:4])    # relu1_2
        self.slice2 = nn.Sequential(*blocks[4:9])    # relu2_2
        self.slice3 = nn.Sequential(*blocks[9:16])   # relu3_3
        self.slice4 = nn.Sequential(*blocks[16:23])  # relu4_3
        self.slice5 = nn.Sequential(*blocks[23:30])  # relu5_3
        self.requires_grad_(requires_grad=False)

    def forward(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return (h1, h2, h3, h4, h5)


class GreyscaleLPIPS(nn.Module):
    """LPIPS for greyscale/topological data with optional 'raw' aggregation.

    ``use_raw=True`` is often preferable for non-natural images since learned
    linear heads are tuned on natural RGB photos.

    Parameters:
        use_raw: If True, average raw per-layer squared diffs (no linear heads).
        clamp_output: Clamp the final loss to ``>= 0``.
        robust_clamp: Clamp inputs to [0, 1] before feature extraction.
        warn_on_clamp: If True, log warnings when inputs fall outside [0, 1].
        freeze: If True, disables grads on all params.
        ckpt_name: Key in URL_MAP/CKPT_MAP for loading LPIPS heads.
        logger: Optional logger for non-intrusive messages/warnings.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        use_raw: bool = True,
        clamp_output: bool = False,
        robust_clamp: bool = True,
        warn_on_clamp: bool = False,
        freeze: bool = True,
        ckpt_name: str = "vgg_lpips",
        logger: logging.Logger | None = None,
    ):
        super().__init__()
        self.use_raw = use_raw
        self.clamp_output = clamp_output
        self.robust_clamp = robust_clamp
        self.warn_on_clamp = warn_on_clamp
        self._logger = logger

        self.scaling_layer = ScalingLayer()
        self.channels = (64, 128, 256, 512, 512)
        self.vgg = VGG16()
        self.linears = nn.ModuleList([NetLinLayer(c) for c in self.channels])

        self._load_from_pretrained(name=ckpt_name)
        if freeze:
            self.requires_grad_(requires_grad=False)


    def forward(self, real_x: th.Tensor, fake_x: th.Tensor) -> th.Tensor:
        """Compute greyscale-aware LPIPS distance between two batches."""
        if self.warn_on_clamp and self._logger is not None:
            with th.no_grad():
                if (fake_x < 0).any() or (fake_x > 1).any():
                    self._logger.warning(
                        "GreyscaleLPIPS: generated input outside [0,1]: [%.4f, %.4f]",
                        float(fake_x.min().item()), float(fake_x.max().item()),
                    )
                if (real_x < 0).any() or (real_x > 1).any():
                    self._logger.warning(
                        "GreyscaleLPIPS: reference input outside [0,1]: [%.4f, %.4f]",
                        float(real_x.min().item()), float(real_x.max().item()),
                    )

        if self.robust_clamp:
            real_x = th.clamp(real_x, 0.0, 1.0)
            fake_x = th.clamp(fake_x, 0.0, 1.0)

        # Promote greyscale -> RGB for VGG features
        if real_x.shape[1] == 1:
            real_x = real_x.repeat(1, 3, 1, 1)
        if fake_x.shape[1] == 1:
            fake_x = fake_x.repeat(1, 3, 1, 1)

        fr = self.vgg(self.scaling_layer(real_x))
        ff = self.vgg(self.scaling_layer(fake_x))
        diffs = [(self._norm_tensor(a) - self._norm_tensor(b)) ** 2 for a, b in zip(fr, ff)]

        if self.use_raw:
            parts = [self._spatial_average(d).mean(dim=1, keepdim=True) for d in diffs]
        else:
            parts = [self._spatial_average(self.linears[i](d)) for i, d in enumerate(diffs)]

        loss = th.stack(parts, dim=0).sum()
        if self.clamp_output:
            loss = th.clamp(loss, min=0.0)
        return loss

    # Helpers
    @staticmethod
    def _norm_tensor(x: th.Tensor) -> th.Tensor:
        """L2-normalize channels per spatial location: BxCxHxW -> BxCxHxW."""
        norm = th.sqrt(th.sum(x**2, dim=1, keepdim=True))
        return x / (norm + 1e-10)

    @staticmethod
    def _spatial_average(x: th.Tensor) -> th.Tensor:
        """Average over spatial dimensions with dims kept: BxCxHxW -> BxCx1x1."""
        return x.mean(dim=(2, 3), keepdim=True)

    def _load_from_pretrained(self, *, name: str) -> None:
        """Load LPIPS linear heads (and any required buffers) from a checkpoint."""
        ckpt = self._get_ckpt_path(name, "vgg_lpips")
        state_dict = th.load(ckpt, map_location=th.device("cpu"), weights_only=True)
        self.load_state_dict(state_dict, strict=False)

    @staticmethod
    def _download(url: str, local_path: str, *, chunk_size: int = 1024) -> None:
        """Stream a file to disk with a progress bar."""
        os.makedirs(os.path.split(local_path)[0], exist_ok=True)
        with requests.get(url, stream=True, timeout=10) as r:
            total_size = int(r.headers.get("content-length", 0))
            with tqdm.tqdm(total=total_size, unit="B", unit_scale=True) as pbar, open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(len(data))

    def _get_ckpt_path(self, name: str, root: str) -> str:
        """Return local path to a pretrained LPIPS checkpoint; download if missing."""
        assert name in URL_MAP, f"Unknown LPIPS checkpoint name: {name!r}"
        path = os.path.join(root, CKPT_MAP[name])
        if not os.path.exists(path):
            if self._logger is not None:
                self._logger.info("Downloading LPIPS weights '%s' from %s to %s", name, URL_MAP[name], path)
            self._download(URL_MAP[name], path)
        return path


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
        self, *,
        device: th.device,

        # CVQGAN parameters
        is_c: bool = False,
        cond_feature_map_dim: int = 4,
        cond_dim: int = 3,
        cond_hidden_dim: int = 256,
        cond_latent_dim: int = 4,
        cond_codebook_vectors: int = 64,

        # VQGAN + Codebook parameters
        encoder_channels: tuple[int, ...],
        encoder_start_resolution: int,
        encoder_attn_resolutions: tuple[int, ...],
        encoder_num_res_blocks: int,
        decoder_channels: tuple[int, ...],
        decoder_start_resolution: int,
        decoder_attn_resolutions: tuple[int, ...],
        decoder_num_res_blocks: int,
        image_channels: int = 1,
        latent_dim: int = 16,
        num_codebook_vectors: int = 256,

    ):
        super().__init__()
        if is_c:
            self.encoder = CondEncoder(
                cond_feature_map_dim,
                cond_dim,
                cond_hidden_dim,
                cond_latent_dim
            ).to(device=device)

            self.decoder = CondDecoder(
                cond_latent_dim,
                cond_dim,
                cond_hidden_dim,
                cond_feature_map_dim
            ).to(device=device)

            self.quant_conv = nn.Conv2d(cond_latent_dim, cond_latent_dim, 1).to(device=device)
            self.post_quant_conv = nn.Conv2d(cond_latent_dim, cond_latent_dim, 1).to(device=device)
        else:
            self.encoder = Encoder(
                encoder_channels,
                encoder_start_resolution,
                encoder_attn_resolutions,
                encoder_num_res_blocks,
                image_channels,
                latent_dim
            ).to(device=device)

            self.decoder = Decoder(
                decoder_channels,
                decoder_start_resolution,
                decoder_attn_resolutions,
                decoder_num_res_blocks,
                image_channels,
                latent_dim
            ).to(device=device)

            self.quant_conv = nn.Conv2d(latent_dim, latent_dim, 1).to(device=device)
            self.post_quant_conv = nn.Conv2d(latent_dim, latent_dim, 1).to(device=device)

        self.codebook = Codebook(
            num_codebook_vectors = cond_codebook_vectors if is_c else num_codebook_vectors,
            latent_dim = cond_latent_dim if is_c else latent_dim
        ).to(device=device)

    def forward(self, imgs: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Full VQGAN forward pass."""
        encoded = self.encoder(imgs)
        quant_encoded = self.quant_conv(encoded)
        quant, indices, q_loss = self.codebook(quant_encoded)
        post_quant = self.post_quant_conv(quant)
        decoded = self.decoder(post_quant)
        return decoded, indices, q_loss

    def encode(self, imgs: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Encode image batch into quantized latent representation."""
        encoded = self.encoder(imgs)
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


###########################################
########## GPT-2 BASE CODE BELOW ##########
###########################################
class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch lacks bias=False support)."""

    def __init__(self, ndim: int, *, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(th.ones(ndim))
        self.bias = nn.Parameter(th.zeros(ndim)) if bias else None

    def forward(self, x: th.Tensor) -> th.Tensor:
        return f.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Causal self-attention with FlashAttention fallback when unavailable."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(f, "scaled_dot_product_attention")
        if not self.flash:
            warnings.warn(
                "Falling back to non-flash attention; PyTorch >= 2.0 enables FlashAttention.",
                stacklevel=2,
            )
            self.register_buffer(
                "bias",
                th.tril(th.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, t, c = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)

        if self.flash:
            y = f.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :t, :t] == 0, float("-inf"))
            att = f.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Feed-forward block used inside Transformer blocks."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Transformer block: LayerNorm -> Self-Attn -> residual; LayerNorm -> MLP -> residual."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 uses 50257; padded to multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # GPT-2 uses biases in Linear/LayerNorm


class GPT(nn.Module):
    """Minimal GPT-2 style Transformer with HF weight import."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                th.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self, *, non_embedding: bool = True) -> int:
        """Return total parameter count (optionally excluding position embeddings)."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer["wpe"].weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            th.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                th.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            th.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: th.Tensor,
        targets: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor | None]:
        """Forward pass returning logits and optional cross-entropy loss."""
        device = idx.device
        _, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}; block size is {self.config.block_size}"
        )
        pos = th.arange(0, t, dtype=th.long, device=device)

        tok_emb = self.transformer["wte"](idx)
        pos_emb = self.transformer["wpe"](pos)
        x = self.transformer["drop"](tok_emb + pos_emb)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)

        logits = self.lm_head(x)
        loss: th.Tensor | None
        if targets is not None:
            loss = f.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None
        return logits, loss

    def crop_block_size(self, block_size: int) -> None:
        """Reduce maximum context length and trim position embeddings."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer["wpe"].weight = nn.Parameter(self.transformer["wpe"].weight[:block_size])
        for block in self.transformer["h"]:
            attn = block.attn
            if hasattr(attn, "bias"):
                attn.bias = attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(
        cls,
        model_type: str,
        override_args: dict[str, float] | None = None,
    ) -> GPT:
        """Load HF GPT-2 weights into this minimal GPT implementation."""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}
        assert all(k == "dropout" for k in override_args), "Only 'dropout' can be overridden"

        cfg_map: dict[str, dict[str, int]] = {
            "gpt2": {"n_layer": 12, "n_head": 12, "n_embd": 768},
            "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024},
            "gpt2-large": {"n_layer": 36, "n_head": 20, "n_embd": 1280},
            "gpt2-xl": {"n_layer": 48, "n_head": 25, "n_embd": 1600},
        }

        # Use object so we can mix int, float, and bool
        config_args: dict[str, object] = dict(cfg_map[model_type])
        config_args.update({"vocab_size": 50257, "block_size": 1024, "bias": True})

        if "dropout" in override_args:
            config_args["dropout"] = float(override_args["dropout"])

        config = GPTConfig(**config_args)  # type: ignore[arg-type]
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = [k for k in sd if not k.endswith(".attn.bias")]

        hf: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = hf.state_dict()
        sd_keys_hf = [
            k
            for k in sd_hf
            if not (k.endswith((".attn.masked_bias", ".attn.bias")))
        ]

        transposed = {"attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with th.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with th.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> th.optim.Optimizer:
        """Create AdamW with decoupled weight decay for matrix weights only."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        dim_threshold = 2
        decay_params = [p for p in param_dict.values() if p.dim() >= dim_threshold]
        nodecay_params = [p for p in param_dict.values() if p.dim() < dim_threshold]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in inspect.signature(th.optim.AdamW).parameters
        use_fused = bool(fused_available and device_type == "cuda")
        extra_args: dict[str, object] = {"fused": True} if use_fused else {}
        return th.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate model FLOPS utilization relative to A100 bf16 peak (312 TFLOPS)."""
        n = self.get_num_params()
        cfg = self.config
        l, h, q, t = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * n + 12 * l * h * q * t
        flops_per_fwdbwd = flops_per_token * t
        flops_per_iter = flops_per_fwdbwd * float(fwdbwd_per_iter)
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_peak = 312e12
        return float(flops_achieved / flops_peak)

    @th.no_grad()
    def generate(
        self,
        idx: th.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> th.Tensor:
        """Autoregressively sample tokens conditioned on idx."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = th.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = f.softmax(logits, dim=-1)
            idx_next = th.multinomial(probs, num_samples=1)
            idx = th.cat((idx, idx_next), dim=1)
        return idx
###########################################
########## GPT-2 BASE CODE ABOVE ##########
###########################################


class VQGANTransformer(nn.Module):
    """Wrapper for VQGAN Stage 2: Transformer.

    Generative component of VQGAN trained on the Stage 1 discrete latent space.

    Parameters:
        conditional (bool): If True, use CVQGAN for conditioning.
        vqgan (VQGAN): Pretrained VQGAN model for primary image encoding/decoding.
        cvqgan (VQGAN): Pretrained CVQGAN model for conditional encoding (if conditional=True).
        image_size (int): Input image size (assumed square).
        decoder_channels (tuple[int, ...]): Decoder channels from the VQGAN model.
        cond_fmap_dim (int): Feature map dimension from the CVQGAN encoder (if conditional=True).
        num_codebook_vectors (int): Number of codebook vectors from the VQGAN model.
        n_layer (int): Number of Transformer layers.
        n_head (int): Number of attention heads in the Transformer.
        n_embd (int): Embedding dimension in the Transformer.
        dropout (float): Dropout rate in the Transformer.
        bias (bool): If True, use bias terms in the Transformer layers.
    """
    def __init__(  # noqa: PLR0913
        self, *,
        conditional: bool = True,
        vqgan: VQGAN,
        cvqgan: VQGAN,
        image_size: int,
        decoder_channels: tuple[int, ...],
        cond_fmap_dim: int,
        num_codebook_vectors: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: int,
        bias: bool = True
    ):
        super().__init__()
        self.sos_token = 0
        self.vqgan = vqgan.eval()
        for param in self.vqgan.parameters():
            param.requires_grad = False

        if conditional:
            self.cvqgan = cvqgan
            for param in self.cvqgan.parameters():
                param.requires_grad = False

        #  block_size is automatically set to the combined sequence length of the VQGAN and CVQGAN
        block_size = (image_size // (2 ** (len(decoder_channels) - 1))) ** 2
        if conditional:
            block_size += cond_fmap_dim ** 2

        #  Create config object for NanoGPT
        transformer_config = GPTConfig(
            vocab_size=num_codebook_vectors,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,    #  Add dropout parameter (default in nanoGPT)
            bias=bias           #  Add bias parameter (default in nanoGPT)
        )
        self.transformer = GPT(transformer_config)
        self.conditional = conditional
        self.sidelen = image_size // (2 ** (len(decoder_channels) - 1))  #  Note: assumes square image

    @th.no_grad()
    def encode_to_z(self, *, x: th.Tensor, is_c: bool = False) -> tuple[th.Tensor, th.Tensor]:
        """Encode images to quantized latent vectors (z) and their indices."""
        if is_c:  #  For the conditional tokens, use the CVQGAN encoder
            quant_z, indices, _ = self.cvqgan.encode(x)
        else:
            quant_z, indices, _ = self.vqgan.encode(x)
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
        logits = logits[:, -indices.shape[1]:]  # Always predict the last 256 tokens

        return logits, target

    def top_k_logits(self, logits: th.Tensor, k: int) -> th.Tensor:
        """Zero out all logits that are not in the top-k."""
        v, _ = th.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @th.no_grad()
    def sample(self, x: th.Tensor, c: th.Tensor, steps: int, temperature: float = 1.0, top_k: int | None = None) -> th.Tensor:
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
                    print("Warning: No finite logits found for sampling")
                    # Make all logits equal (uniform distribution)
                    logits = th.zeros_like(logits)

            probs = f.softmax(logits, dim=-1)

            # In the VQGAN paper we use multinomial sampling (top_k=None, greedy=False)
            ix = th.multinomial(probs, num_samples=1)

            x = th.cat((x, ix), dim=1)

        return x[:, c.shape[1]:]

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

        start_indices = indices[:, :indices.shape[1] // 2]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1], top_k=top_k)
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
