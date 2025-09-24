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
import os
import random
import time
from typing import Optional, TYPE_CHECKING

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
    codebook_vectors: int = 256
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

    # Algorithm-specific: Stage 1 (Conditional AE if the model is conditional)
    cond_dim: int = 3
    """dimensionality of the condition space"""
    cond_hidden_dim: int = 256
    """hidden dimension of the CVQGAN MLP"""
    cond_latent_dim: int = 4
    "individual code dimension for CVQGAN"
    cond_codebook_vectors: int = 256
    """number of vectors in the CVQGAN codebook"""
    cond_feature_map_dim: int = 4
    """feature map dimension for the CVQGAN encoder output"""


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
        pool_size (int): the size of featue buffer
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
        c_fmap_dim (int): feature map dimension for the CVQGAN encoder output
        c_input_dim (int): number of input features
        c_hidden_dim (int): hidden dimension of the CVQGAN MLP
        c_latent_dim (int): individual code dimension for CVQGAN
    """
    def __init__(
        self,
        c_fmap_dim: int,
        c_input_dim: int,
        c_hidden_dim: int,
        c_latent_dim: int
    ):
        super().__init__()
        self.c_fmap_dim = c_fmap_dim
        self.model = nn.Sequential(
            LinearCombo(c_input_dim, c_hidden_dim),
            LinearCombo(c_hidden_dim, c_hidden_dim),
            nn.Linear(c_hidden_dim, c_latent_dim*c_fmap_dim**2)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        encoded = self.model(x)
        s = encoded.shape
        return encoded.view(s[0], s[1]//self.c_fmap_dim**2, self.c_fmap_dim, self.c_fmap_dim)


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
        c_fmap_dim (int): feature map dimension for the CVQGAN encoder output
        c_input_dim (int): number of input features
        c_hidden_dim (int): hidden dimension of the CVQGAN MLP
        c_latent_dim (int): individual code dimension for CVQGAN
    """
    def __init__(
        self,
        c_latent_dim: int,
        c_input_dim: int,
        c_hidden_dim: int,
        c_fmap_dim: int
    ):
        super().__init__()

        self.model = nn.Sequential(
            LinearCombo(c_latent_dim*c_fmap_dim**2, c_hidden_dim),
            LinearCombo(c_hidden_dim, c_hidden_dim),
            nn.Linear(c_hidden_dim, c_input_dim)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x.contiguous().view(len(x), -1))


class Discriminator(nn.Module):
    """PatchGAN-style discriminator.

    Adapted from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538

    Parameters:
        num_filters_last: Number of filters in the last conv layer.
        n_layers: Number of convolutional layers.
        image_channels: Number of channels in the input image.
        image_size: Spatial size (H=W) of the input image.
    """

    def __init__(
        self,
        *,
        num_filters_last: int = 64,
        n_layers: int = 3,
        image_channels: int = 1,
        image_size: int = 128,
    ) -> None:
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

        # Adapter for CVQGAN latent vectors → image
        self.cvqgan_adapter = nn.Sequential(
            nn.Linear(image_channels, image_size),
            nn.ReLU(inplace=True),
            nn.Linear(image_size, image_channels * image_size**2),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (image_channels, image_size, image_size)),
        )

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


    def forward(self, x: th.Tensor, *, is_cvqgan: bool = False) -> th.Tensor:
        """Forward pass with optional CVQGAN adapter."""
        if is_cvqgan:
            x = self.cvqgan_adapter(x)
        return self.model(x)


class ScalingLayer(nn.Module):
    """Channel-wise affine normalization used by LPIPS."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", th.tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", th.tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, x: th.Tensor) -> th.Tensor:
        return (x - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """1x1 conv with dropout (per-layer LPIPS linear head)."""

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)


class VGG16(nn.Module):
    """Torchvision VGG16 feature extractor sliced at LPIPS tap points."""

    def __init__(self) -> None:
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
    ) -> None:
        super().__init__()
        self.use_raw = use_raw
        self.clamp_output = clamp_output
        self.robust_clamp = robust_clamp
        self.warn_on_clamp = warn_on_clamp
        self._logger = logger

        self.scaling_layer = ScalingLayer()
        self.channels = (64, 128, 256, 512, 512)
        self.vgg = VGG16()
        self.lins = nn.ModuleList([NetLinLayer(c) for c in self.channels])

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

        # Promote greyscale → RGB for VGG features
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
            parts = [self._spatial_average(self.lins[i](d)) for i, d in enumerate(diffs)]

        loss = th.stack(parts, dim=0).sum()
        if self.clamp_output:
            loss = th.clamp(loss, min=0.0)
        return loss

    # Helpers
    @staticmethod
    def _norm_tensor(x: th.Tensor) -> th.Tensor:
        """L2-normalize channels per spatial location: BxCxHxW → BxCxHxW."""
        norm = th.sqrt(th.sum(x**2, dim=1, keepdim=True))
        return x / (norm + 1e-10)

    @staticmethod
    def _spatial_average(x: th.Tensor) -> th.Tensor:
        """Average over spatial dimensions with dims kept: BxCxHxW → BxCx1x1."""
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

