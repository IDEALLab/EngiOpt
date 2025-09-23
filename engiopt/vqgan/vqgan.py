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

from dataclasses import dataclass
import os
import random
import time

from einops import rearrange
from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import autograd
from torch import nn
from torch.nn import functional as f
import tqdm
import tyro
import wandb

from engiopt.metrics import dpp_diversity
from engiopt.metrics import mmd
from engiopt.transforms import resize_to
from engiopt.transforms import upsample_nearest


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
    encoder_channels: tuple = (128, 128, 128, 256, 256, 512)
    """list of channel sizes for each encoder layer"""
    encoder_attn_resolutions: tuple = (16,)
    """list of resolutions at which to apply attention in the encoder"""
    encoder_num_res_blocks: int = 2
    """number of residual blocks per encoder layer"""
    encoder_start_resolution: int = 256
    """starting resolution for the encoder"""
    decoder_channels: tuple = (512, 256, 256, 128, 128)
    """list of channel sizes for each decoder layer"""
    decoder_attn_resolutions: tuple = (16,)
    """list of resolutions at which to apply attention in the decoder"""
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

    num_embed: number of codebook entry
    embed_dim: dimensionality of codebook entry
    beta: weight for the commitment loss
    distance: distance for looking up the closest code
    anchor: anchor sampled methods
    first_batch: if true, the offline version of our model
    contras_loss: if true, use the contras_loss to further improve the performance
    """
    def __init__(self, args):
        super().__init__()

        self.num_embed = args.c_num_codebook_vectors if args.is_c else args.num_codebook_vectors
        self.embed_dim = args.c_latent_dim if args.is_c else args.latent_dim
        self.beta = args.beta

        # Fixed parameters from the original implementation
        self.distance = "cos"
        self.anchor = "probrandom"
        self.first_batch = False
        self.contras_loss = False
        self.decay = 0.99
        self.init = False

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

        # quantise and unflatten
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

        # online clustered reinitialisation for unoptimized points
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

    This buffer enables us to initialize the codebook using a history of generated features rather than the ones produced by the latest encoders
    """
    def __init__(self, pool_size, dim=64):
        """Initialize the FeaturePool class.

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (th.rand((pool_size, dim)) * 2 - 1)/ pool_size

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
    """Group Normalization block to be used in VQGAN Encoder and Decoder."""
    def __init__(self, channels):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.gn(x)


class Swish(nn.Module):
    """Swish activation function to be used in VQGAN Encoder and Decoder."""
    def forward(self, x: th.Tensor) -> th.Tensor:
        return x * th.sigmoid(x)


class ResidualBlock(nn.Module):
    """Residual block to be used in VQGAN Encoder and Decoder."""
    def __init__(self, in_channels, out_channels):
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
    """Up-sampling block to be used in VQGAN Decoder."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = f.interpolate(x, scale_factor=2.0)
        return self.conv(x)


class DownSampleBlock(nn.Module):
    """Down-sampling block to be used in VQGAN Encoder."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x: th.Tensor) -> th.Tensor:
        pad = (0, 1, 0, 1)
        x = f.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class NonLocalBlock(nn.Module):
    """Non-local attention block to be used in VQGAN Encoder and Decoder."""
    def __init__(self, channels):
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
    """Regular fully connected layer combo for the CVQGAN if enabled."""
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(alpha)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)


class Encoder(nn.Module):
    """Encoder module for VQGAN.

    Consists of a series of convolutional, residual, and attention blocks arranged using the provided arguments.
    The number of downsample blocks is determined by the length of the encoder channels list minus two.
    For example, if encoder_channels=(128, 128, 128, 128) and the starting resolution is 128, the encoder will downsample the input image twice, from 128x128 to 32x32.
    """
    def __init__(self, args):
        super().__init__()
        channels = args.encoder_channels
        resolution = args.encoder_start_resolution
        layers = [nn.Conv2d(args.image_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for _ in range(args.encoder_num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in args.encoder_attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != len(channels)-2:
                layers.append(DownSampleBlock(channels[i+1]))
                resolution //= 2
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], args.latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)


class CondEncoder(nn.Module):
    """Simpler MLP-based encoder for the CVQGAN if enabled."""
    def __init__(self, args):
        super().__init__()
        self.c_fmap_dim = args.c_fmap_dim
        self.model = nn.Sequential(
            LinearCombo(args.c_input_dim, args.c_hidden_dim),
            LinearCombo(args.c_hidden_dim, args.c_hidden_dim),
            nn.Linear(args.c_hidden_dim, args.c_latent_dim*args.c_fmap_dim**2)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        encoded = self.model(x)
        s = encoded.shape
        return encoded.view(s[0], s[1]//self.c_fmap_dim**2, self.c_fmap_dim, self.c_fmap_dim)


class Decoder(nn.Module):
    """Decoder module for VQGAN.

    Consists of a series of convolutional, residual, and attention blocks arranged using the provided arguments.
    The number of upsample blocks is determined by the length of the decoder channels list minus one.
    For example, if decoder_channels=(128, 128, 128) and the starting resolution is 32, the decoder will upsample the input image twice, from 32x32 to 128x128.
    """
    def __init__(self, args):
        super().__init__()
        in_channels = args.decoder_channels[0]
        resolution = args.decoder_start_resolution
        layers = [nn.Conv2d(args.latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels),
                  NonLocalBlock(in_channels),
                  ResidualBlock(in_channels, in_channels)]

        for i in range(len(args.decoder_channels)):
            out_channels = args.decoder_channels[i]
            for _ in range(args.decoder_num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in args.decoder_attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))

            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, args.image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)


class CondDecoder(nn.Module):
    """Simpler MLP-based decoder for the CVQGAN if enabled."""
    def __init__(self, args):
        super().__init__()

        self.model = nn.Sequential(
            LinearCombo(args.c_latent_dim*args.c_fmap_dim**2, args.c_hidden_dim),
            LinearCombo(args.c_hidden_dim, args.c_hidden_dim),
            nn.Linear(args.c_hidden_dim, args.c_input_dim)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x.contiguous().view(len(x), -1))
