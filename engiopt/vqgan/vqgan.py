# ruff: noqa: F401 # REMOVE THIS LATER
"""Vector Quantized Generative Adversarial Network (VQGAN).

Based on https://github.com/dome272/VQGAN-pytorch with an "Online Clustered Codebook" for better codebook usage from https://github.com/lyndonzheng/CVQ-VAE/blob/main/quantise.py

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

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import autograd
from torch import nn
from torch.nn import functional
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
