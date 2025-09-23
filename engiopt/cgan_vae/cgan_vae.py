"""3D Multiview VAE-GAN - Corrected implementation with proper VAE training.

Created by Christophe Hatterer
Based on https://github.com/bryonkucharski/Multiview-3D-VAE-GAN
Algorithm Based off: https://arxiv.org/pdf/1512.09300
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

from engiopt.metrics import dpp_diversity
from engiopt.metrics import mmd
import wandb


@dataclass
class Args:
    """Command-line arguments for 3D Multiview VAE-GAN."""

    problem_id: str = "heatconduction3d"
    """Problem identifier for 3D engineering design."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this algorithm."""
    export: bool = False
    """Export 3d Volume"""

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

    # Algorithm specific
    n_epochs: int = 300
    """number of epochs of training"""
    batch_size: int = 8
    """size of the batches"""
    lr_gen: float = 0.0025
    """learning rate for the generator"""
    lr_disc: float = 10e-5
    """learning rate for the discriminator"""
    lr_enc: float = 0.001
    """learning rate for the encoder"""
    b1: float = 0.5
    """decay of first order momentum of gradient"""
    b2: float = 0.999
    """decay of first order momentum of gradient"""
    n_cpu: int = 8
    """number of cpu threads to use during batch generation"""
    latent_dim: int = 64
    """dimensionality of the latent space"""
    sample_interval: int = 800
    """interval between volume samples"""

    # VAE specific parameters
    kl_weight: float = 0.01
    """Weight for KL divergence loss"""
    recon_weight: float = 1.0
    """Weight for reconstruction loss"""

    # Multiview parameters
    n_slices: int = 9
    """Number of slices to extract from each volume for training"""

    # Training Updates
    gen_iters: int = 1
    """Number of generator updates per batch"""
    discrim_iters: int = 1
    """Number of discriminator updates per batch"""
    # metrics
    mmd_sigma = 10.0
    """Sigma value for MMD calculations"""
    dpp_sigma = 10.0
    """Sigma value for DPP Calculations"""


def visualize_3d_designs(
    volumes: th.Tensor, conditions: th.Tensor, condition_names: list, save_path: str, max_designs: int = 9
):
    """Visualize 3D volumes by showing cross-sectional slices with a red-white heatmap.

    Args:
        volumes: (N, 1, D, H, W) tensor of 3D designs
        conditions: (N, n_conds) tensor of conditions
        condition_names: List of condition names
        save_path: Path to save the visualization
        max_designs: Maximum number of designs to visualize
    """
    n_designs = min(len(volumes), max_designs)
    volumes = volumes[:n_designs]
    conditions = conditions[:n_designs]

    # Create subplot grid (3 slices per design)
    rows = n_designs
    cols = 3  # XY, XZ, YZ slices
    _, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_designs):
        vol = volumes[i, 0].cpu().numpy()  # Remove channel dimension
        d, h, w = vol.shape  # Use lowercase variable names

        # Use 'Reds_r' colormap: low=red, high=white
        cmap = plt.get_cmap("Reds_r")

        # XY slice (middle Z)
        axes[i, 0].imshow(vol[d // 2, :, :], cmap=cmap, vmin=-1, vmax=1)
        axes[i, 0].set_title(f"Design {i + 1} - XY slice (z={d // 2})")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # XZ slice (middle Y)
        axes[i, 1].imshow(vol[:, h // 2, :], cmap=cmap, vmin=-1, vmax=1)
        axes[i, 1].set_title(f"Design {i + 1} - XZ slice (y={h // 2})")
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        # YZ slice (middle X)
        axes[i, 2].imshow(vol[:, :, w // 2], cmap=cmap, vmin=-1, vmax=1)
        axes[i, 2].set_title(f"Design {i + 1} - YZ slice (x={w // 2})")
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

        # Add condition information as text
        cond_text = []
        for j, name in enumerate(condition_names):
            cond_text.append(f"{name}: {conditions[i, j]:.2f}")
        axes[i, 0].text(
            0.02,
            0.98,
            "\n".join(cond_text),
            transform=axes[i, 0].transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def reparameterize(mu: th.Tensor, logvar: th.Tensor) -> th.Tensor:
    """Reparameterization trick for VAE with numerical stability."""
    # Clamp logvar to prevent extreme values
    logvar = th.clamp(logvar, min=-20, max=20)
    std = th.exp(0.5 * logvar)
    eps = th.randn_like(std)
    return mu + eps * std


def kl_divergence(
    mu: th.Tensor,
    logvar: th.Tensor,
) -> th.Tensor:
    """Calculate KL divergence for VAE with numerical stability."""
    # Clamp logvar to prevent extreme values
    logvar = th.clamp(logvar, min=-20, max=20)
    return -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp())


class Encoder(nn.Module):
    """2D Encoder for VAE - processes 2D slices and produces latent mean and log-variance."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 64,
        num_filters: list[int] | None = None,
    ):
        if num_filters is None:
            num_filters = [32, 64, 128, 256, 512]
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            # Assuming input is around 51x51, we need to handle this size
            # 51 → 25 (with padding to make it work)
            nn.Conv2d(in_channels, num_filters[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(inplace=False),
            # 25 → 12
            nn.Conv2d(num_filters[0], num_filters[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=False),
            # 12 → 6
            nn.Conv2d(num_filters[1], num_filters[2], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=False),
            # 6 → 3
            nn.Conv2d(num_filters[2], num_filters[3], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters[3]),
            nn.ReLU(inplace=False),
            # 3 → 1
            nn.Conv2d(num_filters[3], num_filters[4], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters[4]),
            nn.ReLU(inplace=False),
        )

        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        # Calculate flattened dimension after adaptive pooling
        self.flatten_dim = num_filters[4] * 2 * 2
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Encode input 2D slices to latent distribution parameters."""
        # x should be (B*n_slices, 1, H, W) where H, W are the slice dimensions

        h = self.encoder(x)
        # Use adaptive pooling to ensure consistent output size
        h = self.adaptive_pool(h)
        h = h.view(h.size(0), -1)  # Flatten

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Generator3D(nn.Module):
    """3D Conditional GAN generator that outputs volumetric designs.

    From noise + condition -> 3D volume (e.g., 64x64x64).

    Args:
        latent_dim (int): Dimensionality of the noise (latent) vector.
        n_conds (int): Number of conditional features.
        design_shape (tuple[int, int, int]): Target 3D design shape (D, H, W).
        num_filters (list of int): Number of filters in each upsampling stage.
        out_channels (int): Number of output channels (e.g., 1 for density).
    """

    def __init__(
        self,
        latent_dim: int,
        n_conds: int,
        design_shape: tuple[int, int, int],
        num_filters: list[int] | None = None,  # Extra layer for 3D
        out_channels: int = 1,
    ):
        if num_filters is None:
            num_filters = [512, 256, 128, 64, 32]
        super().__init__()
        self.design_shape = design_shape

        # Path for noise z - start with 4x4x4 volume
        self.z_path = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, num_filters[0] // 2, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(num_filters[0] // 2),
            nn.ReLU(inplace=True),
        )

        # Path for condition c - start with 4x4x4 volume
        self.c_path = nn.Sequential(
            nn.ConvTranspose3d(n_conds, num_filters[0] // 2, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(num_filters[0] // 2),
            nn.ReLU(inplace=True),
        )

        # Upsampling blocks: 4x4x4 -> 8x8x8 -> 16x16x16 -> 32x32x32 -> 64x64x64
        self.up_blocks = nn.Sequential(
            # 4x4x4 -> 8x8x8
            nn.ConvTranspose3d(num_filters[0], num_filters[1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_filters[1]),
            nn.ReLU(inplace=True),
            # 8x8x8 -> 16x16x16
            nn.ConvTranspose3d(num_filters[1], num_filters[2], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_filters[2]),
            nn.ReLU(inplace=True),
            # 16x16x16 -> 32x32x32
            nn.ConvTranspose3d(num_filters[2], num_filters[3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_filters[3]),
            nn.ReLU(inplace=True),
            # 32x32x32 -> 64x64x64
            nn.ConvTranspose3d(num_filters[3], num_filters[4], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_filters[4]),
            nn.ReLU(inplace=True),
            # Final conv without changing spatial size
            nn.Conv3d(num_filters[4], out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(self, z: th.Tensor, c: th.Tensor) -> th.Tensor:
        """Forward pass for the 3D Generator.

        Args:
            z: (B, z_dim, 1, 1, 1) - noise vector
            c: (B, cond_features, 1, 1, 1) - condition vector
        Returns:
            out: (B, out_channels, D, H, W) - 3D design
        """
        # Run noise & condition through separate stems
        z = z.view(z.size(0), z.size(1), 1, 1, 1)  # (B, latent_dim, 1, 1, 1)
        c = c.view(c.size(0), c.size(1), 1, 1, 1)  # (B, n_conds, 1, 1, 1)

        z_feat = self.z_path(z)  # -> (B, num_filters[0]//2, 4, 4, 4)
        c_feat = self.c_path(c)  # -> (B, num_filters[0]//2, 4, 4, 4)

        # Concat along channel dimension
        x = th.cat([z_feat, c_feat], dim=1)  # (B, num_filters[0], 4, 4, 4)

        # Upsample through the main blocks
        return self.up_blocks(x)  # -> (B, out_channels, 128, 128, 128)


class Discriminator3D(nn.Module):
    """3D Conditional GAN discriminator for volumetric designs.

    Takes 3D volumes + conditions and outputs real/fake score.

    Args:
        n_conds (int): Number of conditional channels.
        in_channels (int): Number of input volume channels.
        num_filters (list of int): Number of filters in each downsampling stage.
        out_channels (int): Typically 1 for real/fake score.
    """

    def __init__(
        self,
        n_conds: int,
        in_channels: int = 1,
        num_filters: list[int] | None = None,  # Extra layer for 3D
        out_channels: int = 1,
    ):
        if num_filters is None:
            num_filters = [32, 64, 128, 256, 512]
        super().__init__()

        # Path for 3D design volume
        self.vol_path = nn.Sequential(
            nn.Conv3d(in_channels, num_filters[0] // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Path for condition (expand to match volume size)
        self.cond_path = nn.Sequential(
            nn.Conv3d(n_conds, num_filters[0] // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Downsampling blocks: 64x64x64 -> 32x32x32 -> 16x16x16 -> 8x8x8 -> 4x4x4 -> 1x1x1
        self.down_blocks = nn.Sequential(
            # 32x32x32 -> 16x16x16
            nn.Conv3d(num_filters[0], num_filters[1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_filters[1]),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16x16 -> 8x8x8
            nn.Conv3d(num_filters[1], num_filters[2], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_filters[2]),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8x8 -> 4x4x4
            nn.Conv3d(num_filters[2], num_filters[3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_filters[3]),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4x4 -> 2x2x2
            nn.Conv3d(num_filters[3], num_filters[4], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_filters[4]),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final classification layer
        self.final_conv = nn.Sequential(
            nn.Conv3d(num_filters[4], out_channels, kernel_size=2, stride=1, padding=0, bias=False),
        )

    def forward(self, x: th.Tensor, c: th.Tensor) -> th.Tensor:
        """Forward pass for the 3D Discriminator.

        Args:
            x: (B, in_channels, D, H, W) - 3D design volume
            c: (B, cond_features, 1, 1, 1) - condition vector
        Returns:
            out: (B, out_channels, 1, 1, 1) - real/fake score
        """
        # Expand conditions to match volume spatial dimensions
        c_expanded = c.expand(-1, -1, *x.shape[2:])  # (B, n_conds, D, H, W)

        # Process volume and conditions through separate stems
        x_feat = self.vol_path(x)  # (B, num_filters[0]//2, 32, 32, 32)
        c_feat = self.cond_path(c_expanded)  # (B, num_filters[0]//2, 32, 32, 32)

        # Concat along channel dimension
        h = th.cat([x_feat, c_feat], dim=1)  # (B, num_filters[0], 32, 32, 32)

        # Downsample through blocks
        h = self.down_blocks(h)  # -> (B, num_filters[4], 2, 2, 2)

        # Final classification
        return self.final_conv(h)  # -> (B, out_channels, 1, 1, 1)


def compute_gradient_penalty(discriminator, real_samples, fake_samples, conds, device, lambda_gp=20.0):  # noqa: PLR0913
    """Calculates the gradient penalty loss for WGAN GP."""
    batch_size = real_samples.size(0)
    # Random weight term for interpolation between real and fake samples
    alpha = th.rand(batch_size, 1, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    # Get interpolated samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(requires_grad=True)
    d_interpolates = discriminator(interpolates, conds)
    # For multi-dimensional output, take mean
    fake = th.ones_like(d_interpolates, device=device, requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    return lambda_gp * ((gradient_norm - 1) ** 2).mean()


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Load problem
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    # Extract 3D design space information
    design_shape = problem.design_space.shape
    DESIGN_SHAPE_LEN = 3
    if len(design_shape) != DESIGN_SHAPE_LEN:
        raise ValueError(f"Expected 3D design shape, got {design_shape}")

    conditions = problem.conditions_keys
    n_conds = len(conditions)
    condition_names = [cond[0] for cond in conditions]

    # Setup logging
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), save_code=True, name=run_name)

    # Seeding
    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True

    os.makedirs("images_3d", exist_ok=True)

    # Device
    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    # Enable anomaly detection for debugging
    th.autograd.set_detect_anomaly(True)

    print(f"Using device: {device}")
    print(f"3D Design shape: {design_shape}")
    print(f"Number of conditions: {n_conds}")

    # Loss functions
    reconstruction_loss = th.nn.MSELoss()

    # Initialize models
    encoder = Encoder(latent_dim=args.latent_dim)
    generator = Generator3D(latent_dim=args.latent_dim, n_conds=n_conds, design_shape=design_shape)
    discriminator = Discriminator3D(n_conds)

    # Move to device
    encoder.to(device)
    generator.to(device)
    discriminator.to(device)
    reconstruction_loss.to(device)

    # Print parameters
    enc_params = sum(p.numel() for p in encoder.parameters())
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Encoder parameters: {enc_params:,}")
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")

    # Configure data loader
    training_ds = problem.dataset.with_format("torch", device=device)["train"]

    # Extract 3d designs and conditions
    designs_3d = training_ds["optimal_design"]
    condition_tensors = [training_ds[key] for key in problem.conditions_keys]

    training_ds = th.utils.data.TensorDataset(designs_3d, *condition_tensors)
    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Optimizers for all components
    optimizer_encoder = th.optim.Adam(encoder.parameters(), lr=args.lr_enc, betas=(args.b1, args.b2))
    optimizer_generator = th.optim.Adam(generator.parameters(), lr=args.lr_gen, betas=(args.b1, args.b2))
    optimizer_discriminator = th.optim.Adam(discriminator.parameters(), lr=args.lr_disc, betas=(args.b1, args.b2))

    @th.no_grad()
    def sample_3d_designs(n_designs: int) -> tuple[th.Tensor, th.Tensor]:
        """Sample designs from trained generator."""
        encoder.eval()
        generator.eval()

        # Sample noise with proper 5d shape
        z = th.randn((n_designs, args.latent_dim, 1, 1, 1), device=device)

        # Create condition grid
        all_conditions = th.stack(condition_tensors, dim=1)
        linspaces = [
            th.linspace(all_conditions[:, i].min(), all_conditions[:, i].max(), n_designs, device=device)
            for i in range(all_conditions.shape[1])
        ]
        desired_conds = th.stack(linspaces, dim=1)

        gen_volumes = generator(z, desired_conds.reshape(-1, n_conds, 1, 1, 1))

        encoder.train()
        generator.train()
        return desired_conds, gen_volumes

    # Training loop
    print("Starting 3D VAE-GAN training...")

    mmd_values = []
    dpp_values = []

    for epoch in tqdm.trange(args.n_epochs):
        for i, data in enumerate(dataloader):
            # Extract 3D Designs and conditions
            designs_3d = data[0]  # (B,D,H,W)
            original_shape = designs_3d.shape

            # Add channel dimension: (B, D, H, W) -> (B, 1, D, H, W)
            CHANNEL_DIM = 4
            if len(original_shape) == CHANNEL_DIM:
                designs_3d = designs_3d.unsqueeze(1)  # (B, 1, D, H, W)

            condition_data = data[1:]
            conds = th.stack(condition_data, dim=1)
            conds_expanded = conds.reshape(-1, n_conds, 1, 1, 1)

            batch_size, _, D, H, W = designs_3d.shape

            # --- Extract a subset of XY slices from each volume ---
            n_slices = min(args.n_slices, D)  # You can set n_slices as desired
            slice_indices = th.linspace(0, D - 1, n_slices).long()
            slices = designs_3d[:, :, slice_indices, :, :]  # (B, 1, n_slices, H, W)
            slices = slices.permute(0, 2, 1, 3, 4)  # (B, n_slices, 1, H, W)
            slices = slices.reshape(-1, 1, H, W)  # (B*n_slices, 1, H, W)

            # Pass all slices through encoder
            mu, logvar = encoder(slices)  # (B*D, latent_dim)

            # Compute mean KL divergence over all slices
            mvkl_loss = kl_divergence(mu, logvar) / mu.size(0)  # mu.size(0) == B*D

            # Reshape for aggregation if needed
            mu = mu.view(batch_size, n_slices, -1)  # (B, n_slices, latent_dim)
            logvar = logvar.view(batch_size, n_slices, -1)  # (B, n_slices, latent_dim)

            # Average latent vectors for each volume - Mean Pooling
            mu = mu.mean(dim=1)  # (B, latent_dim)
            logvar = logvar.mean(dim=1)  # (B, latent_dim)

            z_encoded = reparameterize(mu, logvar)

            # Move to device and add channel dimension
            conds = conds.to(device)

            # Reconstruct
            reconstructed = generator(z_encoded, conds)

            if reconstructed.shape[2:] == (51, 51, 51):
                reconstructed = functional.pad(reconstructed, (6, 7, 6, 7, 6, 7), mode="constant", value=0)

            if designs_3d.shape[2:] == (51, 51, 51):
                designs_3d = functional.pad(designs_3d, (6, 7, 6, 7, 6, 7), mode="constant", value=0)
            designs_3d = designs_3d.to(device)

            # VAE losses
            # Create a mask: 1 for real data, 0 for padding
            mask = th.ones_like(designs_3d)
            if original_shape == (51, 51, 51):
                mask = functional.pad(
                    th.ones((batch_size, 1, 51, 51, 51), device=designs_3d.device),
                    (6, 7, 6, 7, 6, 7),
                    mode="constant",
                    value=0,
                )

            # Compute masked loss
            recon_loss = ((reconstructed - designs_3d) ** 2 * mask).sum() / mask.sum()

            # ==================
            # Train Encoder
            # ==================
            optimizer_encoder.zero_grad()
            E_loss = args.recon_weight * recon_loss + args.kl_weight * mvkl_loss
            E_loss.backward(retain_graph=True)
            th.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer_encoder.step()

            # ==================
            # Train Generator
            # ==================
            for _ in range(args.gen_iters):
                # Re-compute reconstruction for generator (to avoid shared graph issues)
                z_encoded_gen = reparameterize(mu.detach(), logvar.detach())
                reconstructed_gen = generator(z_encoded_gen, conds)
                if reconstructed_gen.shape[2:] == (51, 51, 51):
                    reconstructed_gen = functional.pad(reconstructed_gen, (6, 7, 6, 7, 6, 7), mode="constant", value=0)
                recon_loss_gen = reconstruction_loss(reconstructed_gen, designs_3d)

                # GAN loss for generator
                optimizer_generator.zero_grad()
                g_loss_ws = -discriminator(reconstructed_gen, conds_expanded).mean()
                g_loss = g_loss_ws + args.recon_weight * recon_loss_gen
                g_loss.backward()
                th.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_generator.step()

            # ==================
            # Train Discriminator
            # ==================
            for _ in range(args.discrim_iters):
                optimizer_discriminator.zero_grad()
                # Wasserstein discriminator loss
                real_validity = discriminator(designs_3d, conds_expanded)
                fake_validity = discriminator(reconstructed_gen.detach(), conds_expanded)
                # Random noise generation
                z_random = th.randn((batch_size, args.latent_dim), device=device)
                fake_designs = generator(z_random, conds)
                if fake_designs.shape[2:] == (51, 51, 51):
                    fake_designs = functional.pad(fake_designs, (6, 7, 6, 7, 6, 7), mode="constant", value=0)
                fake_validity_random = discriminator(fake_designs.detach(), conds_expanded)
                # Wasserstein loss
                d_loss = -(real_validity.mean() - (fake_validity.mean() + fake_validity_random.mean()) / 2)
                gradient_penalty = compute_gradient_penalty(
                    discriminator, designs_3d, reconstructed_gen.detach(), conds_expanded, device
                )
                d_loss += gradient_penalty
                d_loss.backward()
                th.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optimizer_discriminator.step()

            # --- Discriminator accuracy logging ---
            with th.no_grad():
                real_acc = (real_validity > 0).float().mean().item()
                fake_acc = (fake_validity < 0).float().mean().item()
                fake_rand_acc = (fake_validity_random < 0).float().mean().item()
                disc_acc = 0.5 * (fake_acc + fake_rand_acc)

            # After reconstructed and designs_3d are available and on CPU:
            mmd_value = None
            dpp_value = None
            STEPS = 100
            if i % STEPS / 2 == 0:
                # Generate multiple diverse samples for both MMD and DPP calculation
                generator.eval()
                encoder.eval()
                with th.no_grad():
                    n_samples = 50  # Generate more samples for meaningful comparison
                    # Generate samples for comparison
                    generated_volumes: list[np.ndarray] = []
                    real_volumes: list[np.ndarray] = []
                    for _ in range(n_samples // batch_size + 1):
                        current_batch_size = min(batch_size, n_samples - len(generated_volumes) * batch_size)
                        if current_batch_size <= 0:
                            break
                        # Generate diverse samples with different random noise
                        z_diverse = th.randn((current_batch_size, args.latent_dim, 1, 1, 1), device=device)
                        # Use same conditions for fair comparison
                        conds_diverse = conds[:current_batch_size]
                        diverse_vol = generator(z_diverse, conds_diverse.reshape(-1, n_conds, 1, 1, 1))
                        generated_volumes.append(diverse_vol.detach().cpu().numpy())
                        # Also collect real samples for MMD comparison
                        real_vol = designs_3d[:current_batch_size]
                        real_volumes.append(real_vol.detach().cpu().numpy())
                    # Concatenate all samples
                    all_generated_volumes = np.concatenate(generated_volumes, axis=0)[:n_samples]
                    all_real_volumes = np.concatenate(real_volumes, axis=0)[:n_samples]
                    # Reshape for metric calculations
                    gen_np = all_generated_volumes.reshape(all_generated_volumes.shape[0], -1)
                    real_np = all_real_volumes.reshape(all_real_volumes.shape[0], -1)
                    # Compute MMD between generated and real sample sets
                    mmd_value = mmd(gen_np, real_np, sigma=args.mmd_sigma)
                    # Compute DPP on the generated set
                    dpp_value = dpp_diversity(gen_np, sigma=args.dpp_sigma)
                generator.train()
                encoder.train()
                try:
                    mmd_value = float(mmd_value)
                except (ValueError, TypeError):
                    mmd_value = float("nan")
                if mmd_value is not None:
                    mmd_values.append(mmd_value)
                if dpp_value is not None:
                    dpp_values.append(dpp_value)

            # ----------
            #  Logging
            # ----------
            batches_done = epoch * len(dataloader) + i

            if args.track:
                log_dict = {
                    "E_loss": E_loss.item(),
                    "recon_loss": recon_loss.item(),
                    "g_loss": g_loss.item(),
                    "mvkl_loss": mvkl_loss.item(),
                    "g_loss_ws": g_loss_ws.item(),
                    "d_loss": d_loss.item(),
                    "real_validity": real_validity.mean().item(),
                    "fake_validity": fake_validity.mean().item(),
                    "disc_real_acc": real_acc,
                    "disc_fake_acc": disc_acc,
                    "epoch": epoch,
                    "batch": batches_done,
                }

                if mmd_value is not None:
                    log_dict["mmd"] = mmd_value
                if dpp_value is not None:
                    log_dict["dpp_diversity"] = dpp_value
                    # Log the natural logarithm of DPP for optimization
                    dpp_float = float(dpp_value)
                    tiny_float = float(np.finfo(np.float64).tiny)
                    max_val = max(dpp_float, tiny_float)
                    log_dict["log_dpp"] = np.log(max_val)

                wandb.log(log_dict)

                if i % 10 == 0:
                    print(
                        f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] "
                        f"[E loss: {E_loss.item():.4f}] [D loss: {d_loss.item():.4f}] "
                        f"[Recon: {recon_loss.item():.4f}] [KL: {mvkl_loss.item():.4f}]"
                    )

                if batches_done % args.sample_interval == 0:
                    print("Generating 3D design samples...")

                    desired_conds, volumes_3d = sample_3d_designs(9)

                    img_fname = f"images_3d/{batches_done}.png"
                    visualize_3d_designs(volumes_3d, desired_conds, condition_names, img_fname, max_designs=9)

                    wandb.log({"3d_designs": wandb.Image(img_fname), "sample_step": batches_done})

            # Memory cleanup
            if i % 50 == 0:
                th.cuda.empty_cache() if device.type == "cuda" else None

        # -------------
        # Save models
        # -------------
        if args.save_model and epoch == args.n_epochs - 1:
            print("Saving VAE models...")

            # Save all three models
            th.save(
                {
                    "epoch": epoch,
                    "encoder": encoder.state_dict(),
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optimizer_encoder": optimizer_encoder.state_dict(),
                    "optimizer_generator": optimizer_generator.state_dict(),
                    "optimizer_discriminator": optimizer_discriminator.state_dict(),
                    "args": vars(args),
                    "design_shape": design_shape,
                    "n_conds": n_conds,
                },
                "multiview_3d_vaegan.pth",
            )

            if args.track:
                artifact = wandb.Artifact(f"{args.problem_id}_{args.algo}_models", type="model")
                artifact.add_file("multiview_3d_vaegan.pth")
                wandb.log_artifact(artifact, aliases=[f"seed_{args.seed}"])

            print("3D vae models saved successfully!")

    if args.track:
        if mmd_values:
            final_mmd = np.mean(mmd_values[-10:])
            wandb.log({"mmd": final_mmd, "epoch": args.n_epochs})
        if dpp_values:
            # Use last 100 values if available, otherwise use all
            window = 100 if len(dpp_values) >= 500 else len(dpp_values[-400:])  # noqa: PLR2004
            recent_dpp_values = np.array(dpp_values[-window:])
            # Filter outliers using IQR method
            q1 = np.percentile(recent_dpp_values, 25)
            q3 = np.percentile(recent_dpp_values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            # Keep only values within the IQR bounds
            filtered_dpp_values = recent_dpp_values[(recent_dpp_values >= lower_bound) & (recent_dpp_values <= upper_bound)]
            # Use filtered values if we have enough, otherwise use original
            if len(filtered_dpp_values) >= max(10, len(recent_dpp_values) * 0.5):  # Keep at least 50% or 10 values
                final_dpp = np.mean(filtered_dpp_values)
                print(f"Filtered {len(recent_dpp_values) - len(filtered_dpp_values)} outliers from DPP values")
            else:
                final_dpp = np.mean(recent_dpp_values)
                print("Not enough values after outlier filtering, using original mean")
            wandb.log({"dpp": final_dpp, "epoch": args.n_epochs})
            # Also log the final log_dpp value - handle machine precision values
            wandb.log({"log_dpp": np.log(max(final_dpp, np.finfo(np.float64).tiny)), "epoch": args.n_epochs})
        wandb.finish()

    print("Multiview 3D VAE-GAN training completed!")

    if args.export:
        # ---- Export final generated 3D designs for ParaView ----
        print("Exporting final generated 3D designs for ParaView...")
        generator.eval()
        n_export = 8  # Number of designs to export
        z = th.randn((n_export, args.latent_dim, 1, 1, 1), device=device)
        all_conditions = th.stack(condition_tensors, dim=1)
        linspaces = [
            th.linspace(all_conditions[:, i].min(), all_conditions[:, i].max(), n_export, device=device)
            for i in range(all_conditions.shape[1])
        ]
        export_conds = th.stack(linspaces, dim=1)
        gen_volumes = generator(z, export_conds.reshape(-1, n_conds, 1, 1, 1))
        gen_volumes_np = gen_volumes.squeeze(1).detach().cpu().numpy()

        os.makedirs("paraview_exports", exist_ok=True)
        for i, vol in enumerate(gen_volumes_np):
            np.save(f"paraview_exports/vae_gen3d_{i}.npy", vol)
            # Optionally: save as .vti using pyvista or pyevtk if you want native VTK format

        print(f"Saved {n_export} generated 3D designs to paraview_exports/ as .npy files.")
