"""3D SliceGAN Main Script - Generate 3D volumes by learning 2D slice distributions.

Created by Christophe Hatterer
Based on SliceGAN: https://arxiv.org/abs/2102.07708
Extended from the original cDCGAN framework to handle 3D volumetric engineering designs
through 2D slice generation.
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

# --- PLR2004: Replace magic value 2 with a constant ---
SLICE_AXIS_W = 2
DESIGN_SHAPE_LEN = 3
CHANNEL_DIM = 4


@dataclass
class Args:
    """Command-line arguments for 3D SliceGAN."""

    problem_id: str = "heatconduction3d"  # Assume 3D problem
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
    b1: float = 0.5
    """decay of first order momentum of gradient"""
    b2: float = 0.99
    """decay of first order momentum of gradient"""
    n_cpu: int = 8
    """number of cpu threads to use during batch generation"""
    latent_dim: int = 64
    """dimensionality of the latent space"""
    sample_interval: int = 800
    """interval between volume samples"""

    # SliceGAN specific parameters
    discrim_iters: int = 1
    """Discriminator update"""
    gen_iters: int = 1
    """Generator update"""
    slice_sampling_rate: float = 0.3
    """Fraction of slices to sample during training"""
    use_all_axes: bool = True
    """Use slices from all three axes (XY, XZ, YZ)"""

    # metrics
    mmd_sigma = 10.0
    """Sigma value for MMD calculations"""
    dpp_sigma = 10.0
    """Sigma value for DPP Calculations"""


def extract_random_slices(volumes: th.Tensor, axis: int, n_slices: int | None = None) -> tuple[th.Tensor, th.Tensor]:
    """Extract random slices from 3D volumes and resize to consistent dimensions.

    Args:
        volumes: (B, C, D, H, W) tensor of 3D volumes
        axis: 0=D, 1=H, 2=W (which spatial dimension to slice along)
        n_slices: Optional number of slices to extract per volume
    """
    b, c, d, h, w = volumes.shape  # Use lowercase variable names

    if axis == 0:  # XY slices (slice along D dimension)
        if n_slices is None or n_slices >= d:
            positions = th.arange(d, device=volumes.device, dtype=th.float32)
            positions = positions / max(d - 1, 1)  # Avoid division by zero
            slices = volumes.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
            positions = positions.repeat(b)
        else:
            # Sample random slices efficiently
            pos_indices = th.randint(0, d, (b, n_slices), device=volumes.device)
            positions = pos_indices.float() / max(d - 1, 1)

            # Efficient extraction using advanced indexing
            batch_indices = th.arange(b, device=volumes.device).unsqueeze(1).expand(-1, n_slices)
            slices = volumes[batch_indices.flatten(), :, pos_indices.flatten(), :, :]
            positions = positions.flatten()

    elif axis == 1:  # XZ slices (slice along H dimension)
        if n_slices is None or n_slices >= h:
            positions = th.arange(h, device=volumes.device, dtype=th.float32)
            positions = positions / max(h - 1, 1)
            slices = volumes.permute(0, 3, 1, 2, 4).reshape(b * h, c, d, w)
            positions = positions.repeat(b)
        else:
            pos_indices = th.randint(0, h, (b, n_slices), device=volumes.device)
            positions = pos_indices.float() / max(h - 1, 1)

            batch_indices = th.arange(b, device=volumes.device).unsqueeze(1).expand(-1, n_slices)
            slices = volumes[batch_indices.flatten(), :, :, pos_indices.flatten(), :]
            positions = positions.flatten()

    elif axis == SLICE_AXIS_W:  # YZ slices (slice along W dimension)
        if n_slices is None or n_slices >= w:
            positions = th.arange(w, device=volumes.device, dtype=th.float32)
            positions = positions / max(w - 1, 1)
            slices = volumes.permute(0, 4, 1, 2, 3).reshape(b * w, c, d, h)
            positions = positions.repeat(b)
        else:
            pos_indices = th.randint(0, w, (b, n_slices), device=volumes.device)
            positions = pos_indices.float() / max(w - 1, 1)

            batch_indices = th.arange(b, device=volumes.device).unsqueeze(1).expand(-1, n_slices)
            slices = volumes[batch_indices.flatten(), :, :, :, pos_indices.flatten()]
            positions = positions.flatten()

    return slices, positions


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
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_designs):
        vol = volumes[i, 0].cpu().numpy()  # Remove channel dimension
        d, h, w = vol.shape  # Use lowercase

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


class SliceGenerator3D(nn.Module):
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


class SliceDiscriminator2D(nn.Module):
    """2D Discriminator for slice classification with automatic resizing.

    Takes 2D slice + design conditions + slice position -> real/fake score
    Automatically resizes input to 64x64 regardless of input dimensions.
    """

    def __init__(
        self,
        n_design_conds: int,
        in_channels: int = 1,
        num_filters: list[int] | None = None,
        out_channels: int = 1,
        target_size: int = 64,  # Add target size parameter
    ):
        if num_filters is None:
            num_filters = [64, 128, 256, 512]
        super().__init__()

        self.target_size = target_size

        # Add 1 for slice position conditioning
        total_conds = n_design_conds + 1

        # Initial conv for image
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_filters[0] // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Initial conv for conditions (broadcast to match image size)
        self.cond_conv = nn.Sequential(
            nn.Conv2d(total_conds, num_filters[0] // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Downsampling blocks
        self.down_blocks = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(num_filters[0], num_filters[1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(num_filters[1], num_filters[2], kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8 -> 4x4
            nn.Conv2d(num_filters[2], num_filters[3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final classification
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_filters[3], out_channels, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, x: th.Tensor, design_conds: th.Tensor, slice_pos: th.Tensor) -> th.Tensor:
        """Args:
            x: (B, in_channels, H, W) input slices - can be any size
            design_conds: (B, n_design_conds, 1, 1) design conditions
            slice_pos: (B, 1, 1, 1) normalized slice position.

        Returns:
            out: (B, out_channels, 1, 1) real/fake score.
        """  # noqa: D205
        # Expand conditions to match image spatial dimensions
        all_conds = th.cat([design_conds, slice_pos], dim=1)  # (B, n_design_conds + 1, 1, 1)
        conds_expanded = all_conds.expand(-1, -1, *x.shape[2:])  # (B, n_design_conds + 1, H, W)

        # Process image and conditions
        x_feat = self.img_conv(x)  # (B, num_filters[0]//2, 32, 32)
        c_feat = self.cond_conv(conds_expanded)  # (B, num_filters[0]//2, 32, 32)

        # Combine features
        h = th.cat([x_feat, c_feat], dim=1)  # (B, num_filters[0], 32, 32)

        # Downsample
        h = self.down_blocks(h)  # (B, num_filters[3], 4, 4)

        # Final classification
        return self.final_conv(h)  # (B, out_channels, 1, 1)


def compute_gradient_penalty(  # noqa: PLR0913
    discriminator, real_samples, fake_samples, real_conds, fake_conds, real_pos, fake_pos, device, lambda_gp=10.0
):
    """Calculates the gradient penalty loss for WGAN GP."""
    # Random weight term for interpolation between real and fake samples
    alpha = th.rand(real_samples.size(0), 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)

    # Get interpolated samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(requires_grad=True)

    # Interpolate conditions between real and fake
    alpha_cond = th.rand(real_conds.size(0), 1, 1, 1, device=device)
    alpha_cond = alpha_cond.expand_as(real_conds)
    interpolated_conds = alpha_cond * real_conds + (1 - alpha_cond) * fake_conds

    # Interpolate positions between real and fake
    alpha_pos = th.rand(real_pos.size(0), 1, 1, 1, device=device)
    alpha_pos = alpha_pos.expand_as(real_pos)
    interpolated_pos = alpha_pos * real_pos + (1 - alpha_pos) * fake_pos

    # Get discriminator output for interpolated samples
    d_interpolates = discriminator(interpolates, interpolated_conds, interpolated_pos)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=th.ones_like(d_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Calculate gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    return lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Load 3D problem
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    # Extract 3D design space information
    design_shape = problem.design_space.shape  # Should be (D, H, W) for 3D
    if len(design_shape) != DESIGN_SHAPE_LEN:
        raise ValueError(f"Expected 3D design shape, got {design_shape}")

    conditions = problem.conditions
    n_conds = len(conditions)
    print(n_conds)
    condition_names = [cond[0] for cond in conditions]
    print(condition_names)

    # Setup Logging
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), save_code=True, name=run_name)

    # Seeding
    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True

    os.makedirs("images_slicegan", exist_ok=True)

    # Device selection
    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    print(f"Using device: {device}")
    print(f"3D Design shape: {design_shape}")
    print(f"Number of conditions: {n_conds}")

    # Initialize SliceGAN components
    D, H, W = design_shape

    slice_generator = SliceGenerator3D(latent_dim=args.latent_dim, n_conds=n_conds, design_shape=design_shape)

    # Create slice discriminators for different axes
    slice_discriminators = {}

    # XY slices (vary along Z)
    slice_discriminators["xy"] = SliceDiscriminator2D(n_design_conds=n_conds)

    # XZ slices (vary along Y)
    slice_discriminators["xz"] = SliceDiscriminator2D(n_design_conds=n_conds)

    # YZ slices (vary along X)
    slice_discriminators["yz"] = SliceDiscriminator2D(n_design_conds=n_conds)

    # Move to device
    slice_generator.to(device)
    for disc in slice_discriminators.values():
        disc.to(device)

    # Print model parameters
    gen_params = sum(p.numel() for p in slice_generator.parameters())
    disc_params = sum(sum(p.numel() for p in disc.parameters()) for disc in slice_discriminators.values())
    print(f"Slice Generator parameters: {gen_params:,}")
    print(f"Total Discriminator parameters: {disc_params:,}")

    # Configure data loader
    training_ds = problem.dataset.with_format("torch", device=device)["train"]

    # Extract 3D designs and conditions
    designs_3d = training_ds["optimal_design"]  # Should be (N, D, H, W)
    condition_tensors = [training_ds[key] for key in problem.conditions_keys]

    training_ds = th.utils.data.TensorDataset(designs_3d, *condition_tensors)
    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Create separate optimizers for generator and each discriminator
    optimizer_generator = th.optim.Adam(slice_generator.parameters(), lr=args.lr_gen, betas=(args.b1, args.b2))

    optimizer_discriminators = {}
    for axis_name, disc in slice_discriminators.items():
        optimizer_discriminators[axis_name] = th.optim.Adam(disc.parameters(), lr=args.lr_disc, betas=(args.b1, args.b2))

    @th.no_grad()
    def sample_3d_designs(n_designs: int) -> tuple[th.Tensor, th.Tensor]:
        """Sample n_designs^n_conditions 3D volumes from the SliceGAN."""
        slice_generator.eval()

        # Sample noise with proper 5d shape
        z = th.randn((n_designs, args.latent_dim, 1, 1, 1), device=device, dtype=th.float)

        # Create condition grid
        all_conditions = th.stack(condition_tensors, dim=1)
        linspaces = [
            th.linspace(all_conditions[:, i].min(), all_conditions[:, i].max(), n_designs, device=device)
            for i in range(all_conditions.shape[1])
        ]

        desired_conds = th.stack(linspaces, dim=1)

        # Generate 3D volumes
        gen_volumes = slice_generator(z, desired_conds.reshape(-1, n_conds, 1, 1, 1))

        slice_generator.train()
        return desired_conds, gen_volumes

    # ------------=
    # Training Loop
    # -------------
    print("Starting SliceGAN training...")

    last_disc_acc = dict.fromkeys(slice_discriminators.keys(), 0.0)  # Track discriminator accuracy per axis

    mmd_values = []
    dpp_values = []

    real_slices_dict: dict[str, th.Tensor] = {}
    slice_positions_dict: dict[str, th.Tensor] = {}
    slice_conds_dict: dict[str, th.Tensor] = {}

    for epoch in tqdm.trange(args.n_epochs):
        for i, batch in enumerate(dataloader):
            # Extract 3D designs and conditions
            designs_3d = batch[0]  # (B, D, H, W)

            # Add channel dimension: (B, D, H, W) -> (B, 1, D, H, W)
            if len(designs_3d.shape) == CHANNEL_DIM:
                designs_3d = designs_3d.unsqueeze(1)  # (B, 1, D, H, W)

            # Pad to (B, 1, 64, 64, 64) if needed
            if designs_3d.shape[2:] == (51, 51, 51):
                designs_3d = functional.pad(designs_3d, (6, 7, 6, 7, 6, 7), mode="constant", value=0)

            condition_data = batch[1:]  # List of condition tensors
            conds = th.stack(condition_data, dim=1)  # (B, n_conds)
            conds_expanded = conds.reshape(-1, n_conds, 1, 1, 1)

            # Move to device and add channel dimension to designs
            designs_3d = designs_3d.to(device)
            conds = conds.to(device)

            batch_size, _, D, H, W = designs_3d.shape

            # Calculate number of slices to sample per axis
            n_slices_xy = max(1, int(args.slice_sampling_rate * D))
            n_slices_xz = max(1, int(args.slice_sampling_rate * H))
            n_slices_yz = max(1, int(args.slice_sampling_rate * W))

            # Sample slices from the 3D designs
            real_slices = {}
            slice_positions = {}
            slice_conds = {}

            if "xy" in slice_discriminators:
                xy_slices, xy_positions = extract_random_slices(designs_3d, axis=0, n_slices=n_slices_xy)
                real_slices["xy"] = xy_slices
                slice_positions["xy"] = xy_positions
                slice_conds["xy"] = conds.repeat_interleave(n_slices_xy, dim=0)

            if "xz" in slice_discriminators:
                xz_slices, xz_positions = extract_random_slices(designs_3d, axis=1, n_slices=n_slices_xz)
                real_slices["xz"] = xz_slices
                slice_positions["xz"] = xz_positions
                slice_conds["xz"] = conds.repeat_interleave(n_slices_xz, dim=0)

            if "yz" in slice_discriminators:
                yz_slices, yz_positions = extract_random_slices(designs_3d, axis=2, n_slices=n_slices_yz)
                real_slices["yz"] = yz_slices
                slice_positions["yz"] = yz_positions
                slice_conds["yz"] = conds.repeat_interleave(n_slices_yz, dim=0)

            # ---------------------
            #  Train Discriminators
            # ---------------------
            d_losses = {}
            real_losses = {}
            fake_losses = {}

            for disc_iter in range(args.discrim_iters):  # This should be n_D from algorithm
                # For each axis a = 1, 2, 3 (Line 7)
                for axis_name, discriminator in slice_discriminators.items():
                    optimizer_discriminators[axis_name].zero_grad()

                    # Sample latent vector (Line 5) - MOVED INSIDE THE AXIS LOOP
                    z = th.randn((batch_size, args.latent_dim), device=device)
                    fake_volumes_tensor = slice_generator(z, conds_expanded)
                    fake_volumes: dict[str, th.Tensor] = {"generated": fake_volumes_tensor}

                    if fake_volumes_tensor.shape[2:] == (51, 51, 51):
                        fake_volumes_tensor = functional.pad(
                            fake_volumes_tensor, (6, 7, 6, 7, 6, 7), mode="constant", value=0
                        )

                    axis_idx = {"xy": 0, "xz": 1, "yz": 2}[axis_name]

                    # Determine number of slices for this axis
                    if axis_name == "xy":
                        n_slices = max(1, int(args.slice_sampling_rate * designs_3d.shape[2]))  # D
                    elif axis_name == "xz":
                        n_slices = max(1, int(args.slice_sampling_rate * designs_3d.shape[3]))  # H
                    elif axis_name == "yz":
                        n_slices = max(1, int(args.slice_sampling_rate * designs_3d.shape[4]))  # W

                    # Extract slices from real data
                    real_slices_tensor, real_positions_tensor = extract_random_slices(
                        designs_3d, axis=axis_idx, n_slices=n_slices
                    )
                    real_slice_conds = conds.repeat_interleave(n_slices, dim=0)

                    real_slices_dict[axis_name] = real_slices_tensor
                    slice_positions_dict[axis_name] = real_positions_tensor
                    slice_conds_dict[axis_name] = real_slice_conds

                    # Extract slices from fake data (Line 9)
                    fake_slices, fake_positions = extract_random_slices(
                        fake_volumes_tensor, axis=axis_idx, n_slices=n_slices
                    )
                    fake_slice_conds = conds.repeat_interleave(n_slices, dim=0)

                    # Format conditioning for discriminator
                    real_conds_formatted = real_slice_conds.unsqueeze(-1).unsqueeze(-1)
                    fake_conds_formatted = fake_slice_conds.unsqueeze(-1).unsqueeze(-1)
                    real_pos_formatted = real_positions_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    fake_pos_formatted = fake_positions.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                    # Discriminator outputs
                    out_real = discriminator(real_slices_tensor, real_conds_formatted, real_pos_formatted)
                    out_fake = discriminator(fake_slices, fake_conds_formatted, fake_pos_formatted)

                    # Gradient penalty
                    gradient_penalty = compute_gradient_penalty(
                        discriminator,
                        real_slices_tensor,
                        fake_slices,
                        real_conds_formatted,
                        fake_conds_formatted,
                        real_pos_formatted,
                        fake_pos_formatted,
                        device,
                    )

                    # Wasserstein loss with GP
                    d_loss = out_fake.mean() - out_real.mean() + gradient_penalty
                    if d_loss.dim() > 0:
                        d_loss = d_loss.mean()

                    d_loss.backward()
                    optimizer_discriminators[axis_name].step()

                    # Store losses (only from the last critic iteration for logging)
                    if disc_iter == args.discrim_iters - 1:
                        d_losses[axis_name] = d_loss.item()
                        real_losses[axis_name] = out_real.mean().item()
                        fake_losses[axis_name] = out_fake.mean().item()

                        with th.no_grad():
                            real_acc = (out_real > 0).float().mean()
                            fake_acc = (out_fake < 0).float().mean()
                            last_disc_acc[axis_name] = (real_acc + fake_acc) / 2

            # -----------------
            #  Train Generator
            # -----------------
            for _gen_iter in range(args.gen_iters):
                optimizer_generator.zero_grad()

                # Sample latent vector (Line 17)
                z = th.randn((batch_size, args.latent_dim), device=device)
                # Generate 3D volume (Line 18)
                fake_volumes_tensor = slice_generator(z, conds_expanded)
                fake_volumes = {"generated": fake_volumes_tensor}  # Or appropriate dict key

                total_g_loss = th.tensor(0.0, device=device, requires_grad=True)
                # For each axis a = 1, 2, 3 (Line 19)
                for axis_name, discriminator in slice_discriminators.items():
                    axis_idx = {"xy": 0, "xz": 1, "yz": 2}[axis_name]

                    # Determine number of slices for this axis
                    if axis_name == "xy":
                        n_slices = max(1, int(args.slice_sampling_rate * fake_volumes_tensor.shape[2]))
                    elif axis_name == "xz":
                        n_slices = max(1, int(args.slice_sampling_rate * fake_volumes_tensor.shape[3]))
                    elif axis_name == "yz":
                        n_slices = max(1, int(args.slice_sampling_rate * fake_volumes_tensor.shape[4]))

                    # Extract slices from generated volume (Line 21)
                    fake_slices, fake_positions = extract_random_slices(
                        fake_volumes_tensor, axis=axis_idx, n_slices=n_slices
                    )
                    fake_slice_conds = conds.repeat_interleave(n_slices, dim=0)

                    # Format conditioning
                    fake_conds_formatted = fake_slice_conds.unsqueeze(-1).unsqueeze(-1)
                    fake_pos_formatted = fake_positions.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                    # Generator loss (Line 22)
                    output = discriminator(fake_slices, fake_conds_formatted, fake_pos_formatted)
                    g_loss_axis = -output.mean()  # Negative because we want to maximize discriminator output

                    total_g_loss = total_g_loss + g_loss_axis

                # Update generator parameters (Line 23)
                total_g_loss.backward()
                optimizer_generator.step()

            # Compute MMD and DPP diversity every 100 batches
            mmd_value = None
            dpp_value = None
            if i % 100 == 0:
                gen_np = fake_volumes_tensor.detach().cpu().numpy().reshape(fake_volumes_tensor.size(0), -1)
                real_np = designs_3d.detach().cpu().numpy().reshape(designs_3d.size(0), -1)
                mmd_value = mmd(gen_np, real_np, sigma=args.mmd_sigma)
                dpp_value = dpp_diversity(gen_np, sigma=args.dpp_sigma)
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
                # Log individual axis losses
                log_dict = {}
                for axis_name in slice_discriminators:
                    log_dict[f"d_loss_{axis_name}"] = d_losses.get(axis_name, 0)
                    log_dict[f"real_loss_{axis_name}"] = real_losses.get(axis_name, 0)
                    log_dict[f"fake_loss_{axis_name}"] = fake_losses.get(axis_name, 0)
                    log_dict[f"disc_acc_{axis_name}"] = last_disc_acc[axis_name]

                # Log averages
                log_dict["d_loss_avg"] = sum(d_losses.values()) / len(d_losses) if d_losses else 0
                log_dict["total_g_loss"] = total_g_loss.item() if isinstance(total_g_loss, th.Tensor) else 0
                log_dict["disc_acc_avg"] = sum(last_disc_acc.values()) / len(last_disc_acc)
                log_dict["epoch"] = epoch
                log_dict["batch"] = batches_done

                if mmd_value is not None:
                    log_dict["mmd"] = mmd_value
                if dpp_value is not None:
                    log_dict["dpp_diversity"] = dpp_value

                wandb.log(log_dict)

                if i % 10 == 0:  # Print less frequently
                    avg_d_loss = sum(d_losses.values()) / len(d_losses) if d_losses else 0
                    avg_disc_acc = sum(last_disc_acc.values()) / len(last_disc_acc)
                    g_loss_val = total_g_loss.item() if isinstance(total_g_loss, th.Tensor) else total_g_loss

                    print(
                        f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] "
                        f"[D loss: {avg_d_loss:.4f}] [G loss: {g_loss_val:.4f}] "
                        f"[D acc: {avg_disc_acc:.3f}]"
                    )

                    # Print per-axis details
                    for axis_name in slice_discriminators:
                        if axis_name in d_losses:
                            print(
                                f"  {axis_name.upper()}: D_loss={d_losses[axis_name]:.4f}, "
                                f"D_acc={last_disc_acc[axis_name]:.3f}"
                            )

                # Sample and visualize 3D designs
                if batches_done % args.sample_interval == 0:
                    print("Generating 3D design samples...")

                    gen_conds, gen_volumes = sample_3d_designs(9)

                    img_fname = f"images_slicegan/{batches_done}.png"
                    visualize_3d_designs(gen_volumes, gen_conds, condition_names, img_fname, max_designs=9)

                    # Log to wandb
                    wandb.log({"generated_designs": wandb.Image(img_fname), "sample_step": batches_done})

                    print(f"3D design samples saved for step {batches_done}")

            # Clean up GPU memory periodically
            if i % 50 == 0:
                th.cuda.empty_cache() if device.type == "cuda" else None

        # --------------
        #  Save models
        # --------------
        if args.save_model and epoch == args.n_epochs - 1:
            print("Saving SliceGAN models...")

            # Save generators
            for _axis_name in slice_discriminators:
                ckpt_gen = {
                    "epoch": epoch,
                    "batches_done": epoch * len(dataloader) + len(dataloader) - 1,
                    "generator": slice_generator.state_dict(),
                    "optimizer_generator": optimizer_generator.state_dict(),
                    "loss": total_g_loss,
                    "n_conds": n_conds,
                    "args": vars(args),
                }

                gen_filename = "generator_slicegan.pth"
                th.save(ckpt_gen, gen_filename)

            if args.track:
                artifact_gen = wandb.Artifact(f"{args.problem_id}_{args.algo}_generator_slice_gan_3d", type="model")
                artifact_gen.add_file(gen_filename)
                wandb.log_artifact(artifact_gen, aliases=[f"seed_{args.seed}"])

            # Save discriminators
            for axis_name, discriminator in slice_discriminators.items():
                ckpt_disc = {
                    "epoch": epoch,
                    "batches_done": epoch * len(dataloader) + len(dataloader) - 1,
                    "discriminator": discriminator.state_dict(),
                    "optimizer_discriminator": optimizer_discriminators[axis_name].state_dict(),
                    "loss": d_losses[axis_name],
                    "axis": axis_name,
                    "n_conds": n_conds,
                    "args": vars(args),
                }

                disc_filename = f"discriminator_slice_{axis_name}.pth"
                th.save(ckpt_disc, disc_filename)

                if args.track:
                    artifact_disc = wandb.Artifact(
                        f"{args.problem_id}_{args.algo}_discriminator_slice_{axis_name}", type="model"
                    )
                    artifact_disc.add_file(disc_filename)
                    wandb.log_artifact(artifact_disc, aliases=[f"seed_{args.seed}"])

            print("SliceGAN models saved successfully!")

    if args.track:
        if mmd_values:
            final_mmd = np.mean(mmd_values[-10:])
            wandb.log({"mmd": final_mmd, "epoch": args.n_epochs})
        if dpp_values:
            final_dpp = np.mean(dpp_values[-10:])
            wandb.log({"dpp": final_dpp, "epoch": args.n_epochs})
        wandb.finish()

    print("SliceGAN training completed!")

    if args.export:
        # ---- Export final generated 3D designs for ParaView ----
        print("Exporting final generated 3D designs for ParaView...")
        slice_generator.eval()
        n_export = 8  # Number of designs to export
        z = th.randn((n_export, args.latent_dim, 1, 1, 1), device=device)
        all_conditions = th.stack(condition_tensors, dim=1)
        linspaces = [
            th.linspace(all_conditions[:, i].min(), all_conditions[:, i].max(), n_export, device=device)
            for i in range(all_conditions.shape[1])
        ]
        export_conds = th.stack(linspaces, dim=1)
        gen_volumes = slice_generator(z, export_conds.reshape(-1, n_conds, 1, 1, 1))
        gen_volumes_np = gen_volumes.squeeze(1).detach().cpu().numpy()

        os.makedirs("paraview_exports", exist_ok=True)
        for i, vol in enumerate(gen_volumes_np):
            np.save(f"paraview_exports/slice_gen3d_{i}.npy", vol)
            # Optionally: save as .vti using pyvista or pyevtk if you want native VTK format

        print(f"Saved {n_export} generated 3D designs to paraview_exports/ as .npy files.")
