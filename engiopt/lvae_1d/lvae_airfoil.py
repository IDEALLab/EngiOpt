"""LVAE_DP for airfoil designs with constrained volume minimization.

This implementation uses constraint-based optimization to minimize latent volume
subject to reconstruction accuracy constraints. The encoder/decoder operates on
both the airfoil geometry (coords) AND angle_of_attack.

Optimization Problem:
    minimize: volume_loss (number of active latent dimensions)
    subject to: reconstruction_loss ≤ reconstruction_threshold

No performance constraints since datasets lack performance information.

Architecture:
- Conv1DEncoder: Encodes (coords, angle) → latent code
- SNBezierDecoder: Decodes latent → (coords, angle) using Bezier curves
- LeastVolumeAE_DynamicPruning: Dynamically prunes low-variance dimensions

Supported constraint methods:
- augmented_lagrangian (recommended)
- log_barrier, primal_dual, adaptive, softplus_al, weighted_sum

Note: The EngiBench airfoil problem has a Dict design space with:
  - 'coords': (2, 192) - x,y coordinates of airfoil shape
  - 'angle_of_attack': scalar value (encoded in latent space)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import os
import random
import time

from engibench.utils.all_problems import BUILTIN_PROBLEMS
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import tqdm
import tyro
import wandb

from engiopt.lvae_2d.aes import LeastVolumeAE_DynamicPruning
from engiopt.lvae_2d.constraint_handlers import (
    ConstraintHandler,
    ConstraintLosses,
    ConstraintThresholds,
    create_constraint_handler,
)


@dataclass
class Args:
    # Problem and tracking
    problem_id: str = "airfoil"
    """Problem ID to run. Must be 'airfoil'."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """Algorithm name for tracking purposes."""
    track: bool = True
    """Whether to track with Weights & Biases."""
    wandb_project: str = "lvae"
    """WandB project name."""
    wandb_entity: str | None = None
    """WandB entity name. If None, uses the default entity."""
    seed: int = 1
    """Random seed for reproducibility."""
    save_model: bool = False
    """Whether to save the model after training."""
    sample_interval: int = 500
    """Interval for sampling designs during training."""

    # Training hyperparameters
    n_epochs: int = 2500
    """Number of training epochs."""
    batch_size: int = 128
    """Batch size for training."""
    lr: float = 1e-4
    """Learning rate for the optimizer."""

    # LVAE-specific
    latent_dim: int = 250
    """Dimensionality of the latent space (overestimate)."""
    pruning_epoch: int = 500
    """Epoch to start pruning dimensions."""
    beta: float = 0.9
    """Momentum for EMA of latent statistics."""
    eta: float = 1e-4
    """Scaling factor for the volume loss."""

    # Constraint optimization
    constraint_method: str = "augmented_lagrangian"
    """Constraint method: weighted_sum, augmented_lagrangian, log_barrier, primal_dual, adaptive, softplus_al"""
    reconstruction_threshold: float = 0.001
    """Constraint threshold for reconstruction MSE."""
    volume_warmup_epochs: int = 50
    """Number of epochs to ignore volume loss (prevents early collapse)."""

    # Weighted sum parameters (if using weighted_sum method)
    w_volume: float = 1.0
    """Weight for volume loss in weighted sum method."""
    w_reconstruction: float = 1.0
    """Weight for reconstruction loss in weighted sum method."""

    # Augmented Lagrangian parameters
    alpha_r: float = 0.1
    """Learning rate for reconstruction Lagrange multiplier."""
    mu_r_init: float = 1.0
    """Initial penalty coefficient for reconstruction constraint."""
    mu_r: float = 10.0
    """Final penalty coefficient for reconstruction constraint."""
    warmup_epochs: int = 100
    """Epochs to ramp up penalty coefficients."""

    # Log barrier parameters
    t_init: float = 1.0
    """Initial barrier parameter."""
    t_growth: float = 1.05
    """Barrier parameter growth rate per epoch."""
    t_max: float = 1000.0
    """Maximum barrier parameter."""
    barrier_epsilon: float = 1e-6
    """Safety margin from constraint boundary."""
    fallback_penalty: float = 1e6
    """Penalty when constraints violated."""

    # Primal-dual parameters
    lr_dual: float = 0.01
    """Learning rate for dual variable updates."""
    clip_lambda: float = 100.0
    """Maximum value for dual variables."""

    # Adaptive weight parameters
    adaptation_lr: float = 0.01
    """Learning rate for weight adaptation."""
    update_frequency: int = 10
    """Update weights every N steps."""

    # Softplus AL parameters
    softplus_beta: float = 10.0
    """Smoothness parameter for softplus."""

    # Bezier architecture parameters
    n_control_points: int = 64
    """Number of control points for Bezier curve generation."""
    decoder_lipschitz_scale: float = 1.0
    """Lipschitz constant multiplier for decoder output."""

    # Dynamic pruning
    pruning_strategy: str = "lognorm"
    """Which pruning strategy to use: [plummet, pca_cdf, lognorm, probabilistic]."""
    cdf_threshold: float = 0.99
    """(pca_cdf) Cumulative variance threshold."""
    temperature: float = 1.0
    """(probabilistic) Sampling temperature."""
    plummet_threshold: float = 0.02
    """(plummet) Threshold for pruning dimensions."""
    alpha: float = 0.0
    """(lognorm) Weighting factor for blending reference and current distribution."""
    percentile: float = 0.05
    """(lognorm) Percentile threshold for pruning."""

    # Safeguard parameters
    min_active_dims: int = 0
    """Never prune below this many dims (0 = no limit)."""
    max_prune_per_epoch: int | None = None
    """Max dims to prune in one epoch (None = no limit)."""
    cooldown_epochs: int = 0
    """Epochs to wait between pruning events (0 = no cooldown)."""
    k_consecutive: int = 1
    """Consecutive epochs a dim must be below threshold to be eligible (1 = immediate)."""
    recon_tol: float = float("inf")
    """Relative tolerance to best_val_recon to allow pruning (inf = no constraint)."""


# ============================================================================
# Encoder: Conv1D + MLP for airfoil coordinates
# ============================================================================


class Conv1DEncoder(nn.Module):
    """1D Convolutional encoder for airfoil coordinates + angle of attack.

    Takes (B, 2, 192) airfoil coordinates and (B, 1) angle_of_attack,
    encodes to (B, latent_dim).
    """

    def __init__(self, in_channels: int, in_features: int, latent_dim: int):
        super().__init__()

        # Conv layers: 192 -> 96 -> 48 -> 24 -> 12 -> 6 -> 3
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 4, 2, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 4, 2, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 4, 2, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 1024, 4, 2, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv1d(1024, 2048, 4, 2, 1),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
        )

        # Calculate flattened size: 192 / 2^6 = 3
        m_features = 3 * 2048

        # MLP to latent (concatenates conv features + angle_of_attack)
        self.mlp = nn.Sequential(
            nn.Linear(m_features + 1, 1024),  # +1 for angle_of_attack
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, latent_dim),
        )

    def forward(self, coords: th.Tensor, angle: th.Tensor) -> th.Tensor:
        """Encode airfoil design.

        Args:
            coords: (B, 2, 192) airfoil coordinates
            angle: (B, 1) angle of attack

        Returns:
            z: (B, latent_dim) latent code
        """
        h = self.conv(coords)
        h = h.flatten(1)
        # Concatenate with angle_of_attack
        combined = th.cat([h, angle], dim=1)
        return self.mlp(combined)


# ============================================================================
# Decoder: Bezier-based generator for airfoil coordinates
# ============================================================================


class BezierLayer(nn.Module):
    """Generates Bezier curves using rational Bezier basis functions."""

    def __init__(self, m_features: int, n_control_points: int, n_data_points: int, eps: float = 1e-7):
        super().__init__()
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points
        self.eps = eps

        # Generate intervals (parameter values) from features
        self.generate_intervals = nn.Sequential(
            nn.Linear(m_features, n_data_points - 1),
            nn.Softmax(dim=1),
            nn.ConstantPad1d((1, 0), 0.0),  # Add leading zero
        )

    def forward(
        self, features: th.Tensor, control_points: th.Tensor, weights: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Generate Bezier curves.

        Args:
            features: (B, m_features)
            control_points: (B, 2, n_cp)
            weights: (B, 1, n_cp)

        Returns:
            data_points: (B, 2, n_dp)
            param_vars: (B, 1, n_dp)
            intervals: (B, n_dp)
        """
        # Generate parameter values t in [0, 1]
        intervals = self.generate_intervals(features)  # (B, n_dp)
        pv = th.cumsum(intervals, dim=-1).clamp(0, 1).unsqueeze(1)  # (B, 1, n_dp)

        # Compute Bernstein polynomials
        # B_{i,n}(t) = C(n,i) * t^i * (1-t)^(n-i)
        pw1 = th.arange(0.0, self.n_control_points, device=features.device).view(1, -1, 1)
        pw2 = th.flip(pw1, (1,))

        # Log-space computation for numerical stability
        lbs = (
            pw1 * th.log(pv + self.eps)
            + pw2 * th.log(1 - pv + self.eps)
            + th.lgamma(th.tensor(self.n_control_points, device=features.device) + self.eps).view(1, -1, 1)
            - th.lgamma(pw1 + 1 + self.eps)
            - th.lgamma(pw2 + 1 + self.eps)
        )
        bs = th.exp(lbs)  # (B, n_cp, n_dp)

        # Rational Bezier curve: sum(w_i * P_i * B_i) / sum(w_i * B_i)
        dp = (control_points * weights) @ bs / (weights @ bs + self.eps)  # (B, 2, n_dp)

        return dp, pv, intervals


class SNBezierDecoder(nn.Module):
    """Spectral normalized Bezier decoder.

    Generates airfoil coords (B, 2, 192) and angle_of_attack (B, 1) from latent (B, latent_dim).
    Applies Lipschitz scaling to entire output (coords + angle) to ensure unified constraint.
    """

    def __init__(
        self,
        latent_dim: int,
        n_control_points: int,
        n_data_points: int,
        lipschitz_scale: float = 1.0,
        angle_min: float = 0.0,
        angle_max: float = 1.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points
        self.lipschitz_scale = lipschitz_scale
        self.register_buffer("angle_min", th.tensor(angle_min))
        self.register_buffer("angle_max", th.tensor(angle_max))
        self.eps = eps

        m_features = 256

        # Feature generator for Bezier layer
        self.feature_generator = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 1024)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(1024, 512)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(512, m_features)),
        )

        # Control point and weight generator
        # Dense layers
        deconv_channels = [768, 384, 192, 96]
        n_layers = len(deconv_channels) - 1
        in_width = n_control_points // (2**n_layers)
        in_channels = deconv_channels[0]

        self.cpw_dense = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 1024)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(1024, in_channels * in_width)),
        )

        self.in_channels = in_channels
        self.in_width = in_width

        # Deconvolutional layers
        deconv_layers = []
        for i in range(len(deconv_channels) - 1):
            deconv_layers.extend(
                [
                    spectral_norm(nn.ConvTranspose1d(deconv_channels[i], deconv_channels[i + 1], 4, 2, 1)),
                    nn.BatchNorm1d(deconv_channels[i + 1]),
                    nn.ReLU(),
                ]
            )
        self.deconv = nn.Sequential(*deconv_layers)

        # Output heads for Bezier
        self.cp_gen = nn.Sequential(spectral_norm(nn.Conv1d(deconv_channels[-1], 2, 1)), nn.Tanh())
        self.w_gen = nn.Sequential(spectral_norm(nn.Conv1d(deconv_channels[-1], 1, 1)), nn.Sigmoid())

        # Bezier layer
        self.bezier_layer = BezierLayer(m_features, n_control_points, n_data_points)

        # Separate head for angle_of_attack
        self.angle_head = spectral_norm(nn.Linear(latent_dim, 1))

    def forward(self, z: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Generate airfoil coordinates and angle from latent code.

        Args:
            z: (B, latent_dim)

        Returns:
            coords: (B, 2, n_data_points) airfoil coordinates
            angle: (B, 1) angle of attack (normalized to [0, 1])
        """
        # Generate features for Bezier
        features = self.feature_generator(z)

        # Generate control points and weights
        x = self.cpw_dense(z).view(-1, self.in_channels, self.in_width)
        x = self.deconv(x)
        cp = self.cp_gen(x)  # (B, 2, n_cp)
        w = self.w_gen(x)  # (B, 1, n_cp)

        # Generate Bezier curve
        coords, _, _ = self.bezier_layer(features, cp, w)
        coords = coords * self.lipschitz_scale

        # Generate angle_of_attack (normalized [0, 1])
        angle_norm = th.sigmoid(self.angle_head(z))  # (B, 1) in [0, 1]
        angle_norm = angle_norm * self.lipschitz_scale

        return coords, angle_norm

    def denormalize_angle(self, angle_norm: th.Tensor) -> th.Tensor:
        """Denormalize angle from [0, 1] to original range.

        Args:
            angle_norm: (B, 1) normalized angle in [0, 1]

        Returns:
            angle: (B, 1) angle in original range [angle_min, angle_max]
        """
        return angle_norm * (self.angle_max - self.angle_min + self.eps) + self.angle_min


# ============================================================================
# Constrained LVAE Wrapper
# ============================================================================


class ConstrainedLVAE_Airfoil(LeastVolumeAE_DynamicPruning):
    """Airfoil LVAE with constraint-based volume minimization.

    Wraps LeastVolumeAE_DynamicPruning to support constrained optimization:
        minimize: volume_loss
        subject to: reconstruction_loss ≤ reconstruction_threshold

    No performance constraints since datasets lack performance information.
    Performance loss is set to 0.0 and threshold to infinity.
    """

    def __init__(
        self,
        *args,
        constraint_handler: ConstraintHandler,
        reconstruction_threshold: float,
        coords_mean: th.Tensor,
        coords_std: th.Tensor,
        angle_mean: th.Tensor,
        angle_std: th.Tensor,
        **kwargs,
    ):
        """Initialize constrained LVAE for airfoil designs.

        Args:
            constraint_handler: Constraint optimization method handler
            reconstruction_threshold: Constraint on reconstruction MSE
            coords_mean: Mean for coords normalization (1, 2, 192)
            coords_std: Std for coords normalization (1, 2, 192)
            angle_mean: Mean for angle normalization (1, 1)
            angle_std: Std for angle normalization (1, 1)
            *args, **kwargs: Passed to parent LeastVolumeAE_DynamicPruning
        """
        super().__init__(*args, **kwargs)
        self.constraint_handler = constraint_handler
        self.reconstruction_threshold = reconstruction_threshold
        self.performance_threshold = float("inf")  # Not used (no performance data)

        # Store normalization parameters as buffers (moved to device with model)
        self.register_buffer("coords_mean", coords_mean)
        self.register_buffer("coords_std", coords_std)
        self.register_buffer("angle_mean", angle_mean)
        self.register_buffer("angle_std", angle_std)

    def loss(self, batch, **kwargs):
        """Compute loss components and store for constraint handler.

        Args:
            batch: Tuple of (coords, angle) tensors

        Returns:
            torch.Tensor: [reconstruction_loss, volume_loss] for logging
        """
        coords, angle = batch

        # Encode with pruning mask
        z = self.encoder(coords, angle)
        z[:, self._p] = self._z[self._p]

        # Update moving mean for pruning statistics
        self._update_moving_mean(z)

        # Decode
        coords_hat, angle_hat = self.decoder(z)

        # Compute reconstruction loss with joint normalization
        # Normalize both coords and angle using the same statistics computed from concatenated vector
        # This ensures all elements are on the same scale for fair MSE comparison
        coords_norm = (coords - self.coords_mean) / self.coords_std  # (B, 2, 192)
        coords_hat_norm = (coords_hat - self.coords_mean) / self.coords_std  # (B, 2, 192)
        angle_norm = (angle - self.angle_mean) / self.angle_std  # (B, 1)
        angle_hat_norm = (angle_hat - self.angle_mean) / self.angle_std  # (B, 1)

        # Flatten and concatenate into single vector: (B, 385)
        coords_norm_flat = coords_norm.reshape(coords.shape[0], -1)  # (B, 384)
        coords_hat_norm_flat = coords_hat_norm.reshape(coords_hat.shape[0], -1)  # (B, 384)

        target_vec = th.cat([coords_norm_flat, angle_norm], dim=1)  # (B, 385)
        recon_vec = th.cat([coords_hat_norm_flat, angle_hat_norm], dim=1)  # (B, 385)

        # Single MSE on normalized concatenated vector
        rec_loss = nn.functional.mse_loss(recon_vec, target_vec)

        # Store individual components for separate logging (unnormalized MSE)
        self._coords_mse = nn.functional.mse_loss(coords, coords_hat)
        self._angle_mse = nn.functional.mse_loss(angle, angle_hat)

        # Volume loss with active dimension scaling
        active_ratio = self.dim / len(self._p)
        vol_loss = active_ratio * self.loss_vol(z[:, ~self._p])

        # Store for constraint handler (performance=0 since no performance data)
        self._loss_components = ConstraintLosses(
            volume=vol_loss, reconstruction=rec_loss, performance=th.tensor(0.0, device=vol_loss.device)
        )

        # Return for logging
        return th.stack([rec_loss, vol_loss])

    def compute_total_loss(self):
        """Compute total loss using constraint handler for backprop.

        Returns:
            torch.Tensor: Total loss for optimization
        """
        thresholds = ConstraintThresholds(
            reconstruction=self.reconstruction_threshold,
            performance=self.performance_threshold,  # inf, effectively ignored
        )
        return self.constraint_handler.compute_loss(self._loss_components, thresholds)

    def update_constraint_handler(self):
        """Update constraint handler state (dual variables, barrier params, etc.)."""
        thresholds = ConstraintThresholds(
            reconstruction=self.reconstruction_threshold, performance=self.performance_threshold
        )
        # Detach for handler updates
        detached = ConstraintLosses(
            volume=self._loss_components.volume.detach(),
            reconstruction=self._loss_components.reconstruction.detach(),
            performance=self._loss_components.performance.detach(),
        )
        self.constraint_handler.step(detached, thresholds)

    def epoch_hook(self, epoch, *args, **kwargs):
        """Propagate epoch to constraint handler for warmup/scheduling."""
        super().epoch_hook(epoch, *args, **kwargs)
        self.constraint_handler.epoch_hook(epoch)


# ============================================================================
# Main Training Script
# ============================================================================


if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    # Verify it's airfoil problem with Dict design space
    if not isinstance(problem.design_space, spaces.Dict):
        raise ValueError(f"Expected Dict design space for airfoil, got {type(problem.design_space)}")

    # Get coords shape: (2, 192)
    coords_shape = problem.design_space["coords"].shape
    n_data_points = coords_shape[1]  # 192

    # Logging
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            save_code=True,
            name=run_name,
        )

    # Seeding
    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True

    os.makedirs("images", exist_ok=True)

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    # ---- DataLoader ----
    # Extract coords and angle_of_attack from the airfoil dataset
    problem_dataset = problem.dataset.with_format("torch")
    train_ds = problem_dataset["train"]
    val_ds = problem_dataset["val"]
    test_ds = problem_dataset["test"]

    # Extract coords and angle_of_attack
    coords_train = th.stack([train_ds[i]["optimal_design"]["coords"] for i in range(len(train_ds))])
    coords_val = th.stack([val_ds[i]["optimal_design"]["coords"] for i in range(len(val_ds))])
    coords_test = th.stack([test_ds[i]["optimal_design"]["coords"] for i in range(len(test_ds))])

    angle_train = th.stack([train_ds[i]["optimal_design"]["angle_of_attack"] for i in range(len(train_ds))]).unsqueeze(-1)
    angle_val = th.stack([val_ds[i]["optimal_design"]["angle_of_attack"] for i in range(len(val_ds))]).unsqueeze(-1)
    angle_test = th.stack([test_ds[i]["optimal_design"]["angle_of_attack"] for i in range(len(test_ds))]).unsqueeze(-1)

    # Normalize angle_of_attack to [0, 1] for encoder/decoder
    angle_min = angle_train.min()
    angle_max = angle_train.max()
    eps_angle = 1e-7

    angle_train_norm = (angle_train - angle_min) / (angle_max - angle_min + eps_angle)
    angle_val_norm = (angle_val - angle_min) / (angle_max - angle_min + eps_angle)
    angle_test_norm = (angle_test - angle_min) / (angle_max - angle_min + eps_angle)

    # Compute normalization statistics for concatenated coords+angle vector
    # This ensures coords and angle are on the same scale for MSE computation
    # Concatenate: coords (B, 2, 192) flattened to (B, 384) + angle (B, 1) -> (B, 385)
    coords_train_flat = coords_train.reshape(coords_train.shape[0], -1)  # (N, 384)
    concat_train = th.cat([coords_train_flat, angle_train_norm], dim=1)  # (N, 385)

    # Compute mean and std for standardization
    concat_mean = concat_train.mean(dim=0, keepdim=True)  # (1, 385)
    concat_std = concat_train.std(dim=0, keepdim=True) + 1e-7  # (1, 385)

    # Store for use in loss computation
    # Reshape back to match coords (2, 192) and angle (1) shapes
    coords_mean = concat_mean[:, :384].reshape(1, 2, 192)  # (1, 2, 192)
    coords_std = concat_std[:, :384].reshape(1, 2, 192)  # (1, 2, 192)
    angle_mean = concat_mean[:, 384:]  # (1, 1)
    angle_std = concat_std[:, 384:]  # (1, 1)

    print(f"{'=' * 60}")
    print("Normalization Statistics (for MSE computation)")
    print(f"{'=' * 60}")
    print(f"Coords mean range:   [{coords_mean.min():.6f}, {coords_mean.max():.6f}]")
    print(f"Coords std range:    [{coords_std.min():.6f}, {coords_std.max():.6f}]")
    print(f"Angle mean:          {angle_mean.item():.6f}")
    print(f"Angle std:           {angle_std.item():.6f}")
    print(f"Angle of Attack original range: [{angle_min:.6f}, {angle_max:.6f}]")
    print(f"Angle of Attack normalized range: [{angle_train_norm.min():.6f}, {angle_train_norm.max():.6f}]")
    print(f"{'=' * 60}\n")

    # Build encoder and decoder
    enc = Conv1DEncoder(in_channels=2, in_features=n_data_points, latent_dim=args.latent_dim)
    dec = SNBezierDecoder(
        latent_dim=args.latent_dim,
        n_control_points=args.n_control_points,
        n_data_points=n_data_points,
        lipschitz_scale=args.decoder_lipschitz_scale,
        angle_min=angle_min.item(),
        angle_max=angle_max.item(),
    )

    print(f"Using Bezier-based decoder with {args.n_control_points} control points (Lipschitz scale: {args.decoder_lipschitz_scale})")
    print(f"Coords shape: {coords_shape}")

    # Build pruning parameters
    pruning_params = {}
    if args.pruning_strategy == "plummet":
        pruning_params["threshold"] = args.plummet_threshold
    elif args.pruning_strategy == "pca_cdf":
        pruning_params["threshold"] = args.cdf_threshold
    elif args.pruning_strategy == "probabilistic":
        pruning_params["temperature"] = args.temperature
    elif args.pruning_strategy == "lognorm":
        pruning_params["alpha"] = args.alpha
        pruning_params["threshold"] = args.percentile
    else:
        raise ValueError(f"Unknown pruning_strategy: {args.pruning_strategy}")

    # Create constraint handler
    handler_kwargs = {"device": device, "volume_warmup_epochs": args.volume_warmup_epochs}

    if args.constraint_method == "weighted_sum":
        handler_kwargs.update({
            "w_volume": args.w_volume,
            "w_reconstruction": args.w_reconstruction,
            "w_performance": 0.0,  # Not used
        })
    elif args.constraint_method in ["augmented_lagrangian", "softplus_al"]:
        handler_kwargs.update({
            "alpha_r": args.alpha_r,
            "alpha_p": 0.0,  # Not used (no performance)
            "mu_r_init": args.mu_r_init,
            "mu_p_init": 0.0,
            "mu_r_final": args.mu_r,
            "mu_p_final": 0.0,
            "warmup_epochs": args.warmup_epochs,
        })
        if args.constraint_method == "softplus_al":
            handler_kwargs["beta"] = args.softplus_beta
    elif args.constraint_method == "log_barrier":
        handler_kwargs.update({
            "t_init": args.t_init,
            "t_growth": args.t_growth,
            "t_max": args.t_max,
            "epsilon": args.barrier_epsilon,
            "fallback_penalty": args.fallback_penalty,
        })
    elif args.constraint_method == "primal_dual":
        handler_kwargs.update({
            "lr_dual": args.lr_dual,
            "clip_lambda": args.clip_lambda,
        })
    elif args.constraint_method == "adaptive":
        handler_kwargs.update({
            "w_volume_init": 1.0,
            "w_reconstruction_init": 1.0,
            "w_performance_init": 0.0,
            "adaptation_lr": args.adaptation_lr,
            "update_frequency": args.update_frequency,
        })
    else:
        raise ValueError(f"Unknown constraint_method: {args.constraint_method}")

    constraint_handler = create_constraint_handler(args.constraint_method, **handler_kwargs)

    # Initialize LVAE with dynamic pruning and constraint handler
    lvae = ConstrainedLVAE_Airfoil(
        encoder=enc,
        decoder=dec,
        optimizer=Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr),
        latent_dim=args.latent_dim,
        constraint_handler=constraint_handler,
        reconstruction_threshold=args.reconstruction_threshold,
        coords_mean=coords_mean,
        coords_std=coords_std,
        angle_mean=angle_mean,
        angle_std=angle_std,
        pruning_epoch=args.pruning_epoch,
        beta=args.beta,
        eta=args.eta,
        pruning_strategy=args.pruning_strategy,
        pruning_params=pruning_params,
        min_active_dims=args.min_active_dims,
        max_prune_per_epoch=args.max_prune_per_epoch,
        cooldown_epochs=args.cooldown_epochs,
        k_consecutive=args.k_consecutive,
        recon_tol=args.recon_tol,
    ).to(device)

    loader = DataLoader(TensorDataset(coords_train, angle_train_norm), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(coords_val, angle_val_norm), batch_size=args.batch_size, shuffle=False)

    # ---- Training loop ----
    for epoch in range(args.n_epochs):
        lvae.epoch_hook(epoch=epoch)

        bar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(bar):
            coords_batch = batch[0].to(device)  # (B, 2, 192)
            angle_batch = batch[1].to(device)  # (B, 1)

            lvae.optim.zero_grad()

            # Compute loss components (handled internally by wrapper)
            losses = lvae.loss((coords_batch, angle_batch))  # [rec, vol] for logging

            # Get total loss from constraint handler
            loss = lvae.compute_total_loss()

            # Backprop
            loss.backward()
            lvae.optim.step()

            # Update constraint handler state
            lvae.update_constraint_handler()

            # Get handler metrics for logging
            handler_metrics = lvae.constraint_handler.get_metrics()

            bar.set_postfix(
                {
                    "rec": f"{losses[0].item():.3f}",
                    "vol": f"{losses[1].item():.3f}",
                    "dim": lvae.dim,
                }
            )

            # Log to wandb
            if args.track:
                batches_done = epoch * len(bar) + i
                wandb.log(
                    {
                        "rec_loss": losses[0].item(),
                        "coords_mse": lvae._coords_mse.item(),  # Track coords separately
                        "angle_mse": lvae._angle_mse.item(),  # Track angle separately
                        "vol_loss": losses[1].item(),
                        "total_loss": loss.item(),
                        "active_dims": lvae.dim,
                        "epoch": epoch,
                        **handler_metrics,  # Add constraint handler metrics
                    }
                )
                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(bar)}] "
                    f"[rec loss: {losses[0].item()}] [vol loss: {losses[1].item()}] [active dims: {lvae.dim}]"
                )

                # Sample and visualize at regular intervals
                if batches_done % args.sample_interval == 0:
                    with th.no_grad():
                        # Encode TRAINING designs (use normalized angles) - pruning is based on training data
                        Xs_coords = coords_train.to(device)
                        Xs_angle_norm = angle_train_norm.to(device)
                        z = lvae.encoder(Xs_coords, Xs_angle_norm)
                        # Apply pruning mask (set pruned dimensions to fixed values)
                        z[:, lvae._p] = lvae._z[lvae._p]
                        z_std, idx = th.sort(z.std(0), descending=True)
                        z_mean = z.mean(0)
                        N = (z_std > 0).sum().item()

                        # Generate interpolated designs
                        x_ints_coords = []
                        x_ints_angle = []
                        for alpha in [0, 0.25, 0.5, 0.75, 1]:
                            z_ = (1 - alpha) * z[:25] + alpha * th.roll(z, -1, 0)[:25]
                            coords_, angle_norm_ = lvae.decoder(z_)
                            # Denormalize angles for plotting
                            angle_ = lvae.decoder.denormalize_angle(angle_norm_)
                            x_ints_coords.append(coords_.cpu().numpy())
                            x_ints_angle.append(angle_.cpu().numpy())

                        # Generate random designs
                        z_rand = z_mean.unsqueeze(0).repeat([25, 1])
                        z_rand[:, idx[:N]] += z_std[:N] * th.randn_like(z_rand[:, idx[:N]])
                        coords_rand, angle_rand_norm = lvae.decoder(z_rand)
                        # Denormalize angles for plotting
                        angle_rand = lvae.decoder.denormalize_angle(angle_rand_norm)

                        # Move to CPU (use original unnormalized angles)
                        z_std_cpu = z_std.cpu().numpy()
                        Xs_coords_cpu = Xs_coords.cpu().numpy()
                        Xs_angle_cpu = angle_train[: len(Xs_coords)].cpu().numpy()  # Use original unnormalized angles
                        coords_rand_np = coords_rand.cpu().numpy()
                        angle_rand_np = angle_rand.cpu().numpy()

                    # Plot 1: Latent dimension statistics
                    plt.figure(figsize=(12, 6))
                    plt.subplot(211)
                    plt.bar(np.arange(len(z_std_cpu)), z_std_cpu)
                    plt.yscale("log")
                    plt.xlabel("Latent dimension index")
                    plt.ylabel("Standard deviation")
                    plt.title(f"Number of principal components = {N}")
                    plt.subplot(212)
                    plt.bar(np.arange(N), z_std_cpu[:N])
                    plt.yscale("log")
                    plt.xlabel("Latent dimension index")
                    plt.ylabel("Standard deviation")
                    plt.savefig(f"images/dim_{batches_done}.png")
                    plt.close()

                    # Plot 2: Interpolated airfoil designs
                    fig, axs = plt.subplots(25, 6, figsize=(12, 25))
                    for i, j in product(range(25), range(5)):
                        airfoil = x_ints_coords[j][i]
                        angle_val = x_ints_angle[j][i][0]  # Extract scalar angle value
                        axs[i, j + 1].plot(airfoil[0], airfoil[1], "b-")
                        axs[i, j + 1].set_aspect("equal")
                        axs[i, j + 1].axis("off")
                        axs[i, j + 1].set_title(f"α={angle_val:.2f}", fontsize=8)
                    for ax, alpha in zip(axs[0, 1:], [0, 0.25, 0.5, 0.75, 1]):
                        # Update column headers to show interpolation parameter
                        current_title = ax.get_title()
                        ax.set_title(rf"interp={alpha}" + "\n" + current_title, fontsize=8)
                    for i in range(25):
                        airfoil = Xs_coords_cpu[i]
                        angle_val = Xs_angle_cpu[i][0]  # Extract scalar angle value
                        axs[i, 0].plot(airfoil[0], airfoil[1], "b-")
                        axs[i, 0].set_aspect("equal")
                        axs[i, 0].axis("off")
                        axs[i, 0].set_title(f"α={angle_val:.2f}", fontsize=8)
                    axs[0, 0].set_title("groundtruth\n" + axs[0, 0].get_title(), fontsize=8)
                    fig.tight_layout()
                    plt.savefig(f"images/interp_{batches_done}.png")
                    plt.close()

                    # Plot 3: Random airfoil designs from latent space
                    fig, axs = plt.subplots(5, 5, figsize=(15, 7.5))
                    for k, (i, j) in enumerate(product(range(5), range(5))):
                        airfoil = coords_rand_np[k]
                        angle_val = angle_rand_np[k][0]  # Extract scalar angle value
                        axs[i, j].plot(airfoil[0], airfoil[1], "b-")
                        axs[i, j].set_aspect("equal")
                        axs[i, j].axis("off")
                        axs[i, j].set_title(f"α={angle_val:.2f}", fontsize=8)
                    fig.tight_layout()
                    plt.suptitle("Gaussian random designs from latent space", y=1.0)
                    plt.savefig(f"images/norm_{batches_done}.png")
                    plt.close()

                    # Plot 4: Reconstruction comparison
                    fig, axes = plt.subplots(5, 2, figsize=(10, 15))
                    coords_orig = Xs_coords_cpu[:5]
                    angle_orig = Xs_angle_cpu[:5]
                    coords_recon, angle_recon_norm = lvae.decoder(z[:5])
                    angle_recon = lvae.decoder.denormalize_angle(angle_recon_norm)
                    coords_recon_np = coords_recon.detach().cpu().numpy()
                    angle_recon_np = angle_recon.detach().cpu().numpy()

                    for k in range(5):
                        # Original
                        axes[k, 0].plot(coords_orig[k][0], coords_orig[k][1], "b-")
                        axes[k, 0].set_aspect("equal")
                        axes[k, 0].axis("off")
                        if k == 0:
                            axes[k, 0].set_title("Original")
                        axes[k, 0].text(
                            0.5,
                            -0.1,
                            f"α={angle_orig[k][0]:.2f}",
                            transform=axes[k, 0].transAxes,
                            ha="center",
                            fontsize=8,
                        )

                        # Reconstructed
                        axes[k, 1].plot(coords_recon_np[k][0], coords_recon_np[k][1], "r-")
                        axes[k, 1].set_aspect("equal")
                        axes[k, 1].axis("off")
                        if k == 0:
                            axes[k, 1].set_title("Reconstructed")
                        axes[k, 1].text(
                            0.5,
                            -0.1,
                            f"α={angle_recon_np[k][0]:.2f}",
                            transform=axes[k, 1].transAxes,
                            ha="center",
                            fontsize=8,
                        )

                    plt.tight_layout()
                    plt.savefig(f"images/recon_{batches_done}.png")
                    plt.close()

                    # Log to wandb
                    wandb.log(
                        {
                            "dim_plot": wandb.Image(f"images/dim_{batches_done}.png"),
                            "interp_plot": wandb.Image(f"images/interp_{batches_done}.png"),
                            "norm_plot": wandb.Image(f"images/norm_{batches_done}.png"),
                            "recon_plot": wandb.Image(f"images/recon_{batches_done}.png"),
                        }
                    )

        # ---- Validation ----
        with th.no_grad():
            lvae.eval()
            val_rec = val_vol = 0.0
            n = 0
            for batch_v in val_loader:
                coords_v = batch_v[0].to(device)
                angle_v = batch_v[1].to(device)
                # Encode
                z_v = lvae.encoder(coords_v, angle_v)
                # Apply pruning mask (set pruned dimensions to fixed values)
                z_v[:, lvae._p] = lvae._z[lvae._p]
                # Decode
                coords_hat_v, angle_hat_v = lvae.decoder(z_v)
                # Compute reconstruction loss (coords + angle)
                coords_mse_v = nn.functional.mse_loss(coords_v, coords_hat_v)
                angle_mse_v = nn.functional.mse_loss(angle_v, angle_hat_v)
                rec_loss_v = coords_mse_v + angle_mse_v
                # Volume loss with active dimension scaling
                active_ratio_v = lvae.dim / len(lvae._p)
                vol_loss_v = active_ratio_v * lvae.loss_vol(z_v[:, ~lvae._p])
                bsz = coords_v.size(0)
                val_rec += rec_loss_v.item() * bsz
                val_vol += vol_loss_v.item() * bsz
                n += bsz
        val_rec /= n
        val_vol /= n

        # Compute validation total loss using constraint handler
        val_loss_components = ConstraintLosses(
            volume=th.tensor(val_vol, device=device),
            reconstruction=th.tensor(val_rec, device=device),
            performance=th.tensor(0.0, device=device),
        )
        val_thresholds = ConstraintThresholds(
            reconstruction=args.reconstruction_threshold, performance=float("inf")
        )
        val_total = lvae.constraint_handler.compute_loss(val_loss_components, val_thresholds).item()

        # Pass validation reconstruction loss to pruning logic
        lvae.epoch_report(epoch=epoch, callbacks=[], batch=None, loss=loss, pbar=None, val_recon=val_rec)

        if args.track:
            wandb.log(
                {
                    "epoch": epoch,
                    "val_rec": val_rec,
                    "val_vol_loss": val_vol,
                    "val_total_loss": val_total,
                },
                commit=True,
            )

        th.cuda.empty_cache()
        lvae.train()

        # Save models
        if args.save_model and epoch == args.n_epochs - 1:
            ckpt_lvae = {
                "epoch": epoch,
                "batches_done": batches_done,
                "encoder": lvae.encoder.state_dict(),
                "decoder": lvae.decoder.state_dict(),
                "optimizer": lvae.optim.state_dict(),
                "losses": losses.tolist(),
            }
            th.save(ckpt_lvae, "lvae.pth")
            artifact = wandb.Artifact(f"{args.problem_id}_{args.algo}", type="model")
            artifact.add_file("lvae.pth")
            wandb.log_artifact(artifact, aliases=[f"seed_{args.seed}"])

    wandb.finish()
