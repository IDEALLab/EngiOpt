"""DesignLVAE_DP for airfoil designs with performance prediction.

This script extends d_lvae_1d for the airfoil problem, which has a Dict design space:
- 'coords': (2, 192) - x,y coordinates of airfoil shape
- 'angle_of_attack': scalar value that affects performance

The encoder takes both coords and angle_of_attack and produces a unified latent representation.
The decoder reconstructs both coords and angle_of_attack from the latent code.
The performance predictor uses the first perf_dim latent dimensions to predict lift/drag.

Key design choice: angle_of_attack is encoded in the latent space (not treated as a condition)
because it's part of the design and directly affects aerodynamic performance.
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

from engiopt.lvae_2d.constraint_handlers import ConstraintHandler
from engiopt.lvae_2d.constraint_handlers import ConstraintLosses
from engiopt.lvae_2d.constraint_handlers import ConstraintThresholds
from engiopt.lvae_2d.constraint_handlers import create_constraint_handler
from engiopt.lvae_2d.aes import InterpretableDesignLeastVolumeAE_DP
from engiopt.lvae_2d.utils import SNLinearCombo


@dataclass
class Args:
    # Problem and tracking
    problem_id: str = "airfoil"
    """Problem ID to run. Must be airfoil."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """Algorithm name for tracking purposes."""
    track: bool = True
    """Whether to track with Weights & Biases."""
    wandb_project: str = "d_lvae"
    """WandB project name."""
    wandb_entity: str | None = None
    """WandB entity name. If None, uses the default entity."""
    seed: int = 1
    """Random seed for reproducibility."""
    save_model: bool = False
    """Whether to save the model after training."""
    sample_interval: int = 500
    """Interval for sampling designs during training."""

    # Training parameters
    n_epochs: int = 2500
    """Number of training epochs."""
    batch_size: int = 128
    """Batch size for training."""
    lr: float = 1e-4
    """Learning rate for the optimizer."""

    # LVAE-specific
    latent_dim: int = 100
    """Dimensionality of the latent space (overestimate)."""

    # Bezier architecture parameters
    n_control_points: int = 64
    """Number of control points for Bezier curve generation."""

    # Constraint optimization method
    constraint_method: str = "augmented_lagrangian"
    """Constraint method: weighted_sum, augmented_lagrangian, log_barrier, primal_dual, adaptive, softplus_al"""

    # Constraint thresholds (used by all methods except weighted_sum)
    reconstruction_threshold: float = 0.0001
    """Constraint threshold for reconstruction MSE."""
    performance_threshold: float = 0.001
    """Constraint threshold for performance MSE."""

    # Weighted sum parameters (used when constraint_method='weighted_sum')
    w_volume: float = 1.0
    """Weight for volume loss in weighted sum method."""
    w_reconstruction: float = 1.0
    """Weight for reconstruction loss in weighted sum method."""
    w_performance: float = 1.0
    """Weight for performance loss in weighted sum method."""

    # Augmented Lagrangian parameters (used when constraint_method='augmented_lagrangian' or 'softplus_al')
    alpha_r: float = 0.1
    """Learning rate for reconstruction Lagrange multiplier."""
    alpha_p: float = 0.1
    """Learning rate for performance Lagrange multiplier."""
    mu_r_init: float = 1.0
    """Initial penalty coefficient for reconstruction constraint."""
    mu_p_init: float = 1.0
    """Initial penalty coefficient for performance constraint."""
    mu_r: float = 10.0
    """Penalty coefficient for reconstruction constraint."""
    mu_p: float = 10.0
    """Penalty coefficient for performance constraint."""
    warmup_epochs: int = 100
    """Number of epochs to linearly ramp up penalty coefficients."""

    # Softplus AL parameters (used when constraint_method='softplus_al')
    softplus_beta: float = 10.0
    """Smoothness parameter for softplus activation."""

    # Log barrier parameters (used when constraint_method='log_barrier')
    t_init: float = 1.0
    """Initial barrier parameter."""
    t_growth: float = 1.05
    """Multiplicative growth rate for barrier parameter per epoch."""
    t_max: float = 1000.0
    """Maximum barrier parameter."""
    barrier_epsilon: float = 1e-6
    """Safety margin from constraint boundary."""
    fallback_penalty: float = 1e6
    """Penalty multiplier when constraints are violated."""

    # Primal-dual parameters (used when constraint_method='primal_dual')
    lr_dual: float = 0.01
    """Learning rate for dual variable updates."""
    clip_lambda: float = 100.0
    """Maximum value for dual variables."""

    # Adaptive weight parameters (used when constraint_method='adaptive')
    adaptation_lr: float = 0.01
    """Learning rate for automatic weight adaptation."""
    update_frequency: int = 10
    """Update adaptive weights every N steps."""

    pruning_epoch: int = 50
    """Epoch to start pruning dimensions."""
    beta: float = 0.9
    """Momentum for the pruning ratio calculation."""
    eta: float = 1e-4
    """Scaling factor for the volume loss."""

    # MLP predictor parameters
    predictor_hidden_dims: tuple[int, ...] = (256, 128)
    """Hidden dimensions for the MLP predictor."""
    conditional_predictor: bool = False
    """Whether to include conditions in performance prediction (True) or use only latent codes (False)."""
    perf_dim: int = -1
    """Number of latent dimensions dedicated to performance prediction. If -1, uses all latent_dim dimensions."""
    predictor_lipschitz_scale: float = 1.0
    """Lipschitz constant multiplier for SNMLPPredictor."""
    decoder_lipschitz_scale: float = 1.0
    """Lipschitz constant multiplier for TrueSNDecoder."""

    # Dynamic pruning
    pruning_strategy: str = "plummet"
    """Which pruning strategy to use: [plummet, pca_cdf, lognorm, probabilistic]."""
    cdf_threshold: float = 0.99
    """(pca_cdf) Cumulative variance threshold."""
    temperature: float = 1.0
    """(probabilistic) Sampling temperature."""
    plummet_threshold: float = 0.02
    """(plummet) Threshold for pruning dimensions."""
    alpha: float = 0.2
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


class AirfoilEncoder(nn.Module):
    """Encoder for airfoil designs: coords (2, 192) + angle_of_attack (scalar) → latent vector.

    Uses Conv1D for the airfoil coordinates and concatenates the angle_of_attack
    before the final linear projection to latent space.
    """

    def __init__(self, latent_dim: int, n_data_points: int = 192):
        super().__init__()
        self.n_data_points = n_data_points

        # Conv1D layers for airfoil coordinates (2, 192)
        # 192 -> 96 -> 48 -> 24 -> 12 -> 6 -> 3
        self.conv = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # After 6 stride-2 convs: 192 / 2^6 = 3
        conv_output_size = 2048 * 3

        # MLP to combine conv features + angle_of_attack -> latent
        self.mlp = nn.Sequential(
            nn.Linear(conv_output_size + 1, 1024),  # +1 for angle_of_attack
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
        # Encode coords
        h = self.conv(coords)  # (B, 2048, 3)
        h = h.flatten(1)  # (B, 2048*3)

        # Concatenate with angle_of_attack
        combined = th.cat([h, angle], dim=1)  # (B, 2048*3 + 1)

        # Project to latent space
        return self.mlp(combined)


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
    """Spectral normalized Bezier decoder for airfoil coordinates.

    Generates airfoil coords (B, 2, 192) from latent (B, latent_dim).
    Separate head generates angle_of_attack.
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

        return coords, angle_norm

    def denormalize_angle(self, angle_norm: th.Tensor) -> th.Tensor:
        """Denormalize angle from [0, 1] to original range.

        Args:
            angle_norm: (B, 1) normalized angle in [0, 1]

        Returns:
            angle: (B, 1) angle in original range [angle_min, angle_max]
        """
        return angle_norm * (self.angle_max - self.angle_min + self.eps) + self.angle_min


class SNMLPPredictor(nn.Module):
    """Spectral normalized MLP that predicts performance from latent codes."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...] = (256, 128), lipschitz_scale: float = 1.0
    ):
        super().__init__()
        self.lipschitz_scale = lipschitz_scale
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(SNLinearCombo(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        # Final layer: spectral normalized Linear (no activation)
        layers.append(spectral_norm(nn.Linear(prev_dim, output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Predict performance from latent codes."""
        return self.net(x) * self.lipschitz_scale


class ConfigIDLVAE(InterpretableDesignLeastVolumeAE_DP):
    """Airfoil-specific wrapper with augmented Lagrangian and dynamic pruning."""

    def __init__(
        self,
        *args,
        conditional_predictor: bool = True,
        constraint_handler: ConstraintHandler,
        reconstruction_threshold: float = 0.001,
        performance_threshold: float = 0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.conditional_predictor = conditional_predictor
        self.constraint_handler = constraint_handler
        self.reconstruction_threshold = reconstruction_threshold
        self.performance_threshold = performance_threshold

    def encode(self, x):
        """Encode airfoil design (coords, angle) to latent code."""
        coords, angle = x
        z = self.encoder(coords, angle)
        z[:, self._p] = self._z[self._p]  # Apply pruning mask to freeze pruned dimensions
        return z

    def decode(self, z):
        """Decode latent code to airfoil design (coords, angle)."""
        z[:, self._p] = self._z[self._p]  # Apply pruning mask to freeze pruned dimensions
        return self.decoder(z)

    def loss(self, batch, **kwargs):
        """Compute loss components using constraint handler."""
        coords, angle, c, p = batch
        z = self.encode((coords, angle))
        coords_hat, angle_hat = self.decode(z)

        # Update moving mean for pruning statistics
        self._update_moving_mean(z)

        # Only the first pdim dimensions are used for performance prediction
        pz = z[:, : self.pdim]

        # Conditional or unconditional predictor
        p_hat = self.predictor(th.cat([pz, c], dim=-1)) if self.conditional_predictor else self.predictor(pz)

        # Compute individual loss components
        # Reconstruction loss combines coords and angle
        coords_mse = nn.functional.mse_loss(coords, coords_hat)
        angle_mse = nn.functional.mse_loss(angle, angle_hat)
        reconstruction_loss = coords_mse + angle_mse

        performance_loss = nn.functional.mse_loss(p, p_hat)
        active_ratio = self.dim / len(self._p)
        volume_loss = active_ratio * self.loss_vol(z[:, ~self._p])

        # Store components for handler
        self._loss_components = ConstraintLosses(
            volume=volume_loss,
            reconstruction=reconstruction_loss,
            performance=performance_loss,
        )

        # Return as tensor for logging
        return th.stack([reconstruction_loss, performance_loss, volume_loss])

    def compute_total_loss(self):
        """Compute total loss using constraint handler."""
        thresholds = ConstraintThresholds(
            reconstruction=self.reconstruction_threshold,
            performance=self.performance_threshold,
        )
        return self.constraint_handler.compute_loss(self._loss_components, thresholds)

    def update_constraint_handler(self):
        """Update constraint handler state (e.g., dual variables, barrier parameter)."""
        thresholds = ConstraintThresholds(
            reconstruction=self.reconstruction_threshold,
            performance=self.performance_threshold,
        )
        # Detach loss components for handler updates
        detached_losses = ConstraintLosses(
            volume=self._loss_components.volume.detach(),
            reconstruction=self._loss_components.reconstruction.detach(),
            performance=self._loss_components.performance.detach(),
        )
        self.constraint_handler.step(detached_losses, thresholds)

    def epoch_hook(self, epoch, *args, **kwargs):
        """Update current epoch for constraint handler."""
        super().epoch_hook(epoch, *args, **kwargs)
        self.constraint_handler.epoch_hook(epoch)


if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    # Verify Dict design space
    if not isinstance(problem.design_space, spaces.Dict):
        raise ValueError(f"Expected Dict design space, got {type(problem.design_space)}")

    coords_shape = problem.design_space["coords"].shape  # (2, 192)
    n_data_points = coords_shape[1]
    conditions = problem.conditions_keys
    n_conds = len(conditions)

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

    device = th.device("mps" if th.backends.mps.is_available() else "cuda" if th.cuda.is_available() else "cpu")

    # ============================================================================
    # Data Loading and Preprocessing (BEFORE model initialization)
    # ============================================================================
    hf = problem.dataset.with_format("torch")
    train_ds = hf["train"]
    val_ds = hf["val"]

    # Extract designs (coords + angle_of_attack), conditions, and performance
    # Note: Need to handle Dict design space
    coords_train = th.stack([train_ds[i]["optimal_design"]["coords"] for i in range(len(train_ds))])
    angle_train = th.stack([train_ds[i]["optimal_design"]["angle_of_attack"] for i in range(len(train_ds))]).unsqueeze(-1)
    c_train = (
        th.stack([train_ds[key][:] for key in problem.conditions_keys], dim=-1)
        if n_conds > 0
        else th.empty(len(train_ds), 0)
    )
    p_train = train_ds[problem.objectives_keys[0]][:].unsqueeze(-1)

    coords_val = th.stack([val_ds[i]["optimal_design"]["coords"] for i in range(len(val_ds))])
    angle_val = th.stack([val_ds[i]["optimal_design"]["angle_of_attack"] for i in range(len(val_ds))]).unsqueeze(-1)
    c_val = (
        th.stack([val_ds[key][:] for key in problem.conditions_keys], dim=-1) if n_conds > 0 else th.empty(len(val_ds), 0)
    )
    p_val = val_ds[problem.objectives_keys[0]][:].unsqueeze(-1)

    # Scale performance values
    from sklearn.preprocessing import RobustScaler

    p_scaler = RobustScaler()
    p_train_scaled = th.from_numpy(p_scaler.fit_transform(p_train.numpy())).to(p_train.dtype)
    p_val_scaled = th.from_numpy(p_scaler.transform(p_val.numpy())).to(p_val.dtype)

    print(f"\n{'=' * 60}")
    print("Performance Scaling Statistics")
    print(f"{'=' * 60}")
    print(f"RobustScaler center: {p_scaler.center_[0]:.6f}")
    print(f"RobustScaler scale:  {p_scaler.scale_[0]:.6f}")
    print(f"Original range:      [{p_train.min():.6f}, {p_train.max():.6f}]")
    print(f"Scaled range:        [{p_train_scaled.min():.6f}, {p_train_scaled.max():.6f}]")
    print(f"{'=' * 60}\n")

    # Normalize angle_of_attack to [0, 1] for encoder/decoder
    angle_min = angle_train.min()
    angle_max = angle_train.max()
    eps_angle = 1e-7

    angle_train_norm = (angle_train - angle_min) / (angle_max - angle_min + eps_angle)
    angle_val_norm = (angle_val - angle_min) / (angle_max - angle_min + eps_angle)

    print(f"{'=' * 60}")
    print("Angle of Attack Normalization Statistics")
    print(f"{'=' * 60}")
    print(f"Original range:      [{angle_min:.6f}, {angle_max:.6f}]")
    print(f"Normalized range:    [{angle_train_norm.min():.6f}, {angle_train_norm.max():.6f}]")
    print(f"{'=' * 60}\n")

    # Build decoder with angle normalization parameters
    dec = SNBezierDecoder(
        args.latent_dim,
        args.n_control_points,
        n_data_points,
        lipschitz_scale=args.decoder_lipschitz_scale,
        angle_min=angle_min.item(),
        angle_max=angle_max.item(),
    )

    print(
        f"Using SNBezierDecoder with {args.n_control_points} control points (Lipschitz scale: {args.decoder_lipschitz_scale})"
    )
    print(f"Using SNMLPPredictor with spectral normalization (Lipschitz scale: {args.predictor_lipschitz_scale})")

    # ============================================================================
    # Model Initialization (AFTER data preprocessing)
    # ============================================================================

    # Build encoder
    enc = AirfoilEncoder(args.latent_dim, n_data_points)

    # Build MLP predictor
    perf_dim = args.latent_dim if args.perf_dim == -1 else args.perf_dim
    n_perf = 1

    predictor_input_dim = perf_dim + (n_conds if args.conditional_predictor else 0)
    predictor = SNMLPPredictor(
        input_dim=predictor_input_dim,
        output_dim=n_perf,
        hidden_dims=args.predictor_hidden_dims,
        lipschitz_scale=args.predictor_lipschitz_scale,
    )

    print(f"Performance dimensions: {perf_dim}/{args.latent_dim} latent dimensions")
    print(f"Predictor mode: {'Conditional' if args.conditional_predictor else 'Unconditional'}")

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

    # ============================================================================
    # Create constraint handler based on method
    # ============================================================================

    handler_kwargs = {}

    if args.constraint_method == "weighted_sum":
        handler_kwargs = {
            "w_volume": args.w_volume,
            "w_reconstruction": args.w_reconstruction,
            "w_performance": args.w_performance,
        }
    elif args.constraint_method == "augmented_lagrangian":
        handler_kwargs = {
            "mu_r_init": args.mu_r_init,
            "mu_p_init": args.mu_p_init,
            "mu_r_final": args.mu_r,
            "mu_p_final": args.mu_p,
            "alpha_r": args.alpha_r,
            "alpha_p": args.alpha_p,
            "warmup_epochs": args.warmup_epochs,
        }
    elif args.constraint_method == "log_barrier":
        handler_kwargs = {
            "t_init": args.t_init,
            "t_growth": args.t_growth,
            "t_max": args.t_max,
            "epsilon": args.barrier_epsilon,
            "fallback_penalty": args.fallback_penalty,
        }
    elif args.constraint_method == "primal_dual":
        handler_kwargs = {
            "lr_dual": args.lr_dual,
            "clip_lambda": args.clip_lambda,
        }
    elif args.constraint_method == "adaptive":
        handler_kwargs = {
            "w_volume_init": args.w_volume,
            "w_reconstruction_init": args.w_reconstruction,
            "w_performance_init": args.w_performance,
            "adaptation_lr": args.adaptation_lr,
            "update_frequency": args.update_frequency,
        }
    elif args.constraint_method == "softplus_al":
        handler_kwargs = {
            "beta": args.softplus_beta,
            "mu_r_init": args.mu_r_init,
            "mu_p_init": args.mu_p_init,
            "mu_r_final": args.mu_r,
            "mu_p_final": args.mu_p,
            "alpha_r": args.alpha_r,
            "alpha_p": args.alpha_p,
            "warmup_epochs": args.warmup_epochs,
        }
    else:
        raise ValueError(f"Unknown constraint method: {args.constraint_method}")

    constraint_handler = create_constraint_handler(
        method=args.constraint_method,
        device=device,
        **handler_kwargs,
    )

    # Initialize model with all components
    d_lvae = ConfigIDLVAE(
        encoder=enc,
        decoder=dec,
        predictor=predictor,
        optimizer=Adam(
            list(enc.parameters()) + list(dec.parameters()) + list(predictor.parameters()),
            lr=args.lr,
        ),
        constraint_handler=constraint_handler,
        latent_dim=args.latent_dim,
        perf_dim=perf_dim,
        weights=[1.0, 0.0, 1.0],  # Dummy weights
        pruning_epoch=args.pruning_epoch,
        beta=args.beta,
        eta=args.eta,
        pruning_strategy=args.pruning_strategy,
        pruning_params=pruning_params,
        conditional_predictor=args.conditional_predictor,
        reconstruction_threshold=args.reconstruction_threshold,
        performance_threshold=args.performance_threshold,
        min_active_dims=args.min_active_dims,
        max_prune_per_epoch=args.max_prune_per_epoch,
        cooldown_epochs=args.cooldown_epochs,
        k_consecutive=args.k_consecutive,
        recon_tol=args.recon_tol,
    ).to(device)

    # ============================================================================
    # Create DataLoaders
    # ============================================================================

    loader = DataLoader(
        TensorDataset(coords_train, angle_train_norm, c_train, p_train_scaled),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(coords_val, angle_val_norm, c_val, p_val_scaled),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Training loop
    for epoch in range(args.n_epochs):
        d_lvae.epoch_hook(epoch=epoch)

        bar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(bar):
            coords_batch = batch[0].to(device)
            angle_batch = batch[1].to(device)
            c_batch = batch[2].to(device)
            p_batch = batch[3].to(device)

            d_lvae.optim.zero_grad()

            # Compute loss components
            losses = d_lvae.loss((coords_batch, angle_batch, c_batch, p_batch))

            # Compute total loss using constraint handler
            loss = d_lvae.compute_total_loss()

            loss.backward()
            d_lvae.optim.step()

            # Update constraint handler parameters (e.g., Lagrange multipliers, barrier parameter, etc.)
            d_lvae.update_constraint_handler()

            # Get constraint handler metrics
            handler_metrics = constraint_handler.get_metrics()

            bar.set_postfix(
                {
                    "rec": f"{losses[0].item():.3f}",
                    "perf": f"{losses[1].item():.3f}",
                    "vol": f"{losses[2].item():.3f}",
                    "dim": d_lvae.dim,
                    **{k: f"{v:.2f}" if isinstance(v, float) else v for k, v in handler_metrics.items()},
                }
            )

            # Log to wandb
            if args.track:
                batches_done = epoch * len(bar) + i

                wandb.log(
                    {
                        "rec_loss": losses[0].item(),
                        "perf_loss": losses[1].item(),
                        "vol_loss": losses[2].item(),
                        "total_loss": loss.item(),
                        "active_dims": d_lvae.dim,
                        "epoch": epoch,
                        **handler_metrics,
                    }
                )
                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(bar)}] "
                    f"[rec loss: {losses[0].item():.4f}] [perf loss: {losses[1].item():.4f}] "
                    f"[vol loss: {losses[2].item():.4f}] [active dims: {d_lvae.dim}]"
                )

                # Sample and visualize at regular intervals
                if batches_done % args.sample_interval == 0:
                    with th.no_grad():
                        # Encode TRAINING designs (use normalized angles) - pruning is based on training data
                        Xs_coords = coords_train.to(device)
                        Xs_angle_norm = angle_train_norm.to(device)
                        z = d_lvae.encode((Xs_coords, Xs_angle_norm))
                        z_std, idx = th.sort(z.std(0), descending=True)
                        z_mean = z.mean(0)
                        N = (z_std > 0).sum().item()

                        # Generate interpolated designs
                        x_ints_coords = []
                        x_ints_angle = []
                        for alpha in [0, 0.25, 0.5, 0.75, 1]:
                            z_ = (1 - alpha) * z[:25] + alpha * th.roll(z, -1, 0)[:25]
                            coords_, angle_norm_ = d_lvae.decode(z_)
                            # Denormalize angles for plotting
                            angle_ = d_lvae.decoder.denormalize_angle(angle_norm_)
                            x_ints_coords.append(coords_.cpu().numpy())
                            x_ints_angle.append(angle_.cpu().numpy())

                        # Generate random designs
                        z_rand = z_mean.unsqueeze(0).repeat([25, 1])
                        z_rand[:, idx[:N]] += z_std[:N] * th.randn_like(z_rand[:, idx[:N]])
                        coords_rand, angle_rand_norm = d_lvae.decode(z_rand)
                        # Denormalize angles for plotting
                        angle_rand = d_lvae.decoder.denormalize_angle(angle_rand_norm)

                        # Get performance predictions on TRAINING data
                        pz_train = z[:, :perf_dim]
                        if args.conditional_predictor:
                            p_pred_scaled = d_lvae.predictor(th.cat([pz_train, c_train.to(device)], dim=-1))
                        else:
                            p_pred_scaled = d_lvae.predictor(pz_train)

                        # Inverse transform to get true-scale values for plotting
                        p_actual = p_scaler.inverse_transform(p_train_scaled.cpu().numpy()).flatten()
                        p_predicted = p_scaler.inverse_transform(p_pred_scaled.cpu().numpy()).flatten()

                        # Move tensors to CPU for plotting (use original unnormalized angles)
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

                    # Plot 4: Predicted vs actual performance
                    plt.figure(figsize=(8, 8))
                    plt.scatter(p_actual, p_predicted, alpha=0.5, s=20)
                    min_val = min(p_actual.min(), p_predicted.min())
                    max_val = max(p_actual.max(), p_predicted.max())
                    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="1:1 line")
                    plt.xlabel("Actual Performance")
                    plt.ylabel("Predicted Performance")
                    mse_value = np.mean((p_actual - p_predicted) ** 2)
                    plt.title(f"MSE: {mse_value:.4e}")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.axis("equal")
                    plt.tight_layout()
                    plt.savefig(f"images/perf_pred_vs_actual_{batches_done}.png")
                    plt.close()

                    # Log all plots to wandb
                    wandb.log(
                        {
                            "dim_plot": wandb.Image(f"images/dim_{batches_done}.png"),
                            "interp_plot": wandb.Image(f"images/interp_{batches_done}.png"),
                            "norm_plot": wandb.Image(f"images/norm_{batches_done}.png"),
                            "perf_pred_vs_actual": wandb.Image(f"images/perf_pred_vs_actual_{batches_done}.png"),
                        }
                    )

        # ---- Validation (batched, no graph) ----
        with th.no_grad():
            d_lvae.eval()
            val_rec = val_perf = val_vol = 0.0
            n = 0
            for batch_v in val_loader:
                coords_v = batch_v[0].to(device)
                angle_v = batch_v[1].to(device)
                c_v = batch_v[2].to(device)
                p_v = batch_v[3].to(device)
                vlosses = d_lvae.loss((coords_v, angle_v, c_v, p_v))
                bsz = coords_v.size(0)
                val_rec += vlosses[0].item() * bsz
                val_perf += vlosses[1].item() * bsz
                val_vol += vlosses[2].item() * bsz
                n += bsz
        val_rec /= n
        val_perf /= n
        val_vol /= n

        # Compute validation total loss using constraint handler
        val_losses = ConstraintLosses(
            volume=th.tensor(val_vol, device=device),
            reconstruction=th.tensor(val_rec, device=device),
            performance=th.tensor(val_perf, device=device),
        )
        val_thresholds = ConstraintThresholds(
            reconstruction=args.reconstruction_threshold,
            performance=args.performance_threshold,
        )
        val_total = constraint_handler.compute_loss(val_losses, val_thresholds).item()
        val_rec_violation = max(0.0, val_rec - args.reconstruction_threshold)
        val_perf_violation = max(0.0, val_perf - args.performance_threshold)

        # Pass validation reconstruction loss to pruning logic
        d_lvae.epoch_report(epoch=epoch, callbacks=[], batch=None, loss=loss, pbar=None, val_recon=val_rec)

        if args.track:
            wandb.log(
                {
                    "epoch": epoch,
                    "val_rec": val_rec,
                    "val_perf": val_perf,
                    "val_vol_loss": val_vol,
                    "val_total_loss": val_total,
                    "val_reconstruction_violation": val_rec_violation,
                    "val_performance_violation": val_perf_violation,
                },
                commit=True,
            )

        th.cuda.empty_cache()
        d_lvae.train()

        # Save models at end of training
        if args.save_model and epoch == args.n_epochs - 1:
            ckpt_d_lvae = {
                "epoch": epoch,
                "batches_done": batches_done,
                "encoder": d_lvae.encoder.state_dict(),
                "decoder": d_lvae.decoder.state_dict(),
                "predictor": d_lvae.predictor.state_dict(),
                "optimizer": d_lvae.optim.state_dict(),
                "losses": losses.tolist(),
                "p_scaler": p_scaler,  # Save scaler for inference
                "args": vars(args),  # Save args for reference
            }
            th.save(ckpt_d_lvae, "d_lvae.pth")
            artifact = wandb.Artifact(f"{args.problem_id}_{args.algo}", type="model")
            artifact.add_file("d_lvae.pth")
            wandb.log_artifact(artifact, aliases=[f"seed_{args.seed}"])

    wandb.finish()
