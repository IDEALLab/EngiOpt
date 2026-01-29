"""Constrained LVAE_DP for 2D designs (no performance prediction).

This implementation uses constraint-based optimization to minimize latent volume
subject to reconstruction accuracy constraints. Unlike d_lvae_2d.py, this variant
does NOT include performance prediction - only reconstruction.

Optimization Problem:
    minimize: volume_loss (number of active latent dimensions)
    subject to: reconstruction_loss ≤ reconstruction_threshold

Architecture:
- Encoder: Conv2D encoder (100x100 → latent code)
- TrueSNDecoder: Spectral normalized decoder (latent → 100x100)
- LeastVolumeAE_DynamicPruning: Dynamically prunes low-variance dimensions

Constraint-aware pruning:
- Dimensions are only pruned when reconstruction constraint is satisfied
- Volume loss calculated over entire latent space (not scaled by active dimensions)

Supported constraint methods:
- penalty_method (recommended - simple, robust, auto-adapting)
- weighted_sum, augmented_lagrangian, log_barrier, primal_dual, adaptive, softplus_al
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import os
import random
import time

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import transforms
import tqdm
import tyro
import wandb

from engiopt.lvae_core import (
    ConstraintHandler,
    ConstraintLosses,
    ConstraintThresholds,
    LeastVolumeAE_DynamicPruning,
    TrueSNDeconv2DCombo,
    create_constraint_handler,
    spectral_norm_conv,
)


@dataclass
class Args:
    # Problem and tracking
    problem_id: str = "heatconduction2d"
    """Problem ID to run. Must be one of the built-in problems in engibench."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """Algorithm name for tracking purposes."""
    track: bool = True
    """Whether to track with Weights & Biases."""
    wandb_project: str = "c_lvae"
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
    latent_dim: int = 250
    """Dimensionality of the latent space (overestimate)."""

    # Constraint optimization method
    constraint_method: str = "penalty_method"
    """Constraint method: adaptive_constraint (recommended), penalty_method, weighted_sum, augmented_lagrangian, log_barrier, primal_dual, adaptive, softplus_al"""

    # Constraint thresholds (used by all methods except weighted_sum)
    reconstruction_threshold: float = 0.001
    """Constraint threshold for reconstruction MSE. Volume is minimized subject to reconstruction_loss <= this value."""

    # === Tier 1: Universal Constraint Parameters ===

    # Weighted sum parameters (used when constraint_method='weighted_sum')
    w_volume: float = 1.0
    """Weight for volume loss in weighted sum method."""
    w_reconstruction: float = 1.0
    """Weight for reconstruction loss in weighted sum method."""

    # === Tier 2: Shared Penalty Parameters ===
    # Used by: penalty_method, adaptive_constraint, augmented_lagrangian, log_barrier, softplus_al

    penalty_init: float = 1.0
    """Initial penalty coefficient (low to allow early violations). Used by penalty-based methods.

    Method-specific mappings:
    - penalty_method: penalty_weight_init
    - adaptive_constraint: penalty_init
    - augmented_lagrangian: mu_r_init
    - log_barrier: t_init
    - softplus_al: mu_r_init

    Recommended defaults:
    - adaptive_constraint: 1.0 (volume gating allows low start)
    - penalty_method: 100.0 (no volume gating, needs stronger initial penalty)
    - augmented_lagrangian: 100.0
    - log_barrier: 1.0
    """

    penalty_max: float = 1000.0
    """Maximum penalty coefficient (prevents numerical issues). Used by penalty-based methods.

    Method-specific mappings:
    - penalty_method: penalty_weight_final
    - adaptive_constraint: penalty_max
    - augmented_lagrangian: mu_r
    - log_barrier: t_max
    - softplus_al: mu_r

    Recommended: 1000.0 (works across most applications)
    """

    penalty_growth: float = 1.1
    """Multiplicative growth rate for penalty coefficient. Used by penalty-based methods.

    Method-specific mappings:
    - penalty_method: penalty_growth_rate
    - adaptive_constraint: penalty_growth
    - log_barrier: t_growth

    Recommended: 1.05-1.1 (5-10% increase per epoch/step)
    """

    penalty_warmup_epochs: int = 0
    """Epochs to linearly ramp penalty from penalty_init to penalty_max. Used by some methods.

    Method-specific mappings:
    - penalty_method: warmup_epochs
    - augmented_lagrangian: warmup_epochs

    Set to 0 to disable linear warmup (use reactive growth instead).
    Recommended: 0 for adaptive_constraint (uses reactive growth), 100 for penalty_method/augmented_lagrangian
    """

    volume_warmup_epochs: int = 0
    """Epochs to polynomially (2nd order) ramp volume loss weight from 0 to 1.
    Used by ALL methods via base ConstraintHandler class.

    This prevents early volume collapse by delaying volume loss introduction.
    Recommended: 0 for adaptive_constraint (volume gating is better), 50-100 for penalty_method
    """

    # === Tier 3: Method-Specific Parameters ===

    # Adaptive Constraint specific (used when constraint_method='adaptive_constraint')
    enable_volume_gating: bool = True
    """Enable volume gating to prevent collapse (recommended: True). Set to False for standard penalty method behavior."""
    safety_margin: float = 0.1
    """Relative violation threshold for volume gating (0.1 = 10% of threshold)."""
    penalty_decay: float = 0.95
    """Decay rate for adaptive penalty when constraints satisfied (0.95 = 5% decrease per epoch)."""
    transition_sharpness: float = 2.0
    """Smoothness of volume gate transition (higher = sharper)."""

    # Augmented Lagrangian specific (used when constraint_method='augmented_lagrangian' or 'softplus_al')
    alpha_r: float = 1.0
    """Learning rate for reconstruction Lagrange multiplier (dual ascent)."""

    # Softplus AL specific (used when constraint_method='softplus_al')
    softplus_beta: float = 10.0
    """Smoothness parameter for softplus activation (larger = sharper, closer to ReLU)."""

    # Log Barrier specific (used when constraint_method='log_barrier')
    barrier_epsilon: float = 1e-6
    """Safety margin from constraint boundary."""
    fallback_penalty: float = 1e6
    """Penalty multiplier when constraints are violated (infeasible)."""

    # Primal-Dual specific (used when constraint_method='primal_dual')
    lr_dual: float = 0.01
    """Learning rate for dual variable updates."""
    clip_lambda: float = 100.0
    """Maximum value for dual variables (Lagrange multipliers)."""

    # Adaptive Weight specific (used when constraint_method='adaptive')
    adaptation_lr: float = 0.01
    """Learning rate for automatic weight adaptation."""
    update_frequency: int = 10
    """Update adaptive weights every N steps."""

    pruning_epoch: int = 500
    """Epoch to start pruning dimensions."""
    beta: float = 0.9
    """Momentum for the pruning ratio calculation."""
    eta: float = 1e-4
    """Scaling factor for the volume loss."""
    resize_dimensions: tuple[int, int] = (100, 100)
    """Dimensions to resize input images to before encoding/decoding."""

    decoder_lipschitz_scale: float = 1.0
    """Lipschitz constant multiplier for TrueSNDecoder (1.0 = strict 1-Lipschitz, >1 = relaxed c-Lipschitz)."""

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
    percentile: float = 0.01
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


class Encoder(nn.Module):
    """Input → Latent vector via ConvNet.

    • Input   [100x100]
    • Conv1   [50x50]   (k=4, s=2, p=1)
    • Conv2   [25x25]   (k=4, s=2, p=1)
    • Conv3   [13x13]   (k=3, s=2, p=1)
    • Conv4   [7x7]     (k=3, s=2, p=1)
    • Conv5   [1x1]     (k=7, s=1, p=0).
    """

    def __init__(self, latent_dim: int, design_shape: tuple[int, int], resize_dimensions: tuple[int, int] = (100, 100)):
        super().__init__()
        self.resize_in = transforms.Resize(resize_dimensions)
        self.design_shape = design_shape

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 100→50
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 50→25
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # 25→13
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),  # 13→7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final 7x7 conv produces (B, latent_dim, 1, 1) -> flatten to (B, latent_dim)
        self.to_latent = nn.Conv2d(512, latent_dim, kernel_size=7, stride=1, padding=0, bias=True)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.resize_in(x)  # (B,1,100,100)
        h = self.features(x)  # (B,512,7,7)
        return self.to_latent(h).flatten(1)  # (B,latent_dim)


class TrueSNDecoder(nn.Module):
    """Decoder with spectral normalization for 1-Lipschitz bound.

    Same architecture as standard decoder (7→13→25→50→100) but with spectral
    normalization applied to all linear and convolutional layers.

    • Latent   [latent_dim]
    • Linear   [512*7*7]
    • Reshape  [512x7x7]
    • Deconv1  [256x13x13]  (k=3, s=2, p=1)
    • Deconv2  [128x25x25]  (k=3, s=2, p=1)
    • Deconv3  [64x50x50]   (k=4, s=2, p=1)
    • Deconv4  [1x100x100]  (k=4, s=2, p=1)

    The lipschitz_scale parameter relaxes the strict 1-Lipschitz constraint:
    - lipschitz_scale = 1.0: Strict 1-Lipschitz (default)
    - lipschitz_scale > 1.0: Relaxed c-Lipschitz for higher expressiveness
    """

    def __init__(self, latent_dim: int, design_shape: tuple[int, int], lipschitz_scale: float = 1.0):
        super().__init__()
        self.design_shape = design_shape
        self.resize_out = transforms.Resize(self.design_shape)
        self.lipschitz_scale = lipschitz_scale

        # Spectral normalized linear projection
        self.proj = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 512 * 7 * 7)),
            nn.ReLU(inplace=True),
        )

        # Build deconvolutional layers with spectral normalization
        self.deconv = nn.Sequential(
            # 7→13 (input shape: 7x7)
            TrueSNDeconv2DCombo(
                input_shape=(7, 7),
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            # 13→25 (input shape: 13x13)
            TrueSNDeconv2DCombo(
                input_shape=(13, 13),
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            # 25→50 (input shape: 25x25)
            TrueSNDeconv2DCombo(
                input_shape=(25, 25),
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            # 50→100 (input shape: 50x50) - final layer with sigmoid
            spectral_norm_conv(
                nn.ConvTranspose2d(
                    64,
                    1,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False,
                ),
                (50, 50),
            ),
            nn.Sigmoid(),
        )

    def forward(self, z: th.Tensor) -> th.Tensor:
        """Forward pass through the spectral normalized decoder."""
        x = self.proj(z).view(z.size(0), 512, 7, 7)  # (B,512,7,7)
        x = self.deconv(x)  # (B,1,100,100)
        x = self.resize_out(x)  # (B,1,H_orig,W_orig)
        return x * self.lipschitz_scale


class ConstrainedLVAE(LeastVolumeAE_DynamicPruning):
    """Wrapper around LeastVolumeAE_DynamicPruning with constraint-based optimization.

    This variant omits performance prediction and only minimizes volume subject to
    reconstruction constraints.

    Supports multiple constraint optimization methods via constraint_handler.

    Implements constraint-aware pruning: dimensions are only pruned when reconstruction
    constraint is satisfied.
    """

    def __init__(
        self,
        *args,
        constraint_handler: ConstraintHandler,
        reconstruction_threshold: float = 0.001,
        **kwargs,
    ):
        """Initialize with constraint handler.

        Args:
            constraint_handler: Constraint optimization method handler
            reconstruction_threshold: Constraint on reconstruction MSE
            *args, **kwargs: Passed to parent LeastVolumeAE_DynamicPruning
        """
        super().__init__(*args, **kwargs)
        self.constraint_handler = constraint_handler
        self.reconstruction_threshold = reconstruction_threshold
        self.performance_threshold = float("inf")  # Not used (no performance prediction)
        self._constraints_satisfied = False

    def loss(self, batch, **kwargs):
        """Compute loss components and total loss using constraint handler.

        Args:
            batch: Tensor of designs (B, 1, H, W)

        Returns:
            torch.Tensor: [reconstruction_loss, volume_loss] for logging
        """
        x = batch
        z = self.encode(x)
        x_hat = self.decode(z)

        # Update moving mean for pruning statistics
        self._update_moving_mean(z)

        # Compute individual loss components
        reconstruction_loss = self.loss_rec(x, x_hat)
        # Volume loss calculated over entire latent space (not scaled by active dimension ratio)
        volume_loss = self.loss_vol(z[:, ~self._p]) if (~self._p).any() else th.tensor(0.0, device=z.device)

        # Store components for handler (performance=0 since no performance prediction)
        self._loss_components = ConstraintLosses(
            volume=volume_loss,
            reconstruction=reconstruction_loss,
            performance=th.tensor(0.0, device=volume_loss.device),
        )

        # Return as tensor for logging
        return th.stack([reconstruction_loss, volume_loss])

    def compute_total_loss(self):
        """Compute total loss using constraint handler.

        This should be called after loss() to get the actual loss for backprop.
        """
        thresholds = ConstraintThresholds(
            reconstruction=self.reconstruction_threshold,
            performance=self.performance_threshold,  # inf, effectively ignored
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

    def _check_constraints_satisfied(self) -> bool:
        """Check if reconstruction constraint is satisfied.

        Returns:
            True if reconstruction loss is below threshold, False otherwise.
        """
        if self._loss_components is None:
            return False
        return self._loss_components.reconstruction.item() <= self.reconstruction_threshold

    def _prune_step(self, epoch, val_recon=None):
        """Prune dimensions only if reconstruction constraint is satisfied.

        Overrides parent method to add constraint checking before pruning.

        Args:
            epoch: Current epoch number
            val_recon: Optional validation reconstruction loss
        """
        # Only prune if reconstruction constraint is satisfied
        if not self._check_constraints_satisfied():
            return  # Skip pruning if constraint not satisfied
        super()._prune_step(epoch, val_recon=val_recon)

    def epoch_hook(self, epoch, *args, **kwargs):
        """Update current epoch for constraint handler."""
        super().epoch_hook(epoch, *args, **kwargs)
        self.constraint_handler.epoch_hook(epoch)


if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    design_shape = problem.design_space.shape

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

    # Build encoder and decoder (always use spectral normalization)
    enc = Encoder(args.latent_dim, design_shape, args.resize_dimensions)
    dec = TrueSNDecoder(args.latent_dim, design_shape, lipschitz_scale=args.decoder_lipschitz_scale)
    print(f"Using TrueSNDecoder with spectral normalization (Lipschitz scale: {args.decoder_lipschitz_scale})")

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

    # Create constraint handler based on method
    # Note: unified parameters (penalty_init, penalty_max, penalty_growth, penalty_warmup_epochs)
    # are mapped to method-specific names here
    handler_kwargs = {"device": device, "volume_warmup_epochs": args.volume_warmup_epochs}

    if args.constraint_method == "weighted_sum":
        handler_kwargs.update(
            {
                "w_volume": args.w_volume,
                "w_reconstruction": args.w_reconstruction,
                "w_performance": 0.0,  # Not used (single constraint)
            }
        )

    elif args.constraint_method == "penalty_method":
        handler_kwargs.update(
            {
                "penalty_weight_init": args.penalty_init,  # Unified parameter
                "penalty_weight_final": args.penalty_max,  # Unified parameter
                "penalty_growth_rate": args.penalty_growth,  # Unified parameter
                "warmup_epochs": args.penalty_warmup_epochs,  # Unified parameter
            }
        )

    elif args.constraint_method == "adaptive_constraint":
        handler_kwargs.update(
            {
                "enable_volume_gating": args.enable_volume_gating,
                "safety_margin": args.safety_margin,
                "penalty_init": args.penalty_init,  # Unified parameter
                "penalty_max": args.penalty_max,  # Unified parameter
                "penalty_growth": args.penalty_growth,  # Unified parameter
                "penalty_decay": args.penalty_decay,
                "transition_sharpness": args.transition_sharpness,
            }
        )

    elif args.constraint_method == "augmented_lagrangian":
        handler_kwargs.update(
            {
                "mu_r_init": args.penalty_init,  # Unified parameter
                "mu_p_init": 0.0,  # Not used (single constraint)
                "mu_r_final": args.penalty_max,  # Unified parameter
                "mu_p_final": 0.0,  # Not used (single constraint)
                "alpha_r": args.alpha_r,
                "alpha_p": 0.0,  # Not used (single constraint)
                "warmup_epochs": args.penalty_warmup_epochs,  # Unified parameter
            }
        )

    elif args.constraint_method == "softplus_al":
        handler_kwargs.update(
            {
                "beta": args.softplus_beta,
                "mu_r_init": args.penalty_init,  # Unified parameter
                "mu_p_init": 0.0,  # Not used (single constraint)
                "mu_r_final": args.penalty_max,  # Unified parameter
                "mu_p_final": 0.0,  # Not used (single constraint)
                "alpha_r": args.alpha_r,
                "alpha_p": 0.0,  # Not used (single constraint)
                "warmup_epochs": args.penalty_warmup_epochs,  # Unified parameter
            }
        )

    elif args.constraint_method == "log_barrier":
        handler_kwargs.update(
            {
                "t_init": args.penalty_init,  # Unified parameter
                "t_growth": args.penalty_growth,  # Unified parameter
                "t_max": args.penalty_max,  # Unified parameter
                "epsilon": args.barrier_epsilon,
                "fallback_penalty": args.fallback_penalty,
            }
        )

    elif args.constraint_method == "primal_dual":
        handler_kwargs.update(
            {
                "lr_dual": args.lr_dual,
                "clip_lambda": args.clip_lambda,
            }
        )

    elif args.constraint_method == "adaptive":
        handler_kwargs.update(
            {
                "w_volume_init": args.w_volume,
                "w_reconstruction_init": args.w_reconstruction,
                "w_performance_init": 0.0,  # Not used (single constraint)
                "adaptation_lr": args.adaptation_lr,
                "update_frequency": args.update_frequency,
            }
        )

    else:
        raise ValueError(f"Unknown constraint_method: {args.constraint_method}")

    constraint_handler = create_constraint_handler(
        method=args.constraint_method,
        **handler_kwargs,
    )

    print(f"\n{'=' * 60}")
    print(f"Constraint Method: {args.constraint_method}")
    print(f"Reconstruction Threshold: {args.reconstruction_threshold}")
    print(f"Volume Warmup Epochs: {args.volume_warmup_epochs}")
    print(f"{'=' * 60}\n")

    # Initialize constrained LVAE with dynamic pruning
    c_lvae = ConstrainedLVAE(
        encoder=enc,
        decoder=dec,
        optimizer=Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr),
        latent_dim=args.latent_dim,
        weights=[1.0, 0.0],  # Dummy weights (not directly used, handler controls loss)
        pruning_epoch=args.pruning_epoch,
        beta=args.beta,
        eta=args.eta,
        pruning_strategy=args.pruning_strategy,
        pruning_params=pruning_params,
        constraint_handler=constraint_handler,
        reconstruction_threshold=args.reconstruction_threshold,
        min_active_dims=args.min_active_dims,
        max_prune_per_epoch=args.max_prune_per_epoch,
        cooldown_epochs=args.cooldown_epochs,
        k_consecutive=args.k_consecutive,
        recon_tol=args.recon_tol,
    ).to(device)

    # ---- DataLoader ----
    hf = problem.dataset.with_format("torch")
    train_ds = hf["train"]
    val_ds = hf["val"]
    test_ds = hf["test"]

    x_train = train_ds["optimal_design"][:].unsqueeze(1)
    x_val = val_ds["optimal_design"][:].unsqueeze(1)
    x_test = test_ds["optimal_design"][:].unsqueeze(1)

    loader = DataLoader(TensorDataset(x_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val), batch_size=args.batch_size, shuffle=False)

    # ---- Training loop ----
    for epoch in range(args.n_epochs):
        c_lvae.epoch_hook(epoch=epoch)

        bar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(bar):
            x_batch = batch[0].to(device)
            c_lvae.optim.zero_grad()

            # Compute loss components (rec, vol) for logging
            losses = c_lvae.loss(x_batch)  # [rec, vol]

            # Compute total loss using constraint handler
            loss = c_lvae.compute_total_loss()

            loss.backward()
            c_lvae.optim.step()

            # Update constraint handler state (dual variables, barrier params, etc.)
            c_lvae.update_constraint_handler()

            # Get handler metrics for display
            handler_metrics = c_lvae.constraint_handler.get_metrics()
            postfix_dict = {
                "rec": f"{losses[0].item():.3f}",
                "vol": f"{losses[1].item():.3f}",
                "dim": c_lvae.dim,
            }
            # Add first 2 handler metrics to postfix for display
            for idx, (k, v) in enumerate(handler_metrics.items()):
                if idx < 2:  # Only show first 2 metrics
                    postfix_dict[k.split("/")[-1]] = f"{v:.2f}"
            bar.set_postfix(postfix_dict)

            # Log to wandb
            if args.track:
                batches_done = epoch * len(bar) + i

                log_dict = {
                    "rec_loss": losses[0].item(),
                    "vol_loss": losses[1].item(),
                    "total_loss": loss.item(),
                    "active_dims": c_lvae.dim,
                    "epoch": epoch,
                }
                # Add all handler metrics to log
                log_dict.update(handler_metrics)

                wandb.log(log_dict)
                handler_str = " ".join([f"[{k}: {v:.2f}]" for k, v in list(handler_metrics.items())[:2]])
                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(bar)}] "
                    f"[rec loss: {losses[0].item():.4f}] [vol loss: {losses[1].item():.4f}] "
                    f"[active dims: {c_lvae.dim}] {handler_str}"
                )

                # Sample and visualize at regular intervals
                if batches_done % args.sample_interval == 0:
                    with th.no_grad():
                        # Encode TRAINING designs - pruning is based on training data
                        Xs = x_train.to(device)
                        z = c_lvae.encode(Xs)
                        z_std, idx = th.sort(z.std(0), descending=True)
                        z_mean = z.mean(0)
                        N = (z_std > 0).sum().item()

                        # Generate interpolated designs
                        x_ints = []
                        for alpha in [0, 0.25, 0.5, 0.75, 1]:
                            z_ = (1 - alpha) * z[:25] + alpha * th.roll(z, -1, 0)[:25]
                            x_ints.append(c_lvae.decode(z_).cpu().numpy())

                        # Generate random designs
                        z_rand = z_mean.unsqueeze(0).repeat([25, 1])
                        z_rand[:, idx[:N]] += z_std[:N] * th.randn_like(z_rand[:, idx[:N]])
                        x_rand = c_lvae.decode(z_rand).cpu().numpy()

                        # Move tensors to CPU for plotting
                        z_std_cpu = z_std.cpu().numpy()
                        Xs_cpu = Xs.cpu().numpy()

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

                    # Plot 2: Interpolated designs
                    fig, axs = plt.subplots(25, 6, figsize=(12, 25))
                    for i_row, j in product(range(25), range(5)):
                        axs[i_row, j + 1].imshow(x_ints[j][i_row].reshape(design_shape))
                        axs[i_row, j + 1].axis("off")
                        axs[i_row, j + 1].set_aspect("equal")
                    for ax, alpha in zip(axs[0, 1:], [0, 0.25, 0.5, 0.75, 1]):
                        ax.set_title(rf"$\alpha$ = {alpha}")
                    for i_row in range(25):
                        axs[i_row, 0].imshow(Xs_cpu[i_row].reshape(design_shape))
                        axs[i_row, 0].axis("off")
                        axs[i_row, 0].set_aspect("equal")
                    axs[0, 0].set_title("groundtruth")
                    fig.tight_layout()
                    plt.savefig(f"images/interp_{batches_done}.png")
                    plt.close()

                    # Plot 3: Random designs from latent space
                    fig, axs = plt.subplots(5, 5, figsize=(15, 7.5))
                    for k, (i_row, j) in enumerate(product(range(5), range(5))):
                        axs[i_row, j].imshow(x_rand[k].reshape(design_shape))
                        axs[i_row, j].axis("off")
                        axs[i_row, j].set_aspect("equal")
                    fig.tight_layout()
                    plt.suptitle("Gaussian random designs from latent space")
                    plt.savefig(f"images/norm_{batches_done}.png")
                    plt.close()

                    # Log all plots to wandb
                    wandb.log(
                        {
                            "dim_plot": wandb.Image(f"images/dim_{batches_done}.png"),
                            "interp_plot": wandb.Image(f"images/interp_{batches_done}.png"),
                            "norm_plot": wandb.Image(f"images/norm_{batches_done}.png"),
                        }
                    )

        # ---- Validation (batched, no graph) ----
        with th.no_grad():
            c_lvae.eval()
            val_rec = val_vol = 0.0
            n = 0
            for batch_v in val_loader:
                x_v = batch_v[0].to(device)
                vlosses = c_lvae.loss(x_v)
                bsz = x_v.size(0)
                val_rec += vlosses[0].item() * bsz
                val_vol += vlosses[1].item() * bsz
                n += bsz
        val_rec /= n
        val_vol /= n

        # Compute validation total loss using constraint handler
        val_losses = ConstraintLosses(
            volume=th.tensor(val_vol, device=device),
            reconstruction=th.tensor(val_rec, device=device),
            performance=th.tensor(0.0, device=device),
        )
        val_thresholds = ConstraintThresholds(
            reconstruction=args.reconstruction_threshold,
            performance=float("inf"),
        )
        val_total = c_lvae.constraint_handler.compute_loss(val_losses, val_thresholds).item()

        # Pass validation reconstruction loss to pruning logic
        c_lvae.epoch_report(epoch=epoch, callbacks=[], batch=None, loss=loss, pbar=None, val_recon=val_rec)

        if args.track:
            val_log_dict = {
                "epoch": epoch,
                "val_rec": val_rec,
                "val_vol_loss": val_vol,
                "val_total_loss": val_total,
            }
            wandb.log(val_log_dict, commit=True)

        th.cuda.empty_cache()
        c_lvae.train()

        # Save models at end of training
        if args.save_model and epoch == args.n_epochs - 1:
            ckpt_c_lvae = {
                "epoch": epoch,
                "batches_done": batches_done,
                "encoder": c_lvae.encoder.state_dict(),
                "decoder": c_lvae.decoder.state_dict(),
                "optimizer": c_lvae.optim.state_dict(),
                "losses": losses.tolist(),
                "args": vars(args),  # Save args for reference
            }
            th.save(ckpt_c_lvae, "c_lvae.pth")
            artifact = wandb.Artifact(f"{args.problem_id}_{args.algo}", type="model")
            artifact.add_file("c_lvae.pth")
            wandb.log_artifact(artifact, aliases=[f"seed_{args.seed}"])

    wandb.finish()
