"""LVAE_DP for airfoil designs with FACTORIZED architecture.

This is a factorized variant of lvae_airfoil.py with separate encoder/decoder paths
for geometry (coords) and angle_of_attack. This allows explicit control over which
latent dimensions correspond to which input components.

Optimization Problem:
    minimize: volume_loss (number of active latent dimensions)
    subject to: reconstruction_loss ≤ reconstruction_threshold

Architecture:
- FactorizedAirfoilEncoder: Separate paths for coords → z_geom and angle → z_angle
- FactorizedBezierDecoder: Separate paths for z_geom → coords and z_angle → angle
- LeastVolumeAE_DynamicPruning: Dynamically prunes low-variance dimensions

Key difference from lvae_airfoil.py:
- Explicit latent factorization: z = [z_geom | z_angle]
- Can selectively exclude angle dims from volume loss
- Can protect angle dims from pruning

Note: The EngiBench airfoil problem has a Dict design space with:
  - 'coords': (2, 192) - x,y coordinates of airfoil shape
  - 'angle_of_attack': scalar value (encoded separately in latent space)
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
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import tqdm
import tyro
import wandb

from engiopt.lvae_1d.components_1d import FactorizedBezierDecoder, FactorizedConv1DEncoder, Normalizer
from engiopt.lvae_core import (
    ConstraintHandler,
    ConstraintLosses,
    ConstraintThresholds,
    LeastVolumeAE_DynamicPruning,
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
    eta: float = 0
    """Low volume offset to prevent gradient loss at zero volume."""

    # Volume exclusion for imbalanced Dict spaces
    exclude_angle_from_volume: bool = False
    """Exclude angle latent dimensions from volume loss computation (uses heuristic based on input imbalance)."""

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
    pruning_strategy: str = "plummet"
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
        exclude_angle_from_volume: bool = False,
        **kwargs,
    ):
        """Initialize constrained LVAE for airfoil designs.

        Args:
            constraint_handler: Constraint optimization method handler
            reconstruction_threshold: Constraint on reconstruction MSE
            exclude_angle_from_volume: Exclude angle latent dims from volume loss AND pruning
            *args, **kwargs: Passed to parent LeastVolumeAE_DynamicPruning
        """
        super().__init__(*args, **kwargs)
        self.constraint_handler = constraint_handler
        self.reconstruction_threshold = reconstruction_threshold
        self.performance_threshold = float("inf")  # Not used (no performance data)

        # Volume exclusion settings for imbalanced Dict spaces
        self.exclude_angle_from_volume = exclude_angle_from_volume

    def _prune_step(self, epoch, val_recon=None):
        """Override parent pruning to protect angle dimension when exclude_angle_from_volume=True.

        When excluding angle from volume, we also protect it from being pruned.
        This prevents the high-variance angle dimension from dominating pruning decisions.
        """
        if not self.exclude_angle_from_volume:
            # Use parent's pruning logic (prunes all dimensions normally)
            return super()._prune_step(epoch, val_recon)

        # FACTORIZED VERSION: Protect last dimension (angle) from pruning
        # We temporarily mask it as "pruned" to exclude from candidate selection,
        # then restore it after pruning decision is made

        # Save original state of angle dimension
        angle_was_pruned = self._p[-1].item()

        # Temporarily mark angle as pruned to exclude from candidate selection
        self._p[-1] = True

        # Run parent's pruning logic (will skip angle since it's marked as pruned)
        result = super()._prune_step(epoch, val_recon)

        # Restore angle dimension to active state (unless it was actually pruned before)
        if not angle_was_pruned:
            self._p[-1] = False

        return result

    def loss(self, batch, **kwargs):
        """Compute loss components and store for constraint handler.

        Args:
            batch: Tuple of (coords, angle) tensors (already normalized)

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

        # Compute reconstruction loss (simple MSE on normalized data)
        # Data is already normalized: coords in [-1, 1], angle in [0, 1]
        coords_mse = nn.functional.mse_loss(coords, coords_hat)
        angle_mse = nn.functional.mse_loss(angle, angle_hat)

        # Combined reconstruction loss
        rec_loss = coords_mse + angle_mse

        # Store individual components for separate logging
        self._coords_mse = coords_mse
        self._angle_mse = angle_mse

        # Volume loss with optional angle exclusion
        # FACTORIZED VERSION: We KNOW the last dimension is angle
        if self.exclude_angle_from_volume:
            # Exclude last dimension (angle) from volume computation
            mask_geom = self._p[:-1]  # Geometry pruning mask (exclude last dim)
            z_geom = z[:, :-1]  # Geometry latent dims (first latent_dim-1)

            active_geom = (~mask_geom).sum().item()
            total_geom = len(mask_geom)
            active_ratio_geom = active_geom / total_geom if total_geom > 0 else 0.0

            vol_loss = active_ratio_geom * self.loss_vol(z_geom[:, ~mask_geom])
        else:
            # Original: volume on all dimensions (including angle)
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
# Data Preparation Function
# ============================================================================


def prepare_airfoil_data(problem, batch_size: int):
    """Prepare airfoil dataset with normalization.

    Args:
        problem: EngiBench problem instance
        batch_size: Batch size for dataloaders

    Returns:
        tuple containing:
            - train_loader: DataLoader for training data
            - val_loader: DataLoader for validation data
            - coords_normalizer: Normalizer for coords (to [-1, 1])
            - angle_normalizer: Normalizer for angle of attack (to [0, 1])
            - coords_shape: Shape of coords (2, 192)
            - coords_train_raw: Raw training coords for visualization (N, 2, 192)
            - angle_train_raw: Raw training angles for visualization (N, 1)
    """
    # Extract dataset splits
    problem_dataset = problem.dataset.with_format("torch")
    train_ds = problem_dataset["train"]
    val_ds = problem_dataset["val"]

    # Extract coords and angle_of_attack
    coords_train_raw = th.stack([train_ds[i]["optimal_design"]["coords"] for i in range(len(train_ds))])
    coords_val_raw = th.stack([val_ds[i]["optimal_design"]["coords"] for i in range(len(val_ds))])

    angle_train_raw = th.stack([train_ds[i]["optimal_design"]["angle_of_attack"] for i in range(len(train_ds))]).unsqueeze(
        -1
    )
    angle_val_raw = th.stack([val_ds[i]["optimal_design"]["angle_of_attack"] for i in range(len(val_ds))]).unsqueeze(-1)

    # Create coords normalizer (min-max to [0, 1] to match sigmoid output)
    coords_min = coords_train_raw.min()
    coords_max = coords_train_raw.max()
    coords_normalizer = Normalizer(coords_min, coords_max, target_range=(0.0, 1.0))

    # Create angle normalizer (min-max to [0, 1] to match sigmoid output)
    angle_min = angle_train_raw.min()
    angle_max = angle_train_raw.max()
    angle_normalizer = Normalizer(angle_min, angle_max, target_range=(0.0, 1.0))

    # Normalize data
    coords_train_norm = coords_normalizer.normalize(coords_train_raw)
    coords_val_norm = coords_normalizer.normalize(coords_val_raw)
    angle_train_norm = angle_normalizer.normalize(angle_train_raw)
    angle_val_norm = angle_normalizer.normalize(angle_val_raw)

    # Print statistics
    print(f"{'=' * 60}")
    print("Normalization Statistics")
    print(f"{'=' * 60}")
    print(f"Coords original range: [{coords_min:.6f}, {coords_max:.6f}]")
    print(f"Coords normalized range: [{coords_train_norm.min():.6f}, {coords_train_norm.max():.6f}]")
    print(f"Angle original range: [{angle_min:.6f}, {angle_max:.6f}]")
    print(f"Angle normalized range: [{angle_train_norm.min():.6f}, {angle_train_norm.max():.6f}]")
    print(f"{'=' * 60}\n")

    # Create dataloaders
    train_loader = DataLoader(TensorDataset(coords_train_norm, angle_train_norm), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(coords_val_norm, angle_val_norm), batch_size=batch_size, shuffle=False)

    coords_shape = problem.design_space["coords"].shape

    return (
        train_loader,
        val_loader,
        coords_normalizer,
        angle_normalizer,
        coords_shape,
        coords_train_raw,
        angle_train_raw,
    )


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

    # ---- Prepare Data ----
    (
        train_loader,
        val_loader,
        coords_normalizer,
        angle_normalizer,
        coords_shape,
        coords_train_raw,
        angle_train_raw,
    ) = prepare_airfoil_data(problem, args.batch_size)
    n_data_points = coords_shape[1]  # 192

    # Build factorized encoder and decoder (imported from cmpnts.py)
    enc = FactorizedConv1DEncoder(
        in_channels=2,
        in_features=n_data_points,
        latent_dim=args.latent_dim,
        n_aux=1,  # 1 auxiliary dimension for angle of attack
        use_spectral_norm=True,  # Enable spectral normalization for Lipschitz constraint
    )
    # Note: angle normalization/denormalization is handled externally via angle_normalizer
    dec = FactorizedBezierDecoder(
        latent_dim=args.latent_dim,
        n_aux=1,  # 1 auxiliary dimension for angle of attack
        n_control_points=args.n_control_points,
        n_data_points=n_data_points,
        lipschitz_scale=args.decoder_lipschitz_scale,
    )

    print(
        f"Using Bezier-based decoder with {args.n_control_points} control points (Lipschitz scale: {args.decoder_lipschitz_scale})"
    )
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
        handler_kwargs.update(
            {
                "w_volume": args.w_volume,
                "w_reconstruction": args.w_reconstruction,
                "w_performance": 0.0,  # Not used
            }
        )
    elif args.constraint_method in ["augmented_lagrangian", "softplus_al"]:
        handler_kwargs.update(
            {
                "alpha_r": args.alpha_r,
                "alpha_p": 0.0,  # Not used (no performance)
                "mu_r_init": args.mu_r_init,
                "mu_p_init": 0.0,
                "mu_r_final": args.mu_r,
                "mu_p_final": 0.0,
                "warmup_epochs": args.warmup_epochs,
            }
        )
        if args.constraint_method == "softplus_al":
            handler_kwargs["beta"] = args.softplus_beta
    elif args.constraint_method == "log_barrier":
        handler_kwargs.update(
            {
                "t_init": args.t_init,
                "t_growth": args.t_growth,
                "t_max": args.t_max,
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
                "w_volume_init": 1.0,
                "w_reconstruction_init": 1.0,
                "w_performance_init": 0.0,
                "adaptation_lr": args.adaptation_lr,
                "update_frequency": args.update_frequency,
            }
        )
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
        exclude_angle_from_volume=args.exclude_angle_from_volume,
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

    # ---- Training loop ----
    for epoch in range(args.n_epochs):
        lvae.epoch_hook(epoch=epoch)

        bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")
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
                        # Get normalized training data for encoding
                        coords_train_norm = coords_normalizer.normalize(coords_train_raw).to(device)
                        angle_train_norm = angle_normalizer.normalize(angle_train_raw).to(device)

                        # Encode TRAINING designs - pruning is based on training data
                        z = lvae.encoder(coords_train_norm, angle_train_norm)
                        # Apply pruning mask (set pruned dimensions to fixed values)
                        z[:, lvae._p] = lvae._z[lvae._p]

                        # For variance calculation and plotting, exclude angle dimension if requested
                        if args.exclude_angle_from_volume:
                            # Compute variance only on geometry dimensions (exclude last dim = angle)
                            z_geom = z[:, :-1]
                            z_std_geom, idx_geom = th.sort(z_geom.std(0), descending=True)
                            z_mean = z.mean(0)
                            N = (z_std_geom > 0).sum().item()
                            # For plotting, use geometry variances
                            z_std = z_std_geom
                            idx = idx_geom
                        else:
                            # Original: compute variance over all dimensions
                            z_std, idx = th.sort(z.std(0), descending=True)
                            z_mean = z.mean(0)
                            N = (z_std > 0).sum().item()

                        # Generate interpolated designs
                        x_ints_coords = []
                        x_ints_angle = []
                        for alpha in [0, 0.25, 0.5, 0.75, 1]:
                            z_ = (1 - alpha) * z[:25] + alpha * th.roll(z, -1, 0)[:25]
                            coords_norm_, angle_norm_ = lvae.decoder(z_)
                            # Denormalize for plotting
                            coords_ = coords_normalizer.denormalize(coords_norm_)
                            angle_ = angle_normalizer.denormalize(angle_norm_)
                            x_ints_coords.append(coords_.cpu().numpy())
                            x_ints_angle.append(angle_.cpu().numpy())

                        # Generate random designs
                        z_rand = z_mean.unsqueeze(0).repeat([25, 1])
                        if args.exclude_angle_from_volume:
                            # Sample only geometry dimensions (first N), keep angle at mean
                            z_rand[:, idx_geom[:N]] += z_std_geom[:N] * th.randn_like(z_rand[:, idx_geom[:N]])
                        else:
                            # Sample all dimensions
                            z_rand[:, idx[:N]] += z_std[:N] * th.randn_like(z_rand[:, idx[:N]])
                        coords_rand_norm, angle_rand_norm = lvae.decoder(z_rand)
                        # Denormalize for plotting
                        coords_rand = coords_normalizer.denormalize(coords_rand_norm)
                        angle_rand = angle_normalizer.denormalize(angle_rand_norm)

                        # Move to CPU (use raw training data for visualization)
                        z_std_cpu = z_std.cpu().numpy()
                        coords_train_cpu = coords_train_raw.cpu().numpy()
                        angle_train_cpu = angle_train_raw.cpu().numpy()
                        coords_rand_np = coords_rand.cpu().numpy()
                        angle_rand_np = angle_rand.cpu().numpy()

                    # Plot 1: Latent dimension statistics
                    plt.figure(figsize=(12, 6))
                    plt.subplot(211)
                    plt.bar(np.arange(len(z_std_cpu)), z_std_cpu)
                    plt.yscale("log")
                    plt.xlabel("Latent dimension index")
                    plt.ylabel("Standard deviation")
                    title_suffix = " (geometry only)" if args.exclude_angle_from_volume else ""
                    plt.title(f"Number of principal components = {N}{title_suffix}")
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
                        airfoil = coords_train_cpu[i]
                        angle_val = angle_train_cpu[i][0]  # Extract scalar angle value
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
                    coords_orig = coords_train_cpu[:5]
                    angle_orig = angle_train_cpu[:5]
                    coords_recon_norm, angle_recon_norm = lvae.decoder(z[:5])
                    # Denormalize reconstructions
                    coords_recon = coords_normalizer.denormalize(coords_recon_norm)
                    angle_recon = angle_normalizer.denormalize(angle_recon_norm)
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
        val_thresholds = ConstraintThresholds(reconstruction=args.reconstruction_threshold, performance=float("inf"))
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
