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
    latent_dim: int = 250
    """Dimensionality of the latent space (overestimate)."""

    # Augmented Lagrangian parameters
    reconstruction_threshold: float = 0.001
    """Constraint threshold for reconstruction MSE."""
    performance_threshold: float = 0.01
    """Constraint threshold for performance MSE."""
    alpha_r: float = 0.1
    """Learning rate for reconstruction Lagrange multiplier."""
    alpha_p: float = 0.1
    """Learning rate for performance Lagrange multiplier."""
    mu_r: float = 10.0
    """Penalty coefficient for reconstruction constraint."""
    mu_p: float = 10.0
    """Penalty coefficient for performance constraint."""
    warmup_epochs: int = 100
    """Number of epochs to linearly ramp up penalty coefficients."""

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
    pruning_strategy: str = "lognorm"
    """Which pruning strategy to use: [plummet, pca_cdf, lognorm, probabilistic]."""
    cdf_threshold: float = 0.99
    """(pca_cdf) Cumulative variance threshold."""
    temperature: float = 1.0
    """(probabilistic) Sampling temperature."""
    plummet_threshold: float = 0.02
    """(plummet) Threshold for pruning dimensions."""
    alpha: float = 0.2
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


class AirfoilTrueSNDecoder(nn.Module):
    """Decoder with spectral normalization for airfoil: latent → coords (2, 192) + angle_of_attack.

    Uses spectral normalization for 1-Lipschitz bound.
    Outputs both the airfoil coordinates and the angle_of_attack.
    """

    def __init__(self, latent_dim: int, n_data_points: int = 192, lipschitz_scale: float = 1.0):
        super().__init__()
        self.n_data_points = n_data_points
        self.lipschitz_scale = lipschitz_scale

        # MLP to generate features for coords and angle
        self.feature_mlp = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 512)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Linear(512, 1024)),
            nn.ReLU(inplace=True),
        )

        # Branch 1: Generate angle_of_attack (scalar)
        self.angle_head = spectral_norm(nn.Linear(1024, 1))

        # Branch 2: Generate airfoil coordinates via deconv
        # Project to initial conv shape
        conv_start_size = 3  # Will upsample to 192
        self.coords_proj = nn.Sequential(
            spectral_norm(nn.Linear(1024, 2048 * conv_start_size)),
            nn.ReLU(inplace=True),
        )

        # Deconvolutional layers: 3 -> 6 -> 12 -> 24 -> 48 -> 96 -> 192
        self.deconv = nn.Sequential(
            spectral_norm(nn.ConvTranspose1d(2048, 1024, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            spectral_norm(nn.ConvTranspose1d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            spectral_norm(nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            spectral_norm(nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            spectral_norm(nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            spectral_norm(nn.ConvTranspose1d(64, 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.Tanh(),  # Airfoil coords typically in [-1, 1] range
        )

    def forward(self, z: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Decode latent code to airfoil design.

        Args:
            z: (B, latent_dim)

        Returns:
            coords: (B, 2, 192) airfoil coordinates
            angle: (B, 1) angle of attack
        """
        features = self.feature_mlp(z)  # (B, 1024)

        # Generate angle_of_attack
        angle = th.sigmoid(self.angle_head(features)) * self.lipschitz_scale  # (B, 1)

        # Generate coords
        coords_flat = self.coords_proj(features)  # (B, 2048*3)
        coords_reshaped = coords_flat.view(-1, 2048, 3)  # (B, 2048, 3)
        coords = self.deconv(coords_reshaped) * self.lipschitz_scale  # (B, 2, 192)

        return coords, angle


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
        reconstruction_threshold: float = 0.001,
        performance_threshold: float = 0.01,
        alpha_r: float = 0.1,
        alpha_p: float = 0.1,
        mu_r: float = 10.0,
        mu_p: float = 10.0,
        warmup_epochs: int = 100,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.conditional_predictor = conditional_predictor

        # Augmented Lagrangian parameters
        self.reconstruction_threshold = reconstruction_threshold
        self.performance_threshold = performance_threshold
        self.alpha_r = alpha_r
        self.alpha_p = alpha_p
        self.mu_r_final = mu_r
        self.mu_p_final = mu_p
        self.warmup_epochs = warmup_epochs

        # Lagrange multipliers
        self.register_buffer("lambda_r", th.tensor(0.0))
        self.register_buffer("lambda_p", th.tensor(0.0))
        self._current_epoch = 0

    def encode(self, x):
        """Encode airfoil design (coords, angle) to latent code."""
        coords, angle = x
        return self.encoder(coords, angle)

    def decode(self, z):
        """Decode latent code to airfoil design (coords, angle)."""
        return self.decoder(z)

    def get_penalty_coefficients(self):
        """Get current penalty coefficients with linear warmup."""
        if self._current_epoch < self.warmup_epochs:
            warmup_factor = self._current_epoch / self.warmup_epochs
            mu_r = self.mu_r_final * warmup_factor
            mu_p = self.mu_p_final * warmup_factor
        else:
            mu_r = self.mu_r_final
            mu_p = self.mu_p_final
        return mu_r, mu_p

    def loss(self, batch, **kwargs):
        """Compute augmented Lagrangian loss for airfoil problem."""
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
        reconstruction_loss = coords_mse + angle_mse  # Could weight these differently

        performance_loss = nn.functional.mse_loss(p, p_hat)
        active_ratio = self.dim / len(self._p)
        volume_loss = active_ratio * self.loss_vol(z[:, ~self._p])

        # Store for augmented Lagrangian computation
        self._loss_components = th.stack([reconstruction_loss, performance_loss, volume_loss])
        return self._loss_components

    def compute_augmented_lagrangian_loss(self):
        """Compute total augmented Lagrangian loss from stored components."""
        reconstruction_loss, performance_loss, volume_loss = self._loss_components

        reconstruction_violation = th.clamp(reconstruction_loss - self.reconstruction_threshold, min=0.0)
        performance_violation = th.clamp(performance_loss - self.performance_threshold, min=0.0)

        mu_r, mu_p = self.get_penalty_coefficients()

        total_loss = (
            volume_loss
            + self.lambda_r * reconstruction_violation
            + 0.5 * mu_r * reconstruction_violation**2
            + self.lambda_p * performance_violation
            + 0.5 * mu_p * performance_violation**2
        )

        self._reconstruction_violation = reconstruction_violation.detach()
        self._performance_violation = performance_violation.detach()

        return total_loss

    def update_lagrange_multipliers(self):
        """Update Lagrange multipliers via dual ascent."""
        self.lambda_r = th.clamp(self.lambda_r + self.alpha_r * self._reconstruction_violation, min=0.0)
        self.lambda_p = th.clamp(self.lambda_p + self.alpha_p * self._performance_violation, min=0.0)

    def epoch_hook(self, epoch, *args, **kwargs):
        """Update current epoch for warmup scheduling."""
        super().epoch_hook(epoch, *args, **kwargs)
        self._current_epoch = epoch


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

    # Build encoder and decoder
    enc = AirfoilEncoder(args.latent_dim, n_data_points)
    dec = AirfoilTrueSNDecoder(args.latent_dim, n_data_points, lipschitz_scale=args.decoder_lipschitz_scale)

    print(f"Using AirfoilTrueSNDecoder with spectral normalization (Lipschitz scale: {args.decoder_lipschitz_scale})")
    print(f"Using SNMLPPredictor with spectral normalization (Lipschitz scale: {args.predictor_lipschitz_scale})")

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

    # Initialize model
    d_lvae = ConfigIDLVAE(
        encoder=enc,
        decoder=dec,
        predictor=predictor,
        optimizer=Adam(
            list(enc.parameters()) + list(dec.parameters()) + list(predictor.parameters()),
            lr=args.lr,
        ),
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
        alpha_r=args.alpha_r,
        alpha_p=args.alpha_p,
        mu_r=args.mu_r,
        mu_p=args.mu_p,
        warmup_epochs=args.warmup_epochs,
        min_active_dims=args.min_active_dims,
        max_prune_per_epoch=args.max_prune_per_epoch,
        cooldown_epochs=args.cooldown_epochs,
        k_consecutive=args.k_consecutive,
        recon_tol=args.recon_tol,
    ).to(device)

    # DataLoader
    hf = problem.dataset.with_format("torch")
    train_ds = hf["train"]
    val_ds = hf["val"]
    test_ds = hf["test"]

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

    coords_test = th.stack([test_ds[i]["optimal_design"]["coords"] for i in range(len(test_ds))])
    angle_test = th.stack([test_ds[i]["optimal_design"]["angle_of_attack"] for i in range(len(test_ds))]).unsqueeze(-1)
    c_test = (
        th.stack([test_ds[key][:] for key in problem.conditions_keys], dim=-1) if n_conds > 0 else th.empty(len(test_ds), 0)
    )
    p_test = test_ds[problem.objectives_keys[0]][:].unsqueeze(-1)

    # Scale performance values
    from sklearn.preprocessing import RobustScaler

    p_scaler = RobustScaler()
    p_train_scaled = th.from_numpy(p_scaler.fit_transform(p_train.numpy())).to(p_train.dtype)
    p_val_scaled = th.from_numpy(p_scaler.transform(p_val.numpy())).to(p_val.dtype)
    p_test_scaled = th.from_numpy(p_scaler.transform(p_test.numpy())).to(p_test.dtype)

    print(f"\n{'=' * 60}")
    print("Performance Scaling Statistics")
    print(f"{'=' * 60}")
    print(f"RobustScaler center: {p_scaler.center_[0]:.6f}")
    print(f"RobustScaler scale:  {p_scaler.scale_[0]:.6f}")
    print(f"Original range:      [{p_train.min():.6f}, {p_train.max():.6f}]")
    print(f"Scaled range:        [{p_train_scaled.min():.6f}, {p_train_scaled.max():.6f}]")
    print(f"{'=' * 60}\n")

    loader = DataLoader(
        TensorDataset(coords_train, angle_train, c_train, p_train_scaled),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(coords_val, angle_val, c_val, p_val_scaled),
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

            # Compute augmented Lagrangian loss
            loss = d_lvae.compute_augmented_lagrangian_loss()

            loss.backward()
            d_lvae.optim.step()

            # Update Lagrange multipliers
            d_lvae.update_lagrange_multipliers()

            bar.set_postfix(
                {
                    "rec": f"{losses[0].item():.3f}",
                    "perf": f"{losses[1].item():.3f}",
                    "vol": f"{losses[2].item():.3f}",
                    "dim": d_lvae.dim,
                }
            )

            if args.track:
                batches_done = epoch * len(bar) + i
                mu_r, mu_p = d_lvae.get_penalty_coefficients()

                wandb.log(
                    {
                        "rec_loss": losses[0].item(),
                        "perf_loss": losses[1].item(),
                        "vol_loss": losses[2].item(),
                        "total_loss": loss.item(),
                        "active_dims": d_lvae.dim,
                        "lambda_r": d_lvae.lambda_r.item(),
                        "lambda_p": d_lvae.lambda_p.item(),
                        "mu_r": mu_r,
                        "mu_p": mu_p,
                        "epoch": epoch,
                    }
                )

                # Visualization at intervals
                if batches_done % args.sample_interval == 0:
                    with th.no_grad():
                        Xs_coords = coords_test[:25].to(device)
                        Xs_angle = angle_test[:25].to(device)
                        z = d_lvae.encode((Xs_coords, Xs_angle))
                        z_std, idx = th.sort(z.std(0), descending=True)
                        z_mean = z.mean(0)
                        N = (z_std > 0).sum().item()

                        # Random samples
                        z_rand = z_mean.unsqueeze(0).repeat([25, 1])
                        z_rand[:, idx[:N]] += z_std[:N] * th.randn_like(z_rand[:, idx[:N]])
                        coords_rand, angle_rand = d_lvae.decode(z_rand)

                        coords_rand_np = coords_rand.cpu().numpy()
                        Xs_coords_np = Xs_coords.cpu().numpy()

                    # Plot: Random airfoil designs
                    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
                    axes = axes.flatten()
                    for j in range(25):
                        airfoil = coords_rand_np[j]
                        axes[j].plot(airfoil[0], airfoil[1], "b-")
                        axes[j].set_aspect("equal")
                        axes[j].axis("off")
                        axes[j].set_title(f"α={angle_rand[j].item():.2f}", fontsize=8)
                    plt.tight_layout()
                    plt.suptitle("Random airfoils from latent space")
                    plt.savefig(f"images/airfoils_{batches_done}.png")
                    plt.close()

                    # Plot: Reconstructions
                    coords_recon, angle_recon = d_lvae.decode(z[:5])
                    coords_recon_np = coords_recon.detach().cpu().numpy()

                    fig, axes = plt.subplots(5, 2, figsize=(10, 15))
                    for k in range(5):
                        # Original
                        axes[k, 0].plot(Xs_coords_np[k][0], Xs_coords_np[k][1], "b-")
                        axes[k, 0].set_aspect("equal")
                        axes[k, 0].axis("off")
                        axes[k, 0].set_title(f"Orig α={Xs_angle[k].item():.2f}", fontsize=8)

                        # Reconstructed
                        axes[k, 1].plot(coords_recon_np[k][0], coords_recon_np[k][1], "r-")
                        axes[k, 1].set_aspect("equal")
                        axes[k, 1].axis("off")
                        axes[k, 1].set_title(f"Recon α={angle_recon[k].item():.2f}", fontsize=8)

                    plt.tight_layout()
                    plt.savefig(f"images/recon_{batches_done}.png")
                    plt.close()

                    wandb.log(
                        {
                            "airfoils": wandb.Image(f"images/airfoils_{batches_done}.png"),
                            "recon": wandb.Image(f"images/recon_{batches_done}.png"),
                        }
                    )

        # Validation
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

        val_rec_violation = max(0.0, val_rec - args.reconstruction_threshold)
        val_perf_violation = max(0.0, val_perf - args.performance_threshold)
        mu_r, mu_p = d_lvae.get_penalty_coefficients()
        val_total = (
            val_vol
            + d_lvae.lambda_r.item() * val_rec_violation
            + 0.5 * mu_r * val_rec_violation**2
            + d_lvae.lambda_p.item() * val_perf_violation
            + 0.5 * mu_p * val_perf_violation**2
        )

        d_lvae.epoch_report(epoch=epoch, callbacks=[], batch=None, loss=loss, pbar=None, val_recon=val_rec)

        if args.track:
            wandb.log(
                {
                    "epoch": epoch,
                    "val_rec": val_rec,
                    "val_perf": val_perf,
                    "val_vol_loss": val_vol,
                    "val_total_loss": val_total,
                },
                commit=True,
            )

        th.cuda.empty_cache()
        d_lvae.train()

        # Save model
        if args.save_model and epoch == args.n_epochs - 1:
            ckpt = {
                "epoch": epoch,
                "encoder": d_lvae.encoder.state_dict(),
                "decoder": d_lvae.decoder.state_dict(),
                "predictor": d_lvae.predictor.state_dict(),
                "optimizer": d_lvae.optim.state_dict(),
                "p_scaler": p_scaler,
                "args": vars(args),
            }
            th.save(ckpt, "d_lvae_airfoil.pth")
            artifact = wandb.Artifact(f"{args.problem_id}_{args.algo}", type="model")
            artifact.add_file("d_lvae_airfoil.pth")
            wandb.log_artifact(artifact, aliases=[f"seed_{args.seed}"])

    wandb.finish()
