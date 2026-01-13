"""LVAE_DP for airfoil designs with Bezier-based encoder/decoder architecture.

This implementation uses a Bezier curve generator architecture for the airfoil problem.
The encoder/decoder only operates on the airfoil geometry (coords), not angle_of_attack.

Note: The EngiBench airfoil problem has a Dict design space with:
  - 'coords': (2, 192) - x,y coordinates of airfoil shape
  - 'angle_of_attack': scalar value (not used in this LVAE)
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

from engiopt.lvae_core import LeastVolumeAE_DynamicPruning, polynomial_schedule


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
    w_v: float = 0.01
    """Weight for the volume loss."""
    polynomial_schedule_n: int = 100
    """Number of epochs for the polynomial schedule."""
    polynomial_schedule_p: int = 2
    """Polynomial exponent for the schedule."""
    pruning_epoch: int = 500
    """Epoch to start pruning dimensions."""
    beta: float = 0.9
    """Momentum for EMA of latent statistics."""
    eta: float = 1e-4
    """Scaling factor for the volume loss."""

    # Bezier architecture parameters
    n_control_points: int = 64
    """Number of control points for Bezier curve generation."""

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
    """1D Convolutional encoder for airfoil coordinates.

    Takes (B, 2, 192) airfoil coordinates and encodes to (B, latent_dim).
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

        # MLP to latent
        self.mlp = nn.Sequential(
            nn.Linear(m_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """x: (B, 2, 192) -> (B, latent_dim)"""
        x = self.conv(x)
        x = x.flatten(1)
        return self.mlp(x)


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

    Generates airfoil coords (B, 2, 192) from latent (B, latent_dim).
    """

    def __init__(self, latent_dim: int, n_control_points: int, n_data_points: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points

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

        # Output heads
        self.cp_gen = nn.Sequential(spectral_norm(nn.Conv1d(deconv_channels[-1], 2, 1)), nn.Tanh())
        self.w_gen = nn.Sequential(spectral_norm(nn.Conv1d(deconv_channels[-1], 1, 1)), nn.Sigmoid())

        # Bezier layer
        self.bezier_layer = BezierLayer(m_features, n_control_points, n_data_points)

    def forward(self, z: th.Tensor) -> th.Tensor:
        """Generate airfoil coordinates from latent code.

        Args:
            z: (B, latent_dim)

        Returns:
            coords: (B, 2, n_data_points)
        """
        # Generate features
        features = self.feature_generator(z)

        # Generate control points and weights
        x = self.cpw_dense(z).view(-1, self.in_channels, self.in_width)
        x = self.deconv(x)
        cp = self.cp_gen(x)  # (B, 2, n_cp)
        w = self.w_gen(x)  # (B, 1, n_cp)

        # Generate Bezier curve
        coords, _, _ = self.bezier_layer(features, cp, w)

        return coords


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

    # Build encoder and decoder
    enc = Conv1DEncoder(in_channels=2, in_features=n_data_points, latent_dim=args.latent_dim)
    dec = SNBezierDecoder(latent_dim=args.latent_dim, n_control_points=args.n_control_points, n_data_points=n_data_points)

    print(f"Using Bezier-based decoder with {args.n_control_points} control points")
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

    # Initialize LVAE with dynamic pruning
    lvae = LeastVolumeAE_DynamicPruning(
        encoder=enc,
        decoder=dec,
        optimizer=Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr),
        latent_dim=args.latent_dim,
        weights=polynomial_schedule(
            [1.0, args.w_v],
            N=args.polynomial_schedule_n,
            p=args.polynomial_schedule_p,
            w_init=[1.0, 0],
        ),
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

    # ---- DataLoader ----
    # Extract only coords from the airfoil dataset (ignore angle_of_attack)
    problem_dataset = problem.dataset.with_format("torch")
    train_ds = problem_dataset["train"]
    val_ds = problem_dataset["val"]
    test_ds = problem_dataset["test"]

    # Extract coords only
    coords_train = th.stack([train_ds[i]["optimal_design"]["coords"] for i in range(len(train_ds))])
    coords_val = th.stack([val_ds[i]["optimal_design"]["coords"] for i in range(len(val_ds))])
    coords_test = th.stack([test_ds[i]["optimal_design"]["coords"] for i in range(len(test_ds))])

    loader = DataLoader(TensorDataset(coords_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(coords_val), batch_size=args.batch_size, shuffle=False)

    # ---- Training loop ----
    for epoch in range(args.n_epochs):
        lvae.epoch_hook(epoch=epoch)

        bar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(bar):
            x_batch = batch[0].to(device)  # (B, 2, 192)

            lvae.optim.zero_grad()
            losses = lvae.loss(x_batch)  # [rec, vol]
            loss = (losses * lvae.w).sum()
            loss.backward()
            lvae.optim.step()

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
                        "vol_loss": losses[1].item(),
                        "total_loss": loss.item(),
                        "active_dims": lvae.dim,
                        "epoch": epoch,
                    }
                )
                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(bar)}] "
                    f"[rec loss: {losses[0].item()}] [vol loss: {losses[1].item()}] [active dims: {lvae.dim}]"
                )

                # Sample and visualize at regular intervals
                if batches_done % args.sample_interval == 0:
                    with th.no_grad():
                        # Encode test designs
                        Xs = coords_test[:25].to(device)
                        z = lvae.encode(Xs)
                        z_std, idx = th.sort(z.std(0), descending=True)
                        z_mean = z.mean(0)
                        N = (z_std > 0).sum().item()

                        # Generate interpolated designs
                        x_ints = []
                        for alpha in [0, 0.25, 0.5, 0.75, 1]:
                            z_ = (1 - alpha) * z[:25] + alpha * th.roll(z, -1, 0)[:25]
                            x_ints.append(lvae.decode(z_).detach().cpu().numpy())

                        # Generate random designs
                        z_rand = z_mean.unsqueeze(0).repeat([25, 1])
                        z_rand[:, idx[:N]] += z_std[:N] * th.randn_like(z_rand[:, idx[:N]])
                        coords_rand = lvae.decode(z_rand).cpu().numpy()

                        # Move to CPU
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
                    fig, axs = plt.subplots(25, 6, figsize=(12, 50))
                    for i, j in product(range(25), range(5)):
                        airfoil = x_ints[j][i]  # (2, 192)
                        axs[i, j + 1].plot(airfoil[0], airfoil[1], "b-")
                        axs[i, j + 1].axis("off")
                        axs[i, j + 1].set_aspect("equal")
                    for ax, alpha in zip(axs[0, 1:], [0, 0.25, 0.5, 0.75, 1]):
                        ax.set_title(rf"$\alpha$ = {alpha}")
                    for i in range(25):
                        airfoil = Xs_cpu[i]  # (2, 192)
                        axs[i, 0].plot(airfoil[0], airfoil[1], "b-")
                        axs[i, 0].axis("off")
                        axs[i, 0].set_aspect("equal")
                    axs[0, 0].set_title("groundtruth")
                    fig.tight_layout()
                    plt.savefig(f"images/interp_{batches_done}.png")
                    plt.close()

                    # Plot 3: Random airfoil designs
                    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
                    axes = axes.flatten()
                    for j in range(25):
                        airfoil = coords_rand[j]  # (2, 192)
                        axes[j].plot(airfoil[0], airfoil[1], "b-")
                        axes[j].set_aspect("equal")
                        axes[j].axis("off")
                    plt.tight_layout()
                    plt.suptitle("Random airfoil designs from latent space")
                    plt.savefig(f"images/airfoils_{batches_done}.png")
                    plt.close()

                    # Plot 3: Reconstruction comparison
                    fig, axes = plt.subplots(5, 2, figsize=(10, 15))
                    coords_orig = coords_test[:5].cpu().numpy()
                    coords_recon = lvae.decode(z[:5]).detach().cpu().numpy()

                    for k in range(5):
                        # Original
                        axes[k, 0].plot(coords_orig[k][0], coords_orig[k][1], "b-")
                        axes[k, 0].set_aspect("equal")
                        axes[k, 0].axis("off")
                        if k == 0:
                            axes[k, 0].set_title("Original")

                        # Reconstructed
                        axes[k, 1].plot(coords_recon[k][0], coords_recon[k][1], "r-")
                        axes[k, 1].set_aspect("equal")
                        axes[k, 1].axis("off")
                        if k == 0:
                            axes[k, 1].set_title("Reconstructed")

                    plt.tight_layout()
                    plt.savefig(f"images/recon_{batches_done}.png")
                    plt.close()

                    # Log to wandb
                    wandb.log(
                        {
                            "dim_plot": wandb.Image(f"images/dim_{batches_done}.png"),
                            "interp_plot": wandb.Image(f"images/interp_{batches_done}.png"),
                            "norm_plot": wandb.Image(f"images/airfoils_{batches_done}.png"),
                            "recon_plot": wandb.Image(f"images/recon_{batches_done}.png"),
                        }
                    )

        # ---- Validation ----
        with th.no_grad():
            lvae.eval()
            val_rec = val_vol = 0.0
            n = 0
            for batch_v in val_loader:
                x_v = batch_v[0].to(device)
                vlosses = lvae.loss(x_v)
                bsz = x_v.size(0)
                val_rec += vlosses[0].item() * bsz
                val_vol += vlosses[1].item() * bsz
                n += bsz
        val_rec /= n
        val_vol /= n
        val_total = val_rec + args.w_v * val_vol

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
