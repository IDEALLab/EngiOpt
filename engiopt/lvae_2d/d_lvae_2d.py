"""DesignLVAE_DP for 2D designs with performance prediction.

This script provides a configurable interpretable design autoencoder with performance prediction:
- Encoder: Extracts latent codes from designs
- TrueSNDecoder: Reconstructs designs from latent codes (with spectral normalization)
- SNMLPPredictor: Spectrally normalized MLP that predicts performance from first perf_dim latent codes
- Dynamic Pruning: Prunes low-variance latent dimensions during training

All components use spectral normalization to ensure 1-Lipschitz continuity, enabling:
- Smooth optimization in latent space
- Interpretable performance gradients
- Stable surrogate-based design search

Configuration:
- perf_dim: Number of latent dimensions dedicated to performance (default: all latent_dim dimensions)
  - Use perf_dim < latent_dim for interpretable mode (e.g., --perf-dim 10)
  - Use perf_dim = latent_dim (default) for regular mode
- Conditional (--conditional-predictor, default): Predictor uses [latent[:perf_dim], conditions]
- Unconditional (--no-conditional-predictor): Predictor uses only [latent[:perf_dim]]
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

from engiopt.lvae_2d.aes import InterpretableDesignLeastVolumeAE_DP
from engiopt.lvae_2d.utils import polynomial_schedule
from engiopt.lvae_2d.utils import SNLinearCombo
from engiopt.lvae_2d.utils import spectral_norm_conv
from engiopt.lvae_2d.utils import TrueSNDeconv2DCombo


@dataclass
class Args:
    # Problem and tracking
    problem_id: str = "heatconduction2d"
    """Problem ID to run. Must be one of the built-in problems in engibench."""
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
    w_v: float = 0.01
    """Weight for the volume loss."""
    w_p: float = 1.0
    """Weight for the performance prediction loss."""
    polynomial_schedule_n: int = 100
    """Number of epochs for the polynomial schedule."""
    polynomial_schedule_p: int = 2
    """Polynomial exponent for the schedule."""
    pruning_epoch: int = 500
    """Epoch to start pruning dimensions."""
    beta: float = 0.9
    """Momentum for the pruning ratio calculation."""
    eta: float = 1e-4
    """Scaling factor for the volume loss."""
    resize_dimensions: tuple[int, int] = (100, 100)
    """Dimensions to resize input images to before encoding/decoding."""

    # MLP predictor parameters
    predictor_hidden_dims: tuple[int, ...] = (256, 128)
    """Hidden dimensions for the MLP predictor."""
    conditional_predictor: bool = True
    """Whether to include conditions in performance prediction (True) or use only latent codes (False)."""
    perf_dim: int = -1
    """Number of latent dimensions dedicated to performance prediction. If -1 (default), uses all latent_dim dimensions."""

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
    """

    def __init__(self, latent_dim: int, design_shape: tuple[int, int]):
        super().__init__()
        self.design_shape = design_shape
        self.resize_out = transforms.Resize(self.design_shape)

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
        return self.resize_out(x)  # (B,1,H_orig,W_orig)


class SNMLPPredictor(nn.Module):
    """Spectral normalized MLP that predicts performance from latent codes.

    Enforces 1-Lipschitz continuity to ensure small steps in latent space
    correspond to small steps in performance space. Critical for:
    - Smooth optimization in latent space
    - Interpretable performance gradients
    - Stable surrogate-based design search

    Uses SNLinearCombo (spectral normalized Linear + ReLU) for hidden layers
    and spectral normalized Linear for the output layer.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...] = (256, 128)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(SNLinearCombo(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        # Final layer: spectral normalized Linear (no activation)
        layers.append(spectral_norm(nn.Linear(prev_dim, output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Predict performance from latent codes + conditions."""
        return self.net(x)


class ConfigurableInterpretableDesignLVAE(InterpretableDesignLeastVolumeAE_DP):
    """Wrapper around InterpretableDesignLeastVolumeAE_DP with conditional/unconditional prediction.

    This variant uses only the first perf_dim latent dimensions for performance prediction,
    making those dimensions interpretable as performance-relevant features.
    """

    def __init__(self, *args, conditional_predictor: bool = True, **kwargs):
        """Initialize with conditional flag.

        Args:
            conditional_predictor: If True, predictor uses [z[:perf_dim], c]. If False, uses only [z[:perf_dim]].
            *args, **kwargs: Passed to parent InterpretableDesignLeastVolumeAE_DP
        """
        super().__init__(*args, **kwargs)
        self.conditional_predictor = conditional_predictor

    def loss(self, batch, **kwargs):
        """Compute losses using only first perf_dim latents for performance prediction."""
        x, c, p = batch
        z = self.encode(x)
        x_hat = self.decode(z)

        # Only the first pdim dimensions are used for performance prediction
        pz = z[:, : self.pdim]

        # Conditional: predictor uses first perf_dim latent codes + conditions,
        # otherwise uses only the first perf_dim latent codes
        p_hat = self.predictor(th.cat([pz, c], dim=-1)) if self.conditional_predictor else self.predictor(pz)

        # Normalize targets and predictions
        p_n = self._norm_perf(p)
        p_hat_n = self._norm_perf(p_hat)

        # Note: _update_moving_mean and pruning logic handled by parent class
        active_ratio = self.dim / len(self._p)  # Scale volume loss by active dimension ratio

        return th.stack(
            [
                self.loss_rec(x, x_hat),
                self.loss_rec(p_n, p_hat_n),  # normalized prediction loss
                active_ratio * self.loss_vol(z[:, ~self._p]),
            ]
        )


if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    design_shape = problem.design_space.shape
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

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    # Build encoder and decoder (always use spectral normalization)
    enc = Encoder(args.latent_dim, design_shape, args.resize_dimensions)
    dec = TrueSNDecoder(args.latent_dim, design_shape)
    print("Using TrueSNDecoder with spectral normalization (1-Lipschitz bound)")
    print("Using SNMLPPredictor with spectral normalization for performance prediction")

    # Build MLP predictor
    # Determine perf_dim: if -1 (default), use all latent dimensions
    perf_dim = args.latent_dim if args.perf_dim == -1 else args.perf_dim
    n_perf = 1  # Can be adjusted based on problem

    predictor_input_dim = perf_dim + (n_conds if args.conditional_predictor else 0)
    predictor = SNMLPPredictor(
        input_dim=predictor_input_dim,
        output_dim=n_perf,
        hidden_dims=args.predictor_hidden_dims,
    )

    print(f"Performance dimensions: {perf_dim}/{args.latent_dim} latent dimensions")
    print(f"Predictor mode: {'Conditional' if args.conditional_predictor else 'Unconditional'}")
    print(
        f"Predictor input dimension: {predictor_input_dim} (perf_dim={perf_dim}, n_conds={n_conds if args.conditional_predictor else 0})"
    )

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

    # Initialize DesignLVAE with dynamic pruning
    # Always use interpretable version (generalizes to regular when perf_dim = latent_dim)
    d_lvae = ConfigurableInterpretableDesignLVAE(
        encoder=enc,
        decoder=dec,
        predictor=predictor,
        optimizer=Adam(
            list(enc.parameters()) + list(dec.parameters()) + list(predictor.parameters()),
            lr=args.lr,
        ),
        latent_dim=args.latent_dim,
        perf_dim=perf_dim,
        weights=polynomial_schedule(
            [1.0, args.w_p, args.w_v],
            N=args.polynomial_schedule_n,
            p=args.polynomial_schedule_p,
            w_init=[1.0, 0, 0],
        ),
        pruning_epoch=args.pruning_epoch,
        beta=args.beta,
        eta=args.eta,
        pruning_strategy=args.pruning_strategy,
        pruning_params=pruning_params,
        conditional_predictor=args.conditional_predictor,
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

    # Extract designs, conditions, and performance
    x_train = train_ds["optimal_design"][:].unsqueeze(1)
    c_train = th.stack([train_ds[key][:] for key in problem.conditions_keys], dim=-1)
    p_train = train_ds["optimal_value"][:].unsqueeze(-1)  # (N, 1)

    x_val = val_ds["optimal_design"][:].unsqueeze(1)
    c_val = th.stack([val_ds[key][:] for key in problem.conditions_keys], dim=-1)
    p_val = val_ds["optimal_value"][:].unsqueeze(-1)

    x_test = test_ds["optimal_design"][:].unsqueeze(1)
    c_test = th.stack([test_ds[key][:] for key in problem.conditions_keys], dim=-1)
    p_test = test_ds["optimal_value"][:].unsqueeze(-1)

    loader = DataLoader(
        TensorDataset(x_train, c_train, p_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, c_val, p_val),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Track performance loss separately for plotting
    perf_loss_history = []

    # ---- Training loop ----
    for epoch in range(args.n_epochs):
        d_lvae.epoch_hook(epoch=epoch)

        bar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(bar):
            x_batch = batch[0].to(device)
            c_batch = batch[1].to(device)
            p_batch = batch[2].to(device)

            d_lvae.optim.zero_grad()
            losses = d_lvae.loss((x_batch, c_batch, p_batch))  # [rec, perf, vol]
            loss = (losses * d_lvae.w).sum()  # apply weights
            loss.backward()
            d_lvae.optim.step()

            bar.set_postfix(
                {
                    "rec": f"{losses[0].item():.3f}",
                    "perf": f"{losses[1].item():.3f}",
                    "vol": f"{losses[2].item():.3f}",
                    "dim": d_lvae.dim,
                }
            )

            # Log to wandb
            if args.track:
                batches_done = epoch * len(bar) + i
                perf_loss_history.append(losses[1].item())

                wandb.log(
                    {
                        "rec_loss": losses[0].item(),
                        "perf_loss": losses[1].item(),
                        "vol_loss": losses[2].item(),
                        "total_loss": loss.item(),
                        "active_dims": d_lvae.dim,
                        "epoch": epoch,
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
                        # Encode test designs
                        Xs = x_test.to(device)
                        z = d_lvae.encode(Xs)
                        z_std, idx = th.sort(z.std(0), descending=True)
                        z_mean = z.mean(0)
                        N = (z_std > 0).sum().item()

                        # Generate interpolated designs
                        x_ints = []
                        for alpha in [0, 0.25, 0.5, 0.75, 1]:
                            z_ = alpha * z[:25] + (1 - alpha) * th.roll(z, 1, 0)[:25]
                            x_ints.append(d_lvae.decode(z_).cpu().numpy())

                        # Generate random designs
                        z_rand = z_mean.unsqueeze(0).repeat([25, 1])
                        z_rand[:, idx[:N]] += z_std[:N] * th.randn_like(z_rand[:, idx[:N]])
                        x_rand = d_lvae.decode(z_rand).cpu().numpy()

                        # Get performance predictions
                        pz_test = z[:, :perf_dim]
                        if args.conditional_predictor:
                            p_pred = d_lvae.predictor(th.cat([pz_test, c_test.to(device)], dim=-1))
                        else:
                            p_pred = d_lvae.predictor(pz_test)
                        p_actual = p_test.cpu().numpy().flatten()
                        p_predicted = p_pred.cpu().numpy().flatten()

                        # Move tensors to CPU for plotting
                        z_std_cpu = z_std.cpu().numpy()
                        Xs_cpu = Xs.cpu().numpy()

                    # Plot 1: Performance loss history
                    plt.figure(figsize=(10, 6))
                    plt.plot(perf_loss_history)
                    plt.xlabel("Training Step")
                    plt.ylabel("Performance Loss")
                    plt.title("Performance Prediction Loss over Training")
                    plt.yscale("log")
                    plt.grid(True, alpha=0.3)
                    plt.savefig(f"images/perf_loss_{batches_done}.png")
                    plt.close()

                    # Plot 2: Latent dimension statistics
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

                    # Plot 3: Interpolated designs
                    fig, axs = plt.subplots(25, 6, figsize=(12, 25))
                    for i, j in product(range(25), range(5)):
                        axs[i, j + 1].imshow(x_ints[j][i].reshape(design_shape))
                        axs[i, j + 1].axis("off")
                        axs[i, j + 1].set_aspect("equal")
                    for ax, alpha in zip(axs[0, 1:], [0, 0.25, 0.5, 0.75, 1]):
                        ax.set_title(rf"$\alpha$ = {alpha}")
                    for i in range(25):
                        axs[i, 0].imshow(Xs_cpu[i].reshape(design_shape))
                        axs[i, 0].axis("off")
                        axs[i, 0].set_aspect("equal")
                    axs[0, 0].set_title("groundtruth")
                    fig.tight_layout()
                    plt.savefig(f"images/interp_{batches_done}.png")
                    plt.close()

                    # Plot 4: Random designs from latent space
                    fig, axs = plt.subplots(5, 5, figsize=(15, 7.5))
                    for k, (i, j) in enumerate(product(range(5), range(5))):
                        axs[i, j].imshow(x_rand[k].reshape(design_shape))
                        axs[i, j].axis("off")
                        axs[i, j].set_aspect("equal")
                    fig.tight_layout()
                    plt.suptitle("Gaussian random designs from latent space")
                    plt.savefig(f"images/norm_{batches_done}.png")
                    plt.close()

                    # Plot 5: Predicted vs actual performance
                    plt.figure(figsize=(8, 8))
                    plt.scatter(p_actual, p_predicted, alpha=0.5, s=20)
                    min_val = min(p_actual.min(), p_predicted.min())
                    max_val = max(p_actual.max(), p_predicted.max())
                    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="1:1 line")
                    plt.xlabel("Actual Performance")
                    plt.ylabel("Predicted Performance")
                    plt.title(f"Predicted vs Actual Performance (Step {batches_done})")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.axis("equal")
                    plt.tight_layout()
                    plt.savefig(f"images/perf_pred_vs_actual_{batches_done}.png")
                    plt.close()

                    # Log all plots to wandb
                    wandb.log(
                        {
                            "perf_loss_plot": wandb.Image(f"images/perf_loss_{batches_done}.png"),
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
                x_v = batch_v[0].to(device)
                c_v = batch_v[1].to(device)
                p_v = batch_v[2].to(device)
                vlosses = d_lvae.loss((x_v, c_v, p_v))
                bsz = x_v.size(0)
                val_rec += vlosses[0].item() * bsz
                val_perf += vlosses[1].item() * bsz
                val_vol += vlosses[2].item() * bsz
                n += bsz
        val_rec /= n
        val_perf /= n
        val_vol /= n
        val_total = val_rec + args.w_p * val_perf + args.w_v * val_vol

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
            }
            th.save(ckpt_d_lvae, "d_lvae.pth")
            artifact = wandb.Artifact(f"{args.problem_id}_{args.algo}", type="model")
            artifact.add_file("d_lvae.pth")
            wandb.log_artifact(artifact, aliases=[f"seed_{args.seed}"])

    wandb.finish()
