"""Performance-LVAE for 2D designs with plummet-based dynamic pruning. Adapted from https://github.com/IDEALLab/Least_Volume_ICLR2024.

Configuration:
- perf_dim: Number of latent dimensions dedicated to performance (default: all latent_dim dimensions)
  - Use perf_dim < latent_dim for interpretable mode
  - Use perf_dim = latent_dim (default) for regular mode

For more information, see: https://arxiv.org/abs/2404.17773
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import product
import os
import random
import time

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import RobustScaler
import torch as th
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import transforms
import tqdm
import tyro

from engiopt.lvae_2d.aes import InterpretablePerfLeastVolumeAE_DP
import wandb


@dataclass
class Args:
    """Command-line arguments for Performance-LVAE training."""

    # Problem and tracking
    problem_id: str = "beams2d"
    """Problem ID to run. Must be one of the built-in problems in engibench."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """Algorithm name for tracking purposes."""
    track: bool = True
    """Whether to track with Weights & Biases."""
    wandb_project: str = "engiopt"
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
    n_epochs: int = 2000
    """Number of training epochs."""
    batch_size: int = 128
    """Batch size for training."""
    lr: float = 1e-4
    """Learning rate for the optimizer."""

    # LVAE-specific
    latent_dim: int = 250
    """Dimensionality of the latent space (overestimate)."""
    perf_dim: int = -1
    """Number of latent dimensions dedicated to performance prediction. If -1 (default), uses all latent_dim dimensions."""
    w_reconstruction: float = 1.0
    """Weight for reconstruction loss."""
    w_performance: float = 0.1
    """Weight for performance loss."""
    w_volume: float = 0.001
    """Weight for volume loss."""

    # Pruning parameters
    pruning_epoch: int = 500
    """Epoch to start pruning dimensions."""
    plummet_threshold: float = 0.02
    """Threshold for plummet pruning strategy."""

    # Volume weight warmup
    volume_warmup_epochs: int = 0
    """Epochs to polynomially ramp volume weight from 0 to w_volume. 0 disables warmup."""
    volume_warmup_degree: float = 2.0
    """Polynomial degree for volume weight warmup (1.0=linear, 2.0=quadratic)."""

    # Architecture
    resize_dimensions: tuple[int, int] = (100, 100)
    """Dimensions to resize input images to before encoding/decoding."""
    predictor_hidden_dims: tuple[int, ...] = (256, 128)
    """Hidden dimensions for the MLP predictor."""
    conditional_predictor: bool = True
    """Whether to include conditions in performance prediction (True) or use only latent codes (False)."""


class Encoder(nn.Module):
    """Convolutional encoder for 2D designs.

    Architecture: Input -> Conv layers -> Latent vector
    • Input   [100x100]
    • Conv1   [50x50]   (k=4, s=2, p=1)
    • Conv2   [25x25]   (k=4, s=2, p=1)
    • Conv3   [13x13]   (k=3, s=2, p=1)
    • Conv4   [7x7]     (k=3, s=2, p=1)
    • Conv5   [1x1]     (k=7, s=1, p=0)
    """

    def __init__(
        self,
        latent_dim: int,
        design_shape: tuple[int, int],
        resize_dimensions: tuple[int, int] = (100, 100),
    ) -> None:
        super().__init__()
        self.resize_in = transforms.Resize(resize_dimensions)
        self.design_shape = design_shape

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 100->50
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 50->25
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # 25->13
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),  # 13->7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final 7x7 conv produces (B, latent_dim, 1, 1) -> flatten to (B, latent_dim)
        self.to_latent = nn.Conv2d(512, latent_dim, kernel_size=7, stride=1, padding=0, bias=True)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass through encoder."""
        x = self.resize_in(x)  # (B,1,100,100)
        h = self.features(x)  # (B,512,7,7)
        return self.to_latent(h).flatten(1)  # (B,latent_dim)


class Decoder(nn.Module):
    """Convolutional decoder for 2D designs.

    Architecture: Latent vector -> Deconv layers -> Output
    • Latent   [latent_dim]
    • Linear   [512*7*7]
    • Reshape  [512x7x7]
    • Deconv1  [256x13x13]  (k=3, s=2, p=1)
    • Deconv2  [128x25x25]  (k=3, s=2, p=1)
    • Deconv3  [64x50x50]   (k=4, s=2, p=1)
    • Deconv4  [1x100x100]  (k=4, s=2, p=1)
    """

    def __init__(
        self,
        latent_dim: int,
        design_shape: tuple[int, int],
    ) -> None:
        super().__init__()
        self.design_shape = design_shape
        self.resize_out = transforms.Resize(self.design_shape)

        # Linear projection to spatial features
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, 512 * 7 * 7),
            nn.ReLU(inplace=True),
        )

        # Deconvolutional layers
        self.deconv = nn.Sequential(
            # 7->13
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 13->25
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 25->50
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 50->100
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, z: th.Tensor) -> th.Tensor:
        """Forward pass through decoder."""
        x = self.proj(z).view(z.size(0), 512, 7, 7)  # (B,512,7,7)
        x = self.deconv(x)  # (B,1,100,100)
        return self.resize_out(x)  # (B,1,H_orig,W_orig)


class MLPPredictor(nn.Module):
    """MLP that predicts performance from latent codes + conditions.

    Uses LeakyReLU activations in hidden layers and no activation on output.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = hidden_dim
        # Final layer: Linear (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Predict performance from latent codes + conditions."""
        return self.net(x)


def volume_weight_schedule(  # noqa: PLR0913
    epoch: int, w_rec: float, w_perf: float, w_vol: float, warmup_epochs: int, degree: float
) -> th.Tensor:
    """Compute weights with polynomial ramp on volume weight.

    Args:
        epoch: Current epoch.
        w_rec: Reconstruction weight (constant).
        w_perf: Performance weight (constant).
        w_vol: Final volume weight after warmup.
        warmup_epochs: Epochs to ramp volume weight from 0 to w_vol.
        degree: Polynomial degree (1.0=linear, 2.0=quadratic).

    Returns:
        Tensor [w_rec, w_perf, current_w_vol] where current_w_vol ramps polynomially.
    """
    if warmup_epochs <= 0:
        return th.tensor([w_rec, w_perf, w_vol], dtype=th.float)
    t = min(epoch / warmup_epochs, 1.0)
    return th.tensor([w_rec, w_perf, w_vol * (t**degree)], dtype=th.float)


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

    # Seeding for reproducibility
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    g = th.Generator().manual_seed(args.seed)  # For DataLoader shuffling

    os.makedirs("images", exist_ok=True)

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    # Build encoder and decoder
    enc = Encoder(args.latent_dim, design_shape, args.resize_dimensions)
    dec = Decoder(args.latent_dim, design_shape)

    # Determine perf_dim: if -1 (default), use all latent dimensions
    perf_dim = args.latent_dim if args.perf_dim == -1 else args.perf_dim
    n_perf = 1  # Single performance objective

    # Build MLP predictor (input: perf_dim latent dims + conditions if conditional)
    predictor_input_dim = perf_dim + (n_conds if args.conditional_predictor else 0)
    predictor = MLPPredictor(
        input_dim=predictor_input_dim,
        output_dim=n_perf,
        hidden_dims=args.predictor_hidden_dims,
    )

    print(f"\n{'=' * 60}")
    print("Performance-LVAE Training")
    print(f"Problem: {args.problem_id}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Perf dim: {perf_dim} (first {perf_dim} dims predict performance)")
    print(f"Predictor mode: {'Conditional' if args.conditional_predictor else 'Unconditional'}")
    print(
        f"Predictor input: {predictor_input_dim} (perf_dim={perf_dim}, n_conds={n_conds if args.conditional_predictor else 0})"
    )
    print(f"Pruning epoch: {args.pruning_epoch}")
    print(f"Plummet threshold: {args.plummet_threshold}")
    print(f"Volume warmup epochs: {args.volume_warmup_epochs}")
    print(f"{'=' * 60}\n")

    # Weight schedule (ramps volume weight if warmup_epochs > 0, otherwise constant)
    weights = partial(
        volume_weight_schedule,
        w_rec=args.w_reconstruction,
        w_perf=args.w_performance,
        w_vol=args.w_volume,
        warmup_epochs=args.volume_warmup_epochs,
        degree=args.volume_warmup_degree,
    )

    # Initialize Performance-LVAE with dynamic pruning
    plvae = InterpretablePerfLeastVolumeAE_DP(
        encoder=enc,
        decoder=dec,
        predictor=predictor,
        optimizer=Adam(
            list(enc.parameters()) + list(dec.parameters()) + list(predictor.parameters()),
            lr=args.lr,
        ),
        latent_dim=args.latent_dim,
        perf_dim=perf_dim,
        weights=weights,
        pruning_epoch=args.pruning_epoch,
        plummet_threshold=args.plummet_threshold,
    ).to(device)

    # ---- DataLoader ----
    hf = problem.dataset.with_format("torch")
    train_ds = hf["train"]
    val_ds = hf["val"]

    # Extract designs, conditions, and performance
    x_train = train_ds["optimal_design"][:].unsqueeze(1)
    c_train = th.stack([train_ds[key][:] for key in problem.conditions_keys], dim=-1)
    p_train = train_ds[problem.objectives_keys[0]][:].unsqueeze(-1)  # (N, 1)

    x_val = val_ds["optimal_design"][:].unsqueeze(1)
    c_val = th.stack([val_ds[key][:] for key in problem.conditions_keys], dim=-1)
    p_val = val_ds[problem.objectives_keys[0]][:].unsqueeze(-1)

    # Scale performance values using RobustScaler
    p_scaler = RobustScaler()
    p_train_scaled = th.from_numpy(p_scaler.fit_transform(p_train.numpy())).to(p_train.dtype)
    p_val_scaled = th.from_numpy(p_scaler.transform(p_val.numpy())).to(p_val.dtype)

    # Scale conditions using RobustScaler (if using conditional predictor)
    if args.conditional_predictor:
        c_scaler = RobustScaler()
        c_train_scaled = th.from_numpy(c_scaler.fit_transform(c_train.numpy())).to(c_train.dtype)
        c_val_scaled = th.from_numpy(c_scaler.transform(c_val.numpy())).to(c_val.dtype)
    else:
        # Dummy tensors when not using conditions (won't be used in predictor)
        c_train_scaled = th.zeros(len(x_train), 0)
        c_val_scaled = th.zeros(len(x_val), 0)

    loader = DataLoader(
        TensorDataset(x_train, c_train_scaled, p_train_scaled),
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, c_val_scaled, p_val_scaled),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ---- Training loop ----
    for epoch in range(args.n_epochs):
        plvae.epoch_hook(epoch=epoch)

        bar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(bar):
            x_batch = batch[0].to(device)
            c_batch = batch[1].to(device)
            p_batch = batch[2].to(device)

            plvae.optim.zero_grad()

            # Compute loss components (rec, perf, vol)
            losses = plvae.loss((x_batch, c_batch, p_batch))

            # Weighted sum for backprop
            loss = (losses * plvae.w).sum()
            loss.backward()
            plvae.optim.step()

            bar.set_postfix(
                {
                    "rec": f"{losses[0].item():.4f}",
                    "perf": f"{losses[1].item():.4f}",
                    "vol": f"{losses[2].item():.4f}",
                    "dim": plvae.dim,
                }
            )

            # Log to wandb
            if args.track:
                batches_done = epoch * len(bar) + i

                log_dict = {
                    "rec_loss": losses[0].item(),
                    "perf_loss": losses[1].item(),
                    "vol_loss": losses[2].item(),
                    "total_loss": loss.item(),
                    "active_dims": plvae.dim,
                    "epoch": epoch,
                    "w_volume": plvae.w[2].item(),
                }
                wandb.log(log_dict)

                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(bar)}] "
                    f"[rec loss: {losses[0].item():.4f}] [perf loss: {losses[1].item():.4f}] "
                    f"[vol loss: {losses[2].item():.4f}] [active dims: {plvae.dim}]"
                )

                # Sample and visualize at regular intervals
                if batches_done % args.sample_interval == 0:
                    with th.no_grad():
                        # Encode training designs
                        xs = x_train.to(device)
                        z = plvae.encode(xs)
                        z_std, idx = th.sort(z.std(0), descending=True)
                        z_mean = z.mean(0)
                        n_active = (z_std > 0).sum().item()

                        # Generate interpolated designs
                        x_ints = []
                        for alpha in [0, 0.25, 0.5, 0.75, 1]:
                            z_ = (1 - alpha) * z[:25] + alpha * th.roll(z, -1, 0)[:25]
                            x_ints.append(plvae.decode(z_).cpu().numpy())

                        # Generate random designs
                        z_rand = z_mean.unsqueeze(0).repeat([25, 1])
                        z_rand[:, idx[:n_active]] += z_std[:n_active] * th.randn_like(z_rand[:, idx[:n_active]])
                        x_rand = plvae.decode(z_rand).cpu().numpy()

                        # Get performance predictions on training data
                        pz_train = z[:, :perf_dim]
                        p_pred_scaled = plvae.predictor(th.cat([pz_train, c_train.to(device)], dim=-1))

                        # Inverse transform to get true-scale values for plotting
                        p_actual = p_scaler.inverse_transform(p_train_scaled.cpu().numpy()).flatten()
                        p_predicted = p_scaler.inverse_transform(p_pred_scaled.cpu().numpy()).flatten()

                        # Move tensors to CPU for plotting
                        z_std_cpu = z_std.cpu().numpy()
                        xs_cpu = xs.cpu().numpy()

                    # Plot 1: Latent dimension statistics
                    plt.figure(figsize=(12, 6))
                    plt.subplot(211)
                    plt.bar(np.arange(len(z_std_cpu)), z_std_cpu)
                    plt.yscale("log")
                    plt.xlabel("Latent dimension index")
                    plt.ylabel("Standard deviation")
                    plt.title(f"Number of principal components = {n_active}")
                    plt.subplot(212)
                    plt.bar(np.arange(n_active), z_std_cpu[:n_active])
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
                        axs[i_row, 0].imshow(xs_cpu[i_row].reshape(design_shape))
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
                    plt.grid(visible=True, alpha=0.3)
                    plt.legend()
                    plt.axis("equal")
                    plt.tight_layout()
                    plt.savefig(f"images/perf_pred_vs_actual_{batches_done}.png")
                    plt.close()

                    # Log plots to wandb
                    wandb.log(
                        {
                            "dim_plot": wandb.Image(f"images/dim_{batches_done}.png"),
                            "interp_plot": wandb.Image(f"images/interp_{batches_done}.png"),
                            "norm_plot": wandb.Image(f"images/norm_{batches_done}.png"),
                            "perf_pred_vs_actual": wandb.Image(f"images/perf_pred_vs_actual_{batches_done}.png"),
                        }
                    )

        # ---- Validation ----
        with th.no_grad():
            plvae.eval()
            val_rec = val_perf = val_vol = 0.0
            n = 0
            for batch_v in val_loader:
                x_v = batch_v[0].to(device)
                c_v = batch_v[1].to(device)
                p_v = batch_v[2].to(device)
                vlosses = plvae.loss((x_v, c_v, p_v))
                bsz = x_v.size(0)
                val_rec += vlosses[0].item() * bsz
                val_perf += vlosses[1].item() * bsz
                val_vol += vlosses[2].item() * bsz
                n += bsz
            val_rec /= n
            val_perf /= n
            val_vol /= n

        # Trigger pruning check at end of epoch
        plvae.epoch_report(epoch=epoch, callbacks=[], batch=None, loss=losses, pbar=None)

        if args.track:
            val_log_dict = {
                "epoch": epoch,
                "val_rec": val_rec,
                "val_perf": val_perf,
                "val_vol_loss": val_vol,
            }
            wandb.log(val_log_dict, commit=True)

        th.cuda.empty_cache()
        plvae.train()

        # Save models at end of training
        if args.save_model and epoch == args.n_epochs - 1:
            ckpt_plvae = {
                "epoch": epoch,
                "encoder": plvae.encoder.state_dict(),
                "decoder": plvae.decoder.state_dict(),
                "predictor": plvae.predictor.state_dict(),
                "optimizer": plvae.optim.state_dict(),
                "args": vars(args),
            }
            th.save(ckpt_plvae, "plvae.pth")
            if args.track:
                artifact = wandb.Artifact(f"{args.problem_id}_{args.algo}", type="model")
                artifact.add_file("plvae.pth")
                wandb.log_artifact(artifact, aliases=[f"seed_{args.seed}"])

    if args.track:
        wandb.finish()
