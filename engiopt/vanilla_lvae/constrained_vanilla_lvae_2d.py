"""Constrained LVAE for 2D designs with threshold-gated volume loss.

This variant only applies volume loss when reconstruction is below a threshold,
preventing volume compression from hurting reconstruction quality.

For more information on LVAE, see: https://arxiv.org/abs/2404.17773
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
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import transforms
import tqdm
import tyro

from engiopt.vanilla_lvae.aes import ConstrainedLeastVolumeAE_DP
import wandb


@dataclass
class Args:
    """Command-line arguments for constrained LVAE training."""

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

    # Constraint parameters (uses Normalized MSE = MSE / Var(data) for problem-independence)
    nmse_threshold: float = 0.01
    """Target NMSE threshold (fraction of unexplained variance). 0.01 = 99% variance explained."""
    safety_margin: float = 0.2
    """Fraction below threshold for full volume push (0.2 = 20%)."""
    gate_ema_beta: float = 0.95
    """EMA smoothing factor for NMSE tracking."""
    comfort_multiplier: float = 1.0
    """Volume weight multiplier when in comfort zone. Set >1 to push harder."""
    w_volume: float = 1.0
    """Base volume loss weight (gate modulates this)."""

    # Pruning parameters
    pruning_epoch: int = 500
    """Epoch to start pruning dimensions."""
    pruning_threshold: float = 0.02
    """Threshold for pruning (ratio for plummet, percentile for lognorm)."""
    pruning_strategy: str = "plummet"
    """Pruning strategy to use: 'plummet' or 'lognorm'."""
    alpha: float = 0.0
    """(lognorm only) Blending factor between reference and current distribution."""

    # Architecture
    resize_dimensions: tuple[int, int] = (100, 100)
    """Dimensions to resize input images to before encoding/decoding."""


class Encoder(nn.Module):
    """Convolutional encoder for 2D designs.

    Architecture: Input -> Conv layers -> Latent vector
    * Input   [100x100]
    * Conv1   [50x50]   (k=4, s=2, p=1)
    * Conv2   [25x25]   (k=4, s=2, p=1)
    * Conv3   [13x13]   (k=3, s=2, p=1)
    * Conv4   [7x7]     (k=3, s=2, p=1)
    * Conv5   [1x1]     (k=7, s=1, p=0)
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
    * Latent   [latent_dim]
    * Linear   [512*7*7]
    * Reshape  [512x7x7]
    * Deconv1  [256x13x13]  (k=3, s=2, p=1)
    * Deconv2  [128x25x25]  (k=3, s=2, p=1)
    * Deconv3  [64x50x50]   (k=4, s=2, p=1)
    * Deconv4  [1x100x100]  (k=4, s=2, p=1)
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

    # Initialize constrained LVAE with threshold gating
    lvae = ConstrainedLeastVolumeAE_DP(
        encoder=enc,
        decoder=dec,
        optimizer=Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr),
        latent_dim=args.latent_dim,
        nmse_threshold=args.nmse_threshold,
        safety_margin=args.safety_margin,
        gate_ema_beta=args.gate_ema_beta,
        comfort_multiplier=args.comfort_multiplier,
        weights=[1.0, args.w_volume],
        pruning_epoch=args.pruning_epoch,
        pruning_threshold=args.pruning_threshold,
        pruning_strategy=args.pruning_strategy,
        alpha=args.alpha,
    ).to(device)

    # ---- DataLoader ----
    hf = problem.dataset.with_format("torch")
    train_ds = hf["train"]
    val_ds = hf["val"]

    x_train = train_ds["optimal_design"][:].unsqueeze(1)
    x_val = val_ds["optimal_design"][:].unsqueeze(1)

    # Set data variance for NMSE computation (problem-independent threshold)
    lvae.set_data_variance(x_train)

    print(f"\n{'=' * 60}")
    print("Constrained LVAE Training (NMSE-Gated Volume)")
    print(f"Problem: {args.problem_id}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"NMSE threshold: {args.nmse_threshold} (R² target: {1 - args.nmse_threshold:.2%})")
    print(f"Data variance: {lvae.data_var:.6f}")
    print(f"Safety margin: {args.safety_margin}")
    print(f"Comfort multiplier: {args.comfort_multiplier}")
    print(f"Pruning epoch: {args.pruning_epoch}")
    print(f"Pruning strategy: {args.pruning_strategy}")
    print(f"Pruning threshold: {args.pruning_threshold}")
    if args.pruning_strategy == "lognorm":
        print(f"Alpha (lognorm): {args.alpha}")
    print(f"{'=' * 60}\n")

    loader = DataLoader(TensorDataset(x_train), batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(x_val), batch_size=args.batch_size, shuffle=False)

    # ---- Training loop ----
    for epoch in range(args.n_epochs):
        lvae.epoch_hook(epoch=epoch)

        bar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(bar):
            x_batch = batch[0].to(device)
            lvae.optim.zero_grad()

            # Compute loss components (rec, gated_vol)
            losses = lvae.loss(x_batch)

            # Weighted sum for backprop
            loss = (losses * lvae.w).sum()
            loss.backward()
            lvae.optim.step()

            bar.set_postfix(
                {
                    "rec": f"{losses[0].item():.4f}",
                    "vol": f"{losses[1].item():.4f}",
                    "gate": f"{lvae.gate:.2f}",
                    "dim": lvae.dim,
                }
            )

            # Log to wandb
            if args.track:
                batches_done = epoch * len(bar) + i

                log_dict = {
                    "rec_loss": losses[0].item(),
                    "vol_loss": losses[1].item(),
                    "total_loss": loss.item(),
                    "volume_gate": lvae.gate,
                    "nmse_ema": lvae.nmse_ema,
                    "active_dims": lvae.dim,
                    "epoch": epoch,
                }
                wandb.log(log_dict)

                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(bar)}] "
                    f"[rec: {losses[0].item():.4f}] [nmse: {lvae.nmse_ema:.4f}] "
                    f"[gate: {lvae.gate:.2f}] [dims: {lvae.dim}]"
                )

                # Sample and visualize at regular intervals
                if batches_done % args.sample_interval == 0:
                    with th.no_grad():
                        # Encode training designs
                        xs = x_train.to(device)
                        z = lvae.encode(xs)
                        z_std, idx = th.sort(z.std(0), descending=True)
                        z_mean = z.mean(0)
                        n_active = (z_std > 0).sum().item()

                        # Generate interpolated designs
                        x_ints = []
                        for alpha in [0, 0.25, 0.5, 0.75, 1]:
                            z_ = (1 - alpha) * z[:25] + alpha * th.roll(z, -1, 0)[:25]
                            x_ints.append(lvae.decode(z_).cpu().numpy())

                        # Generate random designs
                        z_rand = z_mean.unsqueeze(0).repeat([25, 1])
                        z_rand[:, idx[:n_active]] += z_std[:n_active] * th.randn_like(z_rand[:, idx[:n_active]])
                        x_rand = lvae.decode(z_rand).cpu().numpy()

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

                    # Log plots to wandb
                    wandb.log(
                        {
                            "dim_plot": wandb.Image(f"images/dim_{batches_done}.png"),
                            "interp_plot": wandb.Image(f"images/interp_{batches_done}.png"),
                            "norm_plot": wandb.Image(f"images/norm_{batches_done}.png"),
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

        # Trigger pruning check at end of epoch
        lvae.epoch_report(epoch=epoch, callbacks=[], batch=None, loss=losses, pbar=None)

        if args.track:
            val_log_dict = {
                "epoch": epoch,
                "val_rec": val_rec,
                "val_vol_loss": val_vol,
            }
            wandb.log(val_log_dict, commit=True)

        th.cuda.empty_cache()
        lvae.train()

        # Save models at end of training
        if args.save_model and epoch == args.n_epochs - 1:
            ckpt_lvae = {
                "epoch": epoch,
                "encoder": lvae.encoder.state_dict(),
                "decoder": lvae.decoder.state_dict(),
                "optimizer": lvae.optim.state_dict(),
                "args": vars(args),
            }
            th.save(ckpt_lvae, "constrained_vanilla_lvae.pth")
            if args.track:
                artifact = wandb.Artifact(f"{args.problem_id}_{args.algo}", type="model")
                artifact.add_file("constrained_vanilla_lvae.pth")
                wandb.log_artifact(artifact, aliases=[f"seed_{args.seed}"])

    if args.track:
        wandb.finish()
