"""Barebones LVAE for 2D designs - simplest possible implementation.

This is a minimal, self-contained implementation of Least Volume Autoencoder
for 2D engineering designs. No dynamic pruning, no complex features - just
the core LVAE concept: reconstruction + volume minimization.
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
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import tqdm
import tyro
import wandb


@dataclass
class Args:
    # Problem and tracking
    problem_id: str = "heatconduction2d"
    """Problem ID to run. Must be one of the built-in problems in engibench."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """Algorithm name for tracking purposes."""
    track: bool = True
    """Whether to track with Weights & Biases."""
    wandb_project: str = "lvae_barebones"
    """WandB project name."""
    wandb_entity: str | None = None
    """WandB entity name. If None, uses the default entity."""
    seed: int = 1
    """Random seed for reproducibility."""
    save_model: bool = False
    """Whether to save the model after training."""
    sample_interval: int = 500
    """Interval for sampling designs during training."""

    # Training
    n_epochs: int = 1000
    """Number of training epochs."""
    batch_size: int = 128
    """Batch size for training."""
    lr: float = 1e-4
    """Learning rate for the optimizer."""

    # LVAE-specific
    latent_dim: int = 64
    """Dimensionality of the latent space."""
    w_rec: float = 1.0
    """Weight for reconstruction loss."""
    w_vol: float = 3e-4
    """Weight for volume loss."""
    eta: float = 1e-4
    """Smoothing factor for volume loss computation."""
    resize_dimensions: tuple[int, int] = (100, 100)
    """Dimensions to resize input images to before encoding/decoding."""


class Encoder(nn.Module):
    """Simple CNN encoder: 100x100 -> latent vector."""

    def __init__(self, latent_dim: int, resize_dimensions: tuple[int, int] = (100, 100)):
        super().__init__()
        self.resize_in = transforms.Resize(resize_dimensions)

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
        """Encode image to latent vector."""
        x = self.resize_in(x)  # (B,1,100,100)
        h = self.features(x)  # (B,512,7,7)
        return self.to_latent(h).flatten(1)  # (B,latent_dim)


class SNDecoder(nn.Module):
    """Spectral normalized CNN decoder: latent vector -> 100x100 image.

    Uses spectral normalization on all linear and convolutional layers
    to enforce 1-Lipschitz bound, which helps stabilize training and
    provides better volume loss behavior.
    """

    def __init__(self, latent_dim: int, design_shape: tuple[int, int]):
        super().__init__()
        self.design_shape = design_shape
        self.resize_out = transforms.Resize(self.design_shape)

        # Spectral normalized projection
        self.proj = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 512 * 7 * 7)),
            nn.ReLU(inplace=True),
        )

        # Spectral normalized deconvolutional layers
        self.deconv = nn.Sequential(
            # 7→13
            spectral_norm(nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=0, bias=False)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 13→25
            spectral_norm(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0, bias=False)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 25→50
            spectral_norm(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 50→100
            spectral_norm(nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)),
            nn.Sigmoid(),
        )

    def forward(self, z: th.Tensor) -> th.Tensor:
        """Decode latent vector to image."""
        x = self.proj(z).view(z.size(0), 512, 7, 7)  # (B,512,7,7)
        x = self.deconv(x)  # (B,1,100,100)
        return self.resize_out(x)  # (B,1,H_orig,W_orig)


class LeastVolumeAE(nn.Module):
    """Barebones Least Volume Autoencoder.

    Combines reconstruction loss with volume loss to learn a compact latent representation.
    Volume loss = geometric mean of latent standard deviations.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        optimizer: th.optim.Optimizer,
        w_rec: float = 1.0,
        w_vol: float = 0.01,
        eta: float = 1e-4,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.w_rec = w_rec
        self.w_vol = w_vol
        self.eta = eta

    def encode(self, x: th.Tensor) -> th.Tensor:
        """Encode design to latent representation."""
        return self.encoder(x)

    def decode(self, z: th.Tensor) -> th.Tensor:
        """Decode latent representation to design."""
        return self.decoder(z)

    def reconstruction_loss(self, x: th.Tensor, x_hat: th.Tensor) -> th.Tensor:
        """MSE reconstruction loss."""
        return th.nn.functional.mse_loss(x_hat, x)

    def volume_loss(self, z: th.Tensor) -> th.Tensor:
        """Volume loss = geometric mean of latent standard deviations.

        Encourages compact latent space by penalizing spread across dimensions.
        """
        std = z.std(dim=0)  # Standard deviation per latent dimension
        return th.exp(th.log(std + self.eta).mean())

    def loss(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Compute total loss = w_rec * reconstruction + w_vol * volume.

        Returns:
            total_loss, reconstruction_loss, volume_loss
        """
        z = self.encode(x)
        x_hat = self.decode(z)

        rec_loss = self.reconstruction_loss(x, x_hat)
        vol_loss = self.volume_loss(z)
        total_loss = self.w_rec * rec_loss + self.w_vol * vol_loss

        return total_loss, rec_loss, vol_loss

    def train_step(self, x: th.Tensor) -> tuple[float, float, float]:
        """Single training step.

        Returns:
            total_loss, reconstruction_loss, volume_loss (as Python floats)
        """
        self.optimizer.zero_grad()
        total_loss, rec_loss, vol_loss = self.loss(x)
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), rec_loss.item(), vol_loss.item()


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Load problem from EngiBench
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    design_shape = problem.design_space.shape

    # Initialize W&B tracking
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            save_code=True,
            name=run_name,
        )

    # Set random seeds for reproducibility
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True

    os.makedirs("images", exist_ok=True)

    # Device selection
    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    print(f"Using device: {device}")

    # Build models
    encoder = Encoder(args.latent_dim, args.resize_dimensions).to(device)
    decoder = SNDecoder(args.latent_dim, design_shape).to(device)
    print("Using SNDecoder with spectral normalization (1-Lipschitz bound)")

    # Build optimizer and LVAE
    optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    lvae = LeastVolumeAE(
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer,
        w_rec=args.w_rec,
        w_vol=args.w_vol,
        eta=args.eta,
    ).to(device)

    # Load dataset
    hf = problem.dataset.with_format("torch")
    train_ds = hf["train"]
    val_ds = hf["val"]

    x_train = train_ds["optimal_design"][:].unsqueeze(1).to(device)
    x_val = val_ds["optimal_design"][:].unsqueeze(1).to(device)

    train_loader = DataLoader(TensorDataset(x_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val), batch_size=args.batch_size, shuffle=False)

    # Training loop
    for epoch in range(args.n_epochs):
        lvae.train()
        epoch_rec_loss = 0.0
        epoch_vol_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0

        bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{args.n_epochs}")
        for i, batch in enumerate(bar):
            x_batch = batch[0]
            total_loss, rec_loss, vol_loss = lvae.train_step(x_batch)

            epoch_total_loss += total_loss
            epoch_rec_loss += rec_loss
            epoch_vol_loss += vol_loss
            n_batches += 1

            bar.set_postfix(
                {
                    "rec": f"{rec_loss:.4f}",
                    "vol": f"{vol_loss:.4f}",
                    "total": f"{total_loss:.4f}",
                }
            )

            # Log to wandb and sample visualizations
            if args.track:
                batches_done = epoch * len(bar) + i
                wandb.log(
                    {
                        "rec_loss": rec_loss,
                        "vol_loss": vol_loss,
                        "total_loss": total_loss,
                        "epoch": epoch,
                    }
                )
                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(bar)}] [rec loss: {rec_loss}] [vol loss: {vol_loss}]"
                )

                # Sample and visualize at regular intervals
                if batches_done % args.sample_interval == 0:
                    from itertools import product

                    with th.no_grad():
                        # Encode TRAINING designs
                        Xs = x_train
                        z = lvae.encode(Xs)
                        z_std, idx = th.sort(z.std(0), descending=True)
                        z_mean = z.mean(0)
                        N = (z_std > 0).sum().item()

                        # Generate interpolated designs
                        x_ints = []
                        for alpha in [0, 0.25, 0.5, 0.75, 1]:
                            z_ = (1 - alpha) * z[:25] + alpha * th.roll(z, -1, 0)[:25]
                            x_ints.append(lvae.decode(z_).cpu().numpy())

                        # Generate random designs
                        z_rand = z_mean.unsqueeze(0).repeat([25, 1])
                        z_rand[:, idx[:N]] += z_std[:N] * th.randn_like(z_rand[:, idx[:N]])
                        x_rand = lvae.decode(z_rand).cpu().numpy()

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
                    for i_plot, j in product(range(25), range(5)):
                        axs[i_plot, j + 1].imshow(x_ints[j][i_plot].reshape(design_shape))
                        axs[i_plot, j + 1].axis("off")
                        axs[i_plot, j + 1].set_aspect("equal")
                    for ax, alpha in zip(axs[0, 1:], [0, 0.25, 0.5, 0.75, 1]):
                        ax.set_title(rf"$\alpha$ = {alpha}")
                    for i_plot in range(25):
                        axs[i_plot, 0].imshow(Xs_cpu[i_plot].reshape(design_shape))
                        axs[i_plot, 0].axis("off")
                        axs[i_plot, 0].set_aspect("equal")
                    axs[0, 0].set_title("groundtruth")
                    fig.tight_layout()
                    plt.savefig(f"images/interp_{batches_done}.png")
                    plt.close()

                    # Plot 3: Random designs from latent space
                    fig, axs = plt.subplots(5, 5, figsize=(15, 7.5))
                    for k, (i_plot, j) in enumerate(product(range(5), range(5))):
                        axs[i_plot, j].imshow(x_rand[k].reshape(design_shape))
                        axs[i_plot, j].axis("off")
                        axs[i_plot, j].set_aspect("equal")
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

        # Average losses over epoch
        epoch_total_loss /= n_batches
        epoch_rec_loss /= n_batches
        epoch_vol_loss /= n_batches

        # Validation
        lvae.eval()
        val_rec_loss = 0.0
        val_vol_loss = 0.0
        val_total_loss = 0.0
        n_val_batches = 0

        with th.no_grad():
            for batch in val_loader:
                x_batch = batch[0]
                total_loss, rec_loss, vol_loss = lvae.loss(x_batch)
                val_total_loss += total_loss.item()
                val_rec_loss += rec_loss.item()
                val_vol_loss += vol_loss.item()
                n_val_batches += 1

        val_total_loss /= n_val_batches
        val_rec_loss /= n_val_batches
        val_vol_loss /= n_val_batches

        # Log to wandb
        if args.track:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/total_loss": epoch_total_loss,
                    "train/rec_loss": epoch_rec_loss,
                    "train/vol_loss": epoch_vol_loss,
                    "val/total_loss": val_total_loss,
                    "val/rec_loss": val_rec_loss,
                    "val/vol_loss": val_vol_loss,
                }
            )


        print(
            f"Epoch {epoch}/{args.n_epochs} - "
            f"Train: total={epoch_total_loss:.4f}, rec={epoch_rec_loss:.4f}, vol={epoch_vol_loss:.4f} - "
            f"Val: total={val_total_loss:.4f}, rec={val_rec_loss:.4f}, vol={val_vol_loss:.4f}"
        )

    # Save model
    if args.save_model:
        checkpoint = {
            "epoch": args.n_epochs - 1,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        th.save(checkpoint, "lvae_barebones.pth")

        if args.track:
            artifact = wandb.Artifact(f"{args.problem_id}_{args.algo}", type="model")
            artifact.add_file("lvae_barebones.pth")
            wandb.log_artifact(artifact, aliases=[f"seed_{args.seed}"])

    if args.track:
        wandb.finish()

    print("Training complete!")
