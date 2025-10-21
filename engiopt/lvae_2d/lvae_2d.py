"""LVAE_DP for 2D designs (no performance or conditioning)."""

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

from engiopt.lvae_2d.aes import LeastVolumeAE_DynamicPruning
from engiopt.lvae_2d.utils import polynomial_schedule
from engiopt.lvae_2d.utils import spectral_norm_conv
from engiopt.lvae_2d.utils import TrueSNDeconv2DCombo
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
    wandb_project: str = "lvae"
    """WandB project name."""
    wandb_entity: str | None = None
    """WandB entity name. If None, uses the default entity."""
    seed: int = 1
    """Random seed for reproducibility."""
    save_model: bool = False
    """Whether to save the model after training."""
    sample_interval: int = 1000  # How often to sample designs
    """Interval for sampling designs during training."""

    # Training dp
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
    """Momentum for the pruning ratio calculation."""
    eta: float = 1e-4
    """Scaling factor for the volume loss."""
    resize_dimensions: tuple[int, int] = (100, 100)
    """Dimensions to resize input images to before encoding/decoding."""
    use_spectral_norm: bool = True
    """Whether to use spectral normalization in the decoder (1-Lipschitz bound)."""

    # Dynamic pruning
    pruning_strategy: str = "plummet"
    """Which pruning strategy to use: [plummet, pca_cdf, lognorm, probabilistic]."""
    cdf_threshold: float = 0.95
    """(pca_cdf) Cumulative variance threshold."""
    temperature: float = 1.0
    """(probabilistic) Sampling temperature."""
    plummet_threshold: float = 0.02
    """(plummet) Threshold for pruning dimensions."""
    alpha: float = 0.2
    """(lognorm) Weighting factor for blending reference and current distribution ()."""
    percentile: float = 0.05
    """(lognorm) Percentile threshold for pruning."""

    # Safeguard parameters (match aes.py defaults - "unsafe" by default)
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
        self.design_shape = design_shape  # Store design shape

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


# ---------- Decoder: 7→13→25→50→100 then (optional) resize to original ----------
class Decoder(nn.Module):
    """Latent vector --> binary pixel image.

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
        self.design_shape = design_shape  # Store design shape
        self.resize_out = transforms.Resize(self.design_shape)
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, 512 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            # 7→13
            nn.ConvTranspose2d(
                512,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 13→25
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 25→50
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 50→100
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
            nn.Sigmoid(),  # use Sigmoid for [0,1] images; use Tanh if you normalize to [-1,1]
        )

    def forward(self, z: th.Tensor) -> th.Tensor:
        x = self.proj(z).view(z.size(0), 512, 7, 7)  # (B,512,7,7)
        x = self.deconv(x)  # (B,1,100,100)
        return self.resize_out(x)  # (B,1,H_orig,W_orig)


# ---------- TrueSNDecoder: Same architecture as Decoder but with spectral normalization ----------
class TrueSNDecoder(nn.Module):
    """Decoder with spectral normalization for 1-Lipschitz bound.

    Same architecture as Decoder (7→13→25→50→100) but with spectral normalization
    applied to all linear and convolutional layers.

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
        # Input shapes for each layer (needed for TrueSNDeconv2DCombo)
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

    enc = Encoder(args.latent_dim, design_shape, args.resize_dimensions)
    if args.use_spectral_norm:
        dec = TrueSNDecoder(args.latent_dim, design_shape)
        print("Using TrueSNDecoder with spectral normalization (1-Lipschitz bound)")
    else:
        dec = Decoder(args.latent_dim, design_shape)
        print("Using standard Decoder")

    # --- Build pruning parameters (match aes.py expectations) ---
    pruning_params = {}
    if args.pruning_strategy == "plummet":
        pruning_params["threshold"] = args.plummet_threshold
    elif args.pruning_strategy == "pca_cdf":
        pruning_params["threshold"] = args.cdf_threshold  # aes.py expects "threshold", not "cdf_threshold"
    elif args.pruning_strategy == "probabilistic":
        pruning_params["temperature"] = args.temperature
    elif args.pruning_strategy == "lognorm":
        pruning_params["alpha"] = args.alpha
        pruning_params["threshold"] = args.percentile  # aes.py expects "threshold" for percentile
    else:
        raise ValueError(f"Unknown pruning_strategy: {args.pruning_strategy}")

    # Initialize LVAE with dynamic pruning and safeguards
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
        # Pass safeguards directly to constructor (not post-hoc)
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
        lvae.epoch_hook(epoch=epoch)

        bar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(bar):
            x_batch = batch[0].to(device)
            lvae.optim.zero_grad()
            losses = lvae.loss(x_batch)  # [rec,prd,vol]
            loss = (losses * lvae.w).sum()  # apply weights
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
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(bar)}] [rec loss: {losses[0].item()}] [vol loss: {losses[1].item()}] [active dims: {lvae.dim}]"
                )
                # This saves a grid image of 25 generated designs every sample_interval
                if batches_done % args.sample_interval == 0:
                    ### Saves an image of the dimensionality of the latent space ###
                    with th.no_grad():
                        Xs = x_test.to(device)
                        z = lvae.encode(Xs)
                        z_std, idx = th.sort(z.std(0), descending=True)
                        z_mean = z.mean(0)
                        N = (z_std > 0).sum().item()

                    plt.figure(figsize=(12, 6))
                    plt.subplot(211)
                    plt.bar(np.arange(len(z_std)), z_std.detach().cpu().numpy())
                    plt.yscale("log")
                    plt.xlabel("Latent dimension index")
                    plt.ylabel("Standard deviation")
                    plt.title(f"Number of principal components = {N}")

                    plt.subplot(212)
                    plt.bar(np.arange(N), z_std[:N].detach().cpu().numpy())
                    plt.yscale("log")
                    plt.xlabel("Latent dimension index")
                    plt.ylabel("Standard deviation")

                    img_fname = f"images/dim_{batches_done}.png"
                    plt.savefig(img_fname)
                    plt.close()
                    wandb.log({"dim_plot": wandb.Image(img_fname)})

                    # Interpolates 25 designs from the test set
                    with th.no_grad():
                        x_ints = []
                        for alpha in [0, 0.25, 0.5, 0.75, 1]:
                            z_ = alpha * z[:25] + (1 - alpha) * th.roll(z, 1, 0)[:25]
                            x_int = lvae.decode(z_).detach().cpu().numpy()
                            x_ints.append(x_int)

                    fig, axs = plt.subplots(25, 6, figsize=(6 * 2, 25 * 1))
                    for i, j in product(range(25), range(5)):  # noqa: PLW2901
                        img = x_ints[j][i].reshape(design_shape[0], design_shape[1])
                        axs[i, j + 1].imshow(img)
                        axs[i, j + 1].axis("off")
                        axs[i, j + 1].set_aspect("equal")

                    for ax, alpha in zip(axs[0, 1:], [0, 0.25, 0.5, 0.75, 1]):
                        ax.set_title(rf"$\alpha$ = {alpha}")

                    for i in range(25):  # noqa: PLW2901
                        axs[i, 0].imshow(Xs.detach().cpu().numpy()[i - 1].reshape(design_shape[0], design_shape[1]))
                        axs[i, 0].axis("off")
                        axs[i, 0].set_aspect("equal")

                    axs[0, 0].set_title("groundtruth")
                    fig.tight_layout()

                    img_fname = f"images/interp_{batches_done}.png"
                    plt.savefig(img_fname)
                    plt.close()
                    wandb.log({"interp_plot": wandb.Image(img_fname)})

                    with th.no_grad():
                        z_rand = z.mean(0).unsqueeze(0).repeat([25, 1])
                        z_rand[:, idx[:N]] += z_std[:N] * th.randn_like(z_rand[:, idx[:N]]) * 1
                        x_rand = lvae.decode(z_rand).detach().cpu().numpy()

                    fig, axs = plt.subplots(5, 5, figsize=(5 * 3, 5 * 1.5))
                    for k, (i, j) in enumerate(product(range(5), range(5))):  # noqa: PLW2901
                        img = x_rand[k].reshape(design_shape[0], design_shape[1])
                        axs[i, j].imshow(img)
                        axs[i, j].axis("off")
                        axs[i, j].set_aspect("equal")

                    fig.tight_layout()
                    plt.suptitle("Gaussian random designs from latent space")
                    img_fname = f"images/norm_{batches_done}.png"
                    plt.savefig(img_fname)
                    plt.close()
                    wandb.log({"norm_plot": wandb.Image(img_fname)})

        # ---- validation (batched, no graph) ----
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
            # single commit per epoch → Hyperband “iteration” == epoch
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

        # --------------
        #  Save models
        # --------------
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
