"""Constrained LVAE for 2D designs with selectable constraint modes.

Provides three strategies for balancing reconstruction and volume compression:
- one_sided: Mutually exclusive (rec OR vol, never both)
- gated: Additive with gating (rec always, vol only when below threshold)
- gradient_balanced: Auto-scaled vol to match rec gradient magnitude

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
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import tqdm
import tyro

from engiopt.vanilla_lvae.aes import ConstrainedLeastVolumeAE_DP
from engiopt.vanilla_lvae.components import Encoder2D
from engiopt.vanilla_lvae.components import TrueSNDecoder2D
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
    n_epochs: int = 10000
    """Number of training epochs."""
    batch_size: int = 128
    """Batch size for training."""
    lr: float = 1e-4
    """Learning rate for the optimizer."""

    # LVAE-specific
    latent_dim: int = 100
    """Dimensionality of the latent space (overestimate)."""

    # Constraint parameters (uses Normalized MSE = MSE / Var(data) for problem-independence)
    nmse_threshold: float = 0.05
    """NMSE ceiling. Training aims to stay at or below this threshold."""
    constraint_mode: str = "one_sided"
    """Constraint mode: 'one_sided' (rec or vol), 'gated' (rec + vol), 'gradient_balanced' (auto-scaled)."""
    w_vol: float = 1.0
    """Volume loss weight (gated mode only)."""
    ema_beta: float = 0.9
    """EMA smoothing for loss tracking (gradient_balanced mode)."""

    # Pruning parameters
    pruning_epoch: int = 500
    """Epoch to start pruning dimensions."""
    pruning_threshold: float = 0.05
    """Threshold for pruning (ratio for plummet, percentile for lognorm)."""
    pruning_strategy: str = "plummet"
    """Pruning strategy to use: 'plummet' or 'lognorm'."""
    alpha: float = 0.0
    """(lognorm only) Blending factor between reference and current distribution."""

    # Architecture
    resize_dimensions: tuple[int, int] = (100, 100)
    """Dimensions to resize input images to before encoding/decoding."""
    lipschitz_scale: float = 1.0
    """Lipschitz bound for spectrally normalized decoder. Controls output scaling."""


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
    enc = Encoder2D(args.latent_dim, design_shape, args.resize_dimensions)
    dec = TrueSNDecoder2D(args.latent_dim, design_shape, lipschitz_scale=args.lipschitz_scale)

    # Initialize constrained LVAE with selectable constraint mode
    lvae = ConstrainedLeastVolumeAE_DP(
        encoder=enc,
        decoder=dec,
        optimizer=Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr),
        latent_dim=args.latent_dim,
        nmse_threshold=args.nmse_threshold,
        constraint_mode=args.constraint_mode,
        w_vol=args.w_vol,
        ema_beta=args.ema_beta,
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
    print("Constrained LVAE Training")
    print(f"Problem: {args.problem_id}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Decoder: TrueSNDecoder2D (lipschitz_scale={args.lipschitz_scale})")
    print(f"NMSE threshold: {args.nmse_threshold} (R² = {1 - args.nmse_threshold:.2%})")
    print(f"Constraint mode: {args.constraint_mode}")
    if args.constraint_mode == "gated":
        print(f"  w_vol: {args.w_vol}")
    elif args.constraint_mode == "gradient_balanced":
        print(f"  ema_beta: {args.ema_beta}")
    print(f"Data variance: {lvae.data_var:.6f}")
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

            # Compute loss (scalar, mode-dependent)
            loss = lvae.loss(x_batch)
            loss.backward()
            lvae.optim.step()

            bar.set_postfix(
                {
                    "rec": f"{lvae.rec_loss:.4f}",
                    "vol": f"{lvae.vol_loss:.4f}",
                    "nmse": f"{lvae.nmse:.4f}",
                    "active": int(lvae.vol_active),
                    "dim": lvae.dim,
                }
            )

            # Log to wandb
            if args.track:
                batches_done = epoch * len(bar) + i

                log_dict = {
                    "rec_loss": lvae.rec_loss,
                    "vol_loss": lvae.vol_loss,
                    "total_loss": loss.item(),
                    "nmse": lvae.nmse,
                    "nmse_threshold": args.nmse_threshold,
                    "vol_active": int(lvae.vol_active),
                    "balance_factor": lvae.balance_factor,
                    "active_dims": lvae.dim,
                    "epoch": epoch,
                    "constraint_mode": args.constraint_mode,
                }
                wandb.log(log_dict)

                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(bar)}] "
                    f"[rec: {lvae.rec_loss:.4f}] [vol: {lvae.vol_loss:.4f}] [nmse: {lvae.nmse:.4f}] "
                    f"[active: {int(lvae.vol_active)}] [dims: {lvae.dim}]"
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
            val_rec = val_vol = val_nmse = 0.0
            n = 0
            for batch_v in val_loader:
                x_v = batch_v[0].to(device)
                _ = lvae.loss(x_v)  # Computes and stores metrics
                bsz = x_v.size(0)
                val_rec += lvae.rec_loss * bsz
                val_vol += lvae.vol_loss * bsz
                val_nmse += lvae.nmse * bsz
                n += bsz
            val_rec /= n
            val_vol /= n
            val_nmse /= n

        # Trigger pruning check at end of epoch
        lvae.epoch_report(epoch=epoch, callbacks=[], batch=None, loss=loss, pbar=None)

        if args.track:
            val_log_dict = {
                "epoch": epoch,
                "val_rec_loss": val_rec,
                "val_vol_loss": val_vol,
                "val_nmse": val_nmse,
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
