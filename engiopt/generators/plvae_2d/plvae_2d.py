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
import math
import os
import random
import time
from typing import Literal

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
import torch as th
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import tqdm
import tyro

from engiopt.checkpoint_store import save_checkpoint_package
from engiopt.core import checkpoint_identity
from engiopt.lvae.aes import InterpretablePerfLeastVolumeAE_DP
from engiopt.lvae.components import Encoder2D
from engiopt.lvae.components import SNMLPPredictor
from engiopt.lvae.components import TrueSNDecoder2D
from engiopt.lvae.data import filter_dataset_by_condition
from engiopt.transforms import get_performance_target
from engiopt.transforms import get_scalar_condition_keys
import wandb


@dataclass
class Args:
    """Command-line arguments for performance-predicting LVAE training."""

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
    hf_entity: str = "IDEALLab"
    """HF org/user where checkpoints are stored."""
    hf_repo_prefix: str = "engiopt"
    """HF repo prefix used for model-family repositories."""
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
    perf_dim: int = -1
    """Number of latent dimensions dedicated to performance prediction. If -1 (default), uses all latent_dim dimensions."""
    w_reconstruction: float = 1.0
    """Weight for reconstruction loss."""
    w_performance: float = 0.1
    """Weight for performance loss."""
    w_volume: float = 0.01
    """Weight for volume loss."""

    # Pruning parameters
    pruning_epoch: int = 500
    """Epoch to start pruning dimensions."""
    pruning_threshold: float = 0.05
    """Threshold for pruning (ratio for plummet, percentile for lognorm)."""
    pruning_strategy: str = "plummet"
    """Pruning strategy: 'plummet' or 'lognorm'."""
    alpha: float = 0.0
    """(lognorm only) Blending factor between reference and current distribution."""

    # Volume weight warmup
    volume_warmup_epochs: int = 0
    """Epochs to polynomially ramp volume weight from 0 to w_volume. 0 disables warmup."""
    volume_warmup_degree: float = 2.0
    """Polynomial degree for volume weight warmup (1.0=linear, 2.0=quadratic)."""

    # Architecture
    resize_dimensions: tuple[int, int] = (100, 100)
    """Dimensions to resize input images to before encoding/decoding."""
    whitening: bool = False
    """Apply PCA-rotation whitening after the encoder to decorrelate latent dimensions
    by construction. Preserves per-dimension variance for volume regularization."""
    predictor_hidden_dims: tuple[int, ...] = (256, 128)
    """Hidden dimensions for the MLP predictor."""
    conditional_predictor: bool = False
    """Whether to include conditions in performance prediction (True) or use only latent codes (False)."""
    decoder_lipschitz_scale: float = 1.0
    """Lipschitz bound for spectrally normalized decoder. Controls output scaling."""
    predictor_lipschitz_ratio: float = 1.0
    """Ratio multiplier for auto-scaled predictor Lipschitz bound. Effective bound is
    ratio * L_dec * sqrt(n_perf / design_dim), which equalizes gradient pressure between
    reconstruction and performance losses regardless of problem dimensionality."""
    perf_scaler: Literal["robust", "quantile"] = "robust"
    """Scaler for performance values. 'robust' preserves cardinal structure (RobustScaler);
    'quantile' maps to N(0,1) via rank transform (QuantileTransformer), making Lipschitz
    bounds ordinal and preventing heavy-tailed objectives from dominating the latent space."""

    # Dataset filtering
    condition_filter_key: str | None = None
    """Condition key to filter dataset on (e.g., 'weight'). None = use all data."""
    condition_filter_value: float | None = None
    """Exact value to match (within tolerance). Mutually exclusive with condition_filter_range."""
    condition_filter_range: tuple[float, float] | None = None
    """Inclusive [lo, hi] range to filter on. Overrides condition_filter_value."""
    condition_filter_tolerance: float = 0.01
    """Tolerance for exact-value matching."""


def volume_weight_schedule(
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
    scalar_cond_keys = get_scalar_condition_keys(problem, problem.dataset["train"])
    n_conds = len(scalar_cond_keys)

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
    enc = Encoder2D(args.latent_dim, design_shape, args.resize_dimensions, whitening=args.whitening)
    dec = TrueSNDecoder2D(args.latent_dim, design_shape, lipschitz_scale=args.decoder_lipschitz_scale)

    # Determine perf_dim: if -1 (default), use all latent dimensions
    perf_dim = args.latent_dim if args.perf_dim == -1 else args.perf_dim
    obj_keys = [name for name, _ in problem.objectives]
    n_perf = len(obj_keys)

    # Build MLP predictor (input: perf_dim latent dims + conditions if conditional)
    # Auto-scale the predictor's Lipschitz bound to equalize gradient pressure
    # with the decoder: the bound is the ratio times the decoder bound, times
    # the square root of n_perf over design_dim.
    design_dim = math.prod(design_shape)
    predictor_lipschitz_scale = (
        args.predictor_lipschitz_ratio * args.decoder_lipschitz_scale * math.sqrt(n_perf / design_dim)
    )

    predictor_input_dim = perf_dim + (n_conds if args.conditional_predictor else 0)
    predictor = SNMLPPredictor(
        input_dim=predictor_input_dim,
        output_dim=n_perf,
        hidden_dims=args.predictor_hidden_dims,
        lipschitz_scale=predictor_lipschitz_scale,
    )

    print(f"\n{'=' * 60}")
    print("Performance-LVAE Training")
    print(f"Problem: {args.problem_id}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Decoder: TrueSNDecoder2D (lipschitz_scale={args.decoder_lipschitz_scale})")
    print(f"Perf dim: {perf_dim} (first {perf_dim} dims predict performance)")
    print(f"Predictor mode: {'Conditional' if args.conditional_predictor else 'Unconditional'}")
    print(
        f"Predictor: SNMLPPredictor (lipschitz_scale={predictor_lipschitz_scale:.6f}, ratio={args.predictor_lipschitz_ratio}, design_dim={design_dim}, n_perf={n_perf})"
    )
    print(
        f"Predictor input: {predictor_input_dim} (perf_dim={perf_dim}, n_conds={n_conds if args.conditional_predictor else 0})"
    )
    print(f"Pruning epoch: {args.pruning_epoch}")
    print(f"Pruning strategy: {args.pruning_strategy}")
    print(f"Pruning threshold: {args.pruning_threshold}")
    if args.pruning_strategy == "lognorm":
        print(f"Alpha (lognorm): {args.alpha}")
    print(f"Volume warmup epochs: {args.volume_warmup_epochs}")
    print(f"{'=' * 60}\n")

    # Log computed predictor_lipschitz_scale to wandb for model reconstruction
    if args.track:
        wandb.config.update({"predictor_lipschitz_scale": predictor_lipschitz_scale})

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
        pruning_threshold=args.pruning_threshold,
        pruning_strategy=args.pruning_strategy,
        alpha=args.alpha,
    ).to(device)

    # ---- DataLoader ----
    raw_train = problem.dataset["train"]
    raw_val = problem.dataset["val"]

    if args.condition_filter_key is not None:
        raw_train = filter_dataset_by_condition(
            raw_train,
            args.condition_filter_key,
            value=args.condition_filter_value,
            value_range=args.condition_filter_range,
            tolerance=args.condition_filter_tolerance,
        )
        raw_val = filter_dataset_by_condition(
            raw_val,
            args.condition_filter_key,
            value=args.condition_filter_value,
            value_range=args.condition_filter_range,
            tolerance=args.condition_filter_tolerance,
        )

    train_ds = raw_train.with_format("torch")
    val_ds = raw_val.with_format("torch")

    # Extract designs, conditions, and performance
    x_train = train_ds["optimal_design"][:].unsqueeze(1)
    c_train = th.stack([train_ds[key][:] for key in scalar_cond_keys], dim=-1)
    p_train = get_performance_target(problem, train_ds)

    x_val = val_ds["optimal_design"][:].unsqueeze(1)
    c_val = th.stack([val_ds[key][:] for key in scalar_cond_keys], dim=-1)
    p_val = get_performance_target(problem, val_ds)

    # Scale performance values
    if args.perf_scaler == "quantile":
        p_scaler = QuantileTransformer(output_distribution="normal", n_quantiles=min(len(p_train), 1000))
    else:
        p_scaler = RobustScaler()
    p_train_scaled = th.from_numpy(p_scaler.fit_transform(p_train.numpy())).to(p_train.dtype)
    p_val_scaled = th.from_numpy(p_scaler.transform(p_val.numpy())).to(p_val.dtype)
    print(
        f"Performance scaler: {args.perf_scaler} | scaled range: [{p_train_scaled.min():.3f}, {p_train_scaled.max():.3f}]"
    )

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
                            x_ints.append(plvae.decode(z_).clamp(0, 1).cpu().numpy())

                        # Generate random designs
                        z_rand = z_mean.unsqueeze(0).repeat([25, 1])
                        z_rand[:, idx[:n_active]] += z_std[:n_active] * th.randn_like(z_rand[:, idx[:n_active]])
                        x_rand = plvae.decode(z_rand).clamp(0, 1).cpu().numpy()

                        # Get performance predictions on training data
                        pz_train = z[:, :perf_dim]
                        p_pred_scaled = plvae.predictor(th.cat([pz_train, c_train_scaled.to(device)], dim=-1))

                        # Inverse transform to get true-scale values for plotting
                        p_actual = p_scaler.inverse_transform(p_train_scaled.cpu().numpy())
                        p_predicted = p_scaler.inverse_transform(p_pred_scaled.cpu().numpy())

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

                    # Plot 4: Predicted vs actual performance (one subplot per objective)
                    fig, axs = plt.subplots(1, n_perf, figsize=(7 * n_perf, 7), squeeze=False)
                    for oi in range(n_perf):
                        ax = axs[0, oi]
                        pa = p_actual[:, oi]
                        pp = p_predicted[:, oi]
                        ax.scatter(pa, pp, alpha=0.5, s=20)
                        lo = min(pa.min(), pp.min())
                        hi = max(pa.max(), pp.max())
                        ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="1:1 line")
                        mse_oi = np.mean((pa - pp) ** 2)
                        ax.set_xlabel("Actual")
                        ax.set_ylabel("Predicted")
                        ax.set_title(f"{obj_keys[oi]}  MSE: {mse_oi:.4e}")
                        ax.set_aspect("equal")
                        ax.grid(visible=True, alpha=0.3)
                        ax.legend()
                    fig.tight_layout()
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
                # Without the mask, a reloaded encoder cannot tell which
                # dimensions were pruned, and every latent distance computed
                # from it would include axes that are constant by construction.
                "pruning_mask": plvae.pruning_mask.cpu(),
                "pruning_frozen_z": plvae.frozen_z.cpu(),
                "args": vars(args),
            }
            th.save(ckpt_plvae, "plvae.pth")
            save_checkpoint_package(
                checkpoint_backend="hf",
                hf_entity=args.hf_entity,
                hf_repo_prefix=args.hf_repo_prefix,
                hf_private=False,
                problem_id=args.problem_id,
                algo=args.algo,
                seed=args.seed,
                checkpoint_files={"plvae.pth": "plvae.pth"},
                run_config=vars(args),
                **checkpoint_identity(args),
                condition_keys=scalar_cond_keys,
                primary_files=["plvae.pth"],
            )

    if args.track:
        wandb.finish()
