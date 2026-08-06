"""Constrained performance-predicting LVAE for 2D designs.

Volume regularization is only trustworthy once the autoencoder actually
reconstructs its inputs -- otherwise the cheapest way to shrink latent volume is
to stop encoding anything. This script therefore optimizes volume subject to two
constraints, in priority order:

1. Reconstruction: `NMSE_rec <= nmse_threshold_rec`
2. Performance prediction: `NMSE_perf <= nmse_threshold_perf`

Reconstruction must be satisfied before performance, and both before volume
optimization begins. A bounded satisfaction gate (`constraint_tau`) softens the
handover so training does not oscillate at the threshold.

The trained model serves two roles: a generator on the leaderboard, and the
measuring instrument for latent-space metrics
(`engiopt.evaluation.metrics.latent`). Both load the same checkpoint package.

For background on least-volume autoencoders see https://arxiv.org/abs/2404.17773

Example:
    `python engiopt/generators/constrained_plvae_2d/constrained_plvae_2d.py
    --problem-id beams2d --seed 1 --save-model`
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import random
import time
from typing import Literal

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.colors as mcolors
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
from engiopt.lvae.aes import ConstrainedPerfLeastVolumeAE_DP
from engiopt.lvae.components import ConditionEncoder2D
from engiopt.lvae.components import Encoder2D
from engiopt.lvae.components import SNMLPPredictor
from engiopt.lvae.components import TrueSNDecoder2D
from engiopt.lvae.data import filter_dataset_by_condition
from engiopt.transforms import get_image_condition_keys
from engiopt.transforms import get_image_condition_shape
from engiopt.transforms import get_performance_target
from engiopt.transforms import get_scalar_condition_keys
from engiopt.transforms import rasterize_index_conditions
import wandb


@dataclass
class Args:
    """Command-line arguments for constrained Performance-LVAE training."""

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

    # Constraint parameters (uses Normalized MSE = MSE / Var(data) for problem-independence)
    nmse_threshold_rec: float = 0.05
    """NMSE threshold for reconstruction. Training aims to stay at or below this. Default: 0.01 (R² = 99%)."""
    nmse_threshold_perf: float = 0.05
    """NMSE threshold for performance prediction. Default: 0.05 (R² = 95%)."""
    constraint_tau: float = 0.3
    """Width of the satisfaction-gate transition as a fraction of the threshold. The gate
    g = sigmoid((nmse/threshold - 1)/tau) weights rec/perf vs. volume: g->1 violated, 0.5 at
    threshold, ->0 satisfied. Default 0.3 clears typical per-batch NMSE noise at batch ~128
    and sits on the stable plateau (not a sensitive knob)."""

    # Pruning parameters
    pruning_epoch: int = 500
    """Epoch to start pruning dimensions."""
    pruning_threshold: float = 0.05
    """Threshold for pruning (ratio for plummet, percentile for lognorm)."""
    pruning_strategy: str = "plummet"
    """Pruning strategy: 'plummet' or 'lognorm'."""
    alpha: float = 0.0
    """(lognorm only) Blending factor between reference and current distribution."""

    # Architecture
    resize_dimensions: tuple[int, int] = (100, 100)
    """Dimensions to resize input images to before encoding/decoding."""
    predictor_hidden_dims: tuple[int, ...] = (256, 128)
    """Hidden dimensions for the MLP predictor."""
    conditional_predictor: bool = False
    """Whether to include conditions in performance prediction (True) or use only latent codes (False)."""
    conditional_decoder: bool = False
    """Whether to condition the decoder on scalar + image conditions for multi-modality measurement."""
    cond_embed_dim: int = 64
    """Dimensionality of image condition embedding (only used when image conditions exist)."""
    whitening: bool = False
    """Apply PCA-rotation whitening after the encoder to decorrelate latent dimensions
    by construction. Preserves per-dimension variance for volume regularization."""
    decoder_lipschitz_scale: float = 1.0
    """Lipschitz bound for spectrally normalized decoder. Controls output scaling."""
    predictor_lipschitz_ratio: float = 1.0
    """Ratio multiplier for auto-scaled predictor Lipschitz bound. Effective bound is
    ratio * L_dec * sqrt(design_dim / n_perf), which strengthens the predictor
    proportionally to the dimensionality gap so it can shape the latent space."""
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


BATCH_WITH_IMAGE_CONDITIONS = 3
"""Batches longer than this carry rasterized image conditions in slot 3."""

MIN_DIMS_FOR_SCATTER = 2
"""A latent scatter plot needs at least two active dimensions."""


if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    design_shape = problem.design_space.shape
    scalar_cond_keys = get_scalar_condition_keys(problem, problem.dataset["train"])
    n_conds = len(scalar_cond_keys)

    # Detect image conditions (e.g., boundary matrices in thermoelastic2d)
    img_cond_keys = get_image_condition_keys(problem, problem.dataset["train"])
    n_img_conds = len(img_cond_keys)
    img_cond_shape: tuple[int, ...] | None = None
    if n_img_conds > 0:
        img_cond_shape = get_image_condition_shape(problem.dataset["train"], img_cond_keys)
        print(f"Found {n_img_conds} image condition(s): {img_cond_keys}, shape={img_cond_shape}")

    # Logging
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={**vars(args), "design_shape": list(design_shape)},
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
    th.use_deterministic_algorithms(mode=True, warn_only=True)
    g = th.Generator().manual_seed(args.seed)  # For DataLoader shuffling

    os.makedirs("images", exist_ok=True)

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    # Build encoder (always unconditional — sees design only)
    enc = Encoder2D(args.latent_dim, design_shape, args.resize_dimensions, whitening=args.whitening)

    # Build condition encoder for image conditions (if any)
    condition_encoder: ConditionEncoder2D | None = None
    img_cond_embed_dim = 0
    if n_img_conds > 0:
        condition_encoder = ConditionEncoder2D(
            n_img_conds=n_img_conds,
            cond_embed_dim=args.cond_embed_dim,
            resize_dimensions=args.resize_dimensions,
        )
        img_cond_embed_dim = args.cond_embed_dim

    # Compute condition dimensions for decoder and predictor
    # Scalar conditions are included when conditional_predictor is on (they get scaled)
    # Image condition embeddings are always included when image conditions exist
    n_scalar_for_cond = n_conds if args.conditional_predictor else 0

    cond_dim_for_decoder = 0
    if args.conditional_decoder:
        cond_dim_for_decoder = n_scalar_for_cond + img_cond_embed_dim

    cond_dim_for_predictor = n_scalar_for_cond + img_cond_embed_dim

    # Build decoder (conditional when cond_dim > 0)
    dec = TrueSNDecoder2D(
        args.latent_dim, design_shape, lipschitz_scale=args.decoder_lipschitz_scale, cond_dim=cond_dim_for_decoder
    )

    # Determine perf_dim: if -1 (default), use all latent dimensions
    perf_dim = args.latent_dim if args.perf_dim == -1 else args.perf_dim
    obj_keys = [name for name, _ in problem.objectives]
    n_perf = len(obj_keys)

    # Build MLP predictor (input: perf_dim latent dims + condition embedding)
    # Auto-scale predictor Lipschitz bound to strengthen predictor proportionally to
    # the dimensionality gap:  L_pred = ratio * L_dec * sqrt(design_dim / n_perf)
    # MSE over design_dim pixels dilutes decoder gradients; this compensates so the
    # predictor has enough capacity to shape the latent space for performance.
    design_dim = math.prod(design_shape)
    predictor_lipschitz_scale = (
        args.predictor_lipschitz_ratio * args.decoder_lipschitz_scale * math.sqrt(design_dim / n_perf)
    )

    predictor_input_dim = perf_dim + cond_dim_for_predictor
    predictor = SNMLPPredictor(
        input_dim=predictor_input_dim,
        output_dim=n_perf,
        hidden_dims=args.predictor_hidden_dims,
        lipschitz_scale=predictor_lipschitz_scale,
    )

    print(f"\n{'=' * 60}")
    print("Constrained Performance-LVAE Training (One-Sided)")
    print(f"Problem: {args.problem_id}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Decoder: TrueSNDecoder2D (lipschitz_scale={args.decoder_lipschitz_scale}, cond_dim={cond_dim_for_decoder})")
    print(f"Decoder mode: {'Conditional' if args.conditional_decoder else 'Unconditional'}")
    print(f"Perf dim: {perf_dim} (first {perf_dim} dims predict performance)")
    print(f"Predictor mode: {'Conditional' if args.conditional_predictor else 'Unconditional'}")
    print(
        f"Predictor: SNMLPPredictor (lipschitz_scale={predictor_lipschitz_scale:.6f}, ratio={args.predictor_lipschitz_ratio}, design_dim={design_dim}, n_perf={n_perf})"
    )
    print(f"Predictor input: {predictor_input_dim} (perf_dim={perf_dim}, cond_dim={cond_dim_for_predictor})")
    if n_img_conds > 0:
        print(f"Image conditions: {n_img_conds} channel(s), embed_dim={img_cond_embed_dim}")
    print(f"NMSE threshold (rec): {args.nmse_threshold_rec} (R² = {1 - args.nmse_threshold_rec:.2%})")
    print(f"NMSE threshold (perf): {args.nmse_threshold_perf} (R² = {1 - args.nmse_threshold_perf:.2%})")
    print(
        f"Constraint gate tau: {args.constraint_tau} (band = threshold x [{1 - args.constraint_tau:.2f}, {1 + args.constraint_tau:.2f}])"
    )
    print(f"Pruning epoch: {args.pruning_epoch}")
    print(f"Pruning strategy: {args.pruning_strategy}")
    print(f"Pruning threshold: {args.pruning_threshold}")
    if args.pruning_strategy == "lognorm":
        print(f"Alpha (lognorm): {args.alpha}")
    if args.whitening:
        print("Latent whitening: enabled (PCA-rotation decorrelation)")
    print(f"{'=' * 60}\n")

    # Log computed predictor_lipschitz_scale to wandb for model reconstruction
    if args.track:
        wandb.config.update({"predictor_lipschitz_scale": predictor_lipschitz_scale})

    # Collect all parameters for optimizer (including condition encoder if present)
    all_params = list(enc.parameters()) + list(dec.parameters()) + list(predictor.parameters())
    if condition_encoder is not None:
        all_params += list(condition_encoder.parameters())

    # Initialize Constrained Performance-LVAE with dynamic pruning
    plvae = ConstrainedPerfLeastVolumeAE_DP(
        encoder=enc,
        decoder=dec,
        predictor=predictor,
        optimizer=Adam(all_params, lr=args.lr),
        latent_dim=args.latent_dim,
        perf_dim=perf_dim,
        nmse_threshold_rec=args.nmse_threshold_rec,
        nmse_threshold_perf=args.nmse_threshold_perf,
        tau=args.constraint_tau,
        pruning_epoch=args.pruning_epoch,
        pruning_threshold=args.pruning_threshold,
        pruning_strategy=args.pruning_strategy,
        alpha=args.alpha,
        conditional_decoder=args.conditional_decoder,
        condition_encoder=condition_encoder,
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

    # Extract image conditions (if any)
    # Image conditions may be dense arrays (H, W) or sparse index arrays (variable length).
    # Sparse index arrays are rasterized into dense binary masks on design_shape.
    ic_train: th.Tensor | None = None
    ic_val: th.Tensor | None = None
    if n_img_conds > 0 and img_cond_shape is not None:
        if len(img_cond_shape) == 1:
            # Sparse node index arrays — rasterize to dense masks on node grid (H+1, W+1)
            node_grid = (design_shape[0] + 1, design_shape[1] + 1)
            _ic_tr = rasterize_index_conditions(raw_train, img_cond_keys, node_grid)
            _ic_va = rasterize_index_conditions(raw_val, img_cond_keys, node_grid)
            print(f"Sparse image conditions rasterized to {node_grid} node grid: {_ic_tr.shape}")
        else:
            # Dense image conditions — stack directly
            _ic_tr = th.stack([train_ds[k][:].float() for k in img_cond_keys], dim=1)
            _ic_va = th.stack([val_ds[k][:].float() for k in img_cond_keys], dim=1)
            print(f"Dense image conditions loaded: {_ic_tr.shape}")
        ic_train = _ic_tr
        ic_val = _ic_va

    # Scale performance values
    if args.perf_scaler == "quantile":
        p_scaler = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(len(p_train), 1000),
            random_state=args.seed,
        )
    else:
        p_scaler = RobustScaler()
    p_train_scaled = th.from_numpy(p_scaler.fit_transform(p_train.numpy())).to(p_train.dtype)
    p_val_scaled = th.from_numpy(p_scaler.transform(p_val.numpy())).to(p_val.dtype)
    print(
        f"Performance scaler: {args.perf_scaler} | scaled range: [{p_train_scaled.min():.3f}, {p_train_scaled.max():.3f}]"
    )

    # Scale conditions using RobustScaler (if using conditional predictor or conditional decoder)
    if args.conditional_predictor or args.conditional_decoder:
        c_scaler = RobustScaler()
        c_train_scaled = th.from_numpy(c_scaler.fit_transform(c_train.numpy())).to(c_train.dtype)
        c_val_scaled = th.from_numpy(c_scaler.transform(c_val.numpy())).to(c_val.dtype)
    else:
        # Dummy tensors when not using conditions (won't be used in predictor or decoder)
        c_train_scaled = th.zeros(len(x_train), 0)
        c_val_scaled = th.zeros(len(x_val), 0)

    # Set data variances for NMSE computation (problem-independent thresholds)
    plvae.set_data_variance(x_train)
    plvae.set_perf_variance(p_train_scaled)

    print(f"Data variance (designs): {plvae.data_var:.6f}")
    print(f"Perf variance (scaled): {plvae.perf_var:.6f}")

    # Build DataLoaders (include image conditions as 4th tensor when present)
    train_tensors = [x_train, c_train_scaled, p_train_scaled]
    val_tensors = [x_val, c_val_scaled, p_val_scaled]
    if ic_train is not None:
        train_tensors.append(ic_train)
        val_tensors.append(ic_val)

    loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(*val_tensors),
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
            ic_batch = batch[3].to(device) if len(batch) > BATCH_WITH_IMAGE_CONDITIONS else None

            plvae.optim.zero_grad()

            # Compute loss (scalar, constraint-dependent)
            batch_tuple = (x_batch, c_batch, p_batch, ic_batch) if ic_batch is not None else (x_batch, c_batch, p_batch)
            loss = plvae.loss(batch_tuple)

            loss.backward()
            plvae.optim.step()

            bar.set_postfix(
                {
                    "rec": f"{plvae.rec_loss:.4f}",
                    "perf": f"{plvae.perf_loss:.4f}",
                    "vol": f"{plvae.vol_loss:.4f}",
                    "nmse_r": f"{plvae.nmse_rec:.4f}",
                    "nmse_p": f"{plvae.nmse_perf:.4f}",
                    "w_vol": f"{plvae.w_vol:.2f}",
                    "dim": plvae.dim,
                }
            )

            # Log to wandb
            if args.track:
                batches_done = epoch * len(bar) + i

                log_dict = {
                    "rec_loss": plvae.rec_loss,
                    "perf_loss": plvae.perf_loss,
                    "vol_loss": plvae.vol_loss,
                    "total_loss": loss.item(),
                    "nmse_rec": plvae.nmse_rec,
                    "nmse_perf": plvae.nmse_perf,
                    "nmse_threshold_rec": args.nmse_threshold_rec,
                    "nmse_threshold_perf": args.nmse_threshold_perf,
                    "g_rec": plvae.g_rec,
                    "g_perf": plvae.g_perf,
                    "w_vol": plvae.w_vol,
                    "vol_active": int(plvae.vol_active),
                    "active_dims": plvae.dim,
                    "epoch": epoch,
                }
                wandb.log(log_dict)

                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(bar)}] "
                    f"[rec: {plvae.rec_loss:.4f}] [perf: {plvae.perf_loss:.4f}] "
                    f"[vol: {plvae.vol_loss:.4f}] [nmse_rec: {plvae.nmse_rec:.4f}] "
                    f"[nmse_perf: {plvae.nmse_perf:.4f}] [w_vol: {plvae.w_vol:.2f}] "
                    f"[dims: {plvae.dim}]"
                )

                # Sample and visualize at regular intervals
                if batches_done % args.sample_interval == 0:
                    with th.no_grad():
                        xs = x_train.to(device)
                        _chunk = 512
                        z = th.cat([plvae.encode(xs[_i0 : _i0 + _chunk]) for _i0 in range(0, len(xs), _chunk)], dim=0)
                        z_std, idx = th.sort(z.std(0), descending=True)
                        z_mean = z.mean(0)
                        n_active = (z_std > 0).sum().item()

                        # Farthest-point sampling for viz indices spread across objectives
                        def _fps(vals: np.ndarray, k: int) -> np.ndarray:
                            sel = [np.argmax(np.linalg.norm(vals - vals.mean(0), axis=1))]
                            for _ in range(k - 1):
                                d = np.min([np.linalg.norm(vals - vals[s], axis=1) for s in sel], axis=0)
                                sel.append(int(np.argmax(d)))
                            return np.array(sel)

                        n_tr_viz = min(10, len(x_train))
                        n_va_viz = min(8, len(x_val))
                        tr_viz_idx = _fps(p_train_scaled.numpy(), n_tr_viz)
                        va_viz_idx = _fps(p_val_scaled.numpy(), n_va_viz)

                        # Condition embeddings (batched to avoid OOM on large datasets)
                        viz_cond_emb = None
                        if args.conditional_decoder or cond_dim_for_predictor > 0:
                            c_all = c_train_scaled.to(device)
                            ic_all = ic_train.to(device) if ic_train is not None else None
                            viz_cond_emb = th.cat(
                                [
                                    plvae.build_cond_embedding(
                                        c_all[_i0 : _i0 + _chunk],
                                        ic_all[_i0 : _i0 + _chunk] if ic_all is not None else None,
                                    )
                                    for _i0 in range(0, len(c_all), _chunk)
                                ],
                                dim=0,
                            )

                        def _viz_decode(z_in, cond=None):
                            if args.conditional_decoder and cond is not None:
                                return plvae.decoder(z_in, cond=cond).clamp(0, 1).cpu().numpy()
                            return plvae.decode(z_in).clamp(0, 1).cpu().numpy()

                        # Interpolated designs between performance-spread pairs
                        z_start, z_end = z[tr_viz_idx], z[np.roll(tr_viz_idx, -1)]
                        c_start = viz_cond_emb[tr_viz_idx] if viz_cond_emb is not None else None
                        c_end = viz_cond_emb[np.roll(tr_viz_idx, -1)] if viz_cond_emb is not None else None
                        x_ints = []
                        for alpha in [0, 0.25, 0.5, 0.75, 1]:
                            z_ = (1 - alpha) * z_start + alpha * z_end
                            c_ = (1 - alpha) * c_start + alpha * c_end if c_start is not None else None
                            x_ints.append(_viz_decode(z_, c_))

                        # Random designs from latent Gaussian
                        z_rand = z_mean.unsqueeze(0).repeat([n_tr_viz, 1])
                        z_rand[:, idx[:n_active]] += z_std[:n_active] * th.randn_like(z_rand[:, idx[:n_active]])
                        x_rand = _viz_decode(z_rand, viz_cond_emb[tr_viz_idx] if viz_cond_emb is not None else None)

                        # Performance predictions
                        pz = z[:, :perf_dim]
                        pred_in = th.cat([pz, viz_cond_emb], dim=-1) if viz_cond_emb is not None else pz
                        p_pred_scaled = plvae.predictor(pred_in)

                        p_actual = p_scaler.inverse_transform(p_train_scaled.cpu().numpy())
                        p_predicted = p_scaler.inverse_transform(p_pred_scaled.cpu().numpy())
                        z_std_cpu = z_std.cpu().numpy()
                        xs_cpu = xs.cpu().numpy()

                    # --- Plots ---

                    # 1: Latent dimension std bars
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
                    ax1.bar(np.arange(len(z_std_cpu)), z_std_cpu)
                    ax1.set_yscale("log")
                    ax1.set_title(f"Number of principal components = {n_active}")
                    ax2.bar(np.arange(n_active), z_std_cpu[:n_active])
                    ax2.set_yscale("log")
                    ax2.set_xlabel("Latent dimension index")
                    for a in (ax1, ax2):
                        a.set_ylabel("Std dev")
                    fig.tight_layout()
                    plt.savefig(f"images/dim_{batches_done}.png")
                    plt.close()

                    # 2: Interpolated designs (GT_start | alphas | GT_end)
                    fig, axs = plt.subplots(n_tr_viz, 7, figsize=(14, 2 * n_tr_viz))
                    xs_start, xs_end = xs_cpu[tr_viz_idx], xs_cpu[np.roll(tr_viz_idx, -1)]
                    for r in range(n_tr_viz):
                        axs[r, 0].imshow(xs_start[r].reshape(design_shape))
                        for j in range(5):
                            axs[r, j + 1].imshow(x_ints[j][r].reshape(design_shape))
                        axs[r, 6].imshow(xs_end[r].reshape(design_shape))
                        for c in range(7):
                            axs[r, c].axis("off")
                    axs[0, 0].set_title("GT start")
                    for ax, a in zip(axs[0, 1:6], [0, 0.25, 0.5, 0.75, 1]):
                        ax.set_title(rf"$\alpha$={a}")
                    axs[0, 6].set_title("GT end")
                    fig.tight_layout()
                    plt.savefig(f"images/interp_{batches_done}.png")
                    plt.close()

                    # 3: Random designs from latent Gaussian
                    nc = 5
                    nr = max(1, (n_tr_viz + nc - 1) // nc)
                    fig, axs = plt.subplots(nr, nc, figsize=(3 * nc, 3 * nr), squeeze=False)
                    for k in range(nr * nc):
                        ax = axs[k // nc, k % nc]
                        if k < n_tr_viz:
                            ax.imshow(x_rand[k].reshape(design_shape))
                        ax.axis("off")
                    fig.suptitle("Gaussian random designs from latent space")
                    fig.tight_layout()
                    plt.savefig(f"images/norm_{batches_done}.png")
                    plt.close()

                    # 4: Predicted vs actual performance (per objective)
                    fig, axs = plt.subplots(1, n_perf, figsize=(7 * n_perf, 7), squeeze=False)
                    for oi in range(n_perf):
                        ax = axs[0, oi]
                        pa, pp = p_actual[:, oi], p_predicted[:, oi]
                        ax.scatter(pa, pp, alpha=0.5, s=20)
                        lim = [min(pa.min(), pp.min()), max(pa.max(), pp.max())]
                        ax.plot(lim, lim, "r--", lw=2, label="1:1")
                        ax.set(xlabel="Actual", ylabel="Predicted", aspect="equal")
                        ax.set_title(f"{obj_keys[oi]}  MSE: {np.mean((pa - pp) ** 2):.4e}")
                        ax.grid(alpha=0.3)
                        ax.legend()
                    fig.tight_layout()
                    plt.savefig(f"images/perf_pred_vs_actual_{batches_done}.png")
                    plt.close()

                    # 5: Top-2 latent dims colored by performance (per objective)
                    if n_active >= MIN_DIMS_FOR_SCATTER:
                        with th.no_grad():
                            z_val_np = plvae.encode(x_val[:].to(device)).cpu().numpy()
                        p_val_actual = p_scaler.inverse_transform(p_val_scaled.numpy())
                        z_tr_np = z.cpu().numpy()
                        d0, d1 = idx[:2].cpu().numpy()

                        fig, axs = plt.subplots(1, n_perf, figsize=(7 * n_perf, 6), squeeze=False)
                        for oi in range(n_perf):
                            ax = axs[0, oi]
                            # Quantile-based colormap normalization for heavy-tailed objectives
                            all_perf = np.concatenate([p_actual[:, oi], p_val_actual[:, oi]])
                            boundaries = np.percentile(all_perf, np.linspace(0, 100, 256))
                            boundaries = np.unique(boundaries)  # remove duplicates from ties
                            norm = mcolors.BoundaryNorm(boundaries, ncolors=256)
                            sc = ax.scatter(
                                z_tr_np[:, d0],
                                z_tr_np[:, d1],
                                c=p_actual[:, oi],
                                s=12,
                                alpha=0.5,
                                cmap="viridis",
                                norm=norm,
                            )
                            ax.scatter(
                                z_val_np[:, d0],
                                z_val_np[:, d1],
                                c=p_val_actual[:, oi],
                                s=40,
                                alpha=0.8,
                                marker="x",
                                cmap="viridis",
                                norm=norm,
                            )
                            if oi == 0:
                                for j in range(n_tr_viz):
                                    ax.annotate(
                                        str(j),
                                        (z_tr_np[tr_viz_idx[j], d0], z_tr_np[tr_viz_idx[j], d1]),
                                        fontsize=7,
                                        alpha=0.7,
                                        color="k",
                                    )
                                for j in range(n_va_viz):
                                    ax.annotate(
                                        f"V{j}",
                                        (z_val_np[va_viz_idx[j], d0], z_val_np[va_viz_idx[j], d1]),
                                        fontsize=7,
                                        fontweight="bold",
                                        color="red",
                                    )
                            ax.set(xlabel=f"z[{d0}]", ylabel=f"z[{d1}]")
                            ax.set_title(obj_keys[oi])
                            fig.colorbar(sc, ax=ax)
                        fig.suptitle(f"Top-2 active dims ({n_active} active) — circles=train, x=val")
                        fig.tight_layout()
                        plt.savefig(f"images/latent_perf_{batches_done}.png")
                        plt.close()
                    else:
                        fig, ax = plt.subplots(figsize=(4, 2))
                        ax.text(0.5, 0.5, f"{n_active} active dim(s)", ha="center", va="center")
                        ax.set_axis_off()
                        plt.savefig(f"images/latent_perf_{batches_done}.png")
                        plt.close()

                    # 6: Validation reconstruction (spread across performance)
                    with th.no_grad():
                        x_viz = x_val[va_viz_idx].to(device)
                        z_viz = plvae.encode(x_viz)
                        viz_val_cond = None
                        if args.conditional_decoder or cond_dim_for_predictor > 0:
                            c_viz = c_val_scaled[va_viz_idx].to(device)
                            ic_viz = ic_val[va_viz_idx].to(device) if ic_val is not None else None
                            viz_val_cond = plvae.build_cond_embedding(c_viz, ic_viz)
                        if args.conditional_decoder and viz_val_cond is not None:
                            x_rec_viz = plvae.decoder(z_viz, cond=viz_val_cond).clamp(0, 1).cpu().numpy()
                        else:
                            x_rec_viz = plvae.decode(z_viz).clamp(0, 1).cpu().numpy()

                    fig, axs = plt.subplots(n_va_viz, 2, figsize=(4, 2 * n_va_viz))
                    for row in range(n_va_viz):
                        axs[row, 0].imshow(x_val[va_viz_idx[row]].numpy().reshape(design_shape))
                        axs[row, 0].axis("off")
                        axs[row, 1].imshow(x_rec_viz[row].reshape(design_shape))
                        axs[row, 1].axis("off")
                    axs[0, 0].set_title("Original")
                    axs[0, 1].set_title("Reconstructed")
                    fig.tight_layout()
                    plt.savefig(f"images/val_recon_{batches_done}.png")
                    plt.close()

                    # Log all plots to wandb
                    wandb.log(
                        {
                            "dim_plot": wandb.Image(f"images/dim_{batches_done}.png"),
                            "interp_plot": wandb.Image(f"images/interp_{batches_done}.png"),
                            "norm_plot": wandb.Image(f"images/norm_{batches_done}.png"),
                            "perf_pred_vs_actual": wandb.Image(f"images/perf_pred_vs_actual_{batches_done}.png"),
                            "latent_perf": wandb.Image(f"images/latent_perf_{batches_done}.png"),
                            "val_reconstruction": wandb.Image(f"images/val_recon_{batches_done}.png"),
                        }
                    )

        # Trigger pruning check BEFORE validation so _vol_active reflects
        # training-time constraint satisfaction, not validation batch state
        plvae.epoch_report(epoch=epoch, callbacks=[], batch=None, loss=loss, pbar=None)

        # ---- Validation ----
        with th.no_grad():
            plvae.eval()
            val_rec = val_perf = val_vol = 0.0
            val_nmse_rec = val_nmse_perf = 0.0
            n = 0
            for batch_v in val_loader:
                x_v = batch_v[0].to(device)
                c_v = batch_v[1].to(device)
                p_v = batch_v[2].to(device)
                ic_v = batch_v[3].to(device) if len(batch_v) > BATCH_WITH_IMAGE_CONDITIONS else None
                val_batch = (x_v, c_v, p_v, ic_v) if ic_v is not None else (x_v, c_v, p_v)
                _ = plvae.loss(val_batch)  # Computes and stores metrics
                bsz = x_v.size(0)
                val_rec += plvae.rec_loss * bsz
                val_perf += plvae.perf_loss * bsz
                val_vol += plvae.vol_loss * bsz
                val_nmse_rec += plvae.nmse_rec * bsz
                val_nmse_perf += plvae.nmse_perf * bsz
                n += bsz
            val_rec /= n
            val_perf /= n
            val_vol /= n
            val_nmse_rec /= n
            val_nmse_perf /= n
            plvae.train()

        if args.track:
            val_log_dict = {
                "epoch": epoch,
                "val_rec": val_rec,
                "val_perf": val_perf,
                "val_vol_loss": val_vol,
                "val_nmse_rec": val_nmse_rec,
                "val_nmse_perf": val_nmse_perf,
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
                "pruning_mask": plvae.pruning_mask.cpu(),
                "pruning_frozen_z": plvae.frozen_z.cpu(),
                "args": vars(args),
            }
            if plvae.condition_encoder is not None:
                ckpt_plvae["condition_encoder"] = plvae.condition_encoder.state_dict()
            th.save(ckpt_plvae, "constrained_plvae.pth")
            # The reconstruction and performance thresholds are part of the run
            # config, so distinct thresholds already produce distinct
            # `config_fingerprint`s. That replaces the old W&B alias scheme
            # (`seed_1_rec0.001_perf0.01`), which encoded hyperparameters in a
            # string only the training script knew how to rebuild.
            save_checkpoint_package(
                checkpoint_backend="hf",
                hf_entity=args.hf_entity,
                hf_repo_prefix=args.hf_repo_prefix,
                hf_private=False,
                problem_id=args.problem_id,
                algo=args.algo,
                seed=args.seed,
                checkpoint_files={"constrained_plvae.pth": "constrained_plvae.pth"},
                run_config=vars(args),
                **checkpoint_identity(args),
                condition_keys=scalar_cond_keys,
                primary_files=["constrained_plvae.pth"],
            )

    if args.track:
        wandb.finish()
