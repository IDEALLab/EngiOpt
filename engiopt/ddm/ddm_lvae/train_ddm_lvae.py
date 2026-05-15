"""
Training script for DDM_LVAE_3D.

Phase 2 training: the LVAE must already be trained and its checkpoint provided.
The DDM UNet is trained from scratch (or from a DDM checkpoint) while the LVAE
encoder + decoder are frozen and used only to compute a pressure supervision
signal on each training step.

Usage
-----
    python -m engiopt.ddm.train_ddm_lvae \
        --lvae_checkpoint path/to/lvae.pth \
        [--ddm_checkpoint  path/to/ddm.pth] \
        [--n_samples 1000] [--seed 0] [--w_pressure 1.0]

Nothing in the original ddm.py, unets.py, or samplers.py is modified.
"""

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from engiopt.bezier_ae.bezier_ae import BezierAutoencoder
from engiopt.ddm.ddm_lvae.ddm_lvae import DDM_LVAE_3D
from engiopt.ddm.unets import Unet_AoAInit3D
from engiopt.ddm import samplers
from engiopt.data_processing.utils import scaler
from engiopt.lvae.lvae import LAE_AoAInit
from engiopt.lvae.unets import LAEEncoder, LAEDecoder
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # BAE
    bae_checkpoint: str = "bezier_ae_best.pt"
    n_control_points: int = 32
    n_data_points: int = 192
    bae_batch_size: int = 32

    # DDM / UNet
    w_dim: int = 15             # spanwise slices in the new dataset
    latent_channels: int = 3
    latent_length: int = 30
    c_dim: int = 4              # DDM conditions on flow params only [mach, reynolds, cl_target, area_ratio]
    batch_size: int = 32
    lr: float = 1e-3
    n_epochs: int = 20000
    num_diffusion_steps: int = 1000
    cosine_schedule: bool = False
    grad_clip: float = None         # max grad norm; None = no clipping
    w_latent: float = 0.0           # weight for soft latent validity penalty
    unet_channels: tuple = (32, 64, 128, 256)  # down/up channel widths; middle = channels[-1]//2
    # w_latent annealing: sigmoid ramp from 0 to w_latent over [anneal_start, anneal_end] epochs
    # Both 0 means no annealing (static w_latent throughout)
    w_latent_anneal_start: int = 0   # epoch at which ramp begins
    w_latent_anneal_end: int = 0     # epoch at which ramp reaches w_latent

    # LVAE
    lvae_checkpoint: str = ""       # required; set via --lvae_checkpoint
    lvae_params_dim: int = 38       # LVAE expects [4 flow + 34 geo] params
    w_pressure: float = 1.0         # weight for the pressure loss term

    # Smoothness regularizer (Option B from original DDM)
    w_smooth: float = 0.0           # weight for t-weighted BAE smoothness loss
    smooth_reg_exp: float = 4.0     # decay rate: penalty ~ exp(-k * t/T)

    # LVAE architecture (must match how the LVAE was trained)
    lvae_w_dim: int = 15
    lvae_lae_latent_dim: int = 64
    lvae_c_dim: int = 38

    # Outputs
    save_dir: str = "results/ddm_lvae"
    model_name: str = "ddm_lvae_v5"

    # Misc
    seed: int = 0
    num_workers: int = 0
    save_every: int = 100


# ---------------------------------------------------------------------------
# Dataset — same as train_ddm.py but carries pressure as a 5th field
# ---------------------------------------------------------------------------

class PrecomputedWingsDatasetWithPressure(Dataset):
    """Pre-encoded BAE latents + pressure coefficients.

    Batch format: (z_opt, aoa, params, z_init, pressure)
    """

    def __init__(self, z_opts, aoas, params, z_inits, pressures,
                 scaler_params=None, scaler_aoas=None, scaler_pressures=None):
        self.z_opts           = z_opts      # [N, 9, 3, L]
        self.aoas             = aoas        # [N, 1]
        self.params           = params      # [N, 4]
        self.z_inits          = z_inits     # [N, 3, L]
        self.pressures        = pressures   # [N, 9, 192]
        self.scaler_params    = scaler_params
        self.scaler_aoas      = scaler_aoas
        self.scaler_pressures = scaler_pressures

    def __len__(self):
        return len(self.z_opts)

    def __getitem__(self, idx):
        z_opt    = self.z_opts[idx].clone()
        aoa      = self.aoas[idx].clone()
        params   = self.params[idx].clone()
        z_init   = self.z_inits[idx].clone()
        pressure = self.pressures[idx].clone()

        if self.scaler_params is not None:
            params = self.scaler_params.transform(params)
        if self.scaler_aoas is not None:
            aoa = self.scaler_aoas.transform(aoa)
        if self.scaler_pressures is not None:
            pressure = self.scaler_pressures.transform(
                pressure.reshape(1, -1)
            ).reshape(pressure.shape)

        return z_opt, aoa, params, z_init, pressure


# ---------------------------------------------------------------------------
# Pre-computation helpers
# ---------------------------------------------------------------------------

def precompute_latents_and_pressure(base_dataset, initial_by_case, bae_model, device):
    """Encode geometry via BAE and collect raw pressure from the dataset.

    Uses the same BAE encoding convention as train_lvae.py (z_ae_mode=True,
    te_shifts for coordinate centering) so that the DDM latent space matches
    the LVAE's expected input space.

    Returns
    -------
    z_opts    : [N, w_dim, 3, L]
    aoas      : [N, 1]
    params    : [N, 4]   — flow params only [mach, reynolds, cl_target, area_ratio]
    z_inits   : [N, 3, L]
    pressures : [N, w_dim, 192]
    """
    print(f"Pre-computing BAE latents for {len(base_dataset)} samples...")

    z_opts_list    = []
    z_inits_list   = []
    aoas_list      = []
    params_list    = []
    pressures_list = []

    bae_model.eval()
    with torch.no_grad():
        for i, item in enumerate(base_dataset):
            if (i + 1) % 100 == 0:
                print(f"  Encoding sample {i + 1}/{len(base_dataset)}...")

            coords    = torch.tensor(item["coords"],    dtype=torch.float32)  # [w, 192, 2]
            te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)  # [w]

            # Centre each slice: remove TE y-shift, fix TE x to 1.0
            # (matches train_lvae.py convention, which is what the LVAE was trained on)
            coords_unshifted = coords.clone()
            coords_unshifted[:, :, 1] -= te_shifts.unsqueeze(1)
            te_x = coords_unshifted[:, 0, 0]
            coords_unshifted[:, :, 0] += (1.0 - te_x).unsqueeze(1)

            # z_init: use root slice of initial (unoptimised) wing if available,
            # otherwise fall back to the root slice of the current item.
            case_num = int(item["case_num"])
            if case_num in initial_by_case:
                init_coords    = torch.tensor(
                    initial_by_case[case_num]["coords"],    dtype=torch.float32
                )
                init_te_shifts = torch.tensor(
                    initial_by_case[case_num]["te_shifts"], dtype=torch.float32
                )
                init_root = init_coords[0].clone()
                init_root[:, 1] -= init_te_shifts[0]
                init_root[:, 0] += (1.0 - init_root[0, 0])
                x_init = init_root.permute(1, 0).unsqueeze(0).to(device)  # [1,2,192]
            else:
                x_init = coords_unshifted[0].permute(1, 0).unsqueeze(0).to(device)

            z_init = bae_model.encode(
                x_init, return_z=True, z_ae_mode=True
            ).squeeze(0).cpu()   # [3, L]

            # z_opt: encode all w_dim centred slices
            z_slices = []
            for s in range(coords_unshifted.shape[0]):
                x_s = coords_unshifted[s].permute(1, 0).unsqueeze(0).to(device)
                z_s = bae_model.encode(
                    x_s, return_z=True, z_ae_mode=True
                ).squeeze(0).cpu()   # [3, L]
                z_slices.append(z_s)
            z_opt = torch.stack(z_slices, dim=0)   # [w, 3, L]

            z_inits_list.append(z_init)
            z_opts_list.append(z_opt)

            aoa = torch.tensor(item["alpha"], dtype=torch.float32)
            if aoa.ndim == 0:
                aoa = aoa.unsqueeze(0)
            aoas_list.append(aoa)

            # 4-dim flow condition vector only — geo params are outputs, not inputs
            flow_params = [item["mach"], item["reynolds"],
                           item["cl_target"], item["area_case_ratio"]]
            params_list.append(torch.tensor(flow_params, dtype=torch.float32))

            pressure = torch.tensor(
                np.array(item["coef_pressure"]), dtype=torch.float32
            )  # [w, 192]
            pressures_list.append(pressure)

    print("Pre-computation complete.")
    return (
        torch.stack(z_opts_list),     # [N, w_dim, 3, L]
        torch.stack(aoas_list),       # [N, 1]
        torch.stack(params_list),     # [N, 4]
        torch.stack(z_inits_list),    # [N, 3, L]
        torch.stack(pressures_list),  # [N, w_dim, 192]
    )


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_bae(cfg: Config) -> BezierAutoencoder:
    model = BezierAutoencoder(
        n_control_points=cfg.n_control_points,
        n_data_points=cfg.n_data_points,
        batch_size=cfg.bae_batch_size,
        auto_batch=True,
    ).to(cfg.device)
    ckpt = torch.load(cfg.bae_checkpoint, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_lvae(cfg: Config, bae_model) -> LAE_AoAInit:
    """Load a trained LVAE from checkpoint.

    LAE_AoAInit.load() replaces self.encoder and self.decoder wholesale from the
    checkpoint, so the shell objects below just need to be structurally valid.
    Their weights are immediately overwritten by load().
    """
    encoder = LAEEncoder(
        w_dim=cfg.lvae_w_dim,
        latent_channels=3,
        latent_length=30,
        lae_latent_dim=cfg.lvae_lae_latent_dim,
        c_dim=cfg.lvae_c_dim,
        c_dim_latent=16,
        down_channels=[64, 128, 256, 512],
        middle_channel=256,
        dropout=0.0,
    )
    decoder = LAEDecoder(
        w_dim=cfg.lvae_w_dim,
        latent_channels=3,
        latent_length=30,
        pressure_length=192,
        perf_dim=2,
        lae_latent_dim=cfg.lvae_lae_latent_dim,
        c_dim=cfg.lvae_c_dim,
        c_dim_latent=16,
        up_channels=[512, 256, 128, 64],
        base_length=4,
        use_pressure=True,
        dropout=0.0,
    )

    lvae = LAE_AoAInit(
        encoder=encoder,
        decoder=decoder,
        sampler=None,
        bae_model=bae_model,
        lae_latent_dim=cfg.lvae_lae_latent_dim,
    )
    lvae.load(cfg.lvae_checkpoint, train_mode=False)
    lvae.encoder.eval()
    lvae.decoder.eval()
    for p in lvae.encoder.parameters():
        p.requires_grad_(False)
    for p in lvae.decoder.parameters():
        p.requires_grad_(False)
    print(f"Loaded and froze LVAE from {cfg.lvae_checkpoint}.")
    return lvae


def build_unet(cfg: Config) -> Unet_AoAInit3D:
    ch = list(cfg.unet_channels)
    return Unet_AoAInit3D(
        w_dim=cfg.w_dim,
        x_latent_channels_2D=cfg.latent_channels,
        N_dim=cfg.latent_length,
        tform_dim=3,
        c_dim=cfg.c_dim,
        down_channels=ch,
        middle_channel=ch[-1] // 2,
        up_channels=list(reversed(ch)),
        upsampling_factor=2,
        block_norms=[True, True, True],
        droput=False,
    ).to(cfg.device)


def build_sampler(cfg: Config):
    return samplers.BaselineSampler_AoA_3D(
        cfg.num_diffusion_steps,
        start_x=1e-4,    end_x=0.02,
        start_alpha=1e-4, end_alpha=0.02,
        cosine=cfg.cosine_schedule,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model: DDM_LVAE_3D, loader: DataLoader,
                    device: str, epoch: int):
    """Returns (total, loss_x, loss_aoa, loss_reg, loss_pressure, loss_latent, grad_norm) — all epoch-averaged."""
    model.unet.train()
    totals = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    total_gnorm = 0.0
    n_steps = 0
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        batch_noised = model._noise_data(batch, device)
        model.optimizer.zero_grad()
        loss, lx, laoa, lreg, lp, llat = model.loss(batch_noised, return_components=True)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            model.unet.parameters(),
            model.grad_clip if (hasattr(model, 'grad_clip') and model.grad_clip is not None) else float('inf'),
        )
        model.optimizer.step()
        model.stats['current_loss'] = loss.item()
        totals[0] += loss.item()
        totals[1] += lx.item()
        totals[2] += laoa.item()
        totals[3] += lreg.item()
        totals[4] += lp.item()
        totals[5] += llat.item()
        total_gnorm += gnorm.item()
        n_steps += 1
    n = max(n_steps, 1)
    return tuple(t / n for t in totals) + (total_gnorm / n,)


@torch.no_grad()
def eval_one_epoch(model: DDM_LVAE_3D, loader: DataLoader, device: str):
    """Returns (total, loss_x, loss_aoa, loss_reg, loss_pressure, loss_latent) — all epoch-averaged."""
    model.unet.eval()
    totals = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    n_steps = 0
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        batch_noised = model._noise_data(batch, device)
        loss, lx, laoa, lreg, lp, llat = model.loss(batch_noised, return_components=True)
        if not torch.isfinite(loss):
            continue
        totals[0] += loss.item()
        totals[1] += lx.item()
        totals[2] += laoa.item()
        totals[3] += lreg.item()
        totals[4] += lp.item()
        totals[5] += llat.item()
        n_steps += 1
    model.unet.train()
    n = max(n_steps, 1)
    return tuple(t / n for t in totals)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train DDM_LVAE_3D")
    parser.add_argument("--lvae_checkpoint", type=str, required=True,
                        help="Path to a trained LVAE .pth checkpoint")
    parser.add_argument("--ddm_checkpoint",  type=str, default=None,
                        help="Optional: resume from an existing DDM_LVAE checkpoint")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Limit number of training samples (ablation)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w_pressure", type=float, default=1.0,
                        help="Weight for the pressure loss term")
    parser.add_argument("--lvae_params_dim", type=int, default=38,
                        help="Condition-vector width expected by the LVAE (default 38 = 4 flow + 34 geo)")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Override the default model name (and save path)")
    parser.add_argument("--cosine_schedule", action="store_true",
                        help="Use cosine noise schedule instead of linear.")
    parser.add_argument("--grad_clip", type=float, default=None,
                        help="Max gradient norm for clip_grad_norm_ (None = disabled).")
    parser.add_argument("--w_latent", type=float, default=0.0,
                        help="Weight for soft latent validity penalty (0 = disabled).")
    parser.add_argument("--w_latent_anneal_start", type=int, default=0,
                        help="Epoch at which w_latent sigmoid ramp begins (default: 0 = no annealing).")
    parser.add_argument("--w_latent_anneal_end", type=int, default=0,
                        help="Epoch at which w_latent reaches its target value (default: 0 = no annealing).")
    parser.add_argument("--w_smooth", type=float, default=0.0,
                        help="Weight for t-weighted BAE smoothness regularizer (0 = disabled).")
    parser.add_argument("--smooth_reg_exp", type=float, default=4.0,
                        help="Decay exponent for smoothness t-weighting: exp(-k*t/T). "
                             "Higher = penalty concentrates more at low-noise steps.")
    parser.add_argument("--unet_channels", type=int, nargs="+", default=None,
                        help="UNet down-channel widths, e.g. --unet_channels 64 64 128 256. "
                             "up-channels and middle are derived automatically. "
                             "Default: 32 64 128 256.")
    parser.add_argument("--n_epochs", type=int, default=None,
                        help="Number of training epochs (default: 20000).")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="engiopt-lvae-ddm",
                        help="W&B project name.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    cfg.seed              = args.seed
    cfg.lvae_checkpoint   = args.lvae_checkpoint
    cfg.w_pressure        = args.w_pressure
    cfg.lvae_params_dim   = args.lvae_params_dim
    cfg.cosine_schedule   = args.cosine_schedule
    cfg.grad_clip         = args.grad_clip
    cfg.w_latent          = args.w_latent
    cfg.w_latent_anneal_start = args.w_latent_anneal_start
    cfg.w_latent_anneal_end   = args.w_latent_anneal_end
    cfg.w_smooth              = args.w_smooth
    cfg.smooth_reg_exp        = args.smooth_reg_exp
    if args.unet_channels is not None:
        cfg.unet_channels = tuple(args.unet_channels)
    if args.n_epochs is not None:
        cfg.n_epochs = args.n_epochs

    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.n_samples is not None:
        cfg.model_name = f"ddm_lvae_n{args.n_samples}_s{cfg.seed}"
        cfg.save_dir   = f"results/ddm_lvae/n{args.n_samples}_s{cfg.seed}"

    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print(f"Device       : {cfg.device}")
    print(f"Model name   : {cfg.model_name}")
    print(f"Save dir     : {cfg.save_dir}")
    print(f"w_pressure   : {cfg.w_pressure}")
    print(f"grad_clip    : {cfg.grad_clip}")
    print(f"w_latent     : {cfg.w_latent}")
    print(f"w_smooth     : {cfg.w_smooth}  (reg_exp={cfg.smooth_reg_exp})")
    print(f"LVAE ckpt    : {cfg.lvae_checkpoint}")

    # 1. Load frozen BAE
    bae_model = load_bae(cfg)
    print("Loaded BAE.")

    # 2. Load frozen LVAE
    lvae_model = load_lvae(cfg, bae_model)

    # 3. Load dataset
    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=cfg.seed)
    all_train    = list(new_dataset["train"])
    initial_by_case = {item["case_num"]: item for item in all_train if item["initial"] == 1}
    base_dataset    = [item for item in all_train if item["final"] == 1]

    all_val          = list(new_dataset["val"])
    val_initial_by_case = {item["case_num"]: item for item in all_val if item["initial"] == 1}
    val_dataset         = [item for item in all_val if item["final"] == 1]

    if args.n_samples is not None:
        rng     = np.random.default_rng(cfg.seed)
        indices = rng.choice(len(base_dataset), size=args.n_samples, replace=False)
        base_dataset = [base_dataset[i] for i in sorted(indices)]
        print(f"Ablation: using {args.n_samples} samples (seed={cfg.seed})")

    print(f"Train split: {len(base_dataset)} final items, "
          f"{len(initial_by_case)} initial cases.")
    print(f"Val split  : {len(val_dataset)} final items.")

    # 4. Normalisation stats — 4 flow params only
    all_params = np.array([
        [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
        for item in base_dataset
    ])
    all_aoas = np.array([float(item["alpha"]) for item in base_dataset])

    params_mean_std = (all_params.mean(axis=0), all_params.std(axis=0))
    aoas_mean_std   = (float(all_aoas.mean()), float(all_aoas.std()))
    scaler_params   = scaler(params_mean_std)
    scaler_aoas     = scaler(aoas_mean_std)

    # 5. Pre-compute BAE latents + collect pressure
    z_opts, aoas, params_all, z_inits, pressures = precompute_latents_and_pressure(
        base_dataset, initial_by_case, bae_model, cfg.device
    )

    # Normalise BAE latents per-channel (same convention as train_ddm.py)
    # z_opts: [N, w_dim, 3, L] → mean/std: [1, 1, 3, 1]
    z_mean = z_opts.mean(dim=(0, 1, 3), keepdim=True)
    z_std  = z_opts.std(dim=(0, 1, 3), keepdim=True)
    z_opts  = (z_opts  - z_mean) / z_std
    z_inits = (z_inits - z_mean[0]) / z_std[0]

    # Normalise pressure with a global scalar (matches the LVAE's scaler_pressures
    # which uses a single global mean/std over all slices and points).
    p_mean_val = float(pressures.mean())
    p_std_val  = float(pressures.std())
    pressures  = (pressures - p_mean_val) / max(p_std_val, 1e-8)

    print(f"z_opts    : [{z_opts.min():.3f}, {z_opts.max():.3f}], "
          f"mean={z_opts.mean():.3f}, std={z_opts.std():.3f}")
    print(f"pressures : [{pressures.min():.3f}, {pressures.max():.3f}], "
          f"mean={pressures.mean():.3f}, std={pressures.std():.3f}")

    # Pressure is already normalised above — no scaler_pressures needed in dataset.
    dataset = PrecomputedWingsDatasetWithPressure(
        z_opts, aoas, params_all, z_inits, pressures,
        scaler_params=scaler_params,
        scaler_aoas=scaler_aoas,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size,
                        shuffle=True, num_workers=cfg.num_workers)

    # Val dataset — precompute using train norm stats (no data leakage).
    val_z_opts, val_aoas, val_params, val_z_inits, val_pressures = \
        precompute_latents_and_pressure(val_dataset, val_initial_by_case, bae_model, cfg.device)
    val_z_opts   = (val_z_opts   - z_mean) / z_std
    val_z_inits  = (val_z_inits  - z_mean[0]) / z_std[0]
    val_pressures = (val_pressures - p_mean_val) / max(p_std_val, 1e-8)
    val_ds = PrecomputedWingsDatasetWithPressure(
        val_z_opts, val_aoas, val_params, val_z_inits, val_pressures,
        scaler_params=scaler_params,
        scaler_aoas=scaler_aoas,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers)

    # 6. Build UNet + sampler
    unet    = build_unet(cfg)
    sampler = build_sampler(cfg)
    print("Built UNet and sampler.")

    # 7. Build DDM_LVAE_3D
    ddm_model = DDM_LVAE_3D(
        unet=unet,
        sampler=sampler,
        bae_model=bae_model,
        lvae_model=lvae_model,
        w_pressure=cfg.w_pressure,
        lvae_params_dim=cfg.lvae_params_dim,
        z_ddm_mean=z_mean,
        z_ddm_std=z_std,
        p_ddm_mean=p_mean_val,
        p_ddm_std=p_std_val,
        params_mean_std=params_mean_std,
        aoas_mean_std=aoas_mean_std,
        name=cfg.model_name,
        opt_lr=cfg.lr,
        opt_betas=(0.9, 0.99),
        weights=(1.0, 1.0),
        apply_reg=False,
        reg_factor=10.0,
        reg_exp=20,
        latent_mean=z_mean,
        latent_std=z_std,
        w_latent=cfg.w_latent,
        w_smooth=cfg.w_smooth,
        smooth_reg_exp=cfg.smooth_reg_exp,
    )

    ddm_model.grad_clip = cfg.grad_clip
    if cfg.w_latent > 0:
        ddm_model.set_latent_valid_range(z_mean, z_std)

    if args.ddm_checkpoint is not None:
        ddm_model.load(args.ddm_checkpoint, train_mode=True)
        print(f"Resumed from DDM checkpoint: {args.ddm_checkpoint}")

    # LR scheduler
    ddm_model.lr_scheduler = 'ReduceLROnPlateau'
    ddm_model.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        ddm_model.optimizer, mode='min', factor=0.5, patience=600,
    )
    print("Built DDM_LVAE_3D model.")

    # W&B init
    use_wandb = args.wandb and _WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity="adelbeke-",
            name=cfg.model_name,
            config={
                "w_pressure":        cfg.w_pressure,
                "lvae_checkpoint":   cfg.lvae_checkpoint,
                "lvae_params_dim":   cfg.lvae_params_dim,
                "weights":           (1.0, 1.0),
                "n_epochs":          cfg.n_epochs,
                "batch_size":        cfg.batch_size,
                "num_diffusion_steps": cfg.num_diffusion_steps,
                "cosine_schedule":   cfg.cosine_schedule,
                "grad_clip":         cfg.grad_clip,
                "w_latent":               cfg.w_latent,
                "w_latent_anneal_start":  cfg.w_latent_anneal_start,
                "w_latent_anneal_end":    cfg.w_latent_anneal_end,
                "w_dim":                  cfg.w_dim,
                "unet_channels":     list(cfg.unet_channels),
                "seed":              cfg.seed,
            },
        )
    elif args.wandb and not _WANDB_AVAILABLE:
        print("Warning: --wandb passed but wandb is not installed. Skipping W&B logging.")

    # 8. Training loop
    anneal_start = cfg.w_latent_anneal_start
    anneal_end   = cfg.w_latent_anneal_end
    w_latent_max = cfg.w_latent
    do_anneal    = (anneal_end > anneal_start) and (w_latent_max > 0)

    def get_w_latent(epoch: int) -> float:
        """Sigmoid ramp: 0 before anneal_start, w_latent_max after anneal_end."""
        if not do_anneal:
            return w_latent_max
        if epoch < anneal_start:
            return 0.0
        if epoch >= anneal_end:
            return w_latent_max
        # Map epoch into [-6, 6] for sigmoid, giving ~0.0025 at start, ~0.9975 at end
        progress = (epoch - anneal_start) / (anneal_end - anneal_start)
        return w_latent_max / (1.0 + math.exp(-12.0 * (progress - 0.5)))

    if do_anneal:
        print(f"w_latent annealing: 0 → {w_latent_max} via sigmoid ramp, "
              f"epochs {anneal_start}–{anneal_end}")

    print(f"Starting training for {cfg.n_epochs} epochs...")
    best_train_loss = float('inf')
    for epoch in range(cfg.n_epochs):
        # Update w_latent on the model before each epoch
        current_w_latent = get_w_latent(epoch)
        ddm_model.w_latent = current_w_latent
        if current_w_latent > 0 and ddm_model._latent_norm_lo is None:
            ddm_model.set_latent_valid_range(z_mean, z_std)

        train_loss, lx, laoa, lreg, lp, llat, gnorm = train_one_epoch(ddm_model, loader, cfg.device, epoch)
        val_loss,  vlx, vlaoa, vlreg, vlp, vllat = eval_one_epoch(ddm_model, val_loader, cfg.device)
        ddm_model.scheduler.step(train_loss if math.isfinite(train_loss) else best_train_loss)

        print(f"Epoch {epoch + 1:05d}/{cfg.n_epochs} | "
              f"Train {train_loss:.4f} (x {lx:.4f} aoa {laoa:.4f} p {lp:.4f} lat {llat:.4f} gnorm {gnorm:.3f}) | "
              f"Val {val_loss:.4f} (x {vlx:.4f} aoa {vlaoa:.4f} p {vlp:.4f} lat {vllat:.4f}) | "
              f"w_lat {current_w_latent:.2e}")

        if use_wandb:
            wandb.log({
                "epoch":               epoch + 1,
                "train_loss":          train_loss,
                "train_loss_x":        lx,
                "train_loss_aoa":      laoa,
                "train_loss_pressure": lp,
                "train_loss_latent":   llat,
                "grad_norm":           gnorm,
                "val_loss":            val_loss,
                "val_loss_x":          vlx,
                "val_loss_aoa":        vlaoa,
                "val_loss_pressure":   vlp,
                "val_loss_latent":     vllat,
                "w_latent":            current_w_latent,
            })

        if math.isfinite(train_loss) and train_loss < best_train_loss:
            best_train_loss = train_loss
            ddm_model.save(cfg.save_dir, suffix="_best")

        if (epoch + 1) % cfg.save_every == 0:
            ddm_model.save(cfg.save_dir)
            print(f"  --> Checkpoint saved at epoch {epoch + 1}.")

    ddm_model.save(cfg.save_dir)
    print("Training complete.")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
