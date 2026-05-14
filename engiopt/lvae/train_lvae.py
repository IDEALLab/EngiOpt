"""
Training script for the LAE — mirrors train_ddm.py.
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")  # headless backend for Slurm
import matplotlib.pyplot as plt

from engiopt.bezier_ae.bezier_ae import BezierAutoencoder
from engiopt.lvae.lvae import LAE_AoAInit
from engiopt.lvae.unets import LAEEncoder, LAEDecoder
from engiopt.lvae import samplers
from engiopt.lvae import plotting as lvae_plotting
from engiopt.data_processing.utils import scaler
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # BAE
    bae_checkpoint: str = "bezier_ae_best.pt"
    n_control_points: int = 32
    n_data_points: int = 192
    bae_batch_size: int = 32

    # Wing geometry
    w_dim: int = 15              # spanwise slices
    latent_channels: int = 3     # BAE latent channels per slice
    latent_length: int = 30      # BAE latent sequence length
    c_dim: int = 38              # condition dimension (4 flow + 34 geometric)
    pressure_length: int = 192   # Cp sample points per slice
    perf_dim: int = 2            # performance outputs: (cd_val, cl_val)

    # LAE latent
    lae_latent_dim: int = 64

    # Training
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.0
    n_epochs: int = 10000
    lambda_lv: float = 7e-5      # least-volume penalty weight
    weights: tuple = (1000, 1, 9, 1, 1)  # w_x, w_alpha, w_eta, w_pressure, w_perf

    # Latent pruning
    prune_every: int = 500       # 0 = disabled
    prune_threshold: float = 0.02

    # Fitted GMM sampler
    n_gmm_components: int = 6    # number of GMM components for the fitted sampler

    # Outputs
    save_dir: str = "results/lvae"
    model_name: str = "lae_aoa_init_3d_v1_repro"

    # Early stopping
    val_every: int = 200          # validate every N epochs
    patience: int = 10            # stop if val loss doesn't improve for this many val checks
    plot_evolution_every: int = 500  # log W&B reconstruction images every N epochs (0 = off)

    # Misc
    seed: int = 0
    num_workers: int = 0
    save_every: int = 100
    milestone_every: int = 1000  # save a separate epoch-stamped checkpoint at these intervals


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PrecomputedWingsDataset(Dataset):
    """Pre-encoded BAE latents with pressure coefficients and performance targets."""

    def __init__(
        self,
        z_opts, aoas, params, eta_ys, pressures, perfs,
        scaler_params=None, scaler_aoas=None,
        scaler_pressures=None, scaler_perfs=None,
    ):
        self.z_opts    = z_opts      # [N, 9, latent_channels, latent_length]
        self.aoas      = aoas        # [N, 1]
        self.params    = params      # [N, 4]
        self.eta_ys    = eta_ys      # [N, 9, 1]
        self.pressures = pressures   # [N, 9, 192]
        self.perfs     = perfs       # [N, 2]  (cd_val, cl_val)

        self.scaler_params    = scaler_params
        self.scaler_aoas      = scaler_aoas
        self.scaler_pressures = scaler_pressures
        self.scaler_perfs     = scaler_perfs

    def __len__(self):
        return len(self.z_opts)

    def __getitem__(self, idx):
        z_opt    = self.z_opts[idx].clone()
        aoa      = self.aoas[idx].clone()
        params   = self.params[idx].clone()
        eta_y    = self.eta_ys[idx].clone()
        pressure = self.pressures[idx].clone()
        perf     = self.perfs[idx].clone()

        if self.scaler_params is not None:
            params = self.scaler_params.transform(params)
        if self.scaler_aoas is not None:
            aoa = self.scaler_aoas.transform(aoa)
        if self.scaler_pressures is not None:
            pressure = self.scaler_pressures.transform(pressure)
        if self.scaler_perfs is not None:
            perf = self.scaler_perfs.transform(perf)

        return z_opt, aoa, params, eta_y, pressure, perf


# ---------------------------------------------------------------------------
# Pre-computation
# ---------------------------------------------------------------------------

def precompute_latents(base_dataset, bae_model, device):
    """Encode entire dataset once with the frozen BAE and collect targets."""
    print(f"Pre-computing BAE latents for {len(base_dataset)} samples...")

    z_opts_list = []
    aoas_list, params_list, eta_ys_list = [], [], []
    pressures_list, perfs_list = [], []

    bae_model.eval()
    with torch.no_grad():
        for i, item in enumerate(base_dataset):
            if (i + 1) % 100 == 0:
                print(f"  Encoding sample {i + 1}/{len(base_dataset)}...")

            coords    = torch.tensor(item["coords"],    dtype=torch.float32)  # [9,192,2]
            te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)  # [9]
            eta_y = te_shifts.unsqueeze(1)                                    # [9,1]

            coords_unshifted = coords.clone()
            coords_unshifted[:, :, 1] -= te_shifts.unsqueeze(1)
            # BAE was trained with TE at x=1.0; shift x so TE x == 1.0
            te_x = coords_unshifted[:, 0, 0]  # [n_slices]
            coords_unshifted[:, :, 0] += (1.0 - te_x).unsqueeze(1)

            z_slices = []
            for s in range(coords_unshifted.shape[0]):
                x_s = coords_unshifted[s].permute(1, 0).unsqueeze(0).to(device)
                z_s = bae_model.encode(x_s, return_z=True, z_ae_mode=True).squeeze(0).cpu()
                z_slices.append(z_s)
            z_opt = torch.stack(z_slices, dim=0)  # [9, latent_channels, latent_length]

            z_opts_list.append(z_opt)
            eta_ys_list.append(eta_y)

            aoa = torch.tensor(item["alpha"], dtype=torch.float32)
            aoa = aoa.unsqueeze(0) if aoa.ndim == 0 else aoa
            aoas_list.append(aoa)

            flow_params = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
            geo_params  = item.get("geo_params", np.zeros(34, dtype=np.float32)).tolist()
            params_list.append(torch.tensor(flow_params + geo_params, dtype=torch.float32))

            pressure = torch.tensor(
                np.array(item["coef_pressure"]), dtype=torch.float32
            )  # [9, 192]
            pressures_list.append(pressure)

            perf = torch.tensor(
                [item["cd_val"], item["cl_val"]], dtype=torch.float32
            )
            perfs_list.append(perf)

    print("Pre-computation complete.")
    return (
        torch.stack(z_opts_list),     # [N, 9, latent_channels, latent_length]
        torch.stack(aoas_list),       # [N, 1]
        torch.stack(params_list),     # [N, 4]
        torch.stack(eta_ys_list),     # [N, 9, 1]
        torch.stack(pressures_list),  # [N, 9, 192]
        torch.stack(perfs_list),      # [N, 2]
    )


# ---------------------------------------------------------------------------
# Model building helpers
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
        p.requires_grad = False
    return model


def build_encoder(cfg: Config) -> LAEEncoder:
    return LAEEncoder(
        w_dim=cfg.w_dim,
        latent_channels=cfg.latent_channels,
        latent_length=cfg.latent_length,
        lae_latent_dim=cfg.lae_latent_dim,
        c_dim=cfg.c_dim,
        c_dim_latent=16,
        down_channels=[64, 128, 256, 512],
        middle_channel=256,
        dropout=cfg.dropout,
    ).to(cfg.device)


def build_decoder(cfg: Config, use_pressure: bool = True) -> LAEDecoder:
    return LAEDecoder(
        w_dim=cfg.w_dim,
        latent_channels=cfg.latent_channels,
        latent_length=cfg.latent_length,
        pressure_length=cfg.pressure_length,
        perf_dim=cfg.perf_dim,
        lae_latent_dim=cfg.lae_latent_dim,
        c_dim=cfg.c_dim,
        c_dim_latent=16,
        up_channels=[512, 256, 128, 64],
        base_length=4,
        use_pressure=use_pressure,
        dropout=cfg.dropout,
    ).to(cfg.device)


def build_sampler(cfg: Config) -> samplers.LAESampler:
    return samplers.LAESampler(latent_dim=cfg.lae_latent_dim)


# ---------------------------------------------------------------------------
# Fitted GMM sampler
# ---------------------------------------------------------------------------

def fit_perf_regressor(
    lae_model: LAE_AoAInit,
    loader: DataLoader,
    device: str,
):
    """Encode training set and fit an MLP regressor z_masked → (cd, cl).

    Fits in de-normalised performance space so predictions are directly
    interpretable.  Stores the regressor and the active-dim mask on the
    model as ``lae_model.perf_regressor`` and
    ``lae_model.perf_regressor_mask``.
    """
    from sklearn.neural_network import MLPRegressor

    was_training = lae_model.encoder.training
    lae_model.encoder.eval()

    zs_masked, perfs_denorm = [], []
    with torch.no_grad():
        for batch in loader:
            z_opt, aoa, params, eta_y, pressure, perf = batch
            z = lae_model.encoder(
                z_opt.to(device),
                params.to(device).float(),
            )
            z_masked = lae_model._apply_mask(z)
            zs_masked.append(z_masked.cpu())
            perf_denorm = lae_model.scaler_perfs.inverse_transform(perf) if lae_model.scaler_perfs is not None else perf
            perfs_denorm.append(perf_denorm)

    if was_training:
        lae_model.encoder.train()

    Z = torch.cat(zs_masked, dim=0).numpy()          # [N, lae_latent_dim]
    P = torch.cat(perfs_denorm, dim=0).numpy()        # [N, 2]

    mask_np = lae_model.active_latent_mask.cpu().numpy()
    Z_active = Z[:, mask_np]                          # [N, n_active]

    reg = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=10000, random_state=0)
    reg.fit(Z_active, P)

    lae_model.perf_regressor      = reg
    lae_model.perf_regressor_mask = mask_np

    train_pred = reg.predict(Z_active)
    rmse_cd = float(((train_pred[:, 0] - P[:, 0]) ** 2).mean() ** 0.5)
    rmse_cl = float(((train_pred[:, 1] - P[:, 1]) ** 2).mean() ** 0.5)
    print(f"Perf regressor fitted — train RMSE: cd={rmse_cd:.4f}  cl={rmse_cl:.4f}")
    return reg, mask_np


def ensure_perf_regressor_fitted(
    lae_model: LAE_AoAInit,
    cfg,
    regressor_path: str,
    n_samples: Optional[int] = None,
) -> None:
    """Load an existing perf regressor or fit one and save it.

    Mirrors the pattern of ``ensure_gmm_fitted``.
    """
    import pickle

    if os.path.exists(regressor_path):
        print(f"Loading perf regressor from {regressor_path}...")
        with open(regressor_path, 'rb') as f:
            data = pickle.load(f)
        stored_mask  = data['mask']
        current_mask = lae_model.active_latent_mask.cpu().numpy()
        if not np.array_equal(stored_mask, current_mask):
            print("  Stored regressor mask does not match current active latent mask — refitting...")
        else:
            lae_model.perf_regressor      = data['regressor']
            lae_model.perf_regressor_mask = stored_mask
            print("Perf regressor loaded.")
            return

    print("No perf regressor found — fitting from training data...")
    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=cfg.seed)
    all_train    = list(new_dataset["train"])
    base_dataset = [item for item in all_train if item["final"] == 1]

    if n_samples is not None:
        rng     = np.random.default_rng(cfg.seed)
        indices = rng.choice(len(base_dataset), size=n_samples, replace=False)
        base_dataset = [base_dataset[i] for i in sorted(indices.tolist())]
        print(f"  Using {n_samples} training samples (seed={cfg.seed})")

    z_opts, aoas, params_all, eta_ys, pressures, perfs = precompute_latents(
        base_dataset, lae_model.bae_model, cfg.device
    )
    dataset = PrecomputedWingsDataset(
        z_opts, aoas, params_all, eta_ys, pressures, perfs,
        scaler_params=lae_model.scaler_params,
        scaler_aoas=lae_model.scaler_aoas,
        scaler_pressures=lae_model.scaler_pressures,
        scaler_perfs=lae_model.scaler_perfs,
    )
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    reg, mask = fit_perf_regressor(lae_model, loader, cfg.device)

    with open(regressor_path, 'wb') as f:
        pickle.dump({'regressor': reg, 'mask': mask}, f)
    print(f"Perf regressor saved to {regressor_path}.")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def compute_val_loss(lae_model: LAE_AoAInit, loader: DataLoader, device: str):
    """Reconstruction loss on the validation set (no gradient).

    Returns (total_loss, raw_geo_mse, raw_pressure_mse) where the raw MSEs
    are unweighted, computed directly in BAE-latent / pressure space.
    """
    lae_model.encoder.eval()
    lae_model.decoder.eval()
    total_loss = 0.0
    geo_mse_sum = 0.0
    pressure_mse_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            outs = lae_model._forward_pass(batch, device)
            (z_opt, z_opt_pred, aoa, alpha_pred,
             eta_y, eta_y_pred, pressure, pressure_pred,
             perf, perf_pred, z) = outs
            loss = lae_model._loss_LAE(
                z_opt, z_opt_pred, aoa, alpha_pred,
                eta_y, eta_y_pred, pressure, pressure_pred,
                perf, perf_pred, z,
            )
            total_loss   += loss.item()
            geo_mse_sum  += nn.functional.mse_loss(z_opt_pred, z_opt).item()
            if getattr(lae_model.decoder, 'use_pressure', True):
                pressure_mse_sum += nn.functional.mse_loss(pressure_pred, pressure).item()
    n = max(len(loader), 1)
    lae_model.encoder.train()
    lae_model.decoder.train()
    return total_loss / n, geo_mse_sum / n, pressure_mse_sum / n


# ---------------------------------------------------------------------------
# Live evolution helpers
# ---------------------------------------------------------------------------

def select_evolution_wings(val_dataset_raw, bae_model, lae_model, device, apply_x_norm=True):
    """Pick one subsonic, one transonic, one supersonic wing from val set (deterministic).

    Returns a dict with pre-encoded tensors ready for reconstruct_batch, plus ground-truth
    airfoils, pressures, perfs, and flow_conditions for plotting.
    """
    buckets = {"subsonic": None, "transonic": None, "supersonic": None}
    for item in val_dataset_raw:
        mach = float(item["mach"])
        key  = "subsonic" if mach < 0.8 else ("transonic" if mach < 1.0 else "supersonic")
        if buckets[key] is None:
            buckets[key] = item
        if all(v is not None for v in buckets.values()):
            break

    selected = [v for v in buckets.values() if v is not None]
    if not selected:
        return None

    gt_airfoils, gt_pressures, gt_perfs = [], [], []
    z_opts, params_scaled_list, te_shifts_list, flow_conditions = [], [], [], []

    for item in selected:
        coords    = torch.tensor(item["coords"],    dtype=torch.float32)
        te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)
        coords_unshifted = coords.clone()
        coords_unshifted[:, :, 1] -= te_shifts.unsqueeze(1)
        if apply_x_norm:
            te_x = coords_unshifted[:, 0, 0]
            coords_unshifted[:, :, 0] += (1.0 - te_x).unsqueeze(1)

        x_opt_slices, z_slices = [], []
        for s in range(coords_unshifted.shape[0]):
            x_s = coords_unshifted[s].permute(1, 0).unsqueeze(0).to(device)
            with torch.no_grad():
                z_s   = bae_model.encode(x_s, return_z=True, z_ae_mode=True)
                dec_s = bae_model.decode_z(z_s, z_ae_mode=True,
                                            denormalize_output=False, normalized_data=True)[0]
            x_opt_slices.append(dec_s.squeeze(0).cpu())
            z_slices.append(z_s.squeeze(0).cpu())

        gt_wing = torch.stack(x_opt_slices)
        gt_wing[:, 1, :] += te_shifts.unsqueeze(1)
        gt_airfoils.append(gt_wing)
        z_opts.append(torch.stack(z_slices))
        te_shifts_list.append(te_shifts)

        flow_p = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
        geo_p  = item.get("geo_params", np.zeros(34, dtype=np.float32)).tolist()
        raw_params = flow_p + geo_p
        c_dim = len(lae_model.scaler_params.mean)
        params = torch.tensor(raw_params[:c_dim], dtype=torch.float32).unsqueeze(0).to(device)
        params_scaled_list.append(lae_model.scaler_params.transform(params))

        gt_pressures.append(torch.tensor(np.array(item["coef_pressure"]), dtype=torch.float32))
        gt_perfs.append(torch.tensor([item["cd_val"], item["cl_val"]], dtype=torch.float32))
        flow_conditions.append({"mach": item["mach"], "reynolds": item["reynolds"],
                                 "cl_target": item["cl_target"]})

    return {
        "gt_airfoils":   torch.stack(gt_airfoils),           # [3, n_slices, 2, 192]
        "gt_pressures":  torch.stack(gt_pressures),           # [3, n_slices, 192]
        "gt_perfs":      torch.stack(gt_perfs),               # [3, 2]
        "z_opts":        torch.stack(z_opts),                 # [3, n_slices, lc, L]
        "params":        torch.cat(params_scaled_list, dim=0),# [3, c_dim]
        "te_shifts":     torch.stack(te_shifts_list),         # [3, n_slices]
        "flow_conditions": flow_conditions,
    }


def make_evolution_image(lae_model, bae_model, evo_wings, device, epoch):
    """Reconstruct the 3 fixed val wings and return a wandb.Image (or None on failure)."""
    if evo_wings is None:
        return None

    lae_model.encoder.eval()
    lae_model.decoder.eval()

    z_opts   = evo_wings["z_opts"]
    params   = evo_wings["params"]
    te_shifts = evo_wings["te_shifts"]

    with torch.no_grad():
        mu      = lae_model.encode(z_opts.to(device), params.to(device))
        z_masked = lae_model._apply_mask(mu)
        z_opt_pred, alpha_pred, eta_y_pred, pressure_pred_norm, perf_pred_norm = lae_model.decode(
            z_masked, params.to(device)
        )

    decoded_slices = []
    for s in range(z_opt_pred.shape[1]):
        z_s   = z_opt_pred[:, s, :, :]
        dec_s = bae_model.decode_z(z_s, z_ae_mode=True,
                                    denormalize_output=False, normalized_data=True)[0]
        decoded_slices.append(dec_s)
    rec_airfoils = torch.stack(decoded_slices, dim=1).cpu()
    rec_airfoils[:, :, 1, :] += eta_y_pred.cpu()

    if lae_model.scaler_pressures is not None:
        rec_pressures = lae_model.scaler_pressures.inverse_transform(pressure_pred_norm.cpu())
    else:
        rec_pressures = pressure_pred_norm.cpu()

    if lae_model.scaler_perfs is not None:
        rec_perfs = lae_model.scaler_perfs.inverse_transform(perf_pred_norm.cpu())
    else:
        rec_perfs = perf_pred_norm.cpu()

    os.makedirs("images", exist_ok=True)
    img_path = f"images/evo_ep{epoch:05d}.png"
    lvae_plotting.plot_airfoil_cp_comparison(
        rec_airfoils, evo_wings["gt_airfoils"],
        rec_pressures, evo_wings["gt_pressures"],
        gen_perfs=rec_perfs, gt_perfs=evo_wings["gt_perfs"],
        sample_indices=[0, 1, 2],
        slice_indices=[0, 7, 14],
        pred_label="Reconstructed",
        flow_conditions=evo_wings["flow_conditions"],
        save_path=img_path,
    )
    print(f"  [EVO] Evolution image created for epoch {epoch}.")
    lae_model.encoder.train()
    lae_model.decoder.train()
    return img_path


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(lae_model: LAE_AoAInit, loader: DataLoader,
                    device: str, epoch: int) -> float:
    lae_model.encoder.train()
    lae_model.decoder.train()
    total_loss = 0.0

    for batch in loader:
        lae_model._update_LAE(epoch, batch, device)
        total_loss += lae_model.stats['current_loss']

        lae_model.stats['train_loss'] = np.append(
            lae_model.stats['train_loss'], lae_model.stats['current_loss'])
        lae_model.stats['train_loss_epoch'] = np.append(
            lae_model.stats['train_loss_epoch'], epoch)

    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of training samples to use (default: all)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--no_lv", action="store_true",
                        help="Disable the least-volume penalty (lambda_lv=0) and pruning. "
                             "Use this to train a plain autoencoder baseline to isolate "
                             "whether the LV penalty is responsible for high cd MSE.")
    parser.add_argument("--lambda_lv", type=float, default=None,
                        help="Override lambda_lv (least-volume penalty weight).")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Override model name / checkpoint filename stem.")
    parser.add_argument("--w_perf", type=float, default=None,
                        help="Override perf loss weight (default: 1). weights=(w_x,w_alpha,w_eta,w_pressure,w_perf).")
    parser.add_argument("--w_pressure", type=float, default=None,
                        help="Override pressure loss weight (default: 1). weights=(w_x,w_alpha,w_eta,w_pressure,w_perf).")
    parser.add_argument("--w_eta", type=float, default=None,
                        help="Override eta_y loss weight (default: 9). weights=(w_x,w_alpha,w_eta,w_pressure,w_perf).")
    parser.add_argument("--w_x", type=float, default=None,
                        help="Override geometry reconstruction weight (default: 1000). weights=(w_x,w_alpha,w_eta,w_pressure,w_perf).")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate (default: 1e-3).")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Override Adam weight decay / L2 regularization (default: 0.0).")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Dropout probability applied after encoder pooling and before decoder heads (default: 0.0).")
    parser.add_argument("--prune_every", type=int, default=None,
                        help="Override prune_every (0 = disable pruning).")
    parser.add_argument("--prune_threshold", type=float, default=None,
                        help="Override prune_threshold (0.0 = keep all dims).")
    parser.add_argument("--no_pressure", action="store_true",
                        help="Remove pressure head from decoder entirely (geometry-only isolation test).")
    parser.add_argument("--flow_only", action="store_true",
                        help="Condition on flow params only (Mach, Reynolds, CL target, area ratio); "
                             "drops the 34 geometric conditioning dims, setting c_dim=4.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override save directory for checkpoints.")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="engiopt-lvae-sweep",
                        help="W&B project name.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    cfg.seed = args.seed

    if args.lambda_lv is not None:
        cfg.lambda_lv = args.lambda_lv
    if args.w_perf is not None:
        w_x, w_alpha, w_eta, w_pressure, _ = cfg.weights
        cfg.weights = (w_x, w_alpha, w_eta, w_pressure, args.w_perf)
        print(f"Override w_perf={args.w_perf}: weights={cfg.weights}")
    if args.w_pressure is not None:
        w_x, w_alpha, w_eta, _, w_perf = cfg.weights
        cfg.weights = (w_x, w_alpha, w_eta, args.w_pressure, w_perf)
        print(f"Override w_pressure={args.w_pressure}: weights={cfg.weights}")
    if args.w_eta is not None:
        w_x, w_alpha, _, w_pressure, w_perf = cfg.weights
        cfg.weights = (w_x, w_alpha, args.w_eta, w_pressure, w_perf)
        print(f"Override w_eta={args.w_eta}: weights={cfg.weights}")
    if args.w_x is not None:
        _, w_alpha, w_eta, w_pressure, w_perf = cfg.weights
        cfg.weights = (args.w_x, w_alpha, w_eta, w_pressure, w_perf)
        print(f"Override w_x={args.w_x}: weights={cfg.weights}")
    if args.lr is not None:
        cfg.lr = args.lr
        print(f"Override lr={args.lr}")
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
        print(f"Override weight_decay={args.weight_decay}")
    if args.dropout is not None:
        cfg.dropout = args.dropout
        print(f"Override dropout={args.dropout}")
    if args.prune_every is not None:
        cfg.prune_every = args.prune_every
    if args.prune_threshold is not None:
        cfg.prune_threshold = args.prune_threshold
    if args.no_lv:
        cfg.lambda_lv   = 0.0
        cfg.prune_every = 0      # no pruning without the LV drive
        cfg.model_name  = f"lae_no_lv_s{args.seed}"
        cfg.save_dir    = f"results/lvae_baseline/no_lv_s{args.seed}"
        print("=== NO-LV BASELINE: lambda_lv=0, pruning disabled ===")
    if args.no_pressure:
        print("=== NO-PRESSURE: pressure head removed from decoder ===")
    if args.flow_only:
        cfg.c_dim = 4
        print("=== FLOW-ONLY: conditioning on 4 flow params only (c_dim=4) ===")
    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir

    if args.n_samples is not None:
        suffix = f"_n{args.n_samples}" if not args.no_lv else f"_n{args.n_samples}"
        cfg.model_name = cfg.model_name + suffix if args.no_lv else f"lae_ablation_n{args.n_samples}_s{args.seed}"
        cfg.save_dir   = (os.path.join(cfg.save_dir, f"n{args.n_samples}")
                          if args.no_lv
                          else f"results/lvae_ablation/n{args.n_samples}_s{args.seed}")

    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    print(f"Using device: {cfg.device}")
    print(f"Model name: {cfg.model_name}")
    print(f"Save dir: {cfg.save_dir}")

    # 1) Frozen BAE
    bae_model = load_bae(cfg)
    print("Loaded BAE.")

    # 2) New dataset
    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=cfg.seed)
    all_train    = list(new_dataset["train"])
    base_dataset = [item for item in all_train if item["final"] == 1]
    all_val      = list(new_dataset["val"])
    val_dataset_raw = [item for item in all_val if item["final"] == 1]

    if args.n_samples is not None:
        rng = np.random.default_rng(cfg.seed)
        indices: np.ndarray = rng.choice(len(base_dataset), size=args.n_samples, replace=False)
        base_dataset = [base_dataset[i] for i in sorted(indices.tolist())]
        print(f"Smoke test: using {args.n_samples} of {len(all_train)} training samples (seed={args.seed})")

    print(f"Train split: {len(base_dataset)} final items.")

    # 3) Normalisation stats for flow params and AoA
    all_params = np.array([
        [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
        + ([] if args.flow_only else item.get("geo_params", np.zeros(34, dtype=np.float32)).tolist())
        for item in base_dataset
    ])
    all_aoas = np.array([float(item["alpha"]) for item in base_dataset])

    params_mean_std = (all_params.mean(axis=0), all_params.std(axis=0))
    aoas_mean_std   = (float(all_aoas.mean()),  float(all_aoas.std()))

    scaler_params = scaler(params_mean_std)
    scaler_aoas   = scaler(aoas_mean_std)

    # 4) Pre-compute latents and targets
    z_opts, aoas, params_all, eta_ys, pressures, perfs = precompute_latents(
        base_dataset, bae_model, cfg.device
    )
    print(f"Val split: {len(val_dataset_raw)} final items.")
    z_opts_val, aoas_val, params_val, eta_ys_val, pressures_val, perfs_val = precompute_latents(
        val_dataset_raw, bae_model, cfg.device
    )
    if args.flow_only:
        params_all = params_all[:, :4]
        params_val = params_val[:, :4]

    # Normalisation stats for pressure and performance
    pressures_np = pressures.numpy()  # [N, 9, 192]
    pressure_mean = pressures_np.mean()
    pressure_std  = pressures_np.std()
    scaler_pressures = scaler((pressure_mean, pressure_std))

    perfs_np = perfs.numpy()          # [N, 2]
    perf_mean = perfs_np.mean(axis=0)
    perf_std  = perfs_np.std(axis=0)
    scaler_perfs = scaler((perf_mean, perf_std))

    print(f"Params mean: {params_mean_std[0]}, std: {params_mean_std[1]}")
    print(f"AoA mean: {aoas_mean_std[0]:.4f}, std: {aoas_mean_std[1]:.4f}")
    print(f"Pressure mean: {pressure_mean:.4f}, std: {pressure_std:.4f}")
    print(f"Perf mean: {perf_mean}, std: {perf_std}")

    dataset = PrecomputedWingsDataset(
        z_opts, aoas, params_all, eta_ys, pressures, perfs,
        scaler_params=scaler_params,
        scaler_aoas=scaler_aoas,
        scaler_pressures=scaler_pressures,
        scaler_perfs=scaler_perfs,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=True)

    val_dataset = PrecomputedWingsDataset(
        z_opts_val, aoas_val, params_val, eta_ys_val, pressures_val, perfs_val,
        scaler_params=scaler_params,
        scaler_aoas=scaler_aoas,
        scaler_pressures=scaler_pressures,
        scaler_perfs=scaler_perfs,
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)

    # 5) Build model
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg, use_pressure=not args.no_pressure)
    sampler = build_sampler(cfg)
    print("Built encoder/decoder.")

    lae_model = LAE_AoAInit(
        encoder=encoder,
        decoder=decoder,
        sampler=sampler,
        bae_model=bae_model,
        lae_latent_dim=cfg.lae_latent_dim,
        params_mean_std=params_mean_std,
        aoas_mean_std=aoas_mean_std,
        pressures_mean_std=(pressure_mean, pressure_std),
        perfs_mean_std=(perf_mean, perf_std),
        name=cfg.model_name,
        lambda_lv=cfg.lambda_lv,
        weights=cfg.weights,
        opt_lr=cfg.lr,
        opt_weight_decay=cfg.weight_decay,
    )
    print("Built LAE model.")

    # Pre-select 3 fixed val wings (subsonic / transonic / supersonic) for W&B evolution plots.
    # Encoded once here so no BAE re-encoding happens each epoch.
    evo_wings = select_evolution_wings(val_dataset_raw, bae_model, lae_model, cfg.device)
    if evo_wings is not None:
        regimes = ["subsonic", "transonic", "supersonic"]
        print(f"[EVO] Fixed val wings selected for evolution plots: "
              + ", ".join(f"{r} Mach={evo_wings['flow_conditions'][i]['mach']:.2f}"
                          for i, r in enumerate(regimes[:len(evo_wings['flow_conditions'])])))

    # W&B init
    use_wandb = args.wandb and _WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity="adelbeke-",
            name=cfg.model_name,
            config={
                "lambda_lv":      cfg.lambda_lv,
                "weights":        cfg.weights,
                "lr":             cfg.lr,
                "lae_latent_dim": cfg.lae_latent_dim,
                "prune_every":    cfg.prune_every,
                "prune_threshold": cfg.prune_threshold,
                "n_epochs":       cfg.n_epochs,
                "seed":           cfg.seed,
                "use_pressure":   not args.no_pressure,
                "c_dim":          cfg.c_dim,
                "weight_decay":          cfg.weight_decay,
                "dropout":               cfg.dropout,
                "plot_evolution_every":  cfg.plot_evolution_every,
            },
        )
    elif args.wandb and not _WANDB_AVAILABLE:
        print("Warning: --wandb passed but wandb is not installed. Skipping W&B logging.")

    # 6) Training loop
    print(f"Starting training for {cfg.n_epochs} epochs "
          f"(checkpoint every {cfg.save_every} epochs, "
          f"val every {cfg.val_every} epochs, patience={cfg.patience})...")

    best_val_loss = float("inf")
    best_val_epoch = 0
    patience_count = 0

    for epoch in range(cfg.n_epochs):
        train_loss = train_one_epoch(lae_model, loader, cfg.device, epoch)
        print(f"Epoch {epoch + 1:05d}/{cfg.n_epochs} | Train Loss {train_loss:.6f}")

        # Periodic latent pruning
        if cfg.prune_every > 0 and (epoch + 1) % cfg.prune_every == 0:
            lae_model.update_active_mask(loader, cfg.device, cfg.prune_threshold)

        if (epoch + 1) % cfg.save_every == 0:
            lae_model.save(cfg.save_dir)
            print(f"  --> Checkpoint saved at epoch {epoch + 1}.")

        if cfg.milestone_every > 0 and (epoch + 1) % cfg.milestone_every == 0:
            orig_name = lae_model.name
            lae_model.name = f"{orig_name}_ep{epoch + 1}"
            lae_model.save(cfg.save_dir)
            lae_model.name = orig_name
            print(f"  --> Milestone checkpoint saved: {lae_model.name}_ep{epoch + 1}.pth")

        # Build evolution image if needed (merged into val log_dict below)
        evo_img = None
        if (cfg.plot_evolution_every > 0
                and (epoch + 1) % cfg.plot_evolution_every == 0):
            evo_img = make_evolution_image(lae_model, bae_model, evo_wings,
                                           cfg.device, epoch + 1)

        # Validation + early stopping
        if (epoch + 1) % cfg.val_every == 0:
            val_loss, raw_geo_mse, raw_pressure_mse = compute_val_loss(lae_model, val_loader, cfg.device)
            n_active = int(lae_model.active_latent_mask.sum().item())
            print(f"  [VAL] Epoch {epoch + 1:05d} | Val Loss {val_loss:.6f} | "
                  f"Best {best_val_loss:.6f} @ epoch {best_val_epoch}")
            if use_wandb:
                log_dict = {
                    "epoch":              epoch + 1,
                    "train_loss":         train_loss,
                    "val_loss":           val_loss,
                    "val_raw_geo_mse":    raw_geo_mse,
                    "val_raw_pressure_mse": raw_pressure_mse,
                    "active_dims":        n_active,
                }
                # per-component val losses if available
                stats = lae_model.stats
                if len(stats["test_loss_recon"]) > 0:
                    log_dict.update({
                        "val_loss_geom":     float(stats["test_loss_recon"][-1]),
                        "val_loss_alpha":    float(stats["test_loss_alpha"][-1]),
                        "val_loss_eta":      float(stats["test_loss_eta_y"][-1]),
                        "val_loss_pressure": float(stats["test_loss_pressure"][-1]),
                        "val_loss_perf":     float(stats["test_loss_perf"][-1]),
                        "val_loss_lv":       float(stats["test_loss_lv"][-1]),
                    })
                if evo_img is not None:
                    log_dict["evolution_reconstruction"] = wandb.Image(evo_img)
                wandb.log(log_dict)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch + 1
                patience_count = 0
                orig_name = lae_model.name
                lae_model.name = f"{orig_name}_best"
                lae_model.save(cfg.save_dir)
                lae_model.name = orig_name
                print(f"  [VAL] --> New best checkpoint saved.")
            else:
                patience_count += 1
                print(f"  [VAL] No improvement ({patience_count}).")

    lae_model.save(cfg.save_dir)
    print(f"Training complete. Best val loss {best_val_loss:.6f} at epoch {best_val_epoch}.")
    if use_wandb:
        wandb.summary["best_val_loss"]  = best_val_loss
        wandb.summary["best_val_epoch"] = best_val_epoch
        wandb.finish()

    # Fit the perf regressor on the final active-dim latent distribution.
    fit_perf_regressor(lae_model, loader, cfg.device)
    reg_path = os.path.join(cfg.save_dir, f"{cfg.model_name}_perf_reg.pkl")
    import pickle
    with open(reg_path, 'wb') as f:
        pickle.dump({'regressor': lae_model.perf_regressor,
                     'mask':      lae_model.perf_regressor_mask}, f)
    print(f"Fitted perf regressor saved to {reg_path}.")


if __name__ == "__main__":
    main()
