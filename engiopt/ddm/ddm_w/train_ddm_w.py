"""
Training script for DDM_W — diffusion in LVAE w-space.

Usage
-----
    python -m engiopt.ddm.ddm_w.train_ddm_w \
        --lvae_checkpoint results/lvae/lae_dropout_0.25_flow_only_best.pth \
        --model_name ddm_w_v1
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
from engiopt.ddm.ddm_w.ddm_w import DDM_W, MLPDenoiser
from engiopt.ddm import samplers
from engiopt.data_processing.utils import scaler
from engiopt.lvae.lvae import LAE_AoAInit
from engiopt.lvae.unets import LAEEncoder, LAEDecoder
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # BAE
    bae_checkpoint: str  = "bezier_ae_best.pt"
    n_control_points: int = 32
    n_data_points: int    = 192
    bae_batch_size: int   = 32

    # LVAE
    lvae_checkpoint: str  = ""
    lvae_w_dim: int       = 15
    lvae_lae_latent_dim: int = 64
    lvae_c_dim: int       = 4
    lvae_params_dim: int  = 4

    # DDM_W
    w_dim: int     = 64    # LVAE latent dimension
    c_dim: int     = 4     # flow params only
    hidden_dims: tuple = (512, 512, 512, 512)
    t_embed_dim: int = 128
    dropout: float = 0.0
    w_pressure: float = 1.0
    w_aoa: float      = 1.0

    # Diffusion
    num_diffusion_steps: int = 1000
    cosine_schedule: bool    = False

    # Training
    batch_size: int  = 64
    lr: float        = 1e-4
    n_epochs: int    = 20000
    grad_clip: float = 1.0
    save_every: int  = 200

    # Outputs
    save_dir: str   = "results/ddm_w"
    model_name: str = "ddm_w_v1"

    seed: int        = 0
    num_workers: int = 0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WDataset(Dataset):
    """Pre-encoded LVAE w vectors."""

    def __init__(self, w_opts, aoas, params, w_inits, pressures,
                 scaler_params=None, scaler_aoas=None):
        self.w_opts    = w_opts      # [N, w_dim]
        self.aoas      = aoas        # [N, 1]
        self.params    = params      # [N, c_dim]
        self.w_inits   = w_inits     # [N, w_dim]
        self.pressures = pressures   # [N, S, 192]
        self.scaler_params = scaler_params
        self.scaler_aoas   = scaler_aoas

    def __len__(self):
        return len(self.w_opts)

    def __getitem__(self, idx):
        w_opt    = self.w_opts[idx].clone()
        aoa      = self.aoas[idx].clone()
        params   = self.params[idx].clone()
        w_init   = self.w_inits[idx].clone()
        pressure = self.pressures[idx].clone()

        if self.scaler_params is not None:
            params = self.scaler_params.transform(params)
        if self.scaler_aoas is not None:
            aoa = self.scaler_aoas.transform(aoa)

        return w_opt, aoa, params, w_init, pressure


# ---------------------------------------------------------------------------
# Pre-computation: BAE → LVAE encoder → w
# ---------------------------------------------------------------------------

def precompute_w(dataset_items, initial_by_case, bae_model, lvae_model,
                 lvae_params_scaler, lvae_params_dim, device):
    """Encode each wing: coords → BAE latents → LVAE encoder → w.

    Returns
    -------
    w_opts    : [N, w_dim]
    aoas      : [N, 1]
    params    : [N, 4]
    w_inits   : [N, w_dim]
    pressures : [N, S, 192]
    """
    print(f"Pre-computing LVAE w encodings for {len(dataset_items)} samples...")

    w_opts_list  = []
    w_inits_list = []
    aoas_list    = []
    params_list  = []
    pressures_list = []

    bae_model.eval()
    lvae_model.encoder.eval()

    with torch.no_grad():
        for i, item in enumerate(dataset_items):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(dataset_items)}...")

            coords    = torch.tensor(item["coords"],    dtype=torch.float32)
            te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)

            # Centre slices (same convention as LVAE training)
            coords_c = coords.clone()
            coords_c[:, :, 1] -= te_shifts.unsqueeze(1)
            te_x = coords_c[:, 0, 0]
            coords_c[:, :, 0] += (1.0 - te_x).unsqueeze(1)

            # BAE encode all slices → z_opt [S, 3, 30]
            z_slices = []
            for s in range(coords_c.shape[0]):
                x_s = coords_c[s].permute(1, 0).unsqueeze(0).to(device)
                z_s = bae_model.encode(x_s, return_z=True, z_ae_mode=True).squeeze(0).cpu()
                z_slices.append(z_s)
            z_opt = torch.stack(z_slices, dim=0).unsqueeze(0).to(device)  # [1, S, 3, 30]

            # LVAE params (padded to lvae_params_dim, normalised by LVAE's own scaler)
            flow = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
            flow_t = torch.tensor(flow, dtype=torch.float32)
            params_list.append(flow_t)

            flow_np = np.array(flow, dtype=np.float32).reshape(1, -1)
            if lvae_params_scaler is not None:
                flow_lvae_np = lvae_params_scaler.transform(flow_np)
            else:
                flow_lvae_np = flow_np
            flow_lvae = torch.tensor(flow_lvae_np, dtype=torch.float32, device=device)  # [1, 4]

            # LVAE encode → w [1, 64]
            lvae_model.encoder.to(device)
            w = lvae_model.encoder(z_opt, flow_lvae)   # [1, 64]
            w_opts_list.append(w.squeeze(0).cpu())

            # z_init: root slice of initial wing
            case_num = int(item["case_num"])
            if case_num in initial_by_case:
                init = initial_by_case[case_num]
                ic   = torch.tensor(init["coords"],    dtype=torch.float32)
                it   = torch.tensor(init["te_shifts"], dtype=torch.float32)
                ir   = ic[0].clone()
                ir[:, 1] -= it[0]
                ir[:, 0] += (1.0 - ir[0, 0])
                x_i  = ir.permute(1, 0).unsqueeze(0).to(device)
                z_i  = bae_model.encode(x_i, return_z=True, z_ae_mode=True)  # [1, 3, 30]
                # Build a "wing" of S copies of root slice for LVAE encoding
                S = z_opt.shape[1]
                z_init_wing = z_i.unsqueeze(1).expand(1, S, -1, -1)  # [1, S, 3, 30]
            else:
                z_init_wing = z_opt  # fall back to target wing

            w_init = lvae_model.encoder(z_init_wing, flow_lvae)  # [1, 64]
            w_inits_list.append(w_init.squeeze(0).cpu())

            aoa = torch.tensor(item["alpha"], dtype=torch.float32)
            aoas_list.append(aoa.unsqueeze(0) if aoa.ndim == 0 else aoa)

            pressure = torch.tensor(np.array(item["coef_pressure"]), dtype=torch.float32)
            pressures_list.append(pressure)

    print("Pre-computation complete.")
    return (
        torch.stack(w_opts_list),    # [N, w_dim]
        torch.stack(aoas_list),      # [N, 1]
        torch.stack(params_list),    # [N, 4]
        torch.stack(w_inits_list),   # [N, w_dim]
        torch.stack(pressures_list), # [N, S, 192]
    )


# ---------------------------------------------------------------------------
# Model loaders (reuse from train_ddm_lvae)
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
    encoder = LAEEncoder(
        w_dim=cfg.lvae_w_dim, latent_channels=3, latent_length=30,
        lae_latent_dim=cfg.lvae_lae_latent_dim, c_dim=4,
        c_dim_latent=16, down_channels=[64, 128, 256, 512],
        middle_channel=256, dropout=0.0,
    )
    decoder = LAEDecoder(
        w_dim=cfg.lvae_w_dim, latent_channels=3, latent_length=30,
        pressure_length=192, perf_dim=2,
        lae_latent_dim=cfg.lvae_lae_latent_dim, c_dim=4,
        c_dim_latent=16, up_channels=[512, 256, 128, 64],
        base_length=4, use_pressure=True, dropout=0.0,
    )
    lvae = LAE_AoAInit(
        encoder=encoder, decoder=decoder, sampler=None,
        bae_model=bae_model, lae_latent_dim=cfg.lvae_lae_latent_dim,
    )
    lvae.load(cfg.lvae_checkpoint, train_mode=False)
    lvae.encoder.eval(); lvae.decoder.eval()
    for p in lvae.encoder.parameters(): p.requires_grad_(False)
    for p in lvae.decoder.parameters(): p.requires_grad_(False)
    print(f"Loaded and froze LVAE from {cfg.lvae_checkpoint}.")
    return lvae


def build_sampler(cfg: Config):
    return samplers.BaselineSampler_AoA_3D(
        cfg.num_diffusion_steps,
        start_x=1e-4, end_x=0.02,
        start_alpha=1e-4, end_alpha=0.02,
        cosine=cfg.cosine_schedule,
    )


# ---------------------------------------------------------------------------
# Train / eval one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model: DDM_W, loader: DataLoader, device: str, grad_clip: float):
    model.denoiser.train()
    total_loss = total_w = total_aoa = total_p = 0.0
    n = 0
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        model.optimizer.zero_grad()
        loss, lw, laoa, lp = model.loss(batch, return_components=True)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), grad_clip)
        model.optimizer.step()
        total_loss += loss.item(); total_w += lw.item()
        total_aoa  += laoa.item(); total_p  += lp.item()
        n += 1
    d = max(n, 1)
    return total_loss / d, total_w / d, total_aoa / d, total_p / d


@torch.no_grad()
def eval_one_epoch(model: DDM_W, loader: DataLoader, device: str):
    model.denoiser.eval()
    total_loss = total_w = total_aoa = total_p = 0.0
    n = 0
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        loss, lw, laoa, lp = model.loss(batch, return_components=True)
        if not torch.isfinite(loss):
            continue
        total_loss += loss.item(); total_w += lw.item()
        total_aoa  += laoa.item(); total_p  += lp.item()
        n += 1
    model.denoiser.train()
    d = max(n, 1)
    return total_loss / d, total_w / d, total_aoa / d, total_p / d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lvae_checkpoint", type=str, required=True)
    p.add_argument("--model_name",      type=str, default=None)
    p.add_argument("--n_epochs",        type=int, default=None)
    p.add_argument("--n_samples",       type=int, default=None)
    p.add_argument("--seed",            type=int, default=0)
    p.add_argument("--w_pressure",      type=float, default=1.0)
    p.add_argument("--w_aoa",           type=float, default=1.0)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--hidden_dims",     type=int, nargs="+", default=None)
    p.add_argument("--wandb",           action="store_true")
    p.add_argument("--wandb_project",   type=str, default="engiopt-ddm-w")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()
    cfg.seed            = args.seed
    cfg.lvae_checkpoint = args.lvae_checkpoint
    cfg.w_pressure      = args.w_pressure
    cfg.w_aoa           = args.w_aoa
    cfg.grad_clip       = args.grad_clip
    cfg.lr              = args.lr
    if args.model_name  is not None: cfg.model_name = args.model_name
    if args.n_epochs    is not None: cfg.n_epochs   = args.n_epochs
    if args.hidden_dims is not None: cfg.hidden_dims = tuple(args.hidden_dims)

    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print(f"Device     : {cfg.device}")
    print(f"Model name : {cfg.model_name}")
    print(f"w_pressure : {cfg.w_pressure}")

    # 1. Frozen models
    bae_model  = load_bae(cfg)
    lvae_model = load_lvae(cfg, bae_model)

    # LVAE params scaler (needed to normalise params before passing to LVAE encoder)
    lvae_params_scaler = getattr(lvae_model, 'scaler_params', None)

    # 2. Dataset
    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=cfg.seed)
    all_train    = list(new_dataset["train"])
    initial_by_case = {item["case_num"]: item for item in all_train if item["initial"] == 1}
    base_dataset    = [item for item in all_train if item["final"]   == 1]

    all_val             = list(new_dataset["val"])
    val_initial_by_case = {item["case_num"]: item for item in all_val if item["initial"] == 1}
    val_dataset         = [item for item in all_val if item["final"]   == 1]

    if args.n_samples is not None:
        rng     = np.random.default_rng(cfg.seed)
        indices = rng.choice(len(base_dataset), size=args.n_samples, replace=False)
        base_dataset = [base_dataset[i] for i in sorted(indices)]
        cfg.model_name = f"ddm_w_n{args.n_samples}_s{cfg.seed}"
        cfg.save_dir   = f"results/ddm_w/n{args.n_samples}_s{cfg.seed}"
        os.makedirs(cfg.save_dir, exist_ok=True)

    print(f"Train: {len(base_dataset)} samples,  Val: {len(val_dataset)} samples")

    # 3. Condition normalization stats (on raw 4-dim flow params)
    all_params_np = np.array([
        [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
        for item in base_dataset
    ])
    all_aoas_np = np.array([float(item["alpha"]) for item in base_dataset])
    params_mean_std = (all_params_np.mean(0), all_params_np.std(0))
    aoas_mean_std   = (float(all_aoas_np.mean()), float(all_aoas_np.std()))
    scaler_params   = scaler(params_mean_std)
    scaler_aoas     = scaler(aoas_mean_std)

    # 4. Pre-compute w encodings
    w_opts, aoas, params_all, w_inits, pressures = precompute_w(
        base_dataset, initial_by_case, bae_model, lvae_model,
        lvae_params_scaler, cfg.lvae_params_dim, cfg.device,
    )

    # Normalise w (zero-mean, unit-std per dim)
    w_mean = w_opts.mean(dim=0, keepdim=True)   # [1, 64]
    w_std  = w_opts.std(dim=0,  keepdim=True).clamp(min=1e-8)
    w_opts_n  = (w_opts  - w_mean) / w_std
    w_inits_n = (w_inits - w_mean) / w_std

    # Pressure normalization
    p_mean = float(pressures.mean())
    p_std  = float(pressures.std())
    pressures_n = (pressures - p_mean) / max(p_std, 1e-8)

    print(f"w_opts   : [{w_opts_n.min():.3f}, {w_opts_n.max():.3f}]  "
          f"mean={w_opts_n.mean():.3f}  std={w_opts_n.std():.3f}")

    dataset = WDataset(w_opts_n, aoas, params_all, w_inits_n, pressures_n,
                       scaler_params=scaler_params, scaler_aoas=scaler_aoas)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size,
                         shuffle=True, num_workers=cfg.num_workers)

    # Val
    val_w, val_aoas, val_params, val_w_inits, val_pressures = precompute_w(
        val_dataset, val_initial_by_case, bae_model, lvae_model,
        lvae_params_scaler, cfg.lvae_params_dim, cfg.device,
    )
    val_w_n      = (val_w      - w_mean) / w_std
    val_w_inits_n = (val_w_inits - w_mean) / w_std
    val_pressures_n = (val_pressures - p_mean) / max(p_std, 1e-8)
    val_ds = WDataset(val_w_n, val_aoas, val_params, val_w_inits_n, val_pressures_n,
                      scaler_params=scaler_params, scaler_aoas=scaler_aoas)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers)

    # 5. Build model
    denoiser = MLPDenoiser(
        w_dim=cfg.w_dim, c_dim=cfg.c_dim,
        hidden_dims=cfg.hidden_dims, t_embed_dim=cfg.t_embed_dim,
        dropout=cfg.dropout,
    ).to(cfg.device)

    sampler = build_sampler(cfg)

    ddm_w = DDM_W(
        denoiser=denoiser,
        lvae_model=lvae_model,
        bae_model=bae_model,
        sampler=sampler,
        w_dim=cfg.w_dim,
        c_dim=cfg.c_dim,
        w_pressure=cfg.w_pressure,
        w_aoa=cfg.w_aoa,
        lvae_params_dim=cfg.lvae_params_dim,
        params_mean_std=params_mean_std,
        aoas_mean_std=aoas_mean_std,
        name=cfg.model_name,
        opt_lr=cfg.lr,
    )
    ddm_w.w_mean    = w_mean
    ddm_w.w_std     = w_std
    ddm_w.p_ddm_mean = p_mean
    ddm_w.p_ddm_std  = p_std

    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"Denoiser parameters: {n_params:,}")

    ddm_w.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        ddm_w.optimizer, mode='min', factor=0.5, patience=600,
    )

    # W&B
    use_wandb = args.wandb and _WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity="adelbeke-",
            name=cfg.model_name,
            config=vars(cfg),
        )

    # 6. Training loop
    print(f"Training for {cfg.n_epochs} epochs...")
    best_val = float('inf')

    for epoch in range(cfg.n_epochs):
        train_loss, lw, laoa, lp = train_one_epoch(
            ddm_w, loader, cfg.device, cfg.grad_clip
        )
        val_loss, vlw, vlaoa, vlp = eval_one_epoch(ddm_w, val_loader, cfg.device)
        ddm_w.scheduler.step(train_loss if math.isfinite(train_loss) else best_val)

        print(f"Epoch {epoch+1:05d}/{cfg.n_epochs} | "
              f"Train {train_loss:.4f} (w {lw:.4f} aoa {laoa:.4f} p {lp:.4f}) | "
              f"Val {val_loss:.4f} (w {vlw:.4f} aoa {vlaoa:.4f} p {vlp:.4f})")

        if use_wandb:
            wandb.log({"epoch": epoch+1,
                       "train_loss": train_loss, "train_w": lw,
                       "train_aoa": laoa, "train_p": lp,
                       "val_loss": val_loss, "val_w": vlw,
                       "val_aoa": vlaoa, "val_p": vlp})

        if math.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            ddm_w.save(cfg.save_dir, suffix="_best")

        if (epoch + 1) % cfg.save_every == 0:
            ddm_w.save(cfg.save_dir)

    ddm_w.save(cfg.save_dir)
    print("Training complete.")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
