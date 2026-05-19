"""
Training script for DDM_PCA — diffusion in PCA-compressed BAE latent space.

Usage
-----
    python -m engiopt.ddm.ddm_pca.train_ddm_pca \
        --model_name ddm_pca_v1 \
        --n_components 64
"""

import argparse
import math
import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from engiopt.bezier_ae.bezier_ae import BezierAutoencoder
from engiopt.ddm.ddm_pca.ddm_pca import DDM_PCA, MLPDenoiser
from engiopt.ddm import samplers
from engiopt.data_processing.utils import scaler
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # BAE
    bae_checkpoint: str   = "bezier_ae_best.pt"
    n_control_points: int = 32
    n_data_points: int    = 192
    bae_batch_size: int   = 32
    n_slices: int         = 15
    bae_latent_channels: int = 3
    bae_latent_length: int   = 30

    # PCA
    n_components: int = 64

    # DDM_PCA
    c_dim: int     = 4
    hidden_dims: tuple = (512, 512, 512, 512)
    t_embed_dim: int = 128
    dropout: float = 0.0
    w_aoa: float   = 1.0

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
    save_dir: str   = "results/ddm_pca"
    model_name: str = "ddm_pca_v1"
    seed: int       = 0
    num_workers: int = 0


class PCADataset(Dataset):
    def __init__(self, z_opts, aoas, params, z_inits,
                 scaler_params=None, scaler_aoas=None):
        self.z_opts  = z_opts    # [N, n_components]
        self.aoas    = aoas      # [N, 1]
        self.params  = params    # [N, 4]
        self.z_inits = z_inits   # [N, n_components]
        self.scaler_params = scaler_params
        self.scaler_aoas   = scaler_aoas

    def __len__(self):
        return len(self.z_opts)

    def __getitem__(self, idx):
        z_opt  = self.z_opts[idx].clone()
        aoa    = self.aoas[idx].clone()
        params = self.params[idx].clone()
        z_init = self.z_inits[idx].clone()
        if self.scaler_params is not None:
            params = self.scaler_params.transform(params)
        if self.scaler_aoas is not None:
            aoa = self.scaler_aoas.transform(aoa)
        return z_opt, aoa, params, z_init


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


def build_sampler(cfg: Config):
    return samplers.BaselineSampler_AoA_3D(
        cfg.num_diffusion_steps,
        start_x=1e-4, end_x=0.02,
        start_alpha=1e-4, end_alpha=0.02,
        cosine=cfg.cosine_schedule,
    )


def precompute_bae_latents(dataset_items, initial_by_case, bae_model, device,
                            n_slices, bae_latent_channels, bae_latent_length):
    """Encode each wing through BAE → flat latent vector [S*C*L]."""
    print(f"Pre-computing BAE latents for {len(dataset_items)} samples...")
    z_opts_list  = []
    z_inits_list = []
    aoas_list    = []
    params_list  = []

    bae_model.eval()
    with torch.no_grad():
        for i, item in enumerate(dataset_items):
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(dataset_items)}...")

            coords    = torch.tensor(item["coords"],    dtype=torch.float32)
            te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)

            coords_c = coords.clone()
            coords_c[:, :, 1] -= te_shifts.unsqueeze(1)
            te_x = coords_c[:, 0, 0]
            coords_c[:, :, 0] += (1.0 - te_x).unsqueeze(1)

            # Encode all slices → [S, C, L]
            z_slices = []
            for s in range(coords_c.shape[0]):
                x_s = coords_c[s].permute(1, 0).unsqueeze(0).to(device)
                z_s = bae_model.encode(x_s, return_z=True, z_ae_mode=True).squeeze(0).cpu()
                z_slices.append(z_s)
            z_opt = torch.stack(z_slices, dim=0)  # [S, C, L]
            z_opts_list.append(z_opt.flatten())   # [S*C*L]

            # Initial wing root slice → replicate S times → flatten
            case_num = int(item["case_num"])
            if case_num in initial_by_case:
                init = initial_by_case[case_num]
                ic   = torch.tensor(init["coords"],    dtype=torch.float32)
                it   = torch.tensor(init["te_shifts"], dtype=torch.float32)
                ir   = ic[0].clone()
                ir[:, 1] -= it[0]; ir[:, 0] += (1.0 - ir[0, 0])
                x_i  = ir.permute(1, 0).unsqueeze(0).to(device)
                z_i  = bae_model.encode(x_i, return_z=True, z_ae_mode=True).squeeze(0).cpu()
                z_init_wing = z_i.unsqueeze(0).expand(n_slices, -1, -1)  # [S, C, L]
            else:
                z_init_wing = z_opt
            z_inits_list.append(z_init_wing.flatten())

            flow = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
            params_list.append(torch.tensor(flow, dtype=torch.float32))
            aoa = torch.tensor(float(item["alpha"]), dtype=torch.float32)
            aoas_list.append(aoa.unsqueeze(0))

    print("Pre-computation complete.")
    return (
        torch.stack(z_opts_list),   # [N, S*C*L]
        torch.stack(aoas_list),     # [N, 1]
        torch.stack(params_list),   # [N, 4]
        torch.stack(z_inits_list),  # [N, S*C*L]
    )


def train_one_epoch(model, loader, device, grad_clip):
    model.denoiser.train()
    total = total_z = total_aoa = 0.0
    n = 0
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        model.optimizer.zero_grad()
        loss, lz, laoa = model.loss(batch, return_components=True)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), grad_clip)
        model.optimizer.step()
        total += loss.item(); total_z += lz.item(); total_aoa += laoa.item()
        n += 1
    d = max(n, 1)
    return total / d, total_z / d, total_aoa / d


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.denoiser.eval()
    total = total_z = total_aoa = 0.0
    n = 0
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        loss, lz, laoa = model.loss(batch, return_components=True)
        if not torch.isfinite(loss):
            continue
        total += loss.item(); total_z += lz.item(); total_aoa += laoa.item()
        n += 1
    model.denoiser.train()
    d = max(n, 1)
    return total / d, total_z / d, total_aoa / d


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",    type=str,   default=None)
    p.add_argument("--n_components",  type=int,   default=64)
    p.add_argument("--n_epochs",      type=int,   default=None)
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--w_aoa",         type=float, default=1.0)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--wandb_project", type=str,   default="engiopt-ddm-pca")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()
    cfg.seed         = args.seed
    cfg.n_components = args.n_components
    cfg.w_aoa        = args.w_aoa
    cfg.grad_clip    = args.grad_clip
    cfg.lr           = args.lr
    if args.model_name is not None: cfg.model_name = args.model_name
    if args.n_epochs   is not None: cfg.n_epochs   = args.n_epochs

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print(f"Device      : {cfg.device}")
    print(f"Model name  : {cfg.model_name}")
    print(f"n_components: {cfg.n_components}")
    print(f"w_aoa       : {cfg.w_aoa}")

    bae_model = load_bae(cfg)

    # Dataset
    new_dataset = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=cfg.seed)
    all_train   = list(new_dataset["train"])
    initial_by_case = {item["case_num"]: item for item in all_train if item["initial"] == 1}
    base_dataset    = [item for item in all_train if item["final"]   == 1]

    all_val             = list(new_dataset["val"])
    val_initial_by_case = {item["case_num"]: item for item in all_val if item["initial"] == 1}
    val_dataset         = [item for item in all_val if item["final"]   == 1]

    print(f"Train: {len(base_dataset)},  Val: {len(val_dataset)}")

    # Pre-compute BAE latents (flat)
    z_opts_flat, aoas, params_all, z_inits_flat = precompute_bae_latents(
        base_dataset, initial_by_case, bae_model, cfg.device,
        cfg.n_slices, cfg.bae_latent_channels, cfg.bae_latent_length,
    )

    # Fit PCA on training set
    print(f"Fitting PCA with {cfg.n_components} components on {z_opts_flat.shape[0]} samples...")
    pca = PCA(n_components=cfg.n_components, random_state=cfg.seed)
    z_opts_np = z_opts_flat.numpy()
    pca.fit(z_opts_np)
    print(f"Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    pca_path = os.path.join(cfg.save_dir, f"{cfg.model_name}_pca.pkl")
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)
    print(f"PCA saved to {pca_path}")

    # Transform to PCA space
    z_opts_pca  = torch.tensor(pca.transform(z_opts_np),            dtype=torch.float32)
    z_inits_pca = torch.tensor(pca.transform(z_inits_flat.numpy()), dtype=torch.float32)

    # Normalise PCA components
    z_mean = z_opts_pca.mean(dim=0, keepdim=True)
    z_std  = z_opts_pca.std(dim=0,  keepdim=True).clamp(min=1e-8)
    z_opts_n  = (z_opts_pca  - z_mean) / z_std
    z_inits_n = (z_inits_pca - z_mean) / z_std
    print(f"PCA z: [{z_opts_n.min():.3f}, {z_opts_n.max():.3f}]  "
          f"mean={z_opts_n.mean():.3f}  std={z_opts_n.std():.3f}")

    # Condition scalers
    all_params_np = np.array([[i["mach"], i["reynolds"], i["cl_target"], i["area_case_ratio"]]
                               for i in base_dataset])
    all_aoas_np   = np.array([float(i["alpha"]) for i in base_dataset])
    params_mean_std = (all_params_np.mean(0), all_params_np.std(0))
    aoas_mean_std   = (float(all_aoas_np.mean()), float(all_aoas_np.std()))
    scaler_params   = scaler(params_mean_std)
    scaler_aoas     = scaler(aoas_mean_std)

    dataset = PCADataset(z_opts_n, aoas, params_all, z_inits_n,
                         scaler_params=scaler_params, scaler_aoas=scaler_aoas)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                         num_workers=cfg.num_workers)

    # Val
    val_z_flat, val_aoas, val_params, val_z_inits_flat = precompute_bae_latents(
        val_dataset, val_initial_by_case, bae_model, cfg.device,
        cfg.n_slices, cfg.bae_latent_channels, cfg.bae_latent_length,
    )
    val_z_pca      = torch.tensor(pca.transform(val_z_flat.numpy()),       dtype=torch.float32)
    val_z_inits_pca = torch.tensor(pca.transform(val_z_inits_flat.numpy()), dtype=torch.float32)
    val_z_n         = (val_z_pca      - z_mean) / z_std
    val_z_inits_n   = (val_z_inits_pca - z_mean) / z_std
    val_ds = PCADataset(val_z_n, val_aoas, val_params, val_z_inits_n,
                        scaler_params=scaler_params, scaler_aoas=scaler_aoas)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)

    # Build model
    denoiser = MLPDenoiser(
        z_dim=cfg.n_components, c_dim=cfg.c_dim,
        hidden_dims=cfg.hidden_dims, t_embed_dim=cfg.t_embed_dim,
        dropout=cfg.dropout,
    ).to(cfg.device)

    sampler = build_sampler(cfg)

    ddm_pca = DDM_PCA(
        denoiser=denoiser, pca=pca, bae_model=bae_model,
        sampler=sampler, z_dim=cfg.n_components, c_dim=cfg.c_dim,
        n_slices=cfg.n_slices,
        bae_latent_channels=cfg.bae_latent_channels,
        bae_latent_length=cfg.bae_latent_length,
        w_aoa=cfg.w_aoa,
        params_mean_std=params_mean_std,
        aoas_mean_std=aoas_mean_std,
        name=cfg.model_name,
        opt_lr=cfg.lr,
    )
    ddm_pca.z_mean = z_mean
    ddm_pca.z_std  = z_std

    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"Denoiser parameters: {n_params:,}")

    ddm_pca.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        ddm_pca.optimizer, mode='min', factor=0.5, patience=600,
    )

    use_wandb = args.wandb and _WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=args.wandb_project, entity="adelbeke-",
                   name=cfg.model_name, config=vars(cfg))

    print(f"Training for {cfg.n_epochs} epochs...")
    best_val = float('inf')

    for epoch in range(cfg.n_epochs):
        train_loss, lz, laoa = train_one_epoch(ddm_pca, loader, cfg.device, cfg.grad_clip)
        val_loss, vlz, vlaoa = eval_one_epoch(ddm_pca, val_loader, cfg.device)
        ddm_pca.scheduler.step(train_loss if math.isfinite(train_loss) else best_val)

        print(f"Epoch {epoch+1:05d}/{cfg.n_epochs} | "
              f"Train {train_loss:.4f} (z {lz:.4f} aoa {laoa:.4f}) | "
              f"Val {val_loss:.4f} (z {vlz:.4f} aoa {vlaoa:.4f})")

        if use_wandb:
            wandb.log({"epoch": epoch+1,
                       "train_loss": train_loss, "train_z": lz, "train_aoa": laoa,
                       "val_loss": val_loss, "val_z": vlz, "val_aoa": vlaoa})

        if math.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            ddm_pca.save(cfg.save_dir, suffix="_best")

        if (epoch + 1) % cfg.save_every == 0:
            ddm_pca.save(cfg.save_dir)

    ddm_pca.save(cfg.save_dir)
    print("Training complete.")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
