import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from engiopt.bezier_ae.bezier_ae import BezierAutoencoder
from engiopt.ddm.ddm import DDM_AoAInit_3D
from engiopt.ddm.unets import Unet_AoAInit3D
from engiopt.ddm import samplers
from engiopt.data_processing.utils import scaler
from engibench.problems.wings3D.v0 import Wings3D


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # BAE
    bae_checkpoint: str = "bezier_ae_best.pt"
    n_control_points: int = 32
    n_data_points: int = 192
    bae_batch_size: int = 32

    # DDM / UNet
    w_dim: int = 9               # number of spanwise slices
    latent_channels: int = 3     # BAE latent channels per slice
    latent_length: int = 30
    c_dim: int = 4
    batch_size: int = 32
    lr: float = 1e-3
    n_epochs: int = 20000
    grad_clip: float = 1.0

    num_diffusion_steps: int = 1000

    # Outputs
    save_dir: str = "results/ddm"
    model_name: str = "ddm_v7hope(250:1)"

    # Misc
    seed: int = 0
    num_workers: int = 0

    # Checkpointing
    save_every: int = 100   # save checkpoint every N epochs


# ---------------------------------------------------------------------------
# Pre-computed dataset — BAE encoding is done ONCE before training starts,
# not on every __getitem__ call. This is the key fix for training speed.
# ---------------------------------------------------------------------------

class PrecomputedWingsDataset(Dataset):
    """
    Holds pre-encoded BAE latents so that no encoding happens during training.
    All tensors are on CPU; the training loop moves them to the device.
    """

    def __init__(self, z_opts, aoas, params, z_inits,
                 scaler_params=None, scaler_aoas=None):
        self.z_opts        = z_opts        # [N, 9, latent_channels, latent_length]
        self.aoas          = aoas          # [N, 1]
        self.params        = params        # [N, 4]
        self.z_inits       = z_inits       # [N, latent_channels, latent_length]  (initial root slice)
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


def precompute_latents(base_dataset, initial_by_case, bae_model, device):
    """
    Run the BAE encoder once over the entire dataset and cache the results.

    For z_opt: center each slice's trailing edge at (1, 0) by subtracting
    coords[s, 0, 1] (the TE y-coordinate, which is the first point) from all
    y-coordinates of that slice before encoding.

    For z_init: use the *initial* (unoptimised) root-slice airfoil, not the
    optimised one, to give the model a realistic starting condition.
    """
    print(f"Pre-computing BAE latents for {len(base_dataset)} samples...")

    z_opts_list  = []
    z_inits_list = []
    aoas_list    = []
    params_list  = []

    bae_model.eval()
    with torch.no_grad():
        for i, item in enumerate(base_dataset):
            if (i + 1) % 100 == 0:
                print(f"  Encoding sample {i + 1}/{len(base_dataset)}...")

            coords = torch.tensor(item["coords"], dtype=torch.float32)  # [9,192,2]

            # Center each slice: subtract the TE y-coordinate (first point) from all y-coords
            coords_centered = coords.clone()
            for s in range(coords_centered.shape[0]):
                coords_centered[s, :, 1] -= coords[s, 0, 1]
                coords_centered[s, :, 0] += (1.0 - coords[s, 0, 0])

            # z_init: initial (unoptimised) root-slice airfoil for this case
            case_num = int(item["case_num"])
            if case_num in initial_by_case:
                init_coords = torch.tensor(
                    initial_by_case[case_num]["coords"], dtype=torch.float32
                )  # [9,192,2]
                # Center the initial root slice as well
                init_root = init_coords[0].clone()
                init_root[:, 1] -= init_root[0, 1]
                x_init = init_root.permute(1, 0).unsqueeze(0).to(device)  # [1,2,192]
            else:
                # Fallback: use root slice of the current (final) item
                x_init = coords_centered[0].permute(1, 0).unsqueeze(0).to(device)
            z_init = bae_model.encode(
                x_init, return_z=True, z_ae_mode=False
            )[:, :, 1:-1].squeeze(0).cpu()   # [3,L] — strip fixed loop-point columns

            # z_opt: encode all 9 centered slices
            z_slices = []
            for s in range(coords_centered.shape[0]):
                x_s = coords_centered[s].permute(1, 0).unsqueeze(0).to(device)  # [1,2,192]
                z_s = bae_model.encode(
                    x_s, return_z=True, z_ae_mode=False
                )[:, :, 1:-1].squeeze(0).cpu()   # [3,L] — strip fixed loop-point columns
                z_slices.append(z_s)
            z_opt = torch.stack(z_slices, dim=0)  # [9,3,L]

            z_inits_list.append(z_init)
            z_opts_list.append(z_opt)

            aoa = torch.tensor(item["alpha"], dtype=torch.float32)
            if aoa.ndim == 0:
                aoa = aoa.unsqueeze(0)   # [1]
            aoas_list.append(aoa)

            params_list.append(torch.tensor(
                [item["mach"], item["reynolds"],
                 item["cl_target"], item["area_case_ratio"]],
                dtype=torch.float32,
            ))

    print("Pre-computation complete.")
    return (
        torch.stack(z_opts_list),   # [N, 9, 3, latent_length]
        torch.stack(aoas_list),     # [N, 1]
        torch.stack(params_list),   # [N, 4]
        torch.stack(z_inits_list),  # [N, 3, latent_length]
    )


# ---------------------------------------------------------------------------
# Model building helpers  (unchanged from original)
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


def build_unet(cfg: Config) -> Unet_AoAInit3D:
    return Unet_AoAInit3D(
        w_dim=cfg.w_dim,
        x_latent_channels_2D=cfg.latent_channels,
        N_dim=cfg.latent_length,
        tform_dim=3,
        c_dim=cfg.c_dim,
        down_channels=[32, 64, 128, 256],
        middle_channel=128,
        up_channels=[256, 128, 64, 32],
        upsampling_factor=2,
        block_norms=[True, True, True],
        droput=False,
    ).to(cfg.device)


def build_sampler(cfg: Config):
    return samplers.BaselineSampler_AoA_3D(
        cfg.num_diffusion_steps,
        start_x=1e-4,   end_x=0.02,
        start_alpha=1e-4, end_alpha=0.02,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(ddm_model: DDM_AoAInit_3D, loader: DataLoader,
                    device: str, epoch: int) -> float:
    ddm_model.unet.train()
    total_loss = 0.0

    for batch in loader:
        batch = tuple(t.to(device) for t in batch)

        ddm_model._update_DDM(epoch, batch, device, ddm_model.bae_model)
        loss_val = ddm_model.stats['current_loss']

        ddm_model.stats['train_loss'] = np.append(
            ddm_model.stats['train_loss'], loss_val)
        ddm_model.stats['train_loss_epoch'] = np.append(
            ddm_model.stats['train_loss_epoch'], epoch)

        total_loss += loss_val

    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of training samples to use (default: all)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    cfg.seed = args.seed

    # Set model name and save dir based on ablation args
    if args.n_samples is not None:
        cfg.model_name = f"ddm_ablation_n{args.n_samples}_s{args.seed}"
        cfg.save_dir = f"results/ablation/n{args.n_samples}_s{args.seed}"

    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print(f"Using device: {cfg.device}")
    print(f"Model name: {cfg.model_name}")
    print(f"Save dir: {cfg.save_dir}")

    # 1) Load frozen BAE
    bae_model = load_bae(cfg)
    print("Loaded BAE.")

    # 2) Load Wings3D dataset (scan once, split into initial and final)
    problem = Wings3D(seed=cfg.seed)
    all_train = list(problem.dataset["train"])
    initial_by_case = {item["case_num"]: item for item in all_train if item["initial"] == 1}
    base_dataset = [item for item in all_train if item["final"] == 1]

    # Subset the dataset if --n_samples is specified
    if args.n_samples is not None:
        rng = np.random.default_rng(cfg.seed)
        indices = rng.choice(len(base_dataset), size=args.n_samples, replace=False)
        base_dataset = [base_dataset[i] for i in sorted(indices)]
        print(f"Ablation: using {args.n_samples} of {len(all_train)} training samples (seed={args.seed})")

    print(f"Loaded Wings3D train split: {len(base_dataset)} final items, "
          f"{len(initial_by_case)} initial cases.")

    # 3) Compute normalisation stats
    all_params = np.array([
        [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
        for item in base_dataset
    ])
    all_aoas = np.array([float(item["alpha"]) for item in base_dataset])

    params_mean_std = (all_params.mean(axis=0), all_params.std(axis=0))
    aoas_mean_std   = (float(all_aoas.mean()), float(all_aoas.std()))

    scaler_params = scaler(params_mean_std)
    scaler_aoas   = scaler(aoas_mean_std)
    print(f"Params mean: {params_mean_std[0]}, std: {params_mean_std[1]}")
    print(f"AoA mean: {aoas_mean_std[0]:.4f}, std: {aoas_mean_std[1]:.4f}")

    # 4) PRE-COMPUTE latents (done once, not every epoch)
    z_opts, aoas, params_all, z_inits = precompute_latents(
        base_dataset, initial_by_case, bae_model, cfg.device
    )

    # Normalize latents per-channel so each of the 3 channels (weights, CP1, CP2)
    # is scaled independently. z_opts: [N, 9, 3, L] → mean/std: [1, 1, 3, 1]
    z_mean = z_opts.mean(dim=(0, 1, 3), keepdim=True)  # [1, 1, 3, 1]
    z_std  = z_opts.std(dim=(0, 1, 3), keepdim=True)   # [1, 1, 3, 1]

    print(f"Before normalization - z_opts range: [{z_opts.min():.3f}, {z_opts.max():.3f}]")
    print(f"Per-channel mean: {z_mean.squeeze().tolist()}")
    print(f"Per-channel std:  {z_std.squeeze().tolist()}")

    z_opts  = (z_opts  - z_mean) / z_std
    # z_inits: [N, 3, L] — squeeze one dim from mean/std to get [1, 3, 1]
    z_inits = (z_inits - z_mean[0]) / z_std[0]

    print(f"After normalization - z_opts range: [{z_opts.min():.3f}, {z_opts.max():.3f}], mean: {z_opts.mean():.3f}, std: {z_opts.std():.3f}")

    ddm_dataset = PrecomputedWingsDataset(
        z_opts, aoas, params_all, z_inits,
        scaler_params=scaler_params,
        scaler_aoas=scaler_aoas,
    )

    ddm_loader = DataLoader(
        ddm_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    # 5) Build UNet
    unet = build_unet(cfg)
    print("Built UNet.")
    print(f"3D model: w_dim={cfg.w_dim}, latent_channels={cfg.latent_channels}, latent_length={cfg.latent_length}")

    # 6) Build sampler
    sampler = build_sampler(cfg)
    print("Built sampler.")

    # 7) Create DDM wrapper
    ddm_model = DDM_AoAInit_3D(
        unet=unet,
        sampler=sampler,
        bae_model=bae_model,
        params_mean_std=params_mean_std,
        aoas_mean_std=aoas_mean_std,
        name=cfg.model_name,
        opt_lr=cfg.lr,
        opt_betas=(0.9, 0.99),
        weights=(250.0, 1.0),  # geometry : AoA
        apply_reg=False,
        reg_factor=10.0,
        reg_exp=20,
        latent_mean=z_mean,
        latent_std=z_std,
    )
    print("Built DDM model.")

    # 7b) LR scheduler: ReduceLROnPlateau
    ddm_model.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        ddm_model.optimizer,
        mode='min',
        factor=0.5,
        patience=600,
    )
    print("LR scheduler: ReduceLROnPlateau (factor=0.5, patience=600).")


    # 8) Training loop with periodic checkpointing
    print(f"Starting training for {cfg.n_epochs} epochs "
          f"(checkpoint every {cfg.save_every} epochs)...")

    for epoch in range(cfg.n_epochs):
        train_loss = train_one_epoch(ddm_model, ddm_loader, cfg.device, epoch)

        ddm_model.scheduler.step(train_loss)

        print(f"Epoch {epoch + 1:05d}/{cfg.n_epochs} | Train Loss {train_loss:.6f}")

        # Save periodically so you never lose more than save_every epochs of work
        if (epoch + 1) % cfg.save_every == 0:
            ddm_model.save(cfg.save_dir)
            print(f"  --> Checkpoint saved at epoch {epoch + 1}.")

    # Final save
    ddm_model.save(cfg.save_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
