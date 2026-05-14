"""Measure the reconstruction floor introduced by the LVAE bottleneck.

Pipeline: coords -> BAE.encode -> z_bae -> LVAE.encode -> z_lvae -> LVAE.decode -> z_bae_pred -> BAE.decode -> coords_pred

We measure MSE at two points:
  1. z_bae vs z_bae_pred  (LVAE latent reconstruction error)
  2. coords vs coords_pred (full round-trip shape MSE through BAE+LVAE+BAE)

This tells us how much error the LVAE bottleneck adds on top of the BAE floor.
"""
import sys
import torch
import numpy as np

sys.path.insert(0, "/cluster/home/adelbeke/EngiOpt")

from engiopt.ddm.ddm_lvae.train_ddm_lvae import Config, load_bae, load_lvae
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

cfg = Config()
cfg.device = device
cfg.bae_checkpoint    = "bezier_ae_best.pt"
cfg.lvae_checkpoint   = "results/lvae/lae_dropout_0.25_best.pth"

bae_model  = load_bae(cfg)
lvae_model = load_lvae(cfg, bae_model)
bae_model.to(device)
lvae_model.to(device)
print(f"BAE loaded from  {cfg.bae_checkpoint}")
print(f"LVAE loaded from {cfg.lvae_checkpoint}")

new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=cfg.seed)
all_test     = list(new_dataset["test"])
test_dataset = [item for item in all_test if item["final"] == 1]
print(f"Test wings: {len(test_dataset)}")

mse_latent = []   # MSE(z_bae, z_bae_pred) — LVAE bottleneck error in latent space
mse_shape  = []   # MSE(coords_bae, coords_lvae) — full round-trip shape error (denormalize=False)

with torch.no_grad():
    for item in test_dataset:
        coords    = torch.tensor(item["coords"],    dtype=torch.float32)  # [S, 192, 2]
        te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)  # [S]
        raw_params = torch.tensor(
            [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]],
            dtype=torch.float32,
        ).unsqueeze(0)  # [1, 4]
        # Pad to full LVAE params dim and scale with LVAE's own scaler
        lvae_params_dim = lvae_model.encoder.c_net[0][0].in_features  # infer from model
        pad = torch.zeros(1, lvae_params_dim - 4)
        lvae_params_raw = torch.cat([raw_params, pad], dim=1)  # [1, lvae_params_dim]
        if hasattr(lvae_model, 'scaler_params') and lvae_model.scaler_params is not None:
            lvae_params_np = lvae_model.scaler_params.transform(lvae_params_raw.numpy())
            lvae_params = torch.tensor(lvae_params_np, dtype=torch.float32).to(device)
        else:
            lvae_params = lvae_params_raw.to(device)
        n_slices  = coords.shape[0]

        # Step 1: BAE encode all slices -> z_bae [1, S, 3, L]
        z_bae_slices = []
        for s in range(n_slices):
            c_s = coords[s].clone()
            c_s[:, 1] -= te_shifts[s]
            x_s = c_s.T.unsqueeze(0).to(device)
            z_s = bae_model.encode(x_s, return_z=True, z_ae_mode=True)  # [1, 3, L]
            z_bae_slices.append(z_s.squeeze(0))
        z_bae = torch.stack(z_bae_slices, dim=0).unsqueeze(0)  # [1, S, 3, L]

        # Step 2: LVAE encode -> decode -> z_bae_pred
        z_lvae    = lvae_model.encode(z_bae, lvae_params)           # [1, latent_dim]
        z_bae_pred, _, _, _, _ = lvae_model.decode(z_lvae, lvae_params)  # [1, S, 3, L]

        mse_latent.append(((z_bae - z_bae_pred) ** 2).mean().item())

        # Step 3: BAE decode z_bae_pred -> coords_pred, compare to BAE-only reconstruction
        for s in range(n_slices):
            z_orig = z_bae[0, s].unsqueeze(0)      # [1, 3, L]
            z_pred = z_bae_pred[0, s].unsqueeze(0) # [1, 3, L]

            dec_orig = bae_model.decode_z(z_orig, z_ae_mode=True, denormalize_output=False)[0].squeeze(0)
            dec_pred = bae_model.decode_z(z_pred, z_ae_mode=True, denormalize_output=False)[0].squeeze(0)

            mse_shape.append(((dec_orig - dec_pred) ** 2).mean().item())

mse_latent = np.array(mse_latent)
mse_shape  = np.array(mse_shape)

print()
print("=" * 60)
print("LVAE RECONSTRUCTION FLOOR")
print("=" * 60)
print(f"  BAE floor (from measure_bae_floor.py, mode B): ~3.26e-05")
print(f"  LVAE latent MSE (z_bae vs z_bae_pred)        : {mse_latent.mean():.4e} ± {mse_latent.std():.4e}")
print(f"  Full round-trip shape MSE (BAE+LVAE+BAE)      : {mse_shape.mean():.4e} ± {mse_shape.std():.4e}")
print()
print("The full round-trip MSE is the hard floor for the LVAE-DDM.")
print("If this >> 1.89e-05, the LVAE bottleneck is the dominant source of error.")
print("=" * 60)
