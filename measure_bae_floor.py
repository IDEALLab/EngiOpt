"""Measure BAE reconstruction floor on the test set.

For each test wing slice:
  1. Apply te_shift centering (same as training)
  2. Encode with BAE
  3. Decode back to coordinates
  4. Compute MSE vs the original (shifted) coordinates

This gives the minimum shape_mse the LVAE-DDM pipeline can ever achieve,
since evaluate_ddm_lvae.py uses BAE-reconstructed coords as ground truth.
We also compute MSE vs raw coords to show the double-counting effect.
"""
import sys
import torch
import numpy as np

sys.path.insert(0, "/cluster/home/adelbeke/EngiOpt")

from engiopt.ddm.ddm_lvae.train_ddm_lvae import Config, load_bae
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

cfg = Config()
cfg.device = device
cfg.bae_checkpoint = "bezier_ae_best.pt"

bae_model = load_bae(cfg)
bae_model.to(device)
bae_model.eval()
print(f"BAE loaded from {cfg.bae_checkpoint}")

new_dataset = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=cfg.seed)
all_test    = list(new_dataset["test"])
test_dataset = [item for item in all_test if item["final"] == 1]
print(f"Test wings: {len(test_dataset)}")

# Mode A: LVAE evaluator style — z_ae_mode=True encode, denormalize_output=True decode
# gt_coords in evaluate_ddm_lvae.py are produced this way, so this IS the floor for LVAE-DDM
mse_lvae_mode = []

# Mode B: 2D DDM evaluator style — z_ae_mode=False encode (strip padding), z_ae_mode=True decode, denormalize_output=False
# gt_coords in evaluate_ddm.py are produced this way; coords are in centered/scaled space
mse_2d_mode = []

with torch.no_grad():
    for item in test_dataset:
        coords    = torch.tensor(item["coords"],    dtype=torch.float32)  # [S, 192, 2]
        te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)  # [S]
        n_slices  = coords.shape[0]

        for s in range(n_slices):
            c_raw   = coords[s].clone()   # [192, 2]

            # --- Mode A: LVAE style ---
            c_lvae = c_raw.clone()
            c_lvae[:, 1] -= te_shifts[s]
            x_a = c_lvae.T.unsqueeze(0).to(device)
            z_a = bae_model.encode(x_a, return_z=True, z_ae_mode=True)
            dec_a = bae_model.decode_z(
                z_a, z_ae_mode=True, denormalize_output=True, normalized_data=False
            )[0].squeeze(0).cpu()
            mse_lvae_mode.append(((dec_a - c_lvae.T) ** 2).mean().item())

            # --- Mode B: 2D DDM style ---
            c_2d = c_raw.clone()
            c_2d[:, 1] -= c_raw[0, 1]                     # y-shift by TE y
            c_2d[:, 0] += (1.0 - c_raw[0, 0])             # x-shift so TE at x=1
            x_b = c_2d.T.unsqueeze(0).to(device)
            z_b = bae_model.encode(x_b, return_z=True, z_ae_mode=False)[:, :, 1:-1]  # strip loop pts
            dec_b = bae_model.decode_z(
                z_b, z_ae_mode=True, denormalize_output=False, normalized_data=False
            )[0].squeeze(0).cpu()
            mse_2d_mode.append(((dec_b - c_2d.T) ** 2).mean().item())

mse_lvae_mode = np.array(mse_lvae_mode)
mse_2d_mode   = np.array(mse_2d_mode)

print()
print("=" * 60)
print("BAE RECONSTRUCTION FLOOR")
print("=" * 60)
print(f"  Mode A (LVAE eval style)  : {mse_lvae_mode.mean():.4e} ± {mse_lvae_mode.std():.4e}")
print(f"    → hard floor for LVAE-DDM shape_mse")
print(f"  Mode B (2D DDM eval style): {mse_2d_mode.mean():.4e} ± {mse_2d_mode.std():.4e}")
print(f"    → hard floor for 2D DDM shape_mse (best achieved: 1.89e-05)")
print("=" * 60)
