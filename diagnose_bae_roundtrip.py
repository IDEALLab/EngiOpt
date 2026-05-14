"""
Diagnose BAE round-trip MSE on the new dataset vs Wings3D.

For each dataset, encodes every airfoil slice through the BAE and decodes it back,
then measures coordinate MSE — this is the hard floor that the LAE shape MSE cannot
beat regardless of how well it trains.
"""

import argparse
import numpy as np
import torch

from engiopt.bezier_ae.bezier_ae import BezierAutoencoder
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BAE_CHECKPOINT  = "bezier_ae_best.pt"
NEW_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
NEW_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

N_CONTROL_POINTS = 32
N_DATA_POINTS    = 192


def load_bae(device: torch.device) -> BezierAutoencoder:
    model = BezierAutoencoder(
        n_control_points=N_CONTROL_POINTS,
        n_data_points=N_DATA_POINTS,
        batch_size=32,
        auto_batch=True,
    ).to(device)
    ckpt = torch.load(BAE_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def bae_roundtrip_mse(coords_raw: torch.Tensor,
                      te_shifts: torch.Tensor,
                      bae: BezierAutoencoder,
                      device: torch.device,
                      apply_x_shift: bool) -> float:
    """
    coords_raw : [n_slices, 192, 2]  (x in col 0, y in col 1)
    te_shifts  : [n_slices]           (TE y offset per slice)
    Returns mean coordinate MSE across all slices after encode → decode.
    """
    coords = coords_raw.clone()
    # remove y te_shift
    coords[:, :, 1] -= te_shifts.unsqueeze(1)

    if apply_x_shift:
        te_x = coords[:, 0, 0]
        coords[:, :, 0] += (1.0 - te_x).unsqueeze(1)

    mses = []
    for s in range(coords.shape[0]):
        x_s = coords[s].permute(1, 0).unsqueeze(0).to(device)  # [1, 2, 192]
        z_s = bae.encode(x_s, return_z=True, z_ae_mode=True)
        rec_s, _, _ = bae.decode_z(z_s, z_ae_mode=True)        # [1, 2, 192]
        mse = ((rec_s - x_s) ** 2).mean().item()
        mses.append(mse)

    return float(np.mean(mses))


def evaluate_dataset(dataset, bae, device, apply_x_shift, label, max_samples=None):
    print(f"\n── {label} (apply_x_shift={apply_x_shift}) ──")
    mses = []
    for i, item in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        coords    = torch.tensor(item["coords"],    dtype=torch.float32)
        te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)
        mse = bae_roundtrip_mse(coords, te_shifts, bae, device, apply_x_shift)
        mses.append(mse)
        if (i + 1) % 100 == 0:
            print(f"  {i+1} samples processed, running mean MSE: {np.mean(mses):.6f}")

    arr = np.array(mses)
    print(f"  Samples : {len(arr)}")
    print(f"  Mean MSE: {arr.mean():.6f}")
    print(f"  Std MSE : {arr.std():.6f}")
    print(f"  Max MSE : {arr.max():.6f}")
    print(f"  Min MSE : {arr.min():.6f}")
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wings3d", action="store_true",
                        help="Also run on Wings3D train split (requires EngiBench)")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    bae = load_bae(device)
    print("BAE loaded.")

    # ── New dataset ──────────────────────────────────────────────────────────
    new_dataset = NewWingsDataset(NEW_SLICES_PKL, NEW_SCALARS_PKL)["train"]
    print(f"New dataset train split: {len(new_dataset)} samples")

    # Without x-shift (what v7 trained with — the broken path)
    evaluate_dataset(new_dataset, bae, device,
                     apply_x_shift=False,
                     label="New dataset — NO x-shift (v7 behaviour)",
                     max_samples=args.max_samples)

    # With x-shift (what v8 trains with — the fixed path)
    evaluate_dataset(new_dataset, bae, device,
                     apply_x_shift=True,
                     label="New dataset — WITH x-shift (v8 behaviour)",
                     max_samples=args.max_samples)

    # ── Wings3D (optional) ───────────────────────────────────────────────────
    if args.wings3d:
        try:
            from engibench.problems.wings3D import v0 as wings3d_problem
            wings3d_train = list(wings3d_problem.dataset["train"])
            print(f"\nWings3D train split: {len(wings3d_train)} samples")

            class Wings3DWrapper:
                def __init__(self, data): self._data = data
                def __iter__(self): return iter(self._data)
                def __len__(self): return len(self._data)

            evaluate_dataset(Wings3DWrapper(wings3d_train), bae, device,
                             apply_x_shift=False,
                             label="Wings3D — NO x-shift (original training)",
                             max_samples=args.max_samples)
        except Exception as e:
            print(f"\nCould not load Wings3D: {e}")


if __name__ == "__main__":
    main()
