import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from engiopt.bezier_ae import BezierAutoencoder, loss_reg_fn
from engiopt.bezier_ae.plotting_bezier_ae import (
    plot_reconstruction,
    plot_control_polygon,
    plot_reconstruction_with_control_points,
)
from engibench.problems.wings3D.v0 import Wings3D


class WingsBezierDataset(Dataset):
    def __init__(self, base_dataset):
        self.samples = []

        for i in range(len(base_dataset)):
            coords = base_dataset[i]["coords"]                 # [9, 192, 2]
            x = torch.tensor(coords, dtype=torch.float32)      # [9, 192, 2]

            # Center trailing edge: shift so TE is at (1, 0) — matches training preprocessing
            for s in range(x.shape[0]):
                x[s, :, 1] -= x[s, 0, 1]
                x[s, :, 0] += (1.0 - x[s, 0, 0])

            x = x.permute(0, 2, 1)                             # [9, 2, 192]

            for j in range(x.shape[0]):
                self.samples.append(x[j])                      # [2, 192]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_latest_run_dir(base_dir="results/bezier_ae"):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"No results directory found at {base_dir}")

    existing = []
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path) and name.startswith("run_"):
            try:
                existing.append((int(name.split("_")[1]), full_path))
            except (IndexError, ValueError):
                pass

    if len(existing) == 0:
        raise FileNotFoundError(f"No run folders found in {base_dir}")

    existing.sort(key=lambda x: x[0])
    return existing[-1][1]


@torch.no_grad()
def evaluate(model, loader, device, reg_fac=0.001):
    model.eval()
    total_loss = 0.0

    for x in loader:
        x = x.to(device)

        y, pv, intvls, cp, w = model(x)
        loss = loss_reg_fn(y, x, cp, w, reg_fac=reg_fac)
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    problem = Wings3D(seed=0)
    base_train_dataset = problem.dataset["train"]

    coords = np.array(base_train_dataset[0]["coords"])  # convert to numpy first
    print("x range:", coords[:,:,0].min(), coords[:,:,0].max())
    print("y range:", coords[:,:,1].min(), coords[:,:,1].max())
    
    full_dataset = WingsBezierDataset(base_train_dataset)


    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(0)
    )

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = BezierAutoencoder(
        n_control_points=32,
        n_data_points=192,
        batch_size=32,
        auto_batch=True,
    ).to(device)

    run_dir = get_latest_run_dir()
    print(f"Using run directory: {run_dir}")

    checkpoint_path = os.path.join(run_dir, "models", "bezier_ae_best.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loss = evaluate(model, val_loader, device, reg_fac=0.001)
    print(f"Validation Loss: {val_loss:.6f}")

    model.eval()

    for i in range(5):
        example_idx = i + 1
        x = val_dataset[i].unsqueeze(0).to(device)  # [1, 2, 192]

        y, pv, intvls, cp, w = model(x)

        x0 = x[0]
        y0 = y[0]
        cp0 = cp[0]

        print(f"Example {example_idx}")
        print("x:", x0.shape)
        print("y:", y0.shape)
        print("cp:", cp0.shape)
        print("w:", w[0].shape)

        plot_reconstruction(
            x0,
            y0,
            title=f"Example {example_idx} Reconstruction",
            filename=f"example_{example_idx}_reconstruction.png",
            run_dir=run_dir,
        )

        plot_control_polygon(
            cp0,
            title=f"Example {example_idx} Control Polygon",
            filename=f"example_{example_idx}_control_polygon.png",
            run_dir=run_dir,
        )

        plot_reconstruction_with_control_points(
            x0,
            y0,
            cp0,
            title=f"Example {example_idx} Reconstruction + Control Points",
            filename=f"example_{example_idx}_reconstruction_with_cp.png",
            run_dir=run_dir,
        )


if __name__ == "__main__":
    main()