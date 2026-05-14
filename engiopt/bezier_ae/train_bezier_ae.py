import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import matplotlib.pyplot as plt
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

            # Center trailing edge: shift so TE is at (1, 0) — matches DDM preprocessing
            for s in range(x.shape[0]):
                x[s, :, 1] -= x[s, 0, 1]
                x[s, :, 0] += (1.0 - x[s, 0, 0])

            x = x.permute(0, 2, 1)                             # [9, 2, 192]

            for j in range(x.shape[0]):
                self.samples.append(x[j])                      # each is [2, 192]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_next_run_dir(base_dir="results/bezier_ae"):
    os.makedirs(base_dir, exist_ok=True)

    existing = []
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path) and name.startswith("run_"):
            try:
                existing.append(int(name.split("_")[1]))
            except (IndexError, ValueError):
                pass

    next_idx = 1 if len(existing) == 0 else max(existing) + 1
    run_dir = os.path.join(base_dir, f"run_{next_idx:03d}")

    os.makedirs(os.path.join(run_dir, "training_curves"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "reconstructions"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "control_polygons"), exist_ok=True)

    return run_dir


def train_one_epoch(model, loader, optimizer, device, reg_fac=0.001):
    model.train()
    total_loss = 0.0

    for x in loader:
        x = x.to(device)

        optimizer.zero_grad()

        y, pv, intvls, cp, w = model(x)
        loss = loss_reg_fn(y, x, cp, w, reg_fac=reg_fac)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loss(model, loader, device, reg_fac=0.001):
    model.eval()
    total_loss = 0.0

    for x in loader:
        x = x.to(device)

        y, pv, intvls, cp, w = model(x)
        loss = loss_reg_fn(y, x, cp, w, reg_fac=reg_fac)

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def save_evaluation_plots(model, val_dataset, device, run_dir, n_examples=5):
    model.eval()

    for i in range(min(n_examples, len(val_dataset))):
        example_idx = i + 1
        x = val_dataset[i].unsqueeze(0).to(device)  # [1, 2, 192]

        y, pv, intvls, cp, w = model(x)

        x0 = x[0]
        y0 = y[0]
        cp0 = cp[0]

        print(f"Saving plots for example {example_idx}")
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    run_dir = make_next_run_dir()
    print(f"Saving training and evaluation outputs to: {run_dir}")

    problem = Wings3D(seed=0)
    base_train_dataset = problem.dataset["train"]
    base_val_dataset = problem.dataset["validation"]

    full_dataset = WingsBezierDataset(list(base_train_dataset) + list(base_val_dataset))
    print("Total slices:", len(full_dataset))

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(0)
    )

    batch_size = 64
    reg_fac = 0.0002
    learning_rate = 1e-3
    n_control_points = 32
    n_data_points = 192
    n_epochs = 500

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BezierAutoencoder(
        n_control_points=n_control_points,
        n_data_points=n_data_points,
        batch_size=batch_size,
        auto_batch=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_epoch = -1

    train_losses = []
    val_losses = []

    best_model_path = os.path.join(run_dir, "models", "bezier_ae_best.pt")

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, reg_fac=reg_fac)
        val_loss = evaluate_loss(model, val_loader, device, reg_fac=reg_fac)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss {train_loss:.6f} | "
            f"Val Loss {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                best_model_path,
            )

    print("Best validation loss:", best_val_loss)
    print("Best epoch:", best_epoch)

    metrics_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch_idx, (tr, va) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([epoch_idx, tr, va])

    print(f"Saved metrics to: {metrics_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()

    curve_path = os.path.join(run_dir, "training_curves", "loss_curve.png")
    plt.savefig(curve_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved loss curve to: {curve_path}")

    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"device: {device}\n")
        f.write(f"total_slices: {len(full_dataset)}\n")
        f.write(f"train_size: {len(train_dataset)}\n")
        f.write(f"val_size: {len(val_dataset)}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"reg_fac: {reg_fac}\n")
        f.write(f"n_control_points: {n_control_points}\n")
        f.write(f"n_data_points: {n_data_points}\n")
        f.write(f"n_epochs: {n_epochs}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val_loss: {best_val_loss}\n")
        f.write(f"best_model_path: {best_model_path}\n")

    print(f"Saved summary to: {summary_path}")

    print("Loading best model checkpoint for final evaluation plots...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_val_loss = evaluate_loss(model, val_loader, device, reg_fac=reg_fac)
    print(f"Final validation loss from best checkpoint: {final_val_loss:.6f}")

    save_evaluation_plots(
        model=model,
        val_dataset=val_dataset,
        device=device,
        run_dir=run_dir,
        n_examples=5,
    )

    print(f"Finished. All outputs saved in: {run_dir}")


if __name__ == "__main__":
    main()