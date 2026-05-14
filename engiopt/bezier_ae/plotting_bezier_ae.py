import os
import matplotlib.pyplot as plt
import torch


RESULTS_DIR = "results/bezier_ae"


def _to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


def plot_reconstruction(x_true, x_pred, title="Reconstruction", filename=None, run_dir=None):
    x_true = _to_cpu(x_true)
    x_pred = _to_cpu(x_pred)

    plt.figure(figsize=(6, 6))
    plt.plot(x_true[0], x_true[1], label="Ground Truth")
    plt.plot(x_pred[0], x_pred[1], label="Prediction")
    plt.legend()
    plt.axis("equal")
    plt.title(title)

    if filename is not None:
        base_dir = run_dir if run_dir is not None else RESULTS_DIR
        save_path = os.path.join(base_dir, "reconstructions", filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_control_polygon(cp, title="Control Polygon", filename=None, run_dir=None):
    cp = _to_cpu(cp)

    plt.figure(figsize=(6, 6))
    plt.plot(cp[0], cp[1], marker="o", linestyle="--", label="Control Polygon")
    plt.legend()
    plt.axis("equal")
    plt.title(title)

    if filename is not None:
        base_dir = run_dir if run_dir is not None else RESULTS_DIR
        save_path = os.path.join(base_dir, "control_polygons", filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_reconstruction_with_control_points(
    x_true,
    x_pred,
    cp,
    title="Reconstruction with Control Points",
    filename=None,
    run_dir=None,
):
    x_true = _to_cpu(x_true)
    x_pred = _to_cpu(x_pred)
    cp = _to_cpu(cp)

    plt.figure(figsize=(6, 6))
    plt.plot(x_true[0], x_true[1], label="Ground Truth")
    plt.plot(x_pred[0], x_pred[1], label="Prediction")
    plt.plot(cp[0], cp[1], marker="o", linestyle="--", label="Control Polygon")
    plt.legend()
    plt.axis("equal")
    plt.title(title)

    if filename is not None:
        base_dir = run_dir if run_dir is not None else RESULTS_DIR
        save_path = os.path.join(base_dir, "reconstructions", filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()