"""Helper utilities for DCC26 workshop notebooks.

All visualization and boilerplate code lives here so that notebook cells
stay short and import a stable packaged module::

    from engiopt.workshops.dcc26.notebook_helpers import *
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import random
from typing import Any

from IPython.display import display as ipy_display
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import torch as th
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

try:
    import ipywidgets as widgets
except ImportError:
    widgets = None

MIN_SHAPE_DIMS = 2
MIN_PAIRWISE_COUNT = 2

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """Set seeds for reproducibility across numpy, python, and torch."""
    random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def pick_device() -> th.device:
    """Pick the best available torch device."""
    if th.backends.mps.is_available():
        return th.device("mps")
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def ensure_dir(path: str) -> str:
    """Create a directory if needed and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: Any, path: str) -> None:
    """Serialize JSON-compatible data to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    """Load JSON data from disk."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Condition introspection
# ---------------------------------------------------------------------------


def _is_array_condition(dataset_train, key: str, n_check: int = 3) -> bool:
    """Heuristic: is this condition key an array (image) rather than a scalar?"""
    for i in range(min(n_check, len(dataset_train[key]))):
        val = dataset_train[key][i]
        arr = np.asarray(val)
        if arr.ndim >= 1 and arr.size > 1:
            return True
    return False


def _split_condition_keys(dataset_train, problem) -> tuple[list[str], list[str]]:
    """Split conditions into (scalar_keys, array_keys)."""
    scalar, array = [], []
    for k in problem.conditions_keys:
        if _is_array_condition(dataset_train, k):
            array.append(k)
        else:
            scalar.append(k)
    return scalar, array


# ---------------------------------------------------------------------------
# NB00 visualizations — uses problem.render() for everything
# ---------------------------------------------------------------------------


def show_design_gallery(
    dataset,
    problem,
    n: int = 8,
    seed: int = 7,
) -> None:
    """Show a grid of random training designs.

    Renders designs directly with imshow in a single figure to avoid
    duplicate-display issues with Jupyter's inline backend.
    """
    rng = np.random.default_rng(seed)
    train = dataset["train"]
    n_total = len(train["optimal_design"])
    n = min(n, n_total)
    ids = rng.choice(n_total, size=n, replace=False)
    scalar_keys, _ = _split_condition_keys(train, problem)

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    design_shape = problem.design_space.shape
    aspect = design_shape[1] / design_shape[0] if len(design_shape) >= MIN_SHAPE_DIMS else 1.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols * aspect, 3.5 * nrows))
    axes = np.atleast_2d(axes)

    for i, idx in enumerate(ids):
        ax = axes[i // ncols, i % ncols]
        d = np.array(train["optimal_design"][int(idx)])
        ax.imshow(d, cmap="gray_r", vmin=0, vmax=1)
        ax.axis("off")
        if scalar_keys:
            cond_str = "\n".join(f"{k}={float(train[k][int(idx)]):.2f}" for k in scalar_keys)
            ax.set_title(cond_str, fontsize=8)

    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")

    fig.suptitle(f"Random training designs ({n} samples)", fontsize=13, y=1.01)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def show_condition_distributions(dataset, problem) -> None:
    """Histogram of each scalar condition key across the training set."""
    train = dataset["train"]
    scalar_keys, _ = _split_condition_keys(train, problem)

    if not scalar_keys:
        print("This problem has no scalar conditions to plot.")
        return

    fig, axes = plt.subplots(1, len(scalar_keys), figsize=(5 * len(scalar_keys), 3.5))
    if len(scalar_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, scalar_keys):
        values = np.array([float(v) for v in train[key]])
        ax.hist(values, bins=30, edgecolor="white", color="steelblue")
        ax.set_xlabel(key, fontsize=11)
        ax.set_ylabel("count")
        ax.set_title(f"Distribution of '{key}'")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def show_valid_vs_violated(
    design,
    violations,
    valid_violations,
    *,
    problem=None,
    cmap: str = "gray_r",
) -> None:
    """Side-by-side rendering: valid config vs violated config.

    Uses ``problem.render()`` if provided, otherwise falls back to imshow.
    """
    if problem is not None:
        # Render via problem.render() for universal support
        result1 = problem.render(design)
        result2 = problem.render(design)

        fig1 = result1[0] if isinstance(result1, tuple) else result1
        fig2 = result2[0] if isinstance(result2, tuple) else result2

        if hasattr(fig1, "savefig"):
            n_valid = len(valid_violations)
            fig1.suptitle(
                f"Valid config — {n_valid} violation(s)",
                color="green" if n_valid == 0 else "red",
                fontsize=12,
            )
            plt.show()
            plt.close(fig1)

        if hasattr(fig2, "savefig"):
            n_bad = len(violations)
            fig2.suptitle(
                f"Bad config — {n_bad} violation(s)",
                color="red" if n_bad > 0 else "green",
                fontsize=12,
            )
            plt.show()
            plt.close(fig2)
    else:
        # Fallback: simple imshow for 2D
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        d = np.array(design)
        ax1.imshow(d, cmap=cmap, vmin=0, vmax=1)
        ax1.set_title(
            f"Valid — {len(valid_violations)} violation(s)", color="green" if len(valid_violations) == 0 else "red"
        )
        ax1.axis("off")
        ax2.imshow(d, cmap=cmap, vmin=0, vmax=1)
        ax2.set_title(f"Bad — {len(violations)} violation(s)", color="red" if len(violations) > 0 else "green")
        ax2.axis("off")
        fig.suptitle("Same design, different conditions", fontsize=13)
        plt.tight_layout()
        plt.show()
        plt.close(fig)


def _build_scalar_condition_sliders(
    scalar_keys: list[str],
    scalar_conds: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Create one range slider per scalar condition."""
    if widgets is None:
        return {}

    sliders: dict[str, Any] = {}
    for key in scalar_keys:
        vals = scalar_conds[key]
        lo, hi = float(np.min(vals)), float(np.max(vals))
        step = max((hi - lo) / 100.0, 1e-6)
        sliders[key] = widgets.FloatRangeSlider(
            value=[lo, hi],
            min=lo,
            max=hi,
            step=step,
            description=key,
            continuous_update=False,
            layout=widgets.Layout(width="500px"),
            style={"description_width": "130px"},
        )
    return sliders


@dataclass(slots=True)
class FilteredGalleryState:
    """Bundle gallery state so the rendering callback stays compact."""

    all_designs: np.ndarray
    scalar_keys: list[str]
    scalar_conds: dict[str, np.ndarray]
    aspect: float
    n_total: int


def _render_filtered_gallery(
    output,
    state: FilteredGalleryState,
    slider_values: dict[str, tuple[float, float]],
) -> None:
    """Render the gallery subset that matches the current slider values."""
    mask = np.ones(state.n_total, dtype=bool)
    for key in state.scalar_keys:
        lo, hi = slider_values[key]
        mask &= (state.scalar_conds[key] >= lo) & (state.scalar_conds[key] <= hi)

    matching_ids = np.where(mask)[0]
    with output:
        output.clear_output(wait=True)
        if len(matching_ids) == 0:
            print("No designs match these conditions. Widen the sliders.")
            return

        n_show = min(8, len(matching_ids))
        rng = np.random.default_rng(42)
        show_ids = rng.choice(matching_ids, size=n_show, replace=False)
        ncols = min(4, n_show)
        nrows = (n_show + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.5 * ncols * state.aspect, 3.5 * nrows),
        )
        axes = np.atleast_2d(axes)

        for i, idx in enumerate(show_ids):
            ax = axes[i // ncols, i % ncols]
            ax.imshow(state.all_designs[idx], cmap="gray_r", vmin=0, vmax=1)
            ax.axis("off")
            cond_str = "\n".join(f"{key}={state.scalar_conds[key][idx]:.2f}" for key in state.scalar_keys)
            ax.set_title(cond_str, fontsize=8)

        for i in range(n_show, nrows * ncols):
            axes[i // ncols, i % ncols].axis("off")

        fig.suptitle(
            f"Matching designs: {len(matching_ids)}/{state.n_total}",
            fontsize=12,
        )
        fig.tight_layout()
        plt.show()
        plt.close(fig)


def interactive_condition_explorer(dataset, problem) -> None:
    """Interactive slider widget to explore designs by scalar condition values.

    Falls back to a static gallery if ipywidgets is unavailable or if
    the problem has no scalar conditions.
    """
    train = dataset["train"]
    scalar_keys, _ = _split_condition_keys(train, problem)
    all_designs = np.array(train["optimal_design"])
    n_total = len(all_designs)

    if widgets is None:
        print("ipywidgets not available — showing static gallery instead.")
        show_design_gallery(dataset, problem)
        return

    # Build scalar condition arrays
    scalar_conds = {k: np.array([float(v) for v in train[k]]) for k in scalar_keys}
    sliders = _build_scalar_condition_sliders(scalar_keys, scalar_conds)
    output = widgets.Output()

    design_shape = problem.design_space.shape
    aspect = design_shape[1] / design_shape[0] if len(design_shape) >= MIN_SHAPE_DIMS else 1.0
    state = FilteredGalleryState(
        all_designs=all_designs,
        scalar_keys=scalar_keys,
        scalar_conds=scalar_conds,
        aspect=aspect,
        n_total=n_total,
    )

    if scalar_keys:
        title = widgets.HTML("<h3>Explore the dataset — drag sliders to filter by condition</h3>")
        slider_box = widgets.VBox(list(sliders.values()))

        def _current_slider_values() -> dict[str, tuple[float, float]]:
            return {key: slider.value for key, slider in sliders.items()}

        def _on_slider_change(_change) -> None:
            _render_filtered_gallery(output, state, _current_slider_values())

        for s in sliders.values():
            s.observe(_on_slider_change, names="value")

        ipy_display(title, slider_box, output)
        _render_filtered_gallery(output, state, _current_slider_values())
    else:
        # No scalar conditions (e.g., PowerElectronics)
        show_design_gallery(dataset, problem)


# ---------------------------------------------------------------------------
# NB01 training helpers
# ---------------------------------------------------------------------------


class WorkshopGenerator(th.nn.Module):
    """Thin wrapper around the EngiOpt CNN generator for the workshop.

    The CNN generator expects 4-D inputs ``(B, C, 1, 1)`` and returns
    ``(B, 1, H, W)``.  This wrapper lets callers pass flat 2-D tensors
    ``(B, C)`` and returns ``(B, H, W)`` — matching the supervised-training
    helpers that operate on numpy arrays of shape ``(N, H, W)``.
    """

    def __init__(self, cnn_generator: th.nn.Module):
        super().__init__()
        self.gen = cnn_generator

    def forward(self, z: th.Tensor, conds: th.Tensor) -> th.Tensor:
        z_4d = z.unsqueeze(-1).unsqueeze(-1)  # (B, z_dim) -> (B, z_dim, 1, 1)
        c_4d = conds.unsqueeze(-1).unsqueeze(-1)  # (B, n_c)  -> (B, n_c,  1, 1)
        out = self.gen(z_4d, c_4d)  # (B, 1, H, W)
        return out.squeeze(1)  # (B, H, W)


@dataclass(slots=True)
class TrainingConfig:
    """Training hyperparameters for the workshop generator."""

    latent_dim: int
    epochs: int = 8
    batch_size: int = 64
    lr: float = 2e-4
    device: th.device | str | None = None
    snapshot_at_epochs: list[int] | None = None
    verbose: bool = True


def train_supervised_generator(
    model,
    train_conditions: np.ndarray,
    train_targets: np.ndarray,
    config: TrainingConfig,
    snapshot_conditions: np.ndarray | None = None,
) -> dict:
    """Train a conditional generator with supervised MSE loss.

    The generator learns to map (noise, conditions) to designs by minimising
    the MSE between its outputs and real optimal designs from the dataset.

    Args:
        model: Generator network. Forward signature: model(noise, conditions).
        train_conditions: (N, n_conds) float32 array.
        train_targets: (N, *design_shape) float32 array, scaled to [-1, 1].
        config: Training hyperparameters.
        snapshot_conditions: If provided, generate designs from these conditions
            at epochs listed in ``config.snapshot_at_epochs``.

    Returns:
        Dict with keys ``losses`` (list[float]) and ``snapshots``
        (list of (epoch, np.ndarray) pairs).
    """
    device = config.device
    if device is None:
        device = th.device("cpu")
    device = th.device(device) if isinstance(device, str) else device
    model = model.to(device)
    model.train()

    optimizer = th.optim.Adam(model.parameters(), lr=config.lr)
    criterion = th.nn.MSELoss()

    conds_t = th.tensor(train_conditions, dtype=th.float32, device=device)
    targets_t = th.tensor(train_targets, dtype=th.float32, device=device)
    dl = DataLoader(
        TensorDataset(conds_t, targets_t),
        batch_size=config.batch_size,
        shuffle=True,
    )

    snap_conds_t = None
    if snapshot_conditions is not None:
        snap_conds_t = th.tensor(snapshot_conditions, dtype=th.float32, device=device)

    losses: list[float] = []
    snapshots: list[tuple[int, np.ndarray]] = []

    snapshot_epochs = config.snapshot_at_epochs or []

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_conds, batch_targets in dl:
            z = th.randn(batch_conds.shape[0], config.latent_dim, device=device)
            fake = model(z, batch_conds)
            loss = criterion(fake.flatten(1), batch_targets.flatten(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(dl)
        losses.append(avg)
        if config.verbose:
            print(f"  Epoch {epoch:3d}/{config.epochs}  |  Loss: {avg:.6f}")

        if epoch in snapshot_epochs and snap_conds_t is not None:
            model.eval()
            with th.no_grad():
                z = th.randn(len(snap_conds_t), config.latent_dim, device=device)
                snap = model(z, snap_conds_t)
                snap_np = ((snap.cpu().numpy() + 1.0) / 2.0).clip(0, 1)
            snapshots.append((epoch, snap_np))

    return {"losses": losses, "snapshots": snapshots}


def generate_designs(
    model,
    conditions: np.ndarray,
    *,
    latent_dim: int,
    device=None,
) -> np.ndarray:
    """Generate designs from conditions using a trained generator.

    Args:
        model: Trained generator model.
        conditions: (N, n_conds) float32 array.
        latent_dim: Dimensionality of the noise vector.
        device: Torch device.

    Returns:
        (N, *design_shape) float32 array in [0, 1].
    """
    if device is None:
        device = th.device("cpu")
    device = th.device(device) if isinstance(device, str) else device
    model.eval()
    conds_t = th.tensor(conditions, dtype=th.float32, device=device)
    with th.no_grad():
        z = th.randn(len(conditions), latent_dim, device=device)
        raw = model(z, conds_t)
    return ((raw.cpu().numpy() + 1.0) / 2.0).clip(0, 1)


def show_training_progression(
    snapshots: list[tuple[int, np.ndarray]],
    baseline_designs: np.ndarray | None = None,
    n_show: int = 4,
) -> None:
    """Visualize how generated designs evolve across training epochs.

    Args:
        snapshots: List of (epoch, designs_array) tuples from training.
        baseline_designs: If provided, show ground-truth in the last row.
        n_show: Number of designs to display per row.
    """
    if not snapshots:
        print("No snapshots to display.")
        return

    n_show = min(n_show, snapshots[0][1].shape[0])
    n_rows = len(snapshots) + (1 if baseline_designs is not None else 0)

    fig, axes = plt.subplots(n_rows, n_show, figsize=(2.8 * n_show, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_show == 1:
        axes = axes[:, np.newaxis]

    for row, (epoch, designs) in enumerate(snapshots):
        for col in range(n_show):
            axes[row, col].imshow(designs[col], cmap="gray_r", vmin=0, vmax=1)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(
            f"Epoch {epoch}",
            fontsize=11,
            rotation=0,
            labelpad=55,
            va="center",
        )

    if baseline_designs is not None:
        row = len(snapshots)
        for col in range(n_show):
            axes[row, col].imshow(baseline_designs[col], cmap="gray_r", vmin=0, vmax=1)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(
            "Ground\ntruth",
            fontsize=11,
            rotation=0,
            labelpad=55,
            va="center",
        )

    fig.suptitle("How the generator learns over training", fontsize=14, y=1.02)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# NB01 visualizations
# ---------------------------------------------------------------------------


def show_training_curve(train_losses: list[float], save_path: str | None = None) -> None:
    """Plot training loss over epochs."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(range(1, len(train_losses) + 1), train_losses, marker="o", linewidth=2, color="#2563eb")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Generator Training Loss", fontsize=14)
    ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.show()
    print(f"Final loss: {train_losses[-1]:.6f}")


def show_gen_vs_baseline(
    gen_designs: np.ndarray,
    baseline_designs: np.ndarray,
    conditions_records: list[dict],
    n_show: int = 8,
    problem=None,
) -> None:
    """Show generated vs baseline designs using ``problem.render()`` if available."""
    n_show = min(n_show, len(gen_designs))

    if problem is not None:
        for i in range(n_show):
            # Scalar conditions only for title
            scalars = {
                k: v
                for k, v in conditions_records[i].items()
                if not isinstance(v, (list, np.ndarray)) or np.asarray(v).size == 1
            }
            cond_str = "  |  ".join(f"{k}: {float(v):.3f}" for k, v in scalars.items())

            result_g = problem.render(gen_designs[i])
            fig_g = result_g[0] if isinstance(result_g, tuple) else result_g
            if hasattr(fig_g, "savefig"):
                fig_g.suptitle(f"Generated {i}  —  {cond_str}", fontsize=10, y=1.02)
                plt.show()
                plt.close(fig_g)

            result_b = problem.render(baseline_designs[i])
            fig_b = result_b[0] if isinstance(result_b, tuple) else result_b
            if hasattr(fig_b, "savefig"):
                fig_b.suptitle(f"Baseline {i}", fontsize=10, y=1.02)
                plt.show()
                plt.close(fig_b)
    else:
        # Fallback: side-by-side imshow grid
        fig, axes = plt.subplots(2, n_show, figsize=(2.2 * n_show, 5.5))
        if n_show == 1:
            axes = axes.reshape(2, 1)
        for i in range(n_show):
            axes[0, i].imshow(gen_designs[i], cmap="gray", vmin=0, vmax=1)
            axes[0, i].axis("off")
            scalars = {
                key: value
                for key, value in conditions_records[i].items()
                if not isinstance(value, (list, np.ndarray)) or np.asarray(value).size == 1
            }
            cond_str = "\n".join(f"{key}: {float(value):.3f}" for key, value in scalars.items())
            axes[0, i].set_title(cond_str, fontsize=8)
            axes[1, i].imshow(baseline_designs[i], cmap="gray", vmin=0, vmax=1)
            axes[1, i].axis("off")
        axes[0, 0].set_ylabel("Generated", fontsize=12, rotation=90, labelpad=10)
        axes[1, 0].set_ylabel("Baseline", fontsize=12, rotation=90, labelpad=10)
        fig.suptitle("Generated (top) vs Baseline (bottom)", fontsize=14, y=1.01)
        fig.tight_layout()
        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# NB02 visualizations
# ---------------------------------------------------------------------------


def show_objective_comparison(results) -> None:
    """Histogram + scatter of generated vs baseline objectives."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.hist(results["gen_obj"], bins=10, alpha=0.7, label="Generated", color="#4C72B0")
    ax1.hist(results["base_obj"], bins=10, alpha=0.7, label="Baseline", color="#DD8452")
    ax1.set_xlabel("Objective (lower is better)")
    ax1.set_ylabel("Count")
    ax1.set_title("Objective distribution")
    ax1.legend()

    colors = results["gen_feasible"].map({True: "#55A868", False: "#C44E52"})
    ax2.scatter(results["base_obj"], results["gen_obj"], alpha=0.8, c=colors, edgecolors="black", linewidths=0.5, s=60)
    lo = min(results["base_obj"].min(), results["gen_obj"].min()) * 0.9
    hi = max(results["base_obj"].max(), results["gen_obj"].max()) * 1.1
    ax2.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, label="y = x")
    ax2.set_xlabel("Baseline objective")
    ax2.set_ylabel("Generated objective")
    ax2.set_title("Per-sample (green=feasible, red=infeasible)")
    ax2.legend()

    fig.tight_layout()
    plt.show()
    plt.close(fig)


def show_feasibility_bars(results) -> None:
    """Bar chart comparing feasibility rates."""
    gen_rate = results["gen_feasible"].mean()
    base_rate = results["base_feasible"].mean()

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Generated", "Baseline"], [gen_rate, base_rate], color=["#4C72B0", "#DD8452"], edgecolor="black")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Feasible fraction")
    ax.set_title("Feasibility rate")
    for bar, val in zip(bars, [gen_rate, base_rate]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, f"{val:.0%}", ha="center", fontweight="bold")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def show_design_comparison_grid(gen_designs, baseline_designs, results, n_show: int = 6, problem=None) -> None:
    """Show generated vs baseline with feasibility annotations.

    Uses ``problem.render()`` if available, otherwise falls back to imshow.
    """
    n_show = min(n_show, len(gen_designs))

    if problem is not None:
        for i in range(n_show):
            feas = "FEASIBLE" if results.iloc[i]["gen_feasible"] else "INFEASIBLE"
            gap = results.iloc[i]["gen_minus_base"]

            result_g = problem.render(gen_designs[i])
            fig_g = result_g[0] if isinstance(result_g, tuple) else result_g
            if hasattr(fig_g, "savefig"):
                color = "green" if results.iloc[i]["gen_feasible"] else "red"
                fig_g.suptitle(f"Generated {i} — {feas}  (gap={gap:.1f})", fontsize=11, color=color, y=1.02)
                plt.show()
                plt.close(fig_g)
    else:
        # Fallback: imshow grid
        fig, axes = plt.subplots(2, n_show, figsize=(2.5 * n_show, 5))
        if n_show == 1:
            axes = axes[:, None]
        for i in range(n_show):
            axes[0, i].imshow(gen_designs[i], cmap="gray", vmin=0, vmax=1, aspect="auto")
            feas = "FEASIBLE" if results.iloc[i]["gen_feasible"] else "INFEASIBLE"
            color = "green" if results.iloc[i]["gen_feasible"] else "red"
            axes[0, i].set_title(f"Gen {i}\n{feas}", fontsize=9, color=color, fontweight="bold")
            axes[0, i].axis("off")
            axes[1, i].imshow(baseline_designs[i], cmap="gray", vmin=0, vmax=1, aspect="auto")
            axes[1, i].set_title(f"Baseline {i}", fontsize=9)
            axes[1, i].axis("off")
        fig.suptitle("Generated vs Baseline", fontsize=13, y=1.02)
        fig.tight_layout()
        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Metric helpers (NB02)
# ---------------------------------------------------------------------------


def mean_pairwise_l2(designs: np.ndarray) -> float:
    """Average L2 distance between all pairs. Measures intra-set diversity."""
    flat = designs.reshape(designs.shape[0], -1)
    n = flat.shape[0]
    if n < MIN_PAIRWISE_COUNT:
        return 0.0
    pairwise = cdist(flat, flat, metric="euclidean")
    upper_triangle = pairwise[np.triu_indices(n, k=1)]
    return float(np.mean(upper_triangle))


def mean_nn_distance_to_reference(designs: np.ndarray, reference: np.ndarray) -> float:
    """Average nearest-neighbor distance to a reference set. Measures novelty."""
    q = designs.reshape(designs.shape[0], -1)
    r = reference.reshape(reference.shape[0], -1)
    nn_dists = []
    for i in range(q.shape[0]):
        d = np.linalg.norm(r - q[i][None, :], axis=1)
        nn_dists.append(float(np.min(d)))
    return float(np.mean(nn_dists))


# ---------------------------------------------------------------------------
# NB02 enhanced visualizations — pedagogical metric exploration
# ---------------------------------------------------------------------------


def show_residual_heatmaps(
    gen_designs: np.ndarray,
    baseline_designs: np.ndarray,
    n_show: int = 6,
) -> None:
    """Pixel-wise absolute difference between generated and baseline designs.

    Three rows: generated, baseline, |residual|.  The residual row uses a
    ``Reds`` colourmap (white = no error, dark red = large error).
    """
    n_show = min(n_show, len(gen_designs))
    fig, axes = plt.subplots(3, n_show, figsize=(2.8 * n_show, 7))
    if n_show == 1:
        axes = axes[:, None]

    for i in range(n_show):
        axes[0, i].imshow(gen_designs[i], cmap="gray_r", vmin=0, vmax=1)
        axes[0, i].set_title(f"Gen {i}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(baseline_designs[i], cmap="gray_r", vmin=0, vmax=1)
        axes[1, i].set_title(f"Baseline {i}", fontsize=9)
        axes[1, i].axis("off")

        diff = np.abs(gen_designs[i].astype(float) - baseline_designs[i].astype(float))
        axes[2, i].imshow(diff, cmap="Reds", vmin=0, vmax=1)
        axes[2, i].set_title(f"|diff| mean={diff.mean():.3f}", fontsize=8)
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Generated", fontsize=11, rotation=90, labelpad=10)
    axes[1, 0].set_ylabel("Baseline", fontsize=11, rotation=90, labelpad=10)
    axes[2, 0].set_ylabel("|Residual|", fontsize=11, rotation=90, labelpad=10)
    fig.suptitle(
        "Pixel-level residuals: where do generated designs differ from baselines?",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def show_objective_residuals(results) -> None:
    """Per-sample bar chart of objective gap (generated - baseline).

    Bars above zero mean the generated design is *worse* (higher compliance).
    """
    gaps = results["gen_minus_base"]
    colors = ["#C44E52" if g > 0 else "#55A868" for g in gaps]

    fig, ax = plt.subplots(figsize=(max(8, len(gaps) * 0.45), 4))
    ax.bar(range(len(gaps)), gaps, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Objective gap (gen - baseline)")
    ax.set_title(
        "Per-sample objective residuals  (green = generated is better, red = worse)",
        fontsize=11,
    )
    ax.set_xticks(range(len(gaps)))
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def show_volfrac_analysis(results, volfrac_tol: float = 0.05) -> None:
    """Volume-fraction target vs actual scatter + error distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = results["gen_feasible"].map({True: "#55A868", False: "#C44E52"})
    ax1.scatter(
        results["target_volfrac"],
        results["gen_volfrac"],
        c=colors,
        edgecolors="black",
        linewidths=0.5,
        s=60,
        alpha=0.8,
    )
    lo = min(results["target_volfrac"].min(), results["gen_volfrac"].min()) - 0.05
    hi = max(results["target_volfrac"].max(), results["gen_volfrac"].max()) + 0.05
    xs = np.linspace(lo, hi, 100)
    ax1.plot(xs, xs, "--", color="gray", linewidth=1, label="Perfect match")
    ax1.fill_between(
        xs,
        xs - volfrac_tol,
        xs + volfrac_tol,
        alpha=0.12,
        color="green",
        label=f"Tolerance (\u00b1{volfrac_tol})",
    )
    ax1.set_xlabel("Target volume fraction")
    ax1.set_ylabel("Generated volume fraction")
    ax1.set_title("Constraint satisfaction: target vs actual volfrac")
    ax1.legend(fontsize=9)

    errors = results["gen_volfrac"] - results["target_volfrac"]
    ax2.hist(errors, bins=15, edgecolor="white", color="#4C72B0", alpha=0.8)
    ax2.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax2.axvline(-volfrac_tol, color="red", linestyle=":", linewidth=1.5, label=f"\u00b1{volfrac_tol}")
    ax2.axvline(volfrac_tol, color="red", linestyle=":", linewidth=1.5)
    ax2.set_xlabel("Volume fraction error (actual \u2212 target)")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of constraint errors")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    plt.show()
    plt.close(fig)


def show_spatial_distribution_comparison(
    gen_designs: np.ndarray,
    baseline_designs: np.ndarray,
    train_reference: np.ndarray | None = None,
) -> None:
    """Compare *where* material is placed on average across design sets.

    For binary/near-binary topology designs, pixel-intensity histograms are
    uninformative (just two spikes at 0 and 1).  Instead we show:

    - **Mean design images**: the average design across each set, revealing
      where material tends to be placed.  Differences highlight spatial
      biases in the generator.
    - **Per-design volume fraction distributions**: how much total material
      each design uses, compared across sets.
    """
    has_train = train_reference is not None
    n_img = 3 if has_train else 2

    fig, axes = plt.subplots(
        1,
        n_img + 1,
        figsize=(4.2 * (n_img + 1), 4),
        gridspec_kw={"width_ratios": [1] * n_img + [1.3]},
        constrained_layout=True,
    )

    # ── Mean design images ───────────────────────────────────────────
    sets: list[tuple[str, np.ndarray, str]] = [
        ("Generated", gen_designs, "#4C72B0"),
        ("Baseline", baseline_designs, "#DD8452"),
    ]
    if has_train:
        assert train_reference is not None
        sets.append(("Training", train_reference, "#55A868"))

    vmin, vmax = 0, 1
    for ax, (label, designs, _color) in zip(axes[:n_img], sets):
        mean_img = designs.mean(axis=0)
        im = ax.imshow(mean_img, cmap="gray_r", vmin=vmin, vmax=vmax)
        ax.set_title(f"Mean {label}\n(n={len(designs)})", fontsize=11)
        ax.axis("off")
    fig.colorbar(im, ax=axes[:n_img].tolist(), shrink=0.75, label="Avg. material density", pad=0.04)

    # ── Per-design volume fraction distributions ─────────────────────
    ax_vf = axes[n_img]
    for label, designs, color in sets:
        vfracs = designs.reshape(designs.shape[0], -1).mean(axis=1)
        ax_vf.hist(vfracs, bins=25, alpha=0.5, density=True, label=label, color=color, edgecolor="white", linewidth=0.3)
    ax_vf.set_xlabel("Volume fraction (per design)")
    ax_vf.set_ylabel("Density")
    ax_vf.set_title("Material-usage\ndistributions", fontsize=11)
    ax_vf.legend(fontsize=9)

    fig.suptitle(
        "Spatial distribution comparison \u2014 where does each set place material?",
        fontsize=13,
        y=1.03,
    )
    plt.show()
    plt.close(fig)


def show_mmd_comparison_bar(
    mmd_gen_base: float,
    mmd_train_base: float,
    mmd_random_base: float,
) -> None:
    """Bar chart placing the generator's MMD in context.

    Shows three reference points so the raw MMD number becomes interpretable:
    - Generated vs baseline (our metric -- same conditions)
    - Train sample vs baseline (retrieval baseline -- no conditioning)
    - Random vs baseline (upper bound / worst case)
    """
    labels = [
        "Generated\nvs Baseline",
        "Train sample\nvs Baseline\n(no conditioning)",
        "Random\nvs Baseline\n(worst case)",
    ]
    values = [mmd_gen_base, mmd_train_base, mmd_random_base]
    colors = ["#4C72B0", "#DD8452", "#C44E52"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5, width=0.55)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(abs(v) for v in values) * 0.02,
            f"{v:.4f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("MMD (lower = more similar)")
    ax.set_title("MMD in context \u2014 where does the generator sit?", fontsize=13)
    ax.set_ylim(0, max(values) * 1.25)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def show_pairwise_distance_heatmap(
    designs: np.ndarray,
    title: str = "Pairwise L2 distance among generated designs",
) -> None:
    """Heatmap of pairwise L2 distances — visual proxy for diversity.

    A uniform warm colour off-diagonal means all designs differ roughly
    equally (good diversity).  Cool/dark blocks reveal clusters of
    near-identical designs (partial mode collapse).
    """
    flat = designs.reshape(designs.shape[0], -1)
    dists = cdist(flat, flat, "euclidean")

    fig, (ax, ax_hist) = plt.subplots(
        1,
        2,
        figsize=(11, 5),
        gridspec_kw={"width_ratios": [1.2, 1]},
    )

    im = ax.imshow(dists, cmap="viridis")
    ax.set_xlabel("Design index")
    ax.set_ylabel("Design index")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, label="L2 distance", fraction=0.046, pad=0.04)

    # Histogram of off-diagonal distances
    triu_idx = np.triu_indices(len(designs), k=1)
    off_diag = dists[triu_idx]
    ax_hist.hist(off_diag, bins=25, edgecolor="white", color="#4C72B0", alpha=0.8)
    ax_hist.axvline(off_diag.mean(), color="#C44E52", linewidth=2, linestyle="--", label=f"Mean = {off_diag.mean():.1f}")
    ax_hist.set_xlabel("Pairwise L2 distance")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Distribution of pairwise distances", fontsize=11)
    ax_hist.legend(fontsize=9)

    fig.tight_layout()
    plt.show()
    plt.close(fig)


def show_embedding_scatter(
    gen_designs: np.ndarray,
    baseline_designs: np.ndarray,
    train_reference: np.ndarray,
) -> None:
    """PCA 2-D projection of generated, baseline, and training designs.

    Uses numpy SVD so there is no sklearn dependency.
    """
    g = gen_designs.reshape(gen_designs.shape[0], -1).astype(np.float64)
    b = baseline_designs.reshape(baseline_designs.shape[0], -1).astype(np.float64)
    t = train_reference.reshape(train_reference.shape[0], -1).astype(np.float64)

    combined = np.vstack([g, b, t])
    mean = combined.mean(axis=0)
    centered = combined - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T

    ng, nb = len(g), len(b)
    pg, pb, pt = proj[:ng], proj[ng : ng + nb], proj[ng + nb :]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(pt[:, 0], pt[:, 1], alpha=0.15, s=15, c="#AAAAAA", label=f"Training ({len(t)})")
    ax.scatter(
        pb[:, 0],
        pb[:, 1],
        alpha=0.7,
        s=50,
        c="#DD8452",
        edgecolors="black",
        linewidths=0.5,
        label=f"Baseline ({nb})",
        marker="s",
    )
    ax.scatter(
        pg[:, 0],
        pg[:, 1],
        alpha=0.8,
        s=60,
        c="#4C72B0",
        edgecolors="black",
        linewidths=0.5,
        label=f"Generated ({ng})",
        marker="o",
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("PCA projection \u2014 where do generated designs live in design space?", fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def show_optimization_trajectories(opt_data: list[dict]) -> None:
    """Plot optimization trajectories showing how generated designs warmstart optimization.

    Each entry in *opt_data* should be a dict with keys:
    ``sample_idx``, ``obj_trajectory`` (list of floats), ``base_obj`` (float).
    """
    n = len(opt_data)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)

    for i, d in enumerate(opt_data):
        ax = axes[0, i]
        objs = d["obj_trajectory"]
        steps = list(range(len(objs)))
        base = d["base_obj"]

        ax.plot(steps, objs, "o-", color="#4C72B0", linewidth=2, markersize=4, label="Optimizer")
        ax.axhline(base, color="#DD8452", linestyle="--", linewidth=1.5, label=f"Baseline = {base:.1f}")
        ax.fill_between(steps, objs, base, alpha=0.12, color="#4C72B0")

        iog = objs[0] - base
        fog = objs[-1] - base
        cog = sum(o - base for o in objs)

        ax.set_title(
            f"Sample {d['sample_idx']}\nIOG={iog:.1f}   FOG={fog:.1f}   COG={cog:.1f}",
            fontsize=10,
        )
        ax.set_xlabel("Optimization step")
        ax.set_ylabel("Objective (compliance)")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Optimization from generated warmstarts \u2014 does the model give the optimizer a head start?",
        fontsize=13,
        y=1.05,
    )
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def show_metric_summary_dashboard(summary_dict: dict) -> None:
    """Multi-panel grouped bar chart summarizing all metric categories."""
    categories = {
        "Simulation\nPerformance": [
            ("Obj gap (gen\u2212base)", summary_dict.get("objective_gap_mean", 0)),
            ("Improvement rate", summary_dict.get("improvement_rate", 0)),
        ],
        "Constraint\nSatisfaction": [
            ("Gen feasible %", summary_dict.get("gen_feasible_rate", 0)),
            ("Base feasible %", summary_dict.get("base_feasible_rate", 0)),
        ],
        "Distributional\nSimilarity": [
            ("MMD", summary_dict.get("mmd", 0)),
        ],
        "Diversity &\nNovelty": [
            ("Diversity (L2)", summary_dict.get("gen_diversity_l2", 0)),
            ("Novelty (NN)", summary_dict.get("gen_novelty_to_train_l2", 0)),
        ],
    }

    fig, axes = plt.subplots(1, len(categories), figsize=(4 * len(categories), 4.5))
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for ax, (cat_name, metrics), color in zip(axes, categories.items(), palette):
        names = [m[0] for m in metrics]
        vals = [m[1] for m in metrics]
        bars = ax.barh(names, vals, color=color, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_width() + max(abs(v) for v in vals) * 0.03,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}" if abs(v) < 1 else f"{v:.1f}",
                va="center",
                fontsize=9,
            )
        ax.set_title(cat_name, fontsize=11, fontweight="bold")
        ax.set_xlim(left=min(0, min(vals) * 1.2))

    fig.suptitle("Evaluation dashboard \u2014 how does the generator perform?", fontsize=14, y=1.03)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
