"""
Visualisation helpers for the LAE.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import scienceplots
    plt.style.use('science')
    plt.rcParams['text.usetex'] = False  # cluster has no LaTeX
except ImportError:
    pass  # scienceplots is optional


# ---------------------------------------------------------------------------
# Low-level image helpers
# ---------------------------------------------------------------------------

def show_tensor_image(image, cpw_mode=False, s=0.5, ax=None):
    d = 3
    if len(image.shape) == d:
        image = image[0, :, :].cpu().numpy()

    c    = None
    cmap = None
    if cpw_mode:
        cmap = 'viridis'
        c    = image[0, :]
    else:
        c = 'k'

    if ax is not None:
        ax.scatter(image[0, :], image[1, :], s=s, c=c, cmap=cmap)
    else:
        plt.scatter(image[0, :], image[1, :], s=s, c=c, cmap=cmap)


def wing_2D_shape_plot(wing, cpw_mode=False, s=0.5, axs=None, dpi=400):
    d = 4
    if len(wing.shape) == d:
        wing = wing[0, :, :].cpu().numpy()
    if isinstance(wing, torch.Tensor):
        wing = wing.cpu().numpy()

    if axs is None:
        _, axs = plt.subplots(2, 1, figsize=(20, 5), dpi=dpi)

    c    = None
    cmap = None
    if cpw_mode:
        cmap = 'viridis'
        c    = wing[0, 0, :]
        x1, y1 = wing[0, 1, :], wing[0, 2, :]
        x2, y2 = wing[-1, 1, :], wing[-1, 2, :]
    else:
        c = 'k'
        x1, y1 = wing[0, 0, :], wing[0, 1, :]
        x2, y2 = wing[-1, 0, :], wing[-1, 1, :]

    axs[0].scatter(x1, y1, s=s, c=c, cmap=cmap)
    axs[1].scatter(x2, y2, s=s, c=c, cmap=cmap)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=14)


def wing_3D_shape_plot(wing, cpw_mode=False, wing_len=2.25, ax=None,
                       alpha=1.0, facecolor='grey', dpi=400, z=None, slice_mode=False):
    d = 4
    if len(wing.shape) == d:
        wing = wing[0, :, :].cpu().numpy()
    if isinstance(wing, torch.Tensor):
        wing = wing.cpu().numpy()

    if ax is None:
        fig = plt.figure(figsize=(20, 5), dpi=dpi)
        ax  = fig.add_subplot(projection='3d')

    if z is None:
        z = np.linspace(0, wing_len, wing.shape[0])

    z_len = np.ones_like(wing[0, 1])
    if cpw_mode:
        for i in range(z.shape[0]):
            ax.scatter(wing[i, 1, :], z[i], wing[i, 2, :], s=0.5,
                       c=wing[i, 0, :], cmap='viridis')
    else:
        for i in range(z.shape[0] - 1):
            if slice_mode:
                ax.plot(wing[i, 0, :], z[i] * z_len, wing[i, 1, :],
                        '-', color=facecolor, alpha=alpha, linewidth=2.0)
                if i == z.shape[0] - 2:
                    ax.plot(wing[i + 1, 0, :], z[i + 1] * z_len, wing[i + 1, 1, :],
                            '-', color=facecolor, alpha=alpha, linewidth=2.0)
            else:
                ax.fill_between(
                    wing[i, 0, :].astype(float),
                    z[i].astype(float) * z_len,
                    wing[i, 1, :].astype(float),
                    wing[i + 1, 0, :].astype(float),
                    z[i + 1].astype(float) * z_len,
                    wing[i + 1, 1, :].astype(float),
                    facecolor=facecolor, alpha=alpha,
                )
    ax.set_aspect('auto')


# ---------------------------------------------------------------------------
# LAE-specific plots
# ---------------------------------------------------------------------------

def plot_loss_curves(stats: dict, save_path: str = None):
    """Plot all LAE training/test loss components.

    Components shown:
        - Total loss (train + test)
        - Geometry latent reconstruction
        - Angle-of-attack reconstruction
        - Trailing-edge y-shift (η_y) reconstruction
        - Pressure coefficient (Cp) reconstruction
        - Aerodynamic performance (cd, cl) reconstruction
        - Least-volume penalty
        - Active latent dimension count over epochs
    """
    epochs_test = stats.get('test_loss_epoch', np.array([]))

    fig, axes = plt.subplots(2, 4, figsize=(22, 8))
    axes = axes.flatten()

    # ── Total loss ──────────────────────────────────────────────────────
    ax = axes[0]
    if len(stats.get('train_loss', [])) > 0:
        ax.plot(stats['train_loss_epoch'], stats['train_loss'],
                label='train', alpha=0.6, linewidth=0.8)
    if len(stats.get('test_loss', [])) > 0:
        ax.plot(epochs_test, stats['test_loss'], label='test', linewidth=1.2)
    ax.set_title('Total loss')
    ax.set_xlabel('epoch')
    ax.legend()

    # ── Reconstruction components ────────────────────────────────────────
    recon_keys = [
        ('test_loss_recon',    'Geometry latent'),
        ('test_loss_alpha',    'AoA (α)'),
        ('test_loss_eta_y',    'η_y (TE shift)'),
        ('test_loss_pressure', 'Pressure (Cp)'),
        ('test_loss_perf',     'Performance (cd, cl)'),
    ]
    for i, (key, label) in enumerate(recon_keys, start=1):
        ax = axes[i]
        data = stats.get(key, np.array([]))
        if len(data) > 0:
            ax.plot(epochs_test, data, linewidth=1.2)
        ax.set_title(label)
        ax.set_xlabel('epoch')

    # ── Least-volume penalty ─────────────────────────────────────────────
    ax = axes[6]
    lv_data = stats.get('test_loss_lv', np.array([]))
    if len(lv_data) > 0:
        ax.plot(epochs_test, lv_data, color='darkorange', linewidth=1.2)
    ax.set_title('Least-volume penalty')
    ax.set_xlabel('epoch')

    # ── Active latent dimensions ─────────────────────────────────────────
    ax = axes[7]
    n_active_data = stats.get('test_n_active_dims', np.array([]))
    if len(n_active_data) > 0:
        ax.plot(epochs_test, n_active_data, color='steelblue',
                drawstyle='steps-post', linewidth=1.2)
    ax.set_title('Active latent dims')
    ax.set_xlabel('epoch')
    ax.set_ylabel('count')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_latent_usage(stats: dict, lae_latent_dim: int = None, save_path: str = None):
    """Plot the active latent dimension count over training epochs."""
    epochs_test   = stats.get('test_loss_epoch', np.array([]))
    n_active_data = stats.get('test_n_active_dims', np.array([]))

    if len(n_active_data) == 0:
        print("No latent-usage stats recorded yet.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs_test, n_active_data, color='steelblue',
            drawstyle='steps-post', linewidth=1.5, label='active dims')
    if lae_latent_dim is not None:
        ax.axhline(lae_latent_dim, color='grey', linestyle='--',
                   linewidth=0.8, label=f'total dims ({lae_latent_dim})')
    ax.set_title('Active latent dimensions over training')
    ax.set_xlabel('epoch')
    ax.set_ylabel('# active dims')
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_reconstruction(original, reconstructed, bae_model, idx=0,
                        save_path=None, device='cpu'):
    """Side-by-side 3-D plots of original vs reconstructed wing.

    Args:
        original:      Tensor [B, 9, latent_channels, L]  (BAE latents)
        reconstructed: Tensor [B, 9, latent_channels, L]
        bae_model:     Frozen BezierAutoencoder.
        idx:           Sample index within the batch to plot.
    """
    bae_model.eval()
    bae_model.to(device)

    def decode_wing(z_wing):
        slices = []
        with torch.no_grad():
            for s in range(z_wing.shape[0]):
                z_s = z_wing[s].unsqueeze(0).to(device)
                dec = bae_model.decode_z(z_s, z_ae_mode=True,
                                          denormalize_output=False,
                                          normalized_data=True)[0]
                slices.append(dec.squeeze(0).cpu())
        return torch.stack(slices)  # [9, 2, 192]

    orig_coords  = decode_wing(original[idx])
    recon_coords = decode_wing(reconstructed[idx])

    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    wing_3D_shape_plot(orig_coords,  ax=ax1, facecolor='steelblue', alpha=0.8, slice_mode=True)
    wing_3D_shape_plot(recon_coords, ax=ax2, facecolor='coral',     alpha=0.8, slice_mode=True)

    ax1.set_title('Original')
    ax2.set_title('Reconstructed (LAE)')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_generated_pressure(pressure: torch.Tensor, sample_idx: int = 0,
                             save_path: str = None):
    """Visualise the generated pressure coefficient (Cp) distribution over slices.

    This is a standalone generated-sample plot — no ground truth is needed or shown.

    Args:
        pressure:   [9, 192] or [N, 9, 192]  (de-normalised Cp values).
        sample_idx: If *pressure* is [N, 9, 192], which sample to plot.
        save_path:  If given, save to this path instead of displaying.
    """
    if pressure.ndim == 3:
        pressure = pressure[sample_idx]     # [9, 192]
    pres_np  = pressure.cpu().numpy()
    n_slices = pres_np.shape[0]
    x        = np.arange(pres_np.shape[1])

    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharey=False)
    axes      = axes.flatten()

    for s in range(n_slices):
        ax = axes[s]
        ax.plot(x, pres_np[s], linewidth=1.0, color='steelblue')
        ax.set_title(f'Slice {s}')
        ax.set_xlabel('point index')
        ax.set_ylabel('Cp')

    plt.suptitle(f'Generated pressure coefficient (Cp) — sample {sample_idx}')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_pressure_comparison(gen_pressure: torch.Tensor, gt_pressure: torch.Tensor,
                              sample_idx: int = 0, save_path: str = None):
    """Compare predicted vs ground-truth pressure coefficients (Cp) for one sample.

    Args:
        gen_pressure: [N, 9, 192] or [9, 192]  (predicted Cp, de-normalised)
        gt_pressure:  [N, 9, 192] or [9, 192]  (ground-truth Cp)
        sample_idx:   Index into the batch dimension (ignored if input is 2-D).
    """
    if gen_pressure.ndim == 3:
        gen_pressure = gen_pressure[sample_idx]   # [9, 192]
        gt_pressure  = gt_pressure[sample_idx]

    gen_np = gen_pressure.cpu().numpy()
    gt_np  = gt_pressure.cpu().numpy()
    n_slices = gen_np.shape[0]

    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharey=False)
    axes = axes.flatten()
    x    = np.arange(gen_np.shape[1])

    for s in range(n_slices):
        ax = axes[s]
        ax.plot(x, gt_np[s],  label='GT',   linewidth=1.0, alpha=0.8)
        ax.plot(x, gen_np[s], label='Pred', linewidth=1.0, alpha=0.8, linestyle='--')
        ax.set_title(f'Slice {s}')
        ax.set_xlabel('point index')
        ax.set_ylabel('Cp')
        if s == 0:
            ax.legend(fontsize=8)

    plt.suptitle('Pressure coefficient (Cp): predicted vs ground truth')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()



def plot_latent_dim_std(stats: dict, save_path: str = None):
    """Plot per-dimension latent std over training epochs (heatmap).

    Requires ``test_latent_dim_std`` to be recorded in *stats*
    (shape [n_eval_points, lae_latent_dim]).

    Shows whether the least-volume penalty is actually compressing individual
    dimensions or just wobbling without reducing the occupied volume.
    """
    dim_std = stats.get('test_latent_dim_std', None)
    if dim_std is None or len(dim_std) == 0:
        print("No per-dimension std data in stats ('test_latent_dim_std' missing).")
        return

    dim_std   = np.array(dim_std)           # [T, D]
    epochs    = stats.get('test_loss_epoch', np.arange(len(dim_std)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    # Heatmap: time × dimension
    ax = axes[0]
    im = ax.imshow(dim_std.T, aspect='auto', origin='lower', cmap='viridis',
                   extent=[epochs[0], epochs[-1], 0, dim_std.shape[1]])
    plt.colorbar(im, ax=ax, label='std')
    ax.set_title('Per-dimension latent std over training')
    ax.set_xlabel('epoch')
    ax.set_ylabel('latent dimension')

    # Mean std per dimension (final snapshot)
    ax = axes[1]
    final_std = dim_std[-1]
    ax.bar(np.arange(len(final_std)), final_std, color='steelblue', width=0.8)
    ax.set_title('Per-dimension std (final epoch)')
    ax.set_xlabel('latent dimension')
    ax.set_ylabel('std')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_airfoil_cp_comparison(
    gen_airfoils: torch.Tensor,
    gt_airfoils: torch.Tensor,
    gen_pressures: torch.Tensor,
    gt_pressures: torch.Tensor,
    gen_perfs=None,
    gt_perfs=None,
    sample_indices=None,
    slice_indices=None,
    save_path: str = None,
    pred_label: str = "Generated",
    flow_conditions=None,   # list of dicts with keys mach, reynolds, cl_target, one per sample
):
    """Plot airfoil shape + Cp distribution side-by-side for root/mid/tip slices.

    For each sample shows:
      - Row 1: 2-D airfoil scatter (GT blue, pred coral)
      - Row 2: Cp vs chord-x split into upper/lower surface
      - Row 3 (if perfs provided): cd and cl bar chart (GT vs pred)

    Args:
        gen_airfoils:   [N, 9, 2, 192]  predicted/generated coordinates.
        gt_airfoils:    [N, 9, 2, 192]  ground-truth coordinates.
        gen_pressures:  [N, 9, 192]     predicted Cp (de-normalised).
        gt_pressures:   [N, 9, 192]     ground-truth Cp (de-normalised).
        gen_perfs:      [N, 2]          predicted (cd, cl) — optional.
        gt_perfs:       [N, 2]          ground-truth (cd, cl) — optional.
        sample_indices: Which samples to plot (default: first 3).
        slice_indices:  Which spanwise slices (default: [0, 4, 8] = root/mid/tip).
        save_path:      Save here if given, otherwise show interactively.
        pred_label:     Label for the predicted (coral) curves, e.g. "Generated" or
                        "Reconstructed". Used in legend, cd/cl annotations, and title.
    """
    def _to_np(t):
        return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t) if t is not None else None

    gen_airfoils  = _to_np(gen_airfoils)
    gt_airfoils   = _to_np(gt_airfoils)
    gen_pressures = _to_np(gen_pressures)
    gt_pressures  = _to_np(gt_pressures)
    gen_perfs     = _to_np(gen_perfs)
    gt_perfs      = _to_np(gt_perfs)

    if sample_indices is None:
        sample_indices = list(range(min(3, gen_airfoils.shape[0])))
    if slice_indices is None:
        slice_indices = [0, 4, 8]

    slice_labels = {sl: f's{sl}' for sl in range(20)}
    n_samples = len(sample_indices)
    n_slices  = len(slice_indices)

    fig, axes = plt.subplots(
        2 * n_samples, n_slices,
        figsize=(5 * n_slices, 4 * n_samples),
        squeeze=False,
    )

    for row_base, si in enumerate(sample_indices):
        ax_shape_row = axes[2 * row_base]
        ax_cp_row    = axes[2 * row_base + 1]

        for col, sl in enumerate(slice_indices):
            label = slice_labels.get(sl, f's{sl}')

            # ── Airfoil shape ───────────────────────────────────────────
            ax = ax_shape_row[col]
            gt_xy  = gt_airfoils[si, sl]   # [2, 192]: row0=x, row1=y
            gen_xy = gen_airfoils[si, sl]

            ax.scatter(gt_xy[0],  gt_xy[1],  s=2, c='steelblue', label='GT',        zorder=2)
            ax.scatter(gen_xy[0], gen_xy[1], s=2, c='coral',     label=pred_label,  zorder=3)

            shape_mse = float(((gen_xy - gt_xy) ** 2).mean())
            title = f'Sample {si} | {label}\nshape MSE={shape_mse:.2e}'
            if col == 0 and flow_conditions is not None and si < len(flow_conditions):
                fc = flow_conditions[si]
                title += (f'\nMach={fc.get("mach", 0):.3f}'
                          f'  Re={fc.get("reynolds", 0):.2e}'
                          f'  cl_t={fc.get("cl_target", 0):.3f}')
            if col == len(slice_indices) // 2 and gen_perfs is not None and gt_perfs is not None:
                gt_cd, gt_cl   = float(gt_perfs[si, 0]),  float(gt_perfs[si, 1])
                gen_cd, gen_cl = float(gen_perfs[si, 0]), float(gen_perfs[si, 1])
                title += (f'\ncd: GT={gt_cd:.4f} {pred_label}={gen_cd:.4f} (err={abs(gen_cd-gt_cd):.4f})'
                          f'\ncl: GT={gt_cl:.4f} {pred_label}={gen_cl:.4f} (err={abs(gen_cl-gt_cl):.4f})')
            ax.set_title(title, fontsize=8)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel('airfoil', fontsize=8)
            if row_base == 0 and col == 0:
                ax.legend(fontsize=7, markerscale=3)

            # ── Cp distribution ─────────────────────────────────────────
            ax = ax_cp_row[col]
            gt_cp  = gt_pressures[si, sl]    # [192]
            gen_cp = gen_pressures[si, sl]
            x_pts  = gt_airfoils[si, sl, 0]  # chord x-coordinates [192]

            le_idx = int(np.argmin(x_pts))
            upper  = np.zeros(len(x_pts), dtype=bool)
            upper[:le_idx + 1] = True
            lower  = ~upper

            for mask, ls, lbl in [(upper, '-', 'GT upper'), (lower, '--', 'GT lower')]:
                if mask.sum() == 0:
                    continue
                idx = np.argsort(x_pts[mask])
                ax.plot(x_pts[mask][idx], gt_cp[mask][idx],
                        color='steelblue', lw=1.0, linestyle=ls, label=lbl)

            for mask, ls in [(upper, '-'), (lower, '--')]:
                if mask.sum() == 0:
                    continue
                idx = np.argsort(x_pts[mask])
                ax.plot(x_pts[mask][idx], gen_cp[mask][idx],
                        color='coral', lw=1.0, linestyle=ls)

            cp_mse = float(((gen_cp - gt_cp) ** 2).mean())
            ax.set_title(f'Cp MSE={cp_mse:.4f}', fontsize=8)
            ax.set_xlabel('x/c', fontsize=8)
            if col == 0:
                ax.set_ylabel('Cp', fontsize=8)
                ax.legend(fontsize=7)

    plt.suptitle(f'Airfoil shape + Cp: GT (blue) vs {pred_label} (coral)', fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_perf_scatter(gen_perf: torch.Tensor, gt_perf: torch.Tensor,
                      save_path: str = None, pred_label: str = "Generated"):
    """Predicted vs ground-truth cd and cl scatter plots across all test wings.

    Args:
        gen_perf:    [N, 2]  columns [cd, cl]  (de-normalised)
        gt_perf:     [N, 2]
        pred_label:  Label for the predicted axis, e.g. "Generated" or "Reconstructed".
    """
    gen_np = gen_perf.cpu().numpy() if isinstance(gen_perf, torch.Tensor) else np.asarray(gen_perf)
    gt_np  = gt_perf.cpu().numpy()  if isinstance(gt_perf,  torch.Tensor) else np.asarray(gt_perf)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ['cd', 'cl']

    for i, (ax, lbl) in enumerate(zip(axes, labels)):
        ax.scatter(gt_np[:, i], gen_np[:, i], s=8, alpha=0.5, color='steelblue')
        lo = min(gt_np[:, i].min(), gen_np[:, i].min())
        hi = max(gt_np[:, i].max(), gen_np[:, i].max())
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8, label='ideal')
        mse = float(((gen_np[:, i] - gt_np[:, i]) ** 2).mean())
        mae = float(np.abs(gen_np[:, i] - gt_np[:, i]).mean())
        ax.set_title(f'{lbl}   MSE={mse:.2e}   MAE={mae:.4f}')
        ax.set_xlabel(f'GT {lbl}')
        ax.set_ylabel(f'{pred_label} {lbl}')
        ax.legend(fontsize=8)

    plt.suptitle(f'Aerodynamic performance: GT vs {pred_label} (N={len(gt_np)} wings)', fontsize=11)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_airfoil_slices_comparison(
    gen_airfoils: torch.Tensor,
    gt_airfoils: torch.Tensor,
    sample_indices=None,
    slice_indices=None,
    save_path: str = None,
):
    """Plot generated vs ground-truth 2-D cross-sections for selected slices and samples.

    Useful for diagnosing *where* along the span the model fails (root / mid / tip).

    Args:
        gen_airfoils:   [N, 9, 2, 192]  generated wing coordinates.
        gt_airfoils:    [N, 9, 2, 192]  ground-truth wing coordinates.
        sample_indices: List of sample indices to plot (default: first 3).
        slice_indices:  List of spanwise slice indices to show (default: [0, 4, 8],
                        i.e. root, mid-wing, tip).
        save_path:      If given, save figure here; otherwise show interactively.
    """
    if isinstance(gen_airfoils, torch.Tensor):
        gen_airfoils = gen_airfoils.cpu().numpy()
    if isinstance(gt_airfoils, torch.Tensor):
        gt_airfoils = gt_airfoils.cpu().numpy()

    if sample_indices is None:
        sample_indices = list(range(min(3, gen_airfoils.shape[0])))
    if slice_indices is None:
        slice_indices = [0, 4, 8]  # root, mid, tip

    slice_labels = {0: 'root', 4: 'mid', 8: 'tip'}

    n_samples = len(sample_indices)
    n_slices  = len(slice_indices)

    fig, axes = plt.subplots(
        n_samples, n_slices,
        figsize=(4 * n_slices, 3 * n_samples),
        squeeze=False,
    )

    for row, si in enumerate(sample_indices):
        for col, sl in enumerate(slice_indices):
            ax = axes[row][col]
            gt  = gt_airfoils[si, sl]   # [2, 192]
            gen = gen_airfoils[si, sl]  # [2, 192]

            ax.scatter(gt[0],  gt[1],  s=1.5, c='steelblue', label='GT',  zorder=2)
            ax.scatter(gen[0], gen[1], s=1.5, c='coral',     label='Gen', zorder=3)

            label = slice_labels.get(sl, f's{sl}')
            mse   = float(((gen - gt) ** 2).mean())
            ax.set_title(f'Sample {si} | {label} (s{sl})\nMSE={mse:.2e}', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            if row == 0 and col == 0:
                ax.legend(fontsize=7, markerscale=4)

    plt.suptitle('Airfoil slices: GT (blue) vs Generated (coral)', fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_reconstruction_error_histograms(
    rec_airfoils: torch.Tensor,
    gt_airfoils: torch.Tensor,
    rec_pressures: torch.Tensor,
    gt_pressures: torch.Tensor,
    rec_perfs: torch.Tensor,
    gt_perfs: torch.Tensor,
    save_path: str = None,
):
    """Histogram of per-sample reconstruction errors across the test set.

    Shows the distribution (not just the mean) of shape MSE, Cp MSE, cd
    absolute error, and cl absolute error, making outliers visible.

    Args:
        rec_airfoils:  [N, 9, 2, 192]  reconstructed coordinates.
        gt_airfoils:   [N, 9, 2, 192]  ground-truth coordinates.
        rec_pressures: [N, 9, 192]     reconstructed Cp (de-normalised).
        gt_pressures:  [N, 9, 192]     ground-truth Cp.
        rec_perfs:     [N, 2]          reconstructed (cd, cl).
        gt_perfs:      [N, 2]          ground-truth (cd, cl).
        save_path:     Save here if given, otherwise show interactively.
    """
    def _np(t):
        return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

    rec_airfoils  = _np(rec_airfoils)
    gt_airfoils   = _np(gt_airfoils)
    rec_pressures = _np(rec_pressures)
    gt_pressures  = _np(gt_pressures)
    rec_perfs     = _np(rec_perfs)
    gt_perfs      = _np(gt_perfs)

    N = rec_airfoils.shape[0]

    # Per-sample errors
    shape_mse = ((rec_airfoils - gt_airfoils) ** 2).reshape(N, -1).mean(axis=1)
    cp_mse    = ((rec_pressures - gt_pressures) ** 2).reshape(N, -1).mean(axis=1)
    cd_err    = np.abs(rec_perfs[:, 0] - gt_perfs[:, 0])
    cl_err    = np.abs(rec_perfs[:, 1] - gt_perfs[:, 1])

    metrics = [
        (shape_mse, "Shape MSE",     "shape MSE"),
        (cp_mse,    "Cp MSE",        "Cp MSE"),
        (cd_err,    "cd |error|",    "cd absolute error"),
        (cl_err,    "cl |error|",    "cl absolute error"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for ax, (values, short, long) in zip(axes, metrics):
        ax.hist(values, bins=40, color='steelblue', edgecolor='white', linewidth=0.4)
        ax.axvline(np.mean(values),   color='coral',  lw=1.5, linestyle='--', label=f'mean={np.mean(values):.3e}')
        ax.axvline(np.median(values), color='orange', lw=1.5, linestyle=':',  label=f'median={np.median(values):.3e}')
        ax.set_title(short, fontsize=10)
        ax.set_xlabel(long, fontsize=8)
        ax.set_ylabel('count', fontsize=8)
        ax.legend(fontsize=7)

    plt.suptitle(f'Reconstruction error distribution  (N={N} test wings)', fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_wing3d_comparison(
    rec_airfoils: torch.Tensor,
    gt_airfoils: torch.Tensor,
    sample_idx: int = 0,
    save_path: str = None,
    title_a: str = "GT",
    title_b: str = "Reconstructed",
):
    """Side-by-side 3-D stacked-slice view of two sets of wing coordinates.

    Args:
        rec_airfoils: [N, n_slices, 2, 192]  first set of coordinates (left panel).
        gt_airfoils:  [N, n_slices, 2, 192]  second set of coordinates (right panel).
        sample_idx:   Which sample to plot.
        save_path:    Save here if given, otherwise show interactively.
        title_a:      Title for the left panel.
        title_b:      Title for the right panel.
    """
    if isinstance(rec_airfoils, torch.Tensor):
        rec_airfoils = rec_airfoils.cpu().numpy()
    if isinstance(gt_airfoils, torch.Tensor):
        gt_airfoils = gt_airfoils.cpu().numpy()

    gt_wing  = gt_airfoils[sample_idx]   # [n_slices, 2, 192]
    rec_wing = rec_airfoils[sample_idx]
    n_slices = gt_wing.shape[0]
    z        = np.linspace(0, 1, n_slices)

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    for i in range(n_slices):
        ax1.plot(gt_wing[i, 0, :],  np.full(192, z[i]), gt_wing[i, 1, :],
                 color='steelblue', linewidth=0.8, alpha=0.7)
        ax2.plot(rec_wing[i, 0, :], np.full(192, z[i]), rec_wing[i, 1, :],
                 color='coral',     linewidth=0.8, alpha=0.7)

    for ax, title in [(ax1, title_a), (ax2, title_b)]:
        ax.set_title(f'Sample {sample_idx} — {title}', fontsize=10)
        ax.set_xlabel('x/c', fontsize=7)
        ax.set_ylabel('span', fontsize=7)
        ax.set_zlabel('y/c', fontsize=7)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_raw_slices_2d(
    raw_coords: torch.Tensor,
    bae_coords: torch.Tensor,
    sample_idx: int = 0,
    save_path: str = None,
):
    """Per-slice 2D diagnostic: raw dataset coords vs BAE roundtrip side by side.

    Each column is one spanwise slice. Top row = raw, bottom row = BAE roundtrip.
    Useful for checking whether raw point ordering / normalization is consistent.

    Args:
        raw_coords: [N, n_slices, 2, 192]  raw dataset coordinates.
        bae_coords: [N, n_slices, 2, 192]  BAE encode→decode coordinates.
        sample_idx: Which sample to plot.
        save_path:  Save here if given, otherwise show interactively.
    """
    if isinstance(raw_coords, torch.Tensor):
        raw_coords = raw_coords.cpu().numpy()
    if isinstance(bae_coords, torch.Tensor):
        bae_coords = bae_coords.cpu().numpy()

    raw  = raw_coords[sample_idx]   # [n_slices, 2, 192]
    bae  = bae_coords[sample_idx]
    n_slices = raw.shape[0]

    fig, axes = plt.subplots(2, n_slices, figsize=(3 * n_slices, 6), squeeze=False)

    for s in range(n_slices):
        # Raw
        ax = axes[0][s]
        ax.scatter(raw[s, 0], raw[s, 1], s=1, c='steelblue')
        ax.set_title(f's{s}', fontsize=7)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        if s == 0:
            ax.set_ylabel('Raw', fontsize=8)

        # BAE roundtrip
        ax = axes[1][s]
        ax.scatter(bae[s, 0], bae[s, 1], s=1, c='coral')
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        if s == 0:
            ax.set_ylabel('BAE roundtrip', fontsize=8)

    plt.suptitle(f'Sample {sample_idx} — Raw coords (blue) vs BAE roundtrip (coral) per slice',
                 fontsize=10)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_generated_wings(generated_airfoils, generated_alphas, save_dir=None):
    """Plot each generated wing (3-D view of all slices)."""
    for idx in range(len(generated_airfoils)):
        fig = plt.figure(figsize=(12, 5))
        ax  = fig.add_subplot(projection='3d')
        wing_3D_shape_plot(generated_airfoils[idx], ax=ax,
                           facecolor='steelblue', alpha=0.7, slice_mode=True)
        alpha_val = (generated_alphas[idx].item()
                     if torch.is_tensor(generated_alphas) else generated_alphas[idx])
        ax.set_title(f'Generated Wing {idx}  (α={alpha_val:.2f}°)')
        ax.set_xlabel('x')
        ax.set_ylabel('z (span)')
        ax.set_zlabel('y')

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'generated_wing_{idx:02d}.png'), dpi=150)
            plt.close()
        else:
            plt.show()
