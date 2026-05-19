"""
Plot generated vs ground-truth wings for a DDM_W checkpoint.

Usage
-----
    python -m engiopt.ddm.ddm_w.plot_wings_w \
        --checkpoint results/ddm_w/ddm_w_v1_best.pth \
        --lvae_checkpoint results/lvae/lae_dropout_0.25_flow_only_best.pth \
        [--n_wings 6] [--seed 0] [--T 1000]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from engiopt.ddm.ddm_w.ddm_w import DDM_W, MLPDenoiser, MLPDenoiserV1
from engiopt.ddm.ddm_w.train_ddm_w import Config, load_bae, load_lvae, build_sampler
from engiopt.ddm.ddm_w.evaluate_ddm_w import precompute_test
from engiopt.ddm.plotting import wing_3D_shape_plot, wing_3D_pressure_plot
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      type=str, required=True)
    p.add_argument("--lvae_checkpoint", type=str, required=True)
    p.add_argument("--n_wings",  type=int, default=6)
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--T",        type=int, default=None)
    p.add_argument("--out",      type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()
    cfg.seed            = args.seed
    cfg.lvae_checkpoint = args.lvae_checkpoint
    device = cfg.device

    # Load frozen models
    bae_model  = load_bae(cfg)
    lvae_model = load_lvae(cfg, bae_model)
    lvae_params_scaler = getattr(lvae_model, 'scaler_params', None)

    # Load DDM_W checkpoint
    ckpt    = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sampler = build_sampler(cfg)

    # Support both old (out_aoa) and new (aoa_head) MLPDenoiser architectures
    saved = ckpt['denoiser']
    if isinstance(saved, nn.Module):
        denoiser = saved
    else:
        denoiser = MLPDenoiser(w_dim=cfg.lvae_lae_latent_dim, c_dim=cfg.c_dim)
        try:
            denoiser.load_state_dict(saved)
        except RuntimeError:
            denoiser = MLPDenoiserV1(w_dim=cfg.lvae_lae_latent_dim, c_dim=cfg.c_dim)
            denoiser.load_state_dict(saved)

    pms = ckpt.get('params_mean_std')
    ams = ckpt.get('aoas_mean_std')

    ddm_w = DDM_W(
        denoiser=denoiser, lvae_model=lvae_model, bae_model=bae_model,
        sampler=sampler, w_dim=cfg.lvae_lae_latent_dim, c_dim=cfg.c_dim,
        w_pressure=ckpt.get('w_pressure', 1.0),
        lvae_params_dim=ckpt.get('lvae_params_dim', cfg.lvae_params_dim),
        params_mean_std=pms, aoas_mean_std=ams,
        name=os.path.splitext(os.path.basename(args.checkpoint))[0],
    )
    ddm_w.load(args.checkpoint, train_mode=False)
    ddm_w.denoiser.to(device)
    print("Model loaded.")

    w_mean = ddm_w.w_mean
    w_std  = ddm_w.w_std

    # Test dataset
    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=cfg.seed)
    all_test     = list(new_dataset["test"])
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}
    test_dataset    = [item for item in all_test if item["final"] == 1]

    gt_coords, gt_aoas, gt_pressures, w_inits, params_norm, te_shifts_list, etas_list = \
        precompute_test(
            test_dataset, initial_by_case, bae_model, lvae_model,
            lvae_params_scaler, cfg.lvae_params_dim, w_mean, w_std,
            ddm_w.scaler_params, ddm_w.scaler_aoas, device,
        )
    n_test   = gt_coords.shape[0]
    n_wings  = min(args.n_wings, n_test)
    print(f"Test set: {n_test} wings, plotting {n_wings}")

    # Generate
    gen_coords, gen_aoas, gen_pressures, gen_te_shifts = ddm_w.generate(
        w_init=w_inits[:n_wings].to(device),
        params=params_norm[:n_wings].to(device),
        device=device,
        T=args.T,
    )

    # Both gen and GT coords are in BAE-centered space (TE at y=0).
    # No te_shift re-application needed for either.

    # Flow params for display
    raw_flow = [
        (item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"])
        for item in test_dataset[:n_wings]
    ]

    # Shared pressure colour scale
    all_p = np.concatenate([
        gt_pressures[:n_wings].numpy().ravel(),
        gen_pressures.numpy().ravel(),
    ])
    pmin = float(np.nanpercentile(all_p, 2))
    pmax = float(np.nanpercentile(all_p, 98))

    wing_len = 2.25
    n_rows   = 4
    fig = plt.figure(figsize=(10 * n_wings, 10 * n_rows), dpi=150)

    for i in range(n_wings):
        mach, re, cl, ar = raw_flow[i]
        cond_str = f"M={mach:.2f}  Re={re/1e6:.2f}M\nCL={cl:.2f}  AR={ar:.2f}"
        aoa_str  = f"{float(np.atleast_1d(gen_aoas)[i]):.1f}°"
        z_pos    = etas_list[i] * wing_len

        ax = fig.add_subplot(n_rows, n_wings, i + 1, projection='3d')
        wing_3D_shape_plot(gen_coords[i].numpy(), ax=ax, facecolor='steelblue', alpha=0.7,
                           z=z_pos, wing_len=wing_len)
        ax.set_title(f"Gen {i+1}  AoA={aoa_str}\n{cond_str}", fontsize=7)
        ax.set_axis_off()

        ax = fig.add_subplot(n_rows, n_wings, n_wings + i + 1, projection='3d')
        wing_3D_shape_plot(gt_coords[i].numpy(), ax=ax, facecolor='coral', alpha=0.7,
                           z=z_pos, wing_len=wing_len)
        ax.set_title(f"GT {i+1}  AoA={gt_aoas[i].item():.1f}°\n{cond_str}", fontsize=7)
        ax.set_axis_off()

        ax = fig.add_subplot(n_rows, n_wings, 2 * n_wings + i + 1, projection='3d')
        wing_3D_pressure_plot(gen_coords[i].numpy(), gen_pressures[i].numpy(),
                              ax=ax, vmin=pmin, vmax=pmax, z=z_pos, wing_len=wing_len)
        ax.set_title(f"Gen Cp {i+1}", fontsize=7)
        ax.set_axis_off()

        ax = fig.add_subplot(n_rows, n_wings, 3 * n_wings + i + 1, projection='3d')
        wing_3D_pressure_plot(gt_coords[i].numpy(), gt_pressures[i].numpy(),
                              ax=ax, vmin=pmin, vmax=pmax, z=z_pos, wing_len=wing_len)
        ax.set_title(f"GT Cp {i+1}", fontsize=7)
        ax.set_axis_off()

    fig.suptitle(
        f"{os.path.basename(args.checkpoint)}  —  rows: gen shape | GT shape | gen Cp | GT Cp",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()

    if args.out is None:
        model_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.out = f"results/plots/wings_{model_name}.png"

    base, ext = os.path.splitext(args.out)
    out_path  = args.out
    counter   = 1
    while os.path.exists(out_path):
        out_path = f"{base}_{counter}{ext}"
        counter += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Saved to {out_path}")

    # Spanwise slice plots
    slices_dir = os.path.join(os.path.dirname(out_path), "spanwise_slices_w")
    os.makedirs(slices_dir, exist_ok=True)
    n_slices   = gen_coords.shape[1]
    n_cols     = min(n_slices, 5)
    n_rows_sp  = (n_slices + n_cols - 1) // n_cols

    for i in range(n_wings):
        mach, re, cl, ar = raw_flow[i]
        gen_np = gen_coords[i].numpy()
        gt_np  = gt_coords[i].numpy()

        fig_sp, axes = plt.subplots(
            n_rows_sp * 2, n_cols,
            figsize=(4 * n_cols, 3 * n_rows_sp * 2),
        )
        axes = np.array(axes).reshape(n_rows_sp * 2, n_cols)

        for s in range(n_slices):
            row_gen = (s // n_cols) * 2
            row_gt  = row_gen + 1
            col     = s % n_cols

            n_pts = gen_np.shape[2]
            half  = n_pts // 2
            for ax, arr, color, label in [
                (axes[row_gen, col], gen_np[s], 'steelblue', f"Gen  slice {s}"),
                (axes[row_gt,  col], gt_np[s],  'coral',     f"GT   slice {s}"),
            ]:
                ax.plot(arr[0, :half],  arr[1, :half],  color=color, lw=1.5)
                ax.plot(arr[0, half:],  arr[1, half:],  color=color, lw=1.5)
                ax.set_title(label, fontsize=7)
                ax.set_aspect('equal')
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.20, 0.20)
                ax.grid(True, lw=0.4)
                ax.tick_params(labelsize=6)

        for s in range(n_slices, n_rows_sp * n_cols):
            axes[(s // n_cols) * 2,     s % n_cols].set_visible(False)
            axes[(s // n_cols) * 2 + 1, s % n_cols].set_visible(False)

        aoa_str = f"{float(np.atleast_1d(gen_aoas)[i]):.1f}"
        fig_sp.suptitle(
            f"Wing {i+1} (DDM_W)  |  M={mach:.2f}  Re={re/1e6:.2f}M  "
            f"CL={cl:.2f}  AR={ar:.2f}  |  gen AoA={aoa_str}°  GT AoA={gt_aoas[i].item():.1f}°\n"
            f"Blue = generated,  Coral = ground truth",
            fontsize=9,
        )
        fig_sp.tight_layout()
        slice_path = os.path.join(slices_dir, f"spanwise_wing_{i+1:02d}.png")
        fig_sp.savefig(slice_path, bbox_inches='tight', dpi=150)
        plt.close(fig_sp)
        print(f"Saved spanwise slices to {slice_path}")


if __name__ == "__main__":
    main()
