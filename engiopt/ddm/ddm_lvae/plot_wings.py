"""
Plot generated vs ground-truth wings for a DDM_LVAE_3D checkpoint.

Usage
-----
    python -m engiopt.ddm.ddm_lvae.plot_wings \
        --checkpoint results/ddm_lvae/ddm_lvae_v6.pth \
        [--n_wings 6] [--seed 0] [--T 100] \
        [--out results/plots/wings_v6.png]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from engiopt.ddm.ddm_lvae.ddm_lvae import DDM_LVAE_3D
from engiopt.ddm.ddm_lvae.train_ddm_lvae import Config, build_sampler, build_unet, load_bae, load_lvae
from engiopt.ddm.ddm_lvae.evaluate_ddm_lvae import precompute_test, generate_batch
from engiopt.ddm.plotting import wing_3D_shape_plot, wing_3D_pressure_plot
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      type=str, required=True)
    parser.add_argument("--lvae_checkpoint", type=str,
                        default="results/lvae/lae_dropout_0.25_best.pth")
    parser.add_argument("--n_wings",  type=int, default=6)
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--T",        type=int, default=None)
    parser.add_argument("--out",      type=str, default=None)
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = Config()
    cfg.seed            = args.seed
    cfg.lvae_checkpoint = args.lvae_checkpoint
    device = cfg.device

    # --- load model ---
    bae_model  = load_bae(cfg)
    lvae_model = load_lvae(cfg, bae_model)
    unet = build_unet(cfg)

    import torch as _torch
    _ckpt_raw = _torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sampler = _ckpt_raw["sampler"]

    ddm_model = DDM_LVAE_3D(
        unet=unet, sampler=sampler,
        bae_model=bae_model, lvae_model=lvae_model,
        w_pressure=1.0, lvae_params_dim=cfg.lvae_params_dim,
        params_mean_std=(np.zeros(cfg.c_dim), np.ones(cfg.c_dim)),
        aoas_mean_std=(0.0, 1.0),
        name=os.path.splitext(os.path.basename(args.checkpoint))[0],
        opt_lr=cfg.lr,
    )
    ddm_model.load(args.checkpoint, train_mode=False)
    ddm_model.unet      = ddm_model.unet.to(device)
    ddm_model.bae_model = ddm_model.bae_model.to(device)
    bae_model = ddm_model.bae_model
    print("Model loaded.")

    # --- test split ---
    new_dataset = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=cfg.seed)
    all_test    = list(new_dataset["test"])
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}
    test_dataset    = [item for item in all_test if item["final"] == 1]

    gt_coords, gt_aoas, gt_pressures, encoded_inits, params_list, etas_list, te_shifts_list = precompute_test(
        test_dataset, initial_by_case, bae_model, ddm_model, device
    )
    n_test = gt_coords.shape[0]
    print(f"Test set: {n_test} wings")

    # --- generate one batch ---
    n_wings = min(args.n_wings, n_test)
    params_batch = torch.cat(params_list[:n_wings], dim=0)
    gen_coords, gen_aoas, gen_pressures = generate_batch(
        ddm_model, bae_model,
        encoded_inits[:n_wings], params_batch, device,
        T=args.T,
    )

    # Re-apply GT te_shifts to both gen and GT coords so wings sit at their
    # original y-positions instead of all being centered at TE y=0.
    # te_shifts[i]: [S] — add to the y-channel (index 1) of each slice.
    for i in range(n_wings):
        shifts = te_shifts_list[i].to(gen_coords.device)  # [S]
        gen_coords[i, :, 1, :] += shifts.unsqueeze(-1)    # [S, 192]
        gt_coords[i,  :, 1, :] += shifts.unsqueeze(-1)
    # gen_coords:     [n_wings, S, 2, 192]
    # gt_coords:      [n_test,  S, 2, 192]
    # gen_pressures:  [n_wings, S, 192]
    # gt_pressures:   [n_test,  S, 192]

    # Collect raw (unscaled) flow params for display: mach, reynolds, cl_target, area_case_ratio
    raw_flow = [
        (item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"])
        for item in test_dataset[:n_wings]
    ]

    # Shared pressure colour scale across all wings for fair comparison
    all_p = np.concatenate([
        gt_pressures[:n_wings].numpy().ravel(),
        gen_pressures.numpy().ravel(),
    ])
    pmin = float(np.nanpercentile(all_p, 2))
    pmax = float(np.nanpercentile(all_p, 98))

    # --- plot: row 1 = gen shape, row 2 = GT shape, row 3 = gen pressure, row 4 = GT pressure ---
    n_rows = 4
    fig = plt.figure(figsize=(10 * n_wings, 10 * n_rows), dpi=150)
    wing_len = 2.25  # physical semi-span used to scale η ∈ [0,1]
    for i in range(n_wings):
        mach, re, cl, ar = raw_flow[i]
        cond_str = f"M={mach:.2f}  Re={re/1e6:.2f}M\nCL={cl:.2f}  AR={ar:.2f}"
        aoa_str  = f"{float(np.atleast_1d(gen_aoas)[i]):.1f}°" if np.ndim(gen_aoas) > 0 else f"{float(gen_aoas):.1f}°"
        # Real spanwise positions scaled to physical semi-span
        z_pos = etas_list[i] * wing_len  # [S]

        # Row 1: generated shape
        ax = fig.add_subplot(n_rows, n_wings, i + 1, projection='3d')
        wing_3D_shape_plot(gen_coords[i].numpy(), ax=ax, facecolor='steelblue', alpha=0.7,
                           z=z_pos, wing_len=wing_len)
        ax.set_title(f"Gen {i+1}  AoA={aoa_str}\n{cond_str}", fontsize=7)
        ax.set_axis_off()

        # Row 2: GT shape
        ax = fig.add_subplot(n_rows, n_wings, n_wings + i + 1, projection='3d')
        wing_3D_shape_plot(gt_coords[i].numpy(), ax=ax, facecolor='coral', alpha=0.7,
                           z=z_pos, wing_len=wing_len)
        ax.set_title(f"GT {i+1}  AoA={gt_aoas[i].item():.1f}°\n{cond_str}", fontsize=7)
        ax.set_axis_off()

        # Row 3: generated pressure colormap
        ax = fig.add_subplot(n_rows, n_wings, 2 * n_wings + i + 1, projection='3d')
        wing_3D_pressure_plot(gen_coords[i].numpy(), gen_pressures[i].numpy(),
                              ax=ax, vmin=pmin, vmax=pmax, z=z_pos, wing_len=wing_len)
        ax.set_title(f"Gen Cp {i+1}", fontsize=7)
        ax.set_axis_off()

        # Row 4: GT pressure colormap
        ax = fig.add_subplot(n_rows, n_wings, 3 * n_wings + i + 1, projection='3d')
        wing_3D_pressure_plot(gt_coords[i].numpy(), gt_pressures[i].numpy(),
                              ax=ax, vmin=pmin, vmax=pmax, z=z_pos, wing_len=wing_len)
        ax.set_title(f"GT Cp {i+1}", fontsize=7)
        ax.set_axis_off()

    fig.suptitle(
        f"{os.path.basename(args.checkpoint)}  —  rows: gen shape | GT shape | gen Cp | GT Cp",
        fontsize=10, y=1.01
    )
    plt.tight_layout()

    if args.out is None:
        model_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.out = f"results/plots/wings_{model_name}.png"

    # Never overwrite an existing plot — append a numeric suffix instead.
    base, ext = os.path.splitext(args.out)
    out_path = args.out
    counter = 1
    while os.path.exists(out_path):
        out_path = f"{base}_{counter}{ext}"
        counter += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Saved to {out_path}")

    # --- spanwise slice plots: one figure per wing ---
    slices_dir = os.path.join(os.path.dirname(out_path), "spanwise_slices")
    os.makedirs(slices_dir, exist_ok=True)
    n_slices = gen_coords.shape[1]
    n_cols = min(n_slices, 5)
    n_rows_sp = (n_slices + n_cols - 1) // n_cols

    for i in range(n_wings):
        mach, re, cl, ar = raw_flow[i]
        gen_np = gen_coords[i].numpy()   # [S, 2, 192]
        gt_np  = gt_coords[i].numpy()    # [S, 2, 192]

        fig_sp, axes = plt.subplots(
            n_rows_sp * 2, n_cols,
            figsize=(4 * n_cols, 3 * n_rows_sp * 2),
        )
        axes = np.array(axes).reshape(n_rows_sp * 2, n_cols)

        for s in range(n_slices):
            row_gen = (s // n_cols) * 2        # gen rows: 0, 2, 4, ...
            row_gt  = row_gen + 1              # GT rows:  1, 3, 5, ...
            col     = s % n_cols

            ax_gen = axes[row_gen, col]
            ax_gt  = axes[row_gt,  col]

            ax_gen.plot(gen_np[s, 0, :], gen_np[s, 1, :], 'steelblue', lw=1.5)
            ax_gen.set_title(f"Gen  slice {s}", fontsize=7)
            ax_gen.set_aspect('equal')
            ax_gen.set_xlim(-0.05, 1.05)
            ax_gen.set_ylim(-0.20, 0.20)
            ax_gen.grid(True, lw=0.4)
            ax_gen.tick_params(labelsize=6)

            ax_gt.plot(gt_np[s, 0, :], gt_np[s, 1, :], 'coral', lw=1.5)
            ax_gt.set_title(f"GT   slice {s}", fontsize=7)
            ax_gt.set_aspect('equal')
            ax_gt.set_xlim(-0.05, 1.05)
            ax_gt.set_ylim(-0.20, 0.20)
            ax_gt.grid(True, lw=0.4)
            ax_gt.tick_params(labelsize=6)

        # Hide unused axes
        for s in range(n_slices, n_rows_sp * n_cols):
            axes[(s // n_cols) * 2,     s % n_cols].set_visible(False)
            axes[(s // n_cols) * 2 + 1, s % n_cols].set_visible(False)

        aoa_str = f"{float(np.atleast_1d(gen_aoas)[i]):.1f}"
        fig_sp.suptitle(
            f"Wing {i+1} spanwise slices  |  M={mach:.2f}  Re={re/1e6:.2f}M  "
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
