"""
Plot generated vs ground-truth wings for a DDM_PCA checkpoint.

Usage
-----
    python -m engiopt.ddm.ddm_pca.plot_wings_pca \
        --checkpoint results/ddm_pca/ddm_pca_v1_best.pth \
        [--n_wings 6] [--seed 0] [--T 1000]
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from engiopt.ddm.ddm_pca.ddm_pca import DDM_PCA, MLPDenoiser
from engiopt.ddm.ddm_pca.train_ddm_pca import Config, load_bae, build_sampler
from engiopt.ddm.ddm_pca.evaluate_ddm_pca import precompute_test
from engiopt.ddm.plotting import wing_3D_shape_plot
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--n_wings",    type=int, default=6)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--T",          type=int, default=None)
    p.add_argument("--out",        type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()
    cfg.seed = args.seed
    device   = cfg.device

    bae_model = load_bae(cfg)

    ckpt     = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    pca_path = args.checkpoint.replace('_best.pth', '_pca.pkl').replace('.pth', '_pca.pkl')
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)

    z_dim = ckpt.get('z_dim', cfg.n_components)
    pms   = ckpt.get('params_mean_std')
    ams   = ckpt.get('aoas_mean_std')

    denoiser = MLPDenoiser(z_dim=z_dim, c_dim=cfg.c_dim).to(device)
    ddm_pca  = DDM_PCA(
        denoiser=denoiser, pca=pca, bae_model=bae_model,
        sampler=build_sampler(cfg), z_dim=z_dim, c_dim=cfg.c_dim,
        n_slices=cfg.n_slices,
        bae_latent_channels=cfg.bae_latent_channels,
        bae_latent_length=cfg.bae_latent_length,
        w_aoa=ckpt.get('w_aoa', 1.0),
        params_mean_std=pms, aoas_mean_std=ams,
        name=os.path.splitext(os.path.basename(args.checkpoint))[0],
    )
    ddm_pca.load(args.checkpoint, train_mode=False)
    ddm_pca.denoiser.to(device)
    print("DDM_PCA loaded.")

    z_mean = ddm_pca.z_mean
    z_std  = ddm_pca.z_std

    new_dataset     = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=cfg.seed)
    all_test        = list(new_dataset["test"])
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}
    test_dataset    = [item for item in all_test if item["final"] == 1]

    gt_coords, gt_aoas, z_inits, params_norm = precompute_test(
        test_dataset, initial_by_case, bae_model, pca,
        z_mean, z_std, ddm_pca.scaler_params, ddm_pca.scaler_aoas,
        cfg, device,
    )
    n_test  = gt_coords.shape[0]
    n_wings = min(args.n_wings, n_test)
    print(f"Test set: {n_test} wings, plotting {n_wings}")

    gen_coords, gen_aoas = ddm_pca.generate(
        z_init=z_inits[:n_wings].to(device),
        params=params_norm[:n_wings].to(device),
        device=device,
        T=args.T,
    )

    raw_flow = [
        (item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"])
        for item in test_dataset[:n_wings]
    ]

    # 3-D overview: 2 rows (gen shape | GT shape)
    n_rows = 2
    fig = plt.figure(figsize=(10 * n_wings, 10 * n_rows), dpi=150)

    # Use equal z-span across wings
    wing_len = 2.25
    for i in range(n_wings):
        mach, re, cl, ar = raw_flow[i]
        cond_str = f"M={mach:.2f}  Re={re/1e6:.2f}M\nCL={cl:.2f}  AR={ar:.2f}"
        aoa_str  = f"{float(np.atleast_1d(gen_aoas)[i]):.1f}°"
        n_slices = gen_coords.shape[1]
        z_pos    = np.linspace(0, wing_len, n_slices)

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

    fig.suptitle(
        f"{os.path.basename(args.checkpoint)}  —  rows: gen shape | GT shape",
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
    plt.close(fig)

    # Spanwise slice plots
    slices_dir = os.path.join(os.path.dirname(out_path), "spanwise_slices_pca")
    os.makedirs(slices_dir, exist_ok=True)
    n_slices  = gen_coords.shape[1]
    n_cols    = min(n_slices, 5)
    n_rows_sp = (n_slices + n_cols - 1) // n_cols

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
                ax.plot(arr[0, :half], arr[1, :half], color=color, lw=1.5)
                ax.plot(arr[0, half:], arr[1, half:], color=color, lw=1.5)
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
            f"Wing {i+1} (DDM_PCA)  |  M={mach:.2f}  Re={re/1e6:.2f}M  "
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
