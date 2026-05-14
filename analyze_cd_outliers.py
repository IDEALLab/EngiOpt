"""
Per-wing cd reconstruction error analysis.

Loads the v2_weighting checkpoint, reconstructs all test wings, then plots
cd absolute error against AoA, Mach, Reynolds, cl_target, area_case_ratio,
and root-slice thickness to identify which regime drives the outliers.

Usage:
    python analyze_cd_outliers.py [--checkpoint PATH] [--top N]
"""

import argparse
import os

import torch as _torch
# Allow loading CUDA checkpoints on CPU-only machines
_orig_load = _torch.load
def _cpu_safe_load(*args, **kwargs):
    if "map_location" not in kwargs:
        kwargs["map_location"] = None if _torch.cuda.is_available() else "cpu"
    return _orig_load(*args, **kwargs)
_torch.load = _cpu_safe_load

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from engibench.problems.wings3D.v0 import Wings3D

from engiopt.lvae.lvae import LAE_AoAInit
from engiopt.lvae.train_lvae import Config, build_encoder, build_decoder, build_sampler, load_bae, ensure_perf_regressor_fitted

SAVE_DIR = "results/lvae_evaluation"
BATCH_SIZE = 64


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/lvae/lae_aoa_init_3d_v2_weighting.pth",
    )
    parser.add_argument("--top", type=int, default=10, help="Number of top outliers to print")
    return parser.parse_args()


def load_model(checkpoint_path, cfg):
    bae_model = load_bae(cfg)
    lae_model = LAE_AoAInit(
        encoder=build_encoder(cfg),
        decoder=build_decoder(cfg),
        sampler=build_sampler(cfg),
        bae_model=bae_model,
        lae_latent_dim=cfg.lae_latent_dim,
        params_mean_std=(0, 1),
        aoas_mean_std=(0, 1),
        name=cfg.model_name,
        opt_lr=cfg.lr,
    )
    lae_model.load(checkpoint_path, train_mode=False)
    lae_model.to(cfg.device)
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    reg_path = os.path.join(os.path.dirname(checkpoint_path), f"{stem}_perf_reg.pkl")
    ensure_perf_regressor_fitted(lae_model, cfg, reg_path)
    return lae_model, lae_model.bae_model


def encode_test_set(test_dataset, lae_model, bae_model, cfg):
    device = cfg.device
    records = []
    z_opts_list, params_list = [], []

    for item in test_dataset:
        coords    = torch.tensor(item["coords"],    dtype=torch.float32)  # [9, 192, 2]
        te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)  # [9]

        coords_unshifted = coords.clone()
        coords_unshifted[:, :, 1] -= te_shifts.unsqueeze(1)

        z_slices = []
        for s in range(9):
            x_s = coords_unshifted[s].permute(1, 0).unsqueeze(0).to(device)
            with torch.no_grad():
                z_s = bae_model.encode(x_s, return_z=True, z_ae_mode=True)
            z_slices.append(z_s.squeeze(0).cpu())
        z_opts_list.append(torch.stack(z_slices))  # [9, latent_ch, L]

        params = torch.tensor(
            [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)
        params_list.append(lae_model.scaler_params.transform(params))

        # Root-slice (s=0) thickness: max(y) - min(y) of original coords
        root_y = coords[0, :, 1].numpy()
        thickness = float(root_y.max() - root_y.min())

        records.append({
            "aoa":              float(item["alpha"]),
            "mach":             float(item["mach"]),
            "reynolds":         float(item["reynolds"]),
            "cl_target":        float(item["cl_target"]),
            "area_case_ratio":  float(item["area_case_ratio"]),
            "thickness":        thickness,
            "gt_cd":            float(item["cd_val"]),
            "gt_cl":            float(item["cl_val"]),
        })

    return records, z_opts_list, params_list


def reconstruct_all(lae_model, bae_model, z_opts_list, params_list, device):
    lae_model.encoder.eval()
    lae_model.decoder.eval()

    rec_perfs = []
    n = len(z_opts_list)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        z_opts_batch = torch.stack(z_opts_list[start:end]).to(device)   # [B, 9, ch, L]
        params_batch = torch.cat(params_list[start:end], dim=0).to(device)

        with torch.no_grad():
            mu       = lae_model.encode(z_opts_batch, params_batch)
            z_masked = lae_model._apply_mask(mu)
            _, _, _, _, perf_pred_norm = lae_model.decode(z_masked, params_batch)

        if lae_model.scaler_perfs is not None:
            perf_out = lae_model.scaler_perfs.inverse_transform(perf_pred_norm.cpu())
        else:
            perf_out = perf_pred_norm.cpu()
        rec_perfs.append(perf_out)

    return torch.cat(rec_perfs, dim=0)  # [N, 2]


def make_scatter_plot(records, cd_abs_errors, save_path):
    fields = [
        ("aoa",             "AoA (deg)"),
        ("mach",            "Mach"),
        ("reynolds",        "Reynolds"),
        ("cl_target",       "CL target"),
        ("area_case_ratio", "Area case ratio"),
        ("thickness",       "Root thickness"),
        ("gt_cd",           "GT cd"),
    ]

    fig, axes = plt.subplots(1, len(fields), figsize=(4 * len(fields), 4))
    fig.suptitle("Reconstruction: cd |error| vs wing parameters", fontsize=13)

    outlier_thresh = np.percentile(cd_abs_errors, 90)

    for ax, (key, label) in zip(axes, fields):
        x = np.array([r[key] for r in records])
        colors = ["tab:red" if e > outlier_thresh else "tab:blue" for e in cd_abs_errors]
        ax.scatter(x, cd_abs_errors, c=colors, s=18, alpha=0.7)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("cd |error|" if ax is axes[0] else "", fontsize=9)
        ax.axhline(np.mean(cd_abs_errors),  color="orange", ls="--", lw=1, label="mean")
        ax.axhline(np.median(cd_abs_errors), color="green",  ls=":",  lw=1, label="median")
        if ax is axes[0]:
            ax.legend(fontsize=7)
        # Pearson r annotation
        if np.std(x) > 0:
            r = float(np.corrcoef(x, cd_abs_errors)[0, 1])
            ax.set_title(f"r={r:+.2f}", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot saved to {save_path}")


def print_top_outliers(records, cd_abs_errors, n=10):
    order = np.argsort(cd_abs_errors)[::-1]
    print(f"\nTop {n} cd outliers (reconstruction):")
    header = (f"{'rank':>4}  {'cd_err':>8}  {'aoa':>6}  {'mach':>5}  "
              f"{'Re':>9}  {'cl_tgt':>6}  {'area':>6}  {'thick':>6}  {'gt_cd':>8}")
    print(header)
    print("-" * len(header))
    for rank, idx in enumerate(order[:n], 1):
        r = records[idx]
        print(
            f"{rank:>4}  {cd_abs_errors[idx]:>8.4f}  {r['aoa']:>6.2f}  {r['mach']:>5.3f}  "
            f"{r['reynolds']:>9.0f}  {r['cl_target']:>6.3f}  {r['area_case_ratio']:>6.3f}  "
            f"{r['thickness']:>6.4f}  {r['gt_cd']:>8.5f}"
        )


def main():
    args = parse_args()
    cfg  = Config()
    cfg.model_name = os.path.splitext(os.path.basename(args.checkpoint))[0]

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {cfg.device}")

    lae_model, bae_model = load_model(args.checkpoint, cfg)

    problem      = Wings3D(seed=cfg.seed)
    test_dataset = [item for item in problem.dataset["test"] if item["final"] == 1]
    print(f"Test wings : {len(test_dataset)}")

    records, z_opts_list, params_list = encode_test_set(test_dataset, lae_model, bae_model, cfg)

    rec_perfs = reconstruct_all(lae_model, bae_model, z_opts_list, params_list, cfg.device)
    gt_cds    = torch.tensor([r["gt_cd"] for r in records])
    cd_abs_errors = (rec_perfs[:, 0] - gt_cds).abs().numpy()

    print(f"\ncd |error|  mean={cd_abs_errors.mean():.4f}  "
          f"median={np.median(cd_abs_errors):.4f}  "
          f"p90={np.percentile(cd_abs_errors, 90):.4f}  "
          f"max={cd_abs_errors.max():.4f}")

    print_top_outliers(records, cd_abs_errors, n=args.top)

    os.makedirs(SAVE_DIR, exist_ok=True)
    stem      = os.path.splitext(os.path.basename(args.checkpoint))[0]
    save_path = os.path.join(SAVE_DIR, f"cd_outlier_analysis_{stem}.png")
    make_scatter_plot(records, cd_abs_errors, save_path)


if __name__ == "__main__":
    main()
