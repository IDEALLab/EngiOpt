"""
Evaluation script for DDM_LVAE_3D checkpoints.

Generates wings on the test split and reports:
  - Shape MSE    (generated vs ground-truth BAE reconstruction)
  - AoA MSE
  - MMD          (averaged over slices and gammas)
  - Vendi score  (normalised)
  - Pressure MSE (generated z through frozen LVAE vs ground-truth pressure)

Usage
-----
    python -m engiopt.ddm.evaluate_ddm_lvae \
        --checkpoint results/ddm_lvae/ddm_lvae_v2.pth \
        [--n_passes 10] [--batch_size 32] [--seed 0]
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import torch

from engiopt.bezier_ae.bezier_ae import BezierAutoencoder
from engiopt.ddm.ddm_lvae.ddm_lvae import DDM_LVAE_3D
from engiopt.ddm.ddm_lvae.train_ddm_lvae import Config, build_sampler, build_unet, load_bae, load_lvae
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

N_FORWARD_PASSES = 10
GAMMAS = [0.5, 25, 50, 100]
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Metric helpers (same as evaluate_ddm.py)
# ---------------------------------------------------------------------------

def gaussian_kernel(x, y, gamma):
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return torch.exp(-gamma * (diff ** 2).sum(-1))


def compute_mmd(generated, real, gamma):
    n, m = generated.shape[0], real.shape[0]
    Kxx = gaussian_kernel(generated, generated, gamma)
    Kyy = gaussian_kernel(real,      real,      gamma)
    Kxy = gaussian_kernel(generated, real,      gamma)
    return (Kxx.sum()/(n*n) - 2*Kxy.sum()/(n*m) + Kyy.sum()/(m*m)).item()


def compute_vendi(samples, gamma):
    valid = torch.isfinite(samples).all(dim=-1)
    samples = samples[valid]
    if samples.shape[0] < 2:
        return float("nan")
    K = gaussian_kernel(samples, samples, gamma) / samples.shape[0]
    K = K + 1e-4 * torch.eye(K.shape[0], device=K.device)
    try:
        ev = torch.linalg.eigvalsh(K).clamp(min=1e-10)
    except torch._C._LinAlgError:
        K2 = K.double().cpu(); K2 = (K2 + K2.T) / 2
        ev = torch.linalg.eigvalsh(K2).clamp(min=1e-10).to(samples.device).float()
    ev = ev / ev.sum()
    return (-(ev * ev.log()).sum()).exp().item()


def compute_metrics(generated, gt_airfoils, gen_aoas, gt_aoas,
                    gen_pressures=None, gt_pressures=None):
    """
    generated    : [N, S, 2, 192]
    gt_airfoils  : [N, S, 2, 192]
    gen_aoas     : [N]
    gt_aoas      : [N]
    gen_pressures: [N, S, 192]  (optional)
    gt_pressures : [N, S, 192]  (optional)
    """
    n_slices = generated.shape[1]
    shape_mse = 0.0
    mmd_vals, vendi_gen_vals, vendi_gt_vals = [], [], []

    for s in range(n_slices):
        gen_s = generated[:, s]       # [N, 2, 192]
        gt_s  = gt_airfoils[:, s]
        shape_mse += ((gen_s - gt_s) ** 2).mean().item()

        gen_flat = gen_s.reshape(gen_s.shape[0], -1)
        gt_flat  = gt_s.reshape(gt_s.shape[0],  -1)
        mmd_slice = [compute_mmd(gen_flat, gt_flat, g) for g in GAMMAS]
        vg = [compute_vendi(gen_flat, g) for g in GAMMAS]
        vr = [compute_vendi(gt_flat,  g) for g in GAMMAS]
        mmd_vals.append(float(np.mean(mmd_slice)))
        vendi_gen_vals.append(float(np.nanmean(vg)))
        vendi_gt_vals.append(float(np.nanmean(vr)))

    shape_mse /= n_slices
    mmd        = float(np.mean(mmd_vals))
    vendi_gt   = float(np.mean(vendi_gt_vals))
    vendi_norm = float(np.mean(vendi_gen_vals)) / vendi_gt if vendi_gt > 0 else 0.0
    aoa_mse    = ((gen_aoas - gt_aoas) ** 2).mean().item()

    out = {"shape_mse": shape_mse, "aoa_mse": aoa_mse, "mmd": mmd, "vendi": vendi_norm}

    if gen_pressures is not None and gt_pressures is not None:
        out["pressure_mse"] = ((gen_pressures - gt_pressures) ** 2).mean().item()

    return out


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def generate_batch(ddm_model, bae_model, encoded_inits_batch, params_batch, device, T=None):
    """Returns (decoded_coords [B,S,2,192], aoas [B], pressure [B,S,192])."""
    B = params_batch.shape[0]
    S = ddm_model.unet.w_dim if hasattr(ddm_model.unet, 'w_dim') else encoded_inits_batch[0].shape[0]

    noise_x     = torch.randn(B, S, 3, 30, device=device)
    noise_alpha = torch.randn(B, 1,        device=device)

    lat_mean = ddm_model.latent_mean
    lat_std  = ddm_model.latent_std
    if isinstance(lat_mean, torch.Tensor):
        lat_mean_full = lat_mean.to(device)   # [1, 1, 3, 1] — for all-slice denorm
        lat_std_full  = lat_std.to(device)
        lat_mean_init = lat_mean_full[0]       # [1, 3, 1]   — for z_init only
        lat_std_init  = lat_std_full[0]
    else:
        lat_mean_full = lat_mean
        lat_std_full  = lat_std
        lat_mean_init = lat_mean
        lat_std_init  = lat_std

    encoded_batch      = torch.cat(encoded_inits_batch, dim=0)          # [B, 3, L]
    encoded_batch_norm = (encoded_batch - lat_mean_init) / lat_std_init

    with torch.no_grad():
        gen_z, gen_alpha = ddm_model.sampler.sample_airfoil(
            model=ddm_model.unet,
            noise_x=noise_x,
            noise_alpha=noise_alpha,
            c=params_batch,
            x0=encoded_batch_norm,
            T=T,
        )
        n_nan_z = (~torch.isfinite(gen_z)).any(dim=(2,3)).any(dim=1).sum().item()
        if n_nan_z > 0:
            print(f"  [debug] gen_z has {n_nan_z}/{B} non-finite samples (pre-denorm)")

        # Pressure via frozen LVAE (using DDM's params — geo dims will be zero-padded
        # unless c_dim==38, in which case they are passed directly).
        pressure = ddm_model._lvae_pressure(gen_z, params_batch)  # [B, S, 192]

        # Denormalize latents and decode geometry — use full [1,1,3,1] stats
        gen_z_raw = gen_z * lat_std_full + lat_mean_full

        # Diagnostic: report how far generated latents stray from valid BAE ranges.
        # Valid ranges (from training data observations):
        #   ch0 (Bezier weights): [0.1, 2.0]
        #   ch1 (CP x):           [-0.002, 1.0]
        #   ch2 (CP y):           [-0.108, 0.146]
        valid_ranges = [
            ("ch0_bezier_w", 0.05,  2.1),
            ("ch1_cp_x",    -0.1,   1.1),
            ("ch2_cp_y",    -0.15,  0.20),
        ]
        for ci, (name, lo, hi) in enumerate(valid_ranges):
            ch = gen_z_raw[:, :, ci, :]
            frac_below = (ch < lo).float().mean().item()
            frac_above = (ch > hi).float().mean().item()
            ch_min = ch.min().item()
            ch_max = ch.max().item()
            print(f"  [latent diag] {name}: range [{ch_min:.4f}, {ch_max:.4f}]  "
                  f"out-of-range: {100*(frac_below+frac_above):.1f}%  "
                  f"(below {lo}: {100*frac_below:.1f}%, above {hi}: {100*frac_above:.1f}%)")

        gen_z_raw[:, :, 0, :] = gen_z_raw[:, :, 0, :].clamp(0.05, 2.1)
        gen_z_raw[:, :, 1, :] = gen_z_raw[:, :, 1, :].clamp(-0.1,  1.1)
        gen_z_raw[:, :, 2, :] = gen_z_raw[:, :, 2, :].clamp(-0.15, 0.20)
        decoded_slices = []
        for s in range(S):
            z_s = gen_z_raw[:, s].clone()
            dec_s = bae_model.decode_z(
                z_s, z_ae_mode=True, denormalize_output=False, normalized_data=False
            )[0]
            decoded_slices.append(dec_s)
        decoded = torch.stack(decoded_slices, dim=1)  # [B, S, 2, 192]

    gen_alpha_np = ddm_model.scaler_aoas.inverse_transform(
        gen_alpha.cpu().numpy()
    ).squeeze()
    return decoded.cpu(), gen_alpha_np, pressure.cpu()


# ---------------------------------------------------------------------------
# Pre-compute ground truth for test split
# ---------------------------------------------------------------------------

def precompute_test(test_dataset, initial_by_case, bae_model, ddm_model, device):
    """Returns gt_coords, gt_aoas, gt_pressures, encoded_inits, params_scaled, etas, te_shifts_list."""
    gt_coords_list    = []
    gt_aoas_list      = []
    gt_pressures_list = []
    encoded_inits     = []
    params_list       = []
    etas_list         = []
    te_shifts_list    = []

    lat_mean = ddm_model.latent_mean
    lat_std  = ddm_model.latent_std
    if isinstance(lat_mean, torch.Tensor):
        lat_mean = lat_mean.to(device)
        lat_std  = lat_std.to(device)

    bae_model.to(device)

    for item in test_dataset:
        coords     = torch.tensor(item["coords"],        dtype=torch.float32)  # [S, 192, 2]
        te_shifts  = torch.tensor(item["te_shifts"],     dtype=torch.float32)  # [S]
        pressure   = torch.tensor(item["coef_pressure"], dtype=torch.float32)  # [S, 192]
        n_slices   = coords.shape[0]

        # BAE encode each slice using te_shifts centering (matches training)
        gt_slices = []
        for s in range(n_slices):
            c_s = coords[s].clone()            # [192, 2]
            c_s[:, 1] -= te_shifts[s]          # y-center at TE
            x_s = c_s.T.unsqueeze(0).to(device)  # [1, 2, 192]
            with torch.no_grad():
                z_s = bae_model.encode(x_s, return_z=True, z_ae_mode=True)
                dec_s = bae_model.decode_z(
                    z_s, z_ae_mode=True, denormalize_output=False, normalized_data=False
                )[0]
            gt_slices.append(dec_s.squeeze(0).cpu())
        gt_coords_list.append(torch.stack(gt_slices))  # [S, 2, 192]
        gt_pressures_list.append(pressure)

        # z_init: BAE encoding of the initial (unoptimized) root slice
        case_num = int(item["case_num"])
        if case_num in initial_by_case:
            init_item  = initial_by_case[case_num]
            init_coords = torch.tensor(init_item["coords"],    dtype=torch.float32)
            init_te     = torch.tensor(init_item["te_shifts"], dtype=torch.float32)
            init_root   = init_coords[0].clone()
            init_root[:, 1] -= init_te[0]
        else:
            init_root = coords[0].clone()
            init_root[:, 1] -= te_shifts[0]

        x_init = init_root.T.unsqueeze(0).to(device)
        with torch.no_grad():
            z_init = bae_model.encode(x_init, return_z=True, z_ae_mode=True)
        if torch.isnan(z_init).any() or torch.isinf(z_init).any():
            print(f"[WARNING] Skipping case {case_num} — NaN/Inf in z_init")
            gt_coords_list.pop(); gt_pressures_list.pop()
            continue

        flow_params = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
        # Use only as many params as the model's scaler was trained on (4 for flow-only
        # models, 38 for models conditioned on geo params as well).
        n_params = ddm_model.scaler_params.mean.shape[0]
        if n_params > 4:
            geo_params = item["geo_params"].tolist() if "geo_params" in item else []
        else:
            geo_params = []
        params = torch.tensor(
            flow_params + geo_params,
            dtype=torch.float32,
        ).unsqueeze(0).to(device)
        params_scaled = ddm_model.scaler_params.transform(params)

        gt_aoas_list.append(float(item["alpha"]))
        encoded_inits.append(z_init)
        params_list.append(params_scaled)
        etas_list.append(item["transforms"].copy())  # [S] span positions η ∈ [0, 1]
        te_shifts_list.append(te_shifts)             # [S] TE y-offsets for this wing

    gt_coords    = torch.stack(gt_coords_list)    # [N, S, 2, 192]
    gt_pressures = torch.stack(gt_pressures_list) # [N, S, 192]
    gt_aoas      = torch.tensor(gt_aoas_list)

    return gt_coords, gt_aoas, gt_pressures, encoded_inits, params_list, etas_list, te_shifts_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DDM_LVAE_3D")
    parser.add_argument("--checkpoint",      type=str, required=True)
    parser.add_argument("--lvae_checkpoint", type=str,
                        default="results/lvae/lae_dropout_0.25_best.pth")
    parser.add_argument("--n_passes",   type=int, default=N_FORWARD_PASSES)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--T", type=int, default=None,
                        help="Number of diffusion steps (default: model's full T)")
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = Config()
    cfg.seed             = args.seed
    cfg.lvae_checkpoint  = args.lvae_checkpoint
    device = cfg.device
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")

    # Load frozen BAE + LVAE (needed to reconstruct DDM_LVAE_3D)
    bae_model  = load_bae(cfg)
    lvae_model = load_lvae(cfg, bae_model)

    unet = build_unet(cfg)

    # Load the sampler that was saved inside the checkpoint so that the noise
    # schedule (cosine vs linear, beta range, etc.) matches what the model was
    # trained with.  build_sampler(cfg) would use default Config() settings and
    # silently produce a schedule mismatch, causing NaN outputs at inference.
    import torch as _torch
    _ckpt_raw = _torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sampler = _ckpt_raw["sampler"]

    ddm_model = DDM_LVAE_3D(
        unet=unet,
        sampler=sampler,
        bae_model=bae_model,
        lvae_model=lvae_model,
        w_pressure=1.0,
        lvae_params_dim=cfg.lvae_params_dim,
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

    # Test split
    new_dataset = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=cfg.seed)
    all_test    = list(new_dataset["test"])
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}
    test_dataset    = [item for item in all_test if item["final"]   == 1]
    print(f"Test split : {len(test_dataset)} wings")

    # Pre-compute ground truth
    print("Pre-computing ground truth...")
    gt_coords, gt_aoas, gt_pressures, encoded_inits, params_list, _, _ = precompute_test(
        test_dataset, initial_by_case, bae_model, ddm_model, device
    )
    n_test = gt_coords.shape[0]
    print(f"Valid test samples: {n_test}")

    # N forward passes
    all_metrics = []
    for pass_idx in range(args.n_passes):
        print(f"Pass {pass_idx + 1}/{args.n_passes}...")
        gen_coords_list    = []
        gen_aoas_list      = []
        gen_pressures_list = []

        for start in range(0, n_test, args.batch_size):
            end          = min(start + args.batch_size, n_test)
            params_batch = torch.cat(params_list[start:end], dim=0)

            coords_b, aoas_b, pres_b = generate_batch(
                ddm_model, bae_model,
                encoded_inits[start:end], params_batch, device,
                T=args.T,
            )
            gen_coords_list.append(coords_b)
            gen_pressures_list.append(pres_b)
            gen_aoas_list.extend(np.atleast_1d(aoas_b).tolist())

        gen_coords    = torch.cat(gen_coords_list)    # [N, S, 2, 192]
        gen_pressures = torch.cat(gen_pressures_list) # [N, S, 192]
        gen_aoas_t    = torch.tensor(gen_aoas_list)

        # Drop samples that are non-finite or have exploded to unreasonable magnitudes
        valid_coords = torch.isfinite(gen_coords).all(dim=(1, 2, 3)) & (gen_coords.abs().amax(dim=(1,2,3)) < 1e3)
        valid_aoa    = torch.isfinite(gen_aoas_t) & (gen_aoas_t.abs() < 1e3)
        valid_pres   = torch.isfinite(gen_pressures).all(dim=(1, 2)) & (gen_pressures.abs().amax(dim=(1,2)) < 1e3)
        print(f"  [debug] valid coords={valid_coords.sum()}, aoa={valid_aoa.sum()}, pres={valid_pres.sum()}")
        valid = valid_coords & valid_aoa & valid_pres
        if (~valid).any():
            print(f"  Dropping {(~valid).sum().item()} NaN samples")
        gen_coords    = gen_coords[valid]
        gen_pressures = gen_pressures[valid]
        gen_aoas_t    = gen_aoas_t[valid]
        gt_c          = gt_coords[valid]
        gt_p          = gt_pressures[valid]
        gt_a          = gt_aoas[valid]

        if gen_coords.shape[0] == 0:
            print("  Skipping pass — no valid samples.")
            continue

        metrics = compute_metrics(gen_coords, gt_c, gen_aoas_t, gt_a, gen_pressures, gt_p)
        all_metrics.append(metrics)
        print(
            f"  Shape MSE {metrics['shape_mse']:.2e} | "
            f"AoA MSE {metrics['aoa_mse']:.4f} | "
            f"MMD {metrics['mmd']:.4f} | "
            f"Vendi {metrics['vendi']:.3f} | "
            f"Pressure MSE {metrics.get('pressure_mse', float('nan')):.4f}"
        )

    # Aggregate — report median (robust to rare outlier passes) alongside mean±std
    keys = ["shape_mse", "aoa_mse", "mmd", "vendi", "pressure_mse"]
    results = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m]
        if vals:
            results[k] = {
                "mean":   float(np.mean(vals)),
                "std":    float(np.std(vals)),
                "median": float(np.median(vals)),
                "all":    vals,
            }

    print("\n" + "=" * 60)
    print(f"RESULTS  (median | mean ± std over {len(all_metrics)} passes)")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:<16}: median {v['median']:.4e}  |  mean {v['mean']:.4e} ± {v['std']:.4e}")
    print("=" * 60)

    # Save
    save_dir = os.path.join("results", "evaluation")
    os.makedirs(save_dir, exist_ok=True)
    timestamp   = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_name  = os.path.splitext(os.path.basename(args.checkpoint))[0]
    base_path   = os.path.join(save_dir, f"eval_lvae_{model_name}_{timestamp}")

    with open(base_path + ".txt", "w") as f:
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"Test wings : {n_test}\n")
        f.write(f"Passes     : {len(all_metrics)}\n\n")
        for k, v in results.items():
            f.write(f"{k:<16}: median {v['median']:.4e}  |  mean {v['mean']:.4e} ± {v['std']:.4e}\n")

    with open(base_path + ".json", "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "n_test":     n_test,
            "n_passes":   len(all_metrics),
            "results":    {k: {kk: vv for kk, vv in v.items() if kk != "all"} for k, v in results.items()},
        }, f, indent=2)

    print(f"\nSaved to {base_path}.txt / .json")


if __name__ == "__main__":
    main()
