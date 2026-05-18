"""
Evaluation script for DDM_W checkpoints.

Usage
-----
    python -m engiopt.ddm.ddm_w.evaluate_ddm_w \
        --checkpoint results/ddm_w/ddm_w_v1_best.pth \
        --lvae_checkpoint results/lvae/lae_dropout_0.25_flow_only_best.pth \
        [--n_passes 10] [--seed 0]
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import torch

from engiopt.ddm.ddm_w.ddm_w import DDM_W, MLPDenoiser
from engiopt.ddm.ddm_w.train_ddm_w import (
    Config, load_bae, load_lvae, build_sampler, precompute_w,
)
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

N_FORWARD_PASSES = 10
GAMMAS = [0.5, 25, 50, 100]


# ---------------------------------------------------------------------------
# Metric helpers (identical to evaluate_ddm_lvae.py)
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
    n_slices  = generated.shape[1]
    shape_mse = 0.0
    mmd_vals, vendi_gen_vals, vendi_gt_vals = [], [], []

    for s in range(n_slices):
        gen_s = generated[:, s]
        gt_s  = gt_airfoils[:, s]
        shape_mse += ((gen_s - gt_s) ** 2).mean().item()
        gen_flat = gen_s.reshape(gen_s.shape[0], -1)
        gt_flat  = gt_s.reshape(gt_s.shape[0],  -1)
        mmd_vals.append(float(np.mean([compute_mmd(gen_flat, gt_flat, g) for g in GAMMAS])))
        vendi_gen_vals.append(float(np.nanmean([compute_vendi(gen_flat, g) for g in GAMMAS])))
        vendi_gt_vals.append(float(np.nanmean([compute_vendi(gt_flat,  g) for g in GAMMAS])))

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
# Ground-truth pre-computation
# ---------------------------------------------------------------------------

def precompute_test(test_dataset, initial_by_case, bae_model, lvae_model,
                    lvae_params_scaler, lvae_params_dim, w_mean, w_std,
                    scaler_params, scaler_aoas, device):
    """
    Returns parallel lists/tensors for the test set:
      gt_coords    : [N, S, 2, 192]
      gt_aoas      : [N]
      gt_pressures : [N, S, 192]
      w_inits_norm : [N, w_dim]   — normalised w_init for each test wing
      params_norm  : [N, c_dim]   — normalised flow params
      te_shifts    : list of [S] tensors
      etas         : list of [S] tensors (spanwise positions)
    """
    gt_coords_list  = []
    gt_aoas_list    = []
    gt_pressures_list = []
    w_inits_list    = []
    params_list     = []
    te_shifts_list  = []
    etas_list       = []

    bae_model.eval()
    lvae_model.encoder.eval()

    with torch.no_grad():
        for item in test_dataset:
            coords    = torch.tensor(item["coords"],    dtype=torch.float32)
            te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)
            pressure  = torch.tensor(item["coef_pressure"], dtype=torch.float32)

            S = coords.shape[0]

            # BAE encode (centred)
            coords_c = coords.clone()
            coords_c[:, :, 1] -= te_shifts.unsqueeze(1)
            te_x = coords_c[:, 0, 0]
            coords_c[:, :, 0] += (1.0 - te_x).unsqueeze(1)

            gt_slices = []
            for s in range(S):
                x_s = coords_c[s].permute(1, 0).unsqueeze(0).to(device)
                z_s = bae_model.encode(x_s, return_z=True, z_ae_mode=True)
                dec, _, _ = bae_model.decode_z(z_s, z_ae_mode=True,
                                               denormalize_output=False,
                                               normalized_data=False)
                gt_slices.append(dec.squeeze(0).cpu())
            gt_coords_list.append(torch.stack(gt_slices))

            gt_pressures_list.append(pressure)
            gt_aoas_list.append(torch.tensor(float(item["alpha"])))
            te_shifts_list.append(te_shifts)

            # eta (spanwise positions)
            if "eta" in item:
                etas_list.append(torch.tensor(item["eta"], dtype=torch.float32))
            else:
                etas_list.append(torch.linspace(0, 1, S))

            # w_init for this test wing
            case_num = int(item["case_num"])
            flow = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
            flow_np = np.array(flow, dtype=np.float32).reshape(1, -1)
            if lvae_params_scaler is not None:
                flow_lvae_np = lvae_params_scaler.transform(flow_np)
            else:
                flow_lvae_np = flow_np
            flow_lvae = torch.tensor(flow_lvae_np, dtype=torch.float32, device=device)

            if case_num in initial_by_case:
                init = initial_by_case[case_num]
                ic   = torch.tensor(init["coords"],    dtype=torch.float32)
                it   = torch.tensor(init["te_shifts"], dtype=torch.float32)
                ir   = ic[0].clone()
                ir[:, 1] -= it[0]; ir[:, 0] += (1.0 - ir[0, 0])
                x_i = ir.permute(1, 0).unsqueeze(0).to(device)
                z_i = bae_model.encode(x_i, return_z=True, z_ae_mode=True)
                z_init_wing = z_i.unsqueeze(1).expand(1, S, -1, -1)
            else:
                # Encode target wing as fallback
                z_slices2 = []
                for s in range(S):
                    x_s = coords_c[s].permute(1, 0).unsqueeze(0).to(device)
                    z_slices2.append(bae_model.encode(x_s, return_z=True, z_ae_mode=True).squeeze(0))
                z_init_wing = torch.stack(z_slices2, dim=0).unsqueeze(0)

            lvae_model.encoder.to(device)
            w_init = lvae_model.encoder(z_init_wing, flow_lvae).squeeze(0).cpu()
            w_init_norm = (w_init - w_mean.squeeze(0)) / w_std.squeeze(0)
            w_inits_list.append(w_init_norm)

            flow_t = torch.tensor(flow, dtype=torch.float32)
            params_norm = scaler_params.transform(flow_t)
            params_list.append(params_norm)

    return (
        torch.stack(gt_coords_list),    # [N, S, 2, 192]
        torch.stack(gt_aoas_list),      # [N]
        torch.stack(gt_pressures_list), # [N, S, 192]
        torch.stack(w_inits_list),      # [N, w_dim]
        torch.stack(params_list),       # [N, c_dim]
        te_shifts_list,
        etas_list,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      type=str, required=True)
    p.add_argument("--lvae_checkpoint", type=str, required=True)
    p.add_argument("--n_passes",        type=int, default=N_FORWARD_PASSES)
    p.add_argument("--seed",            type=int, default=0)
    p.add_argument("--T",               type=int, default=None)
    p.add_argument("--out_dir",         type=str, default="results/evaluation")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = Config()
    cfg.lvae_checkpoint = args.lvae_checkpoint

    device = cfg.device
    os.makedirs(args.out_dir, exist_ok=True)

    # Load frozen models
    bae_model  = load_bae(cfg)
    lvae_model = load_lvae(cfg, bae_model)
    lvae_params_scaler = getattr(lvae_model, 'scaler_params', None)

    # Load DDM_W checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    denoiser = MLPDenoiser(w_dim=cfg.lvae_lae_latent_dim, c_dim=cfg.c_dim).to(device)
    sampler  = build_sampler(cfg)

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
    print("DDM_W loaded.")

    w_mean = ddm_w.w_mean  # [1, 64]
    w_std  = ddm_w.w_std

    # Test dataset
    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_test     = list(new_dataset["test"])
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}
    test_dataset    = [item for item in all_test if item["final"] == 1]
    print(f"Test set: {len(test_dataset)} wings")

    gt_coords, gt_aoas, gt_pressures, w_inits, params_norm, te_shifts_list, etas_list = \
        precompute_test(
            test_dataset, initial_by_case, bae_model, lvae_model,
            lvae_params_scaler, cfg.lvae_params_dim, w_mean, w_std,
            ddm_w.scaler_params, ddm_w.scaler_aoas, device,
        )
    N = gt_coords.shape[0]
    print(f"Pre-computed GT for {N} test wings.")

    # Generate N_passes times and average metrics
    all_coords     = []
    all_aoas       = []
    all_pressures  = []
    all_te_shifts  = []

    for pass_i in range(args.n_passes):
        coords_pass, aoas_pass, pres_pass, te_pass = ddm_w.generate(
            w_init=w_inits.to(device),
            params=params_norm.to(device),
            device=device,
            T=args.T,
        )
        all_coords.append(coords_pass)
        all_aoas.append(aoas_pass)
        all_pressures.append(pres_pass)
        all_te_shifts.append(te_pass)
        print(f"  Pass {pass_i+1}/{args.n_passes} done.")

    gen_coords    = torch.stack(all_coords,    dim=0).mean(0)  # [N, S, 2, 192]
    gen_aoas      = torch.stack(all_aoas,      dim=0).mean(0)  # [N]
    gen_pressures = torch.stack(all_pressures, dim=0).mean(0)  # [N, S, 192]
    gen_te_shifts = torch.stack(all_te_shifts, dim=0).mean(0)  # [N, S]

    # Re-apply te_shifts to both gen and GT coords for fair comparison
    for i in range(N):
        shifts = gen_te_shifts[i]               # [S]  — from LVAE eta_y_pred
        gen_coords[i, :, 1, :] += shifts.unsqueeze(-1)
        gt_shifts = te_shifts_list[i]
        gt_coords[i,  :, 1, :] += gt_shifts.unsqueeze(-1)

    metrics = compute_metrics(
        gen_coords, gt_coords, gen_aoas, gt_aoas,
        gen_pressures, gt_pressures,
    )

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    # Save results
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    base = os.path.join(args.out_dir, f"eval_{stem}_{ts}")

    metrics["checkpoint"] = args.checkpoint
    metrics["n_passes"]   = args.n_passes
    metrics["n_test"]     = N

    with open(base + ".json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(base + ".txt", "w") as f:
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"n_passes   : {args.n_passes}\n")
        f.write(f"n_test     : {N}\n\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")

    print(f"Saved to {base}.json / .txt")


if __name__ == "__main__":
    main()
