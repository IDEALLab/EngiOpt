"""
Evaluation script for DDM_PCA checkpoints.

Usage
-----
    python -m engiopt.ddm.ddm_pca.evaluate_ddm_pca \
        --checkpoint results/ddm_pca/ddm_pca_v1_best.pth \
        [--n_passes 10] [--seed 0] [--T 1000]
"""

import argparse
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import torch

from engiopt.ddm.ddm_pca.ddm_pca import DDM_PCA, MLPDenoiser
from engiopt.ddm.ddm_pca.train_ddm_pca import (
    Config, load_bae, build_sampler, precompute_bae_latents,
)
from engiopt.data_processing.utils import scaler
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

N_FORWARD_PASSES = 10
GAMMAS = [0.5, 25, 50, 100]


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


def compute_metrics(generated, gt_airfoils, gen_aoas, gt_aoas):
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

    return {"shape_mse": shape_mse, "aoa_mse": aoa_mse, "mmd": mmd, "vendi": vendi_norm}


def precompute_test(test_dataset, initial_by_case, bae_model, pca,
                    z_mean, z_std, scaler_params, scaler_aoas,
                    cfg, device):
    """Encode test set through BAE → PCA, return normalised z_inits and gt coords."""
    gt_coords_list = []
    gt_aoas_list   = []
    z_inits_list   = []
    params_list    = []

    bae_model.eval()
    with torch.no_grad():
        for item in test_dataset:
            coords    = torch.tensor(item["coords"],    dtype=torch.float32)
            te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)

            coords_c = coords.clone()
            coords_c[:, :, 1] -= te_shifts.unsqueeze(1)
            te_x = coords_c[:, 0, 0]
            coords_c[:, :, 0] += (1.0 - te_x).unsqueeze(1)

            S = coords_c.shape[0]
            gt_slices = []
            z_slices  = []
            for s in range(S):
                x_s = coords_c[s].permute(1, 0).unsqueeze(0).to(device)
                z_s = bae_model.encode(x_s, return_z=True, z_ae_mode=True)
                dec, _, _ = bae_model.decode_z(z_s, z_ae_mode=True,
                                               denormalize_output=False,
                                               normalized_data=False)
                gt_slices.append(dec.squeeze(0).cpu())
                z_slices.append(z_s.squeeze(0).cpu())
            gt_coords_list.append(torch.stack(gt_slices))
            gt_aoas_list.append(torch.tensor(float(item["alpha"])))

            # w_init via PCA
            case_num = int(item["case_num"])
            if case_num in initial_by_case:
                init = initial_by_case[case_num]
                ic   = torch.tensor(init["coords"],    dtype=torch.float32)
                it   = torch.tensor(init["te_shifts"], dtype=torch.float32)
                ir   = ic[0].clone()
                ir[:, 1] -= it[0]; ir[:, 0] += (1.0 - ir[0, 0])
                x_i  = ir.permute(1, 0).unsqueeze(0).to(device)
                z_i  = bae_model.encode(x_i, return_z=True, z_ae_mode=True).squeeze(0).cpu()
                z_init_flat = z_i.unsqueeze(0).expand(S, -1, -1).flatten()
            else:
                z_init_flat = torch.stack(z_slices).flatten()

            z_init_pca  = torch.tensor(pca.transform(z_init_flat.unsqueeze(0).numpy()),
                                       dtype=torch.float32).squeeze(0)
            z_init_norm = (z_init_pca - z_mean.squeeze(0)) / z_std.squeeze(0)
            z_inits_list.append(z_init_norm)

            flow = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
            flow_t = torch.tensor(flow, dtype=torch.float32)
            params_norm = scaler_params.transform(flow_t)
            params_list.append(params_norm)

    return (
        torch.stack(gt_coords_list),
        torch.stack(gt_aoas_list),
        torch.stack(z_inits_list),
        torch.stack(params_list),
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--n_passes",   type=int, default=N_FORWARD_PASSES)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--T",          type=int, default=None)
    p.add_argument("--out_dir",    type=str, default="results/evaluation")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = Config()
    device = cfg.device
    os.makedirs(args.out_dir, exist_ok=True)

    bae_model = load_bae(cfg)

    # Load checkpoint + PCA
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    # PCA is saved once per model name (without _best suffix)
    pca_path = args.checkpoint.replace('_best.pth', '_pca.pkl').replace('.pth', '_pca.pkl')
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)

    z_dim = ckpt.get('z_dim', cfg.n_components)
    pms   = ckpt.get('params_mean_std')
    ams   = ckpt.get('aoas_mean_std')

    denoiser = MLPDenoiser(z_dim=z_dim, c_dim=cfg.c_dim).to(device)

    ddm_pca = DDM_PCA(
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

    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=args.seed)
    all_test     = list(new_dataset["test"])
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}
    test_dataset    = [item for item in all_test if item["final"] == 1]
    print(f"Test set: {len(test_dataset)} wings")

    gt_coords, gt_aoas, z_inits, params_norm = precompute_test(
        test_dataset, initial_by_case, bae_model, pca,
        z_mean, z_std, ddm_pca.scaler_params, ddm_pca.scaler_aoas,
        cfg, device,
    )
    N = gt_coords.shape[0]
    print(f"Pre-computed GT for {N} test wings.")

    all_coords = []
    all_aoas   = []
    for pass_i in range(args.n_passes):
        coords_pass, aoas_pass = ddm_pca.generate(
            z_init=z_inits.to(device),
            params=params_norm.to(device),
            device=device, T=args.T,
        )
        all_coords.append(coords_pass)
        all_aoas.append(aoas_pass)
        print(f"  Pass {pass_i+1}/{args.n_passes} done.")

    gen_coords = torch.stack(all_coords, dim=0).mean(0)
    gen_aoas   = torch.stack(all_aoas,   dim=0).mean(0)

    metrics = compute_metrics(gen_coords, gt_coords, gen_aoas, gt_aoas)

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

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
