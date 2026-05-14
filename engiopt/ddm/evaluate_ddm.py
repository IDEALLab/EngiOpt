import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import torch
from engibench.problems.wings3D.v0 import Wings3D

from engiopt.ddm.ddm import DDM_AoAInit_3D
from engiopt.ddm.train_ddm import Config, build_sampler, build_unet, load_bae

# ── Evaluation settings ───────────────────────────────────────────────────────
N_FORWARD_PASSES = 10
GAMMAS = [0.5, 25, 50, 100]
VOLUME_THRESHOLD = 0.75
BATCH_SIZE = 32  # tune down if you hit OOM, up if GPU memory is underused


# ── Metric helpers ────────────────────────────────────────────────────────────

def gaussian_kernel(x, y, gamma):
    """x: [N, D], y: [M, D] → [N, M]"""
    diff = x.unsqueeze(1) - y.unsqueeze(0)      # [N, M, D]
    sq_dist = (diff ** 2).sum(-1)               # [N, M]
    return torch.exp(-gamma * sq_dist)


def compute_mmd(generated, real, gamma):
    """Maximum Mean Discrepancy with Gaussian kernel."""
    n, m = generated.shape[0], real.shape[0]
    Kxx = gaussian_kernel(generated, generated, gamma)
    Kyy = gaussian_kernel(real, real, gamma)
    Kxy = gaussian_kernel(generated, real, gamma)
    return (Kxx.sum() / (n * n) - 2 * Kxy.sum() / (n * m) + Kyy.sum() / (m * m)).item()


def compute_vendi(samples, gamma):
    """
    Vendi Score: exp(Shannon entropy of normalised eigenvalues of kernel matrix).
    Normalised by dividing by Vendi of the real set to get a value in [0, 1].
    """
    valid = torch.isfinite(samples).all(dim=-1)
    samples = samples[valid]
    if samples.shape[0] < 2:
        return float("nan")

    K = gaussian_kernel(samples, samples, gamma)
    K = K / samples.shape[0]
    K = K + 1e-4 * torch.eye(K.shape[0], device=K.device)

    try:
        eigenvalues = torch.linalg.eigvalsh(K).clamp(min=1e-10)
    except torch._C._LinAlgError:
        K_cpu = K.double().cpu()
        K_cpu = (K_cpu + K_cpu.T) / 2
        eigenvalues = torch.linalg.eigvalsh(K_cpu).clamp(min=1e-10).to(samples.device).float()

    eigenvalues = eigenvalues / eigenvalues.sum()
    ##eigenvalues = torch.linalg.eigvalsh(K).clamp(min=1e-10)
    ##eigenvalues = eigenvalues / eigenvalues.sum()
    return (-(eigenvalues * eigenvalues.log()).sum()).exp().item()


def compute_metrics_for_pass(generated, gt_airfoils, gen_aoas, gt_aoas):
    """
    generated:   [N, 9, 2, 192]
    gt_airfoils: [N, 9, 2, 192]
    gen_aoas:    [N]
    gt_aoas:     [N]
    """
    # Shape MSE — averaged per slice then across slices
    shape_mse = 0.0
    mmd_vals_all, vendi_gen_vals_all, vendi_gt_vals_all = [], [], []

    for s in range(9):
        gen_s = generated[:, s, :, :]   # [N, 2, 192]
        gt_s  = gt_airfoils[:, s, :, :] # [N, 2, 192]

        # MSE for this slice
        shape_mse += ((gen_s - gt_s) ** 2).mean().item()

        # Flatten for kernel metrics: [N, 384]
        gen_flat = gen_s.reshape(gen_s.shape[0], -1)
        gt_flat  = gt_s.reshape(gt_s.shape[0], -1)

        mmd_slice, vendi_gen_slice, vendi_gt_slice = [], [], []
        for gamma in GAMMAS:
            mmd_slice.append(compute_mmd(gen_flat, gt_flat, gamma))
            vendi_gen_slice.append(compute_vendi(gen_flat, gamma))
            vendi_gt_slice.append(compute_vendi(gt_flat, gamma))

        mmd_vals_all.append(float(np.mean(mmd_slice)))
        vendi_gen_vals_all.append(float(np.mean(vendi_gen_slice)))
        vendi_gt_vals_all.append(float(np.mean(vendi_gt_slice)))

    # Average across slices
    shape_mse /= 9
    mmd = float(np.mean(mmd_vals_all))
    vendi_gen = float(np.mean(vendi_gen_vals_all))
    vendi_gt  = float(np.mean(vendi_gt_vals_all))
    vendi_normalised = vendi_gen / vendi_gt if vendi_gt > 0 else 0.0

    # AoA MSE
    aoa_mse = ((gen_aoas - gt_aoas) ** 2).mean().item()

    return {
        "shape_mse": shape_mse,
        "aoa_mse": aoa_mse,
        "mmd": mmd,
        "vendi": vendi_normalised,
    }

# ── Batched generation helper ─────────────────────────────────────────────────

def generate_batch(ddm_model, bae_model, encoded_inits_batch, params_batch, device):
    B = params_batch.shape[0]
    noise_x     = torch.randn(B, 9, 3, 30, device=device)
    noise_alpha = torch.randn(B, 1,        device=device)

    # Normalize x0 to match the latent space seen during training
    lat_mean = ddm_model.latent_mean
    lat_std  = ddm_model.latent_std
    if isinstance(lat_mean, torch.Tensor):
        lat_mean = lat_mean.to(device)
        lat_std  = lat_std.to(device)
        # encoded_batch: [B, 3, L] — use [1, 3, 1] slice of [1, 1, 3, 1]
        lat_mean_init = lat_mean[0]
        lat_std_init  = lat_std[0]
    else:
        lat_mean_init = lat_mean
        lat_std_init  = lat_std
    encoded_batch = torch.cat(encoded_inits_batch, dim=0)
    encoded_batch_norm = (encoded_batch - lat_mean_init) / lat_std_init

    with torch.no_grad():
        gen_z, gen_alpha = ddm_model.sampler.sample_airfoil(
            model=ddm_model.unet,
            noise_x=noise_x,
            noise_alpha=noise_alpha,
            c=params_batch,
            x0=encoded_batch_norm,
            T=None,
        )
        # Decode centered geometry slices
        # gen_z is in normalized latent space — denormalize before decoding
        gen_z_raw = gen_z * lat_std + lat_mean
        print(f"[DEBUG] gen_z mean={gen_z.mean():.3f} std={gen_z.std():.3f}")
        print(f"[DEBUG] gen_z_raw mean={gen_z_raw.mean():.3f} std={gen_z_raw.std():.3f}")
        print(f"[DEBUG] lat_mean={lat_mean}, lat_std={lat_std}")
        decoded_slices = []
        for s in range(9):
            z_s = gen_z_raw[:, s, :, :].clone()
            z_s[:, 0, :] = z_s[:, 0, :].clamp(0.1, 2.0)    # weight channel
            z_s[:, 1:, :] = z_s[:, 1:, :].clamp(-2.228, 3.117) # CP channels
            dec_s = bae_model.decode_z(
                z_s,
                z_ae_mode=True,
                denormalize_output=False,
                normalized_data=False,
            )[0]  # [B, 2, 192]
            decoded_slices.append(dec_s)
        decoded = torch.stack(decoded_slices, dim=1)  # [B, 9, 2, 192]

    gen_alpha_rescaled = ddm_model.scaler_aoas.inverse_transform(
        gen_alpha.cpu().numpy()
    )
    return decoded.cpu(), gen_alpha_rescaled.squeeze()


# ── Main evaluation ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint file. If omitted, uses default model.")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Training data size (used only for labelling output files)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed used during training (used only for labelling output files)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    # Resolve checkpoint path and model name
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        cfg.model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        cfg.save_dir = os.path.dirname(checkpoint_path)
    else:
        cfg.model_name = "ddm_v7hope(250:1)"
        cfg.save_dir = "results/ddm"
        checkpoint_path = os.path.join(cfg.save_dir, f"{cfg.model_name}.pth")

    device = cfg.device
    print(f"Using device: {device}")
    print(f"Loading model: {cfg.model_name}")

    # Load model
    bae_model = load_bae(cfg)
    unet = build_unet(cfg)
    sampler = build_sampler(cfg)

    ddm_model = DDM_AoAInit_3D(
        unet=unet,
        sampler=sampler,
        bae_model=bae_model,
        params_mean_std=(0, 1),
        aoas_mean_std=(0, 1),
        name=cfg.model_name,
        opt_lr=cfg.lr,
    )

    print(f"Loading checkpoint from {checkpoint_path}...")
    ddm_model.load(checkpoint_path, train_mode=False)
    ddm_model.unet = ddm_model.unet.to(device)
    ddm_model.bae_model = ddm_model.bae_model.to(device)
    bae_model = ddm_model.bae_model

    # Load test split
    problem = Wings3D(seed=cfg.seed)
    test_dataset = [item for item in problem.dataset["test"] if item["final"] == 1]
    initial_by_case = {item["case_num"]: item for item in problem.dataset["test"] if item["initial"] == 1}
    n_test = len(test_dataset)
    print(f"Test set: {n_test} wings, {len(initial_by_case)} initial cases")

    # ── Pre-compute ground truth and encoded inits ────────────────────────────
    print("Pre-computing ground truth encodings...")
    gt_airfoils = []
    gt_aoas = []
    encoded_inits = []
    params_list = []

    for i in range(n_test):
        item = test_dataset[i]
        coords = torch.tensor(item["coords"], dtype=torch.float32)  # [9, 192, 2]

        coords_centered = coords.clone()
        for s in range(9):
            coords_centered[s, :, 1] -= coords[s, 0, 1]
            coords_centered[s, :, 0] += (1.0 - coords[s, 0, 0])

        x_opt_slices = []
        for s in range(9):
            x_s = coords_centered[s].permute(1, 0).unsqueeze(0).to(device)  # [1, 2, 192]
            with torch.no_grad():
                z_s = bae_model.encode(x_s, return_z=True, z_ae_mode=False)[:, :, 1:-1]
                dec_s = bae_model.decode_z(z_s, z_ae_mode=True,
                                            denormalize_output=False,
                                            normalized_data=False)[0]
            x_opt_slices.append(dec_s.squeeze(0).cpu())  # [2, 192]

        x_opt = torch.stack(x_opt_slices, dim=0)  # [9, 2, 192]

        # Use the initial (unoptimized) root slice as z_init — matches training
        case_num = int(item["case_num"])
        if case_num in initial_by_case:
            init_coords = torch.tensor(initial_by_case[case_num]["coords"], dtype=torch.float32)
            init_root = init_coords[0].clone()
        else:
            print(f"[WARNING] No initial wing found for case {case_num}, falling back to final root slice")
            init_root = coords[0].clone()
        init_root[:, 1] -= init_root[0, 1]  # y-shift only, no x-shift (matches training)
        x_init = init_root.permute(1, 0).unsqueeze(0).to(device)

        with torch.no_grad():
            z_init = bae_model.encode(x_init, return_z=True, z_ae_mode=False)[:, :, 1:-1]
        if torch.isnan(z_init).any() or torch.isinf(z_init).any():
            print(f"[WARNING] Skipping test sample {i} — NaN/Inf in encoded init")
            continue
        params = torch.tensor(
            [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)
        params_scaled = ddm_model.scaler_params.transform(params)

        gt_airfoils.append(x_opt)
        gt_aoas.append(float(item["alpha"]))
        encoded_inits.append(z_init)
        params_list.append(params_scaled)

    gt_airfoils = torch.stack(gt_airfoils)   # [N, 9, 2, 192]
    gt_aoas_t = torch.tensor(gt_aoas)

    # ── 10 forward passes (batched) ───────────────────────────────────────────
    pass_metrics = []

    for pass_idx in range(N_FORWARD_PASSES):
        print(f"Forward pass {pass_idx + 1}/{N_FORWARD_PASSES}...")
        gen_airfoils = []
        gen_aoas = []

        for start in range(0, n_test, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_test)
            params_batch = torch.cat(params_list[start:end], dim=0)  # [B, 4]

            airfoil_batch, aoa_batch = generate_batch(
                ddm_model, bae_model,
                encoded_inits[start:end], params_batch, device,
            )
            gen_airfoils.append(airfoil_batch)
            # aoa_batch may be scalar if B==1, so wrap safely
            gen_aoas.extend(np.atleast_1d(aoa_batch).tolist())

        gen_airfoils_t = torch.cat(gen_airfoils)   # [N, 9, 2, 192]
        gen_aoas_t = torch.tensor(gen_aoas)

        nan_mask = ~torch.isfinite(gen_airfoils_t).all(dim=(1, 2, 3))
        if nan_mask.any():
            print(f"  [WARNING] {nan_mask.sum().item()} NaN/Inf generated samples dropped before metrics")
            valid = ~nan_mask
            gen_airfoils_t = gen_airfoils_t[valid]
            gen_aoas_t = gen_aoas_t[valid]
            gt_airfoils_pass = gt_airfoils[valid]
            gt_aoas_pass = gt_aoas_t[valid]
        else:
            gt_airfoils_pass = gt_airfoils
            gt_aoas_pass = gt_aoas_t

        metrics = compute_metrics_for_pass(
            gen_airfoils_t, gt_airfoils_pass, gen_aoas_t, gt_aoas_pass
        )
        pass_metrics.append(metrics)

        print(
            f"  Shape MSE: {metrics['shape_mse']:.2e} | "
            f"AoA MSE: {metrics['aoa_mse']:.3f} | "
            f"MMD: {metrics['mmd']:.4f} | "
            f"Vendi: {metrics['vendi']:.3f}"
        )

    # ── Aggregate results ─────────────────────────────────────────────────────
    keys = ["shape_mse", "aoa_mse", "mmd", "vendi"]
    results = {
        k: (
            float(np.mean([m[k] for m in pass_metrics])),
            float(np.std([m[k] for m in pass_metrics])),
        )
        for k in keys
    }

    print("\n" + "=" * 55)
    print("EVALUATION RESULTS (mean ± std over 10 passes)")
    print("=" * 55)
    print(f"  Shape MSE : {results['shape_mse'][0]:.2e} ± {results['shape_mse'][1]:.2e}")
    print(f"  AoA MSE   : {results['aoa_mse'][0]:.4f} ± {results['aoa_mse'][1]:.4f}")
    print(f"  MMD       : {results['mmd'][0]:.4f} ± {results['mmd'][1]:.4f}")
    print(f"  Vendi     : {results['vendi'][0]:.4f} ± {results['vendi'][1]:.4f}")
    print("=" * 55)

    # ── Save results ──────────────────────────────────────────────────────────
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("results", "evaluation")
    os.makedirs(save_dir, exist_ok=True)

    # Build a label for the output files
    if args.n_samples is not None and args.seed is not None:
        label = f"n{args.n_samples}_s{args.seed}"
    elif args.n_samples is not None:
        label = f"n{args.n_samples}"
    else:
        label = timestamp

    results_path = os.path.join(save_dir, f"eval_{label}.txt")
    with open(results_path, "w") as f:
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test wings: {n_test}\n")
        f.write(f"Forward passes: {N_FORWARD_PASSES}\n")
        f.write(f"Gammas: {GAMMAS}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n\n")
        f.write("RESULTS (mean ± std)\n")
        f.write(f"Shape MSE : {results['shape_mse'][0]:.2e} ± {results['shape_mse'][1]:.2e}\n")
        f.write(f"AoA MSE   : {results['aoa_mse'][0]:.4f} ± {results['aoa_mse'][1]:.4f}\n")
        f.write(f"MMD       : {results['mmd'][0]:.4f} ± {results['mmd'][1]:.4f}\n")
        f.write(f"Vendi     : {results['vendi'][0]:.4f} ± {results['vendi'][1]:.4f}\n")

    # Save JSON for aggregation script
    json_path = os.path.join(save_dir, f"eval_{label}.json")
    with open(json_path, "w") as f:
        json.dump({
            "checkpoint": checkpoint_path,
            "n_samples": args.n_samples,
            "seed": args.seed,
            "n_test": n_test,
            "results": {k: {"mean": v[0], "std": v[1]} for k, v in results.items()},
        }, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()