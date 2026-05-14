"""
Evaluation script for the LAE.
Computes geometry metrics (shape MSE, AoA MSE, MMD, Vendi) and
the new pressure / performance metrics (pressure MSE, cd MSE, cl MSE).
"""

import os
from datetime import datetime, timezone

import numpy as np
import torch
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

_SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
_SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

from engiopt.lvae.lvae import LAE_AoAInit
from engiopt.lvae.train_lvae import Config, build_encoder, build_decoder, build_sampler, load_bae, ensure_perf_regressor_fitted
from engiopt.lvae import plotting as lvae_plotting

# ── Evaluation settings ───────────────────────────────────────────────────────
N_FORWARD_PASSES = 1   # deterministic model — multiple passes are identical
GAMMAS = [0.5, 25, 50, 100]
BATCH_SIZE = 64


# ── Geometry metric helpers ───────────────────────────────────────────────────

def gaussian_kernel(x, y, gamma):
    diff   = x.unsqueeze(1) - y.unsqueeze(0)
    sq_dist = (diff ** 2).sum(-1)
    return torch.exp(-gamma * sq_dist)


def compute_mmd(generated, real, gamma):
    n, m = generated.shape[0], real.shape[0]
    Kxx  = gaussian_kernel(generated, generated, gamma)
    Kyy  = gaussian_kernel(real,      real,      gamma)
    Kxy  = gaussian_kernel(generated, real,      gamma)
    return (Kxx.sum() / (n * n) - 2 * Kxy.sum() / (n * m) + Kyy.sum() / (m * m)).item()


def compute_vendi(samples, gamma):
    K           = gaussian_kernel(samples, samples, gamma) / samples.shape[0]
    eigenvalues = torch.linalg.eigvalsh(K).clamp(min=1e-10)
    eigenvalues = eigenvalues / eigenvalues.sum()
    return (-(eigenvalues * eigenvalues.log()).sum()).exp().item()


def compute_geometry_metrics(generated, gt_airfoils, gen_aoas, gt_aoas):
    """
    generated:   [N, 9, 2, 192]
    gt_airfoils: [N, 9, 2, 192]
    gen_aoas:    [N]
    gt_aoas:     [N]
    """
    shape_mse = 0.0
    mmd_vals, vendi_gen_vals, vendi_gt_vals = [], [], []

    n_slices = generated.shape[1]
    for s in range(n_slices):
        gen_s    = generated[:, s, :, :]
        gt_s     = gt_airfoils[:, s, :, :]
        shape_mse += ((gen_s - gt_s) ** 2).mean().item()

        gen_flat = gen_s.reshape(gen_s.shape[0], -1)
        gt_flat  = gt_s.reshape(gt_s.shape[0], -1)

        mmd_s, vendi_gen_s, vendi_gt_s = [], [], []
        for gamma in GAMMAS:
            mmd_s.append(compute_mmd(gen_flat, gt_flat, gamma))
            vendi_gen_s.append(compute_vendi(gen_flat, gamma))
            vendi_gt_s.append(compute_vendi(gt_flat,  gamma))

        mmd_vals.append(float(np.mean(mmd_s)))
        vendi_gen_vals.append(float(np.mean(vendi_gen_s)))
        vendi_gt_vals.append(float(np.mean(vendi_gt_s)))

    shape_mse /= n_slices
    mmd       = float(np.mean(mmd_vals))
    vendi_gen = float(np.mean(vendi_gen_vals))
    vendi_gt  = float(np.mean(vendi_gt_vals))
    vendi_normalised = vendi_gen / vendi_gt if vendi_gt > 0 else 0.0
    aoa_mse = ((gen_aoas - gt_aoas) ** 2).mean().item()

    return {"shape_mse": shape_mse, "aoa_mse": aoa_mse,
            "mmd": mmd, "vendi": vendi_normalised}


# ── Pressure and performance metric helpers ───────────────────────────────────

def compute_pressure_metrics(gen_pressure: torch.Tensor, gt_pressure: torch.Tensor):
    """
    gen_pressure: [N, 9, 192]
    gt_pressure:  [N, 9, 192]  (raw, un-normalised)
    Returns per-wing-average pressure MSE and per-slice MSE array.
    """
    diff        = gen_pressure - gt_pressure          # [N, n_slices, 192]
    pressure_mse = (diff ** 2).mean().item()
    slice_mse   = [(diff[:, s, :] ** 2).mean().item() for s in range(diff.shape[1])]
    return {
        "pressure_mse":       pressure_mse,
        "pressure_slice_mse": slice_mse,
    }


def compute_perf_metrics(gen_perf: torch.Tensor, gt_perf: torch.Tensor):
    """
    gen_perf: [N, 2]  columns are [cd, cl]
    gt_perf:  [N, 2]  (raw, un-normalised)
    """
    diff     = gen_perf - gt_perf
    perf_mse = (diff ** 2).mean().item()
    cd_mse   = (diff[:, 0] ** 2).mean().item()
    cl_mse   = (diff[:, 1] ** 2).mean().item()
    return {"perf_mse": perf_mse, "cd_mse": cd_mse, "cl_mse": cl_mse}


# ── Batched generation helper ─────────────────────────────────────────────────

def generate_batch(
    lae_model: LAE_AoAInit,
    bae_model,
    params_batch,
    device,
):
    """Run the model on one batch, returning decoded geometry, AoA, pressure, and perf.

    Samples z from the GMM prior to generate wings.
    Pressure and perf predictions are de-normalised before return.
    """
    B = params_batch.shape[0]

    lae_model.encoder.eval()
    lae_model.decoder.eval()
    with torch.no_grad():
        # Sample z from the GMM prior (or zeros if sampler not fitted)
        try:
            z_np = lae_model.sampler.sample(B)
            z = torch.tensor(z_np, dtype=torch.float32, device=device)
        except Exception:
            z = torch.zeros(B, lae_model.lae_latent_dim, device=device)
        z = lae_model._apply_mask(z)
        gen_z, gen_alpha, gen_eta_y, pressure_pred_norm, perf_pred_norm = lae_model.decoder(
            z, params_batch
        )

    # Decode geometry latents via BAE
    decoded_slices = []
    for s in range(gen_z.shape[1]):
        z_s   = gen_z[:, s, :, :]
        dec_s = bae_model.decode_z(
            z_s, z_ae_mode=True, denormalize_output=False, normalized_data=True
        )[0]
        decoded_slices.append(dec_s)
    decoded = torch.stack(decoded_slices, dim=1).clone()   # [B, n_slices, 2, 192]
    decoded[:, :, 1, :] += gen_eta_y                        # apply y-shift

    # De-normalise — always pass tensors so inverse_transform returns tensors.
    gen_alpha_t   = lae_model.scaler_aoas.inverse_transform(gen_alpha.cpu())   # tensor

    if lae_model.scaler_pressures is not None:
        pressure_pred = lae_model.scaler_pressures.inverse_transform(pressure_pred_norm.cpu())
    else:
        pressure_pred = pressure_pred_norm.cpu()

    if lae_model.scaler_perfs is not None:
        perf_pred = lae_model.scaler_perfs.inverse_transform(perf_pred_norm.cpu())
    else:
        perf_pred = perf_pred_norm.cpu()

    return decoded.cpu(), gen_alpha_t.squeeze().numpy(), pressure_pred, perf_pred


def reconstruct_batch(
    lae_model: LAE_AoAInit,
    bae_model,
    z_opts_batch,    # [B, 9, latent_ch, L] — BAE-encoded optimized slices
    params_batch,    # [B, c_dim] scaled
    device,
    te_shifts_batch=None,  # [B, 9] ground-truth TE y-offsets for forced alignment
):
    """Encode real optimized wings through the LAE encoder → decode.
    If te_shifts_batch is provided, use it for y-alignment (forced tip alignment)
    instead of the decoder's eta_y_pred."""
    lae_model.encoder.eval()
    lae_model.decoder.eval()
    with torch.no_grad():
        mu      = lae_model.encode(
            z_opts_batch.to(device), params_batch.to(device)
        )
        z_masked = lae_model._apply_mask(mu)
        z_opt_pred, alpha_pred, eta_y_pred, pressure_pred_norm, perf_pred_norm = lae_model.decode(
            z_masked, params_batch.to(device)
        )

    decoded_slices = []
    for s in range(z_opt_pred.shape[1]):
        z_s   = z_opt_pred[:, s, :, :]
        dec_s = bae_model.decode_z(
            z_s, z_ae_mode=True, denormalize_output=False, normalized_data=True
        )[0]
        decoded_slices.append(dec_s)
    decoded = torch.stack(decoded_slices, dim=1).cpu()   # [B, n_slices, 2, 192]
    if te_shifts_batch is not None:
        decoded[:, :, 1, :] += te_shifts_batch.unsqueeze(-1)
    else:
        decoded[:, :, 1, :] += eta_y_pred.cpu()

    alpha_deg = lae_model.scaler_aoas.inverse_transform(alpha_pred.cpu())

    if lae_model.scaler_pressures is not None:
        pressure_out = lae_model.scaler_pressures.inverse_transform(pressure_pred_norm.cpu())
    else:
        pressure_out = pressure_pred_norm.cpu()

    if lae_model.scaler_perfs is not None:
        perf_out = lae_model.scaler_perfs.inverse_transform(perf_pred_norm.cpu())
    else:
        perf_out = perf_pred_norm.cpu()

    return decoded, alpha_deg.squeeze().numpy(), pressure_out, perf_out


# ── Main evaluation ───────────────────────────────────────────────────────────

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a .pth checkpoint (overrides config default)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of training samples used (needed to fit sampler if missing)")
    parser.add_argument("--refit", action="store_true",
                        help="Delete and refit perf regressor from the current checkpoint")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for x0 shuffle (0 = matched x0, default; non-zero = shuffled)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override checkpoint save directory from config")
    parser.add_argument("--no_x_norm", action="store_true",
                        help="Skip the TE x-normalisation fix when building GT (use for v1-v7 checkpoints trained without it)")
    parser.add_argument("--no_force_align", action="store_true",
                        help="Use decoder eta_y_pred for y-alignment instead of ground-truth te_shifts (tests real spatial learning)")
    return parser.parse_args()


def diagnose_train_vs_test(lae_model, bae_model, cfg, device, n_samples=250, n_diag=30, apply_x_norm=True, force_align=True):
    """Two-part diagnostic: conditional generation (encoder-based z) vs reconstruction (encoded z).

    Generation (encoder z from x0):
      - Low train MSE + high test MSE → overfitting
      - High both → x0 inputs don't cover the data distribution

    Reconstruction (encode through LAE → decode):
      - High train MSE → model hasn't learned to reconstruct training data
      - Low train, high test → overfitting
      - Low both → model is fine; diversity comes from varied x0 inputs
    """
    new_dataset  = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=cfg.seed)
    all_train = list(new_dataset["train"])
    base_dataset = [item for item in all_train if item["final"] == 1]

    rng     = np.random.default_rng(cfg.seed)
    indices = rng.choice(len(base_dataset), size=n_samples, replace=False)
    train_subset = [base_dataset[i] for i in sorted(indices)][:n_diag]

    test_all    = list(new_dataset["test"])
    test_subset = [item for item in test_all if item["final"] == 1][:n_diag]

    def encode_items(items):
        """Returns gt_airfoils, gt_aoas, z_opts, params_list, gt_pressures, gt_perfs, machs, te_shifts_list."""
        gt_airfoils, params_list, gt_aoas, z_opts = [], [], [], []
        gt_pressures, gt_perfs, machs, te_shifts_list = [], [], [], []
        for item in items:
            coords    = torch.tensor(item["coords"],    dtype=torch.float32)
            te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)
            coords_unshifted = coords.clone()
            coords_unshifted[:, :, 1] -= te_shifts.unsqueeze(1)
            if apply_x_norm:
                te_x = coords_unshifted[:, 0, 0]
                coords_unshifted[:, :, 0] += (1.0 - te_x).unsqueeze(1)
            x_opt_slices, z_slices = [], []
            for s in range(coords_unshifted.shape[0]):
                x_s = coords_unshifted[s].permute(1, 0).unsqueeze(0).to(device)
                with torch.no_grad():
                    z_s   = bae_model.encode(x_s, return_z=True, z_ae_mode=True)
                    dec_s = bae_model.decode_z(z_s, z_ae_mode=True,
                                               denormalize_output=False, normalized_data=True)[0]
                x_opt_slices.append(dec_s.squeeze(0).cpu())
                z_slices.append(z_s.squeeze(0).cpu())
            gt_wing = torch.stack(x_opt_slices)
            gt_wing[:, 1, :] += te_shifts.unsqueeze(1)
            gt_airfoils.append(gt_wing)
            z_opts.append(torch.stack(z_slices))
            te_shifts_list.append(te_shifts)
            flow_p = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
            geo_p  = item.get("geo_params", np.zeros(34, dtype=np.float32)).tolist()
            c_dim  = len(lae_model.scaler_params.mean)
            params = torch.tensor((flow_p + geo_p)[:c_dim], dtype=torch.float32).unsqueeze(0).to(device)
            params_scaled = lae_model.scaler_params.transform(params)
            gt_aoas.append(float(item["alpha"]))
            params_list.append(params_scaled)
            gt_pressures.append(torch.tensor(np.array(item["coef_pressure"]), dtype=torch.float32))
            gt_perfs.append(torch.tensor([item["cd_val"], item["cl_val"]], dtype=torch.float32))
            machs.append(float(item["mach"]))
        return (torch.stack(gt_airfoils), torch.tensor(gt_aoas),
                torch.stack(z_opts), params_list,
                torch.stack(gt_pressures), torch.stack(gt_perfs),
                np.array(machs), torch.stack(te_shifts_list))

    def reconstruct_batch(z_opts_batch, params_batch, te_shifts_batch):
        """Encode through LAE → apply mask → decode. Returns airfoils, AoAs, pressure, perf.
        If force_align=True, uses ground-truth te_shifts for y-alignment instead of eta_y_pred."""
        with torch.no_grad():
            mu       = lae_model.encode(
                z_opts_batch.to(device), params_batch.to(device)
            )
            z_masked = lae_model._apply_mask(mu)
            z_opt_pred, alpha_pred, eta_y_pred, pressure_pred, perf_pred = lae_model.decode(
                z_masked, params_batch.to(device)
            )
        # Decode each slice via BAE
        decoded_slices = []
        for s in range(z_opt_pred.shape[1]):
            z_s   = z_opt_pred[:, s, :, :]
            dec_s = bae_model.decode_z(z_s, z_ae_mode=True,
                                        denormalize_output=False, normalized_data=True)[0]
            decoded_slices.append(dec_s)
        decoded = torch.stack(decoded_slices, dim=1).cpu()   # [B, n_slices, 2, 192]
        if force_align:
            decoded[:, :, 1, :] += te_shifts_batch.unsqueeze(-1)
        else:
            decoded[:, :, 1, :] += eta_y_pred.cpu()
        alpha_deg = lae_model.scaler_aoas.inverse_transform(alpha_pred.cpu())
        # De-normalise pressure and perf
        if lae_model.scaler_pressures is not None:
            pressure_out = lae_model.scaler_pressures.inverse_transform(pressure_pred.cpu())
        else:
            pressure_out = pressure_pred.cpu()
        if lae_model.scaler_perfs is not None:
            perf_out = lae_model.scaler_perfs.inverse_transform(perf_pred.cpu())
        else:
            perf_out = perf_pred.cpu()
        return decoded, alpha_deg.squeeze().numpy(), pressure_out, perf_out

    print(f"\n[DIAG] Encoding {n_diag} train samples...")
    gt_train, gt_aoas_train, z_opts_train, params_train, gt_press_train, gt_perf_train, machs_train, te_shifts_train = encode_items(train_subset)
    print(f"[DIAG] Encoding {n_diag} test samples...")
    gt_test, gt_aoas_test, z_opts_test, params_test, gt_press_test, gt_perf_test, machs_test, te_shifts_test = encode_items(test_subset)

    params_batch_train = torch.cat(params_train, dim=0)
    params_batch_test  = torch.cat(params_test,  dim=0)

    # ── Generative pass (encoder z from x0) ─────────────────────────────────
    gen_train, gen_aoas_train, gen_press_train, gen_perf_train = generate_batch(
        lae_model, bae_model, params_batch_train, device)
    gen_test,  gen_aoas_test,  gen_press_test,  gen_perf_test  = generate_batch(
        lae_model, bae_model, params_batch_test,  device)

    # ── Reconstruction pass (encode through LAE → decode) ────────────────────
    rec_train, rec_aoas_train, rec_press_train, rec_perf_train = reconstruct_batch(
        z_opts_train, params_batch_train, te_shifts_train)
    rec_test,  rec_aoas_test,  rec_press_test,  rec_perf_test  = reconstruct_batch(
        z_opts_test,  params_batch_test,  te_shifts_test)

    def mse(a, b): return ((a - b) ** 2).mean().item()
    def aoa_mse(gen, gt): return ((torch.tensor(np.atleast_1d(gen)) - gt) ** 2).mean().item()
    def cd_mse(perf_pred, perf_gt): return ((perf_pred[:, 0] - perf_gt[:, 0]) ** 2).mean().item()
    def cl_mse(perf_pred, perf_gt): return ((perf_pred[:, 1] - perf_gt[:, 1]) ** 2).mean().item()

    print("\n[DIAG] ── Generation (z sampled from prior) ───────────────────────────────────")
    print(f"[DIAG]  Train  shape MSE : {mse(gen_train, gt_train):.4e}  AoA MSE : {aoa_mse(gen_aoas_train, gt_aoas_train):.4f}  "
          f"pressure MSE : {mse(gen_press_train, gt_press_train):.4f}  cd MSE : {cd_mse(gen_perf_train, gt_perf_train):.4f}  cl MSE : {cl_mse(gen_perf_train, gt_perf_train):.6f}")
    print(f"[DIAG]  Test   shape MSE : {mse(gen_test,  gt_test ):.4e}  AoA MSE : {aoa_mse(gen_aoas_test,  gt_aoas_test ):.4f}  "
          f"pressure MSE : {mse(gen_press_test,  gt_press_test ):.4f}  cd MSE : {cd_mse(gen_perf_test,  gt_perf_test ):.4f}  cl MSE : {cl_mse(gen_perf_test,  gt_perf_test ):.6f}")
    print("[DIAG] ── Reconstruction (encode → decode) ──────────────────────────────────")
    print(f"[DIAG]  Train  shape MSE : {mse(rec_train, gt_train):.4e}  AoA MSE : {aoa_mse(rec_aoas_train, gt_aoas_train):.4f}  "
          f"pressure MSE : {mse(rec_press_train, gt_press_train):.4f}  cd MSE : {cd_mse(rec_perf_train, gt_perf_train):.4f}  cl MSE : {cl_mse(rec_perf_train, gt_perf_train):.6f}")
    print(f"[DIAG]  Test   shape MSE : {mse(rec_test,  gt_test ):.4e}  AoA MSE : {aoa_mse(rec_aoas_test,  gt_aoas_test ):.4f}  "
          f"pressure MSE : {mse(rec_press_test,  gt_press_test ):.4f}  cd MSE : {cd_mse(rec_perf_test,  gt_perf_test ):.4f}  cl MSE : {cl_mse(rec_perf_test,  gt_perf_test ):.6f}")

    rec_train_mse = mse(rec_train, gt_train)
    rec_test_mse  = mse(rec_test,  gt_test)
    if rec_train_mse > 0.05:
        print("[DIAG]  → RECONSTRUCTION BUG or undertrained: model can't reconstruct training samples")
    elif rec_train_mse < rec_test_mse / 5:
        print("[DIAG]  → OVERFITTING: reconstructs train well but not test")
    else:
        print("[DIAG]  → Reconstruction OK")

    print("[DIAG] ── Reconstruction by flow regime (train vs test) ──────────────────────")
    diag_regimes = [
        ("Subsonic   (Mach < 0.8)", machs_train < 0.8,                                   machs_test < 0.8),
        ("Transonic  (0.8-1.0)   ", (machs_train >= 0.8) & (machs_train < 1.0),          (machs_test >= 0.8) & (machs_test < 1.0)),
        ("Supersonic (Mach >= 1.0)", machs_train >= 1.0,                                  machs_test >= 1.0),
    ]
    rec_train_np = rec_train.numpy()
    gt_train_np  = gt_train.numpy()
    rec_test_np  = rec_test.numpy()
    gt_test_np   = gt_test.numpy()
    for label, tr_mask, te_mask in diag_regimes:
        tr_mse = float(((rec_train_np[tr_mask] - gt_train_np[tr_mask]) ** 2).mean()) if tr_mask.any() else float("nan")
        te_mse = float(((rec_test_np[te_mask]  - gt_test_np[te_mask])  ** 2).mean()) if te_mask.any() else float("nan")
        print(f"[DIAG]  {label}: train n={tr_mask.sum():2d} MSE={tr_mse:.2e}  |  test n={te_mask.sum():2d} MSE={te_mse:.2e}")
    print("[DIAG] ────────────────────────────────────────────────────────────────────────\n")


def main():
    args   = parse_args()
    cfg    = Config()
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    device = cfg.device
    print(f"Using device: {device}")

    bae_model = load_bae(cfg)
    encoder   = build_encoder(cfg)
    decoder   = build_decoder(cfg)
    sampler   = build_sampler(cfg)

    lae_model = LAE_AoAInit(
        encoder=encoder,
        decoder=decoder,
        sampler=sampler,
        bae_model=bae_model,
        lae_latent_dim=cfg.lae_latent_dim,
        params_mean_std=(0, 1),
        aoas_mean_std=(0, 1),
        name=cfg.model_name,
        opt_lr=cfg.lr,
    )

    checkpoint_path = args.checkpoint if args.checkpoint else os.path.join(cfg.save_dir, f"{cfg.model_name}.pth")
    print(f"Loading checkpoint from {checkpoint_path}...")
    lae_model.load(checkpoint_path, train_mode=False)
    lae_model.to(device)

    stem     = os.path.splitext(os.path.basename(checkpoint_path))[0]
    reg_path = os.path.join(os.path.dirname(checkpoint_path), f"{stem}_perf_reg.pkl")

    if args.refit and os.path.exists(reg_path):
        os.remove(reg_path)
        print(f"Deleted stale {reg_path}")

    ensure_perf_regressor_fitted(lae_model, cfg, reg_path, n_samples=args.n_samples)

    bae_model = lae_model.bae_model
    apply_x_norm = not args.no_x_norm
    print(f"[INFO] x-norm fix: {'ON (v8+)' if apply_x_norm else 'OFF (--no_x_norm, for v1-v7)'}")

    diagnose_train_vs_test(lae_model, bae_model, cfg, device, apply_x_norm=apply_x_norm,
                           force_align=not args.no_force_align)

    new_dataset      = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=cfg.seed)
    test_all         = list(new_dataset["test"])
    initial_by_case  = {item["case_num"]: item for item in test_all if item["initial"] == 1}
    test_dataset     = [item for item in test_all if item["final"] == 1]
    n_test           = len(test_dataset)
    print(f"Test set: {n_test} wings, {len(initial_by_case)} initial cases")

    print("Pre-computing ground truth encodings...")
    gt_airfoils, gt_aoas, gt_pressures, gt_perfs = [], [], [], []
    raw_coords_list = []
    params_list, z_opts_list, te_shifts_list = [], [], []
    flow_conditions = []   # mach, reynolds, cl_target per test wing

    for item in test_dataset:
        coords    = torch.tensor(item["coords"],    dtype=torch.float32)  # [9, 192, 2]
        te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)  # [9]

        # Store raw coords as [n_slices, 2, 192] (same layout as gt_airfoils)
        raw_coords_list.append(coords.permute(0, 2, 1).clone())

        coords_unshifted = coords.clone()
        coords_unshifted[:, :, 1] -= te_shifts.unsqueeze(1)
        if apply_x_norm:
            te_x = coords_unshifted[:, 0, 0]
            coords_unshifted[:, :, 0] += (1.0 - te_x).unsqueeze(1)

        x_opt_slices, z_opt_slices = [], []
        for s in range(coords_unshifted.shape[0]):
            x_s = coords_unshifted[s].permute(1, 0).unsqueeze(0).to(device)
            with torch.no_grad():
                z_s   = bae_model.encode(x_s, return_z=True, z_ae_mode=True)
                dec_s = bae_model.decode_z(z_s, z_ae_mode=True,
                                            denormalize_output=False,
                                            normalized_data=True)[0]
            x_opt_slices.append(dec_s.squeeze(0).cpu())
            z_opt_slices.append(z_s.squeeze(0).cpu())   # [latent_ch, L]

        gt_wing = torch.stack(x_opt_slices)         # [n_slices, 2, 192]
        gt_wing[:, 1, :] += te_shifts.unsqueeze(1)
        gt_airfoils.append(gt_wing)
        z_opts_list.append(torch.stack(z_opt_slices))  # [9, latent_ch, L]

        flow_p = [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
        geo_p  = item.get("geo_params", np.zeros(34, dtype=np.float32)).tolist()
        c_dim  = len(lae_model.scaler_params.mean)
        params = torch.tensor((flow_p + geo_p)[:c_dim], dtype=torch.float32).unsqueeze(0).to(device)
        params_scaled = lae_model.scaler_params.transform(params)

        gt_aoas.append(float(item["alpha"]))
        params_list.append(params_scaled)
        te_shifts_list.append(te_shifts)
        flow_conditions.append({
            "mach":       item["mach"],
            "reynolds":   item["reynolds"],
            "cl_target":  item["cl_target"],
        })

        # Pressure and performance ground truth (raw, un-normalised)
        gt_pressures.append(torch.tensor(
            np.array(item["coef_pressure"]), dtype=torch.float32
        ))  # [9, 192]
        gt_perfs.append(torch.tensor(
            [item["cd_val"], item["cl_val"]], dtype=torch.float32
        ))  # [2]

    raw_coords_t   = torch.stack(raw_coords_list)  # [N, n_slices, 2, 192]
    gt_airfoils_t  = torch.stack(gt_airfoils)   # [N, 9, 2, 192]
    gt_aoas_t      = torch.tensor(gt_aoas)
    gt_pressures_t = torch.stack(gt_pressures)  # [N, 9, 192]
    gt_perfs_t     = torch.stack(gt_perfs)       # [N, 2]
    z_opts_t       = torch.stack(z_opts_list)    # [N, 9, latent_ch, L]

    print("[INFO] seed=0 — standard evaluation (no x0 conditioning)")

    # ── Diagnostic: raw dataset coords vs BAE roundtrip (GT) ────────────────
    for sample_idx in range(min(3, gt_airfoils_t.shape[0])):
        raw_vs_bae_path = os.path.join(
            os.path.join("results", "lvae_evaluation"),
            f"raw_vs_bae_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}_s{sample_idx}.png",
        )
        lvae_plotting.plot_wing3d_comparison(
            raw_coords_t, gt_airfoils_t,
            sample_idx=sample_idx,
            save_path=raw_vs_bae_path,
            title_a="Raw dataset coords",
            title_b="BAE roundtrip (GT used in eval)",
        )
        print(f"[DEBUG] Raw vs BAE plot saved to {raw_vs_bae_path}")

    # ── Diagnostic: per-slice 2D raw coords for first 3 samples ─────────────
    diag_save_dir = os.path.join("results", "lvae_evaluation")
    for sample_idx in range(min(3, raw_coords_t.shape[0])):
        raw_2d_path = os.path.join(diag_save_dir,
            f"raw_slices2d_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}_s{sample_idx}.png")
        lvae_plotting.plot_raw_slices_2d(
            raw_coords_t, gt_airfoils_t,
            sample_idx=sample_idx,
            save_path=raw_2d_path,
        )
        print(f"[DEBUG] Raw 2D slice diagnostic saved to {raw_2d_path}")

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_dir  = os.path.join("results", "lvae_evaluation")
    os.makedirs(save_dir, exist_ok=True)

    pass_metrics = []
    for pass_idx in range(N_FORWARD_PASSES):
        print(f"Forward pass {pass_idx + 1}/{N_FORWARD_PASSES}...")
        gen_airfoils, gen_aoas   = [], []
        gen_pressures, gen_perfs = [], []

        for start in range(0, n_test, BATCH_SIZE):
            end          = min(start + BATCH_SIZE, n_test)
            params_batch = torch.cat(params_list[start:end], dim=0)

            airfoil_batch, aoa_batch, pressure_batch, perf_batch = generate_batch(
                lae_model, bae_model, params_batch, device,
            )
            gen_airfoils.append(airfoil_batch)
            gen_aoas.extend(np.atleast_1d(aoa_batch).tolist())
            gen_pressures.append(pressure_batch)
            gen_perfs.append(perf_batch)

        gen_airfoils_t  = torch.cat(gen_airfoils)
        gen_aoas_t      = torch.tensor(gen_aoas)
        gen_pressures_t = torch.cat(gen_pressures)
        gen_perfs_t     = torch.cat(gen_perfs)

        if pass_idx == 0:
            print(f"[DEBUG] gen_aoas   min={gen_aoas_t.min():.3f} max={gen_aoas_t.max():.3f} mean={gen_aoas_t.mean():.3f}")
            print(f"[DEBUG] gt_aoas    min={gt_aoas_t.min():.3f} max={gt_aoas_t.max():.3f} mean={gt_aoas_t.mean():.3f}")
            print(f"[DEBUG] gen_perfs  min={gen_perfs_t.min():.3f} max={gen_perfs_t.max():.3f} mean={gen_perfs_t.mean():.3f}")
            print(f"[DEBUG] gt_perfs   min={gt_perfs_t.min():.3f} max={gt_perfs_t.max():.3f} mean={gt_perfs_t.mean():.3f}")
            # ── Airfoil shape + Cp: root / mid / tip (supervisor diagnostic) ──
            cp_plot_path = os.path.join(save_dir, f"airfoil_cp_{timestamp}.png")
            lvae_plotting.plot_airfoil_cp_comparison(
                gen_airfoils_t, gt_airfoils_t,
                gen_pressures_t, gt_pressures_t,
                gen_perfs=gen_perfs_t, gt_perfs=gt_perfs_t,
                sample_indices=list(range(min(3, gen_airfoils_t.shape[0]))),
                slice_indices=list(range(15)),
                save_path=cp_plot_path,
                flow_conditions=flow_conditions,
            )
            print(f"[DEBUG] Airfoil+Cp plot saved to {cp_plot_path}")

            # ── Geometry-only slice comparison ──────────────────────────
            slice_plot_path = os.path.join(save_dir, f"slice_comparison_{timestamp}.png")
            lvae_plotting.plot_airfoil_slices_comparison(
                gen_airfoils_t, gt_airfoils_t,
                sample_indices=list(range(min(5, gen_airfoils_t.shape[0]))),
                slice_indices=[0, 7, 14],
                save_path=slice_plot_path,
            )
            print(f"[DEBUG] Slice comparison plot saved to {slice_plot_path}")

        # ── Generation metrics ───────────────────────────────────────────────
        geom_m     = compute_geometry_metrics(gen_airfoils_t, gt_airfoils_t,
                                              gen_aoas_t, gt_aoas_t)
        pressure_m = compute_pressure_metrics(gen_pressures_t, gt_pressures_t)
        perf_m     = compute_perf_metrics(gen_perfs_t, gt_perfs_t)

        # ── Reconstruction pass ──────────────────────────────────────────────
        rec_airfoils, rec_aoas   = [], []
        rec_pressures, rec_perfs = [], []
        te_shifts_t = torch.stack(te_shifts_list)   # [N, 9]
        for start in range(0, n_test, BATCH_SIZE):
            end              = min(start + BATCH_SIZE, n_test)
            params_batch     = torch.cat(params_list[start:end], dim=0)
            z_opts_batch     = z_opts_t[start:end]               # [B, 9, latent_ch, L]
            te_shifts_batch  = te_shifts_t[start:end]            # [B, 9]

            r_airfoil, r_aoa, r_pressure, r_perf = reconstruct_batch(
                lae_model, bae_model, z_opts_batch, params_batch, device,
                te_shifts_batch=te_shifts_batch,
            )
            rec_airfoils.append(r_airfoil)
            rec_aoas.extend(np.atleast_1d(r_aoa).tolist())
            rec_pressures.append(r_pressure)
            rec_perfs.append(r_perf)

        rec_airfoils_t  = torch.cat(rec_airfoils)
        rec_aoas_t      = torch.tensor(rec_aoas)
        rec_pressures_t = torch.cat(rec_pressures)
        rec_perfs_t     = torch.cat(rec_perfs)

        if pass_idx == 0:
            # ── Reconstruction: airfoil+Cp plot ─────────────────────────────
            rec_cp_plot_path = os.path.join(save_dir, f"rec_airfoil_cp_{timestamp}.png")
            lvae_plotting.plot_airfoil_cp_comparison(
                rec_airfoils_t, gt_airfoils_t,
                rec_pressures_t, gt_pressures_t,
                gen_perfs=rec_perfs_t, gt_perfs=gt_perfs_t,
                sample_indices=list(range(min(3, rec_airfoils_t.shape[0]))),
                slice_indices=list(range(15)),
                save_path=rec_cp_plot_path,
                pred_label="Reconstructed",
                flow_conditions=flow_conditions,
            )
            print(f"[DEBUG] Reconstruction airfoil+Cp plot saved to {rec_cp_plot_path}")

            # ── Reconstruction: geometry-only slice comparison ───────────────
            rec_slice_plot_path = os.path.join(save_dir, f"rec_slice_comparison_{timestamp}.png")
            lvae_plotting.plot_airfoil_slices_comparison(
                rec_airfoils_t, gt_airfoils_t,
                sample_indices=list(range(min(5, rec_airfoils_t.shape[0]))),
                slice_indices=[0, 7, 14],
                save_path=rec_slice_plot_path,
            )
            print(f"[DEBUG] Reconstruction slice comparison plot saved to {rec_slice_plot_path}")

            # ── Reconstruction: 3-D wing stacked view ───────────────────────
            for sample_idx in range(min(3, rec_airfoils_t.shape[0])):
                wing3d_path = os.path.join(save_dir, f"rec_wing3d_{timestamp}_s{sample_idx}.png")
                lvae_plotting.plot_wing3d_comparison(
                    rec_airfoils_t, gt_airfoils_t,
                    sample_idx=sample_idx,
                    save_path=wing3d_path,
                )
                print(f"[DEBUG] 3-D wing plot saved to {wing3d_path}")

            # ── Reconstruction: error distribution histograms ────────────────
            hist_plot_path = os.path.join(save_dir, f"rec_error_hist_{timestamp}.png")
            lvae_plotting.plot_reconstruction_error_histograms(
                rec_airfoils_t, gt_airfoils_t,
                rec_pressures_t, gt_pressures_t,
                rec_perfs_t, gt_perfs_t,
                save_path=hist_plot_path,
            )
            print(f"[DEBUG] Reconstruction error histogram saved to {hist_plot_path}")

            # ── cd/cl scatter: generation ────────────────────────────────
            gen_scatter_path = os.path.join(save_dir, f"gen_perf_scatter_{timestamp}.png")
            lvae_plotting.plot_perf_scatter(
                gen_perfs_t, gt_perfs_t,
                save_path=gen_scatter_path,
                pred_label="Generated",
            )
            print(f"[DEBUG] Generation perf scatter saved to {gen_scatter_path}")

            # ── cd/cl scatter: reconstruction ────────────────────────────
            rec_scatter_path = os.path.join(save_dir, f"rec_perf_scatter_{timestamp}.png")
            lvae_plotting.plot_perf_scatter(
                rec_perfs_t, gt_perfs_t,
                save_path=rec_scatter_path,
                pred_label="Reconstructed",
            )
            print(f"[DEBUG] Reconstruction perf scatter saved to {rec_scatter_path}")

        rec_geom_m     = compute_geometry_metrics(rec_airfoils_t, gt_airfoils_t,
                                                  rec_aoas_t, gt_aoas_t)
        rec_pressure_m = compute_pressure_metrics(rec_pressures_t, gt_pressures_t)
        rec_perf_m     = compute_perf_metrics(rec_perfs_t, gt_perfs_t)
        # Prefix all reconstruction metric keys with "rec_"
        rec_metrics = {f"rec_{k}": v for k, v in {
            **rec_geom_m, **rec_pressure_m, **rec_perf_m
        }.items()}

        metrics = {**geom_m, **pressure_m, **perf_m, **rec_metrics}
        pass_metrics.append(metrics)
        print(
            f"  [GEN ] Shape MSE={metrics['shape_mse']:.2e} | AoA MSE={metrics['aoa_mse']:.3f} | "
            f"MMD={metrics['mmd']:.4f} | Vendi={metrics['vendi']:.3f} | "
            f"Pressure MSE={metrics['pressure_mse']:.4f} | "
            f"cd MSE={metrics['cd_mse']:.6f} | cl MSE={metrics['cl_mse']:.6f}"
        )
        print(
            f"  [REC ] Shape MSE={metrics['rec_shape_mse']:.2e} | AoA MSE={metrics['rec_aoa_mse']:.3f} | "
            f"MMD={metrics['rec_mmd']:.4f} | Vendi={metrics['rec_vendi']:.3f} | "
            f"Pressure MSE={metrics['rec_pressure_mse']:.4f} | "
            f"cd MSE={metrics['rec_cd_mse']:.6f} | cl MSE={metrics['rec_cl_mse']:.6f}"
        )

    gen_keys = ["shape_mse", "aoa_mse", "mmd", "vendi",
                "pressure_mse", "perf_mse", "cd_mse", "cl_mse"]
    rec_keys = [f"rec_{k}" for k in gen_keys]

    # Single deterministic pass — just unwrap the one entry
    metrics = pass_metrics[0]
    slice_mse_mean     = np.array(metrics["pressure_slice_mse"])
    rec_slice_mse_mean = np.array(metrics["rec_pressure_slice_mse"])

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)
    print("── Generation ──────────────────────────────────────────────")
    print(f"  Shape MSE    : {metrics['shape_mse']:.2e}")
    print(f"  AoA MSE      : {metrics['aoa_mse']:.4f}")
    print(f"  MMD          : {metrics['mmd']:.4f}")
    print(f"  Vendi        : {metrics['vendi']:.4f}")
    print(f"  Pressure MSE : {metrics['pressure_mse']:.4f}")
    print(f"  cd MSE       : {metrics['cd_mse']:.6f}")
    print(f"  cl MSE       : {metrics['cl_mse']:.6f}")
    print(f"  Pressure slice MSE (s0..s8): "
          + " ".join(f"{v:.4f}" for v in slice_mse_mean))
    print("── Reconstruction ──────────────────────────────────────────")
    print(f"  Shape MSE    : {metrics['rec_shape_mse']:.2e}")
    print(f"  AoA MSE      : {metrics['rec_aoa_mse']:.4f}")
    print(f"  MMD          : {metrics['rec_mmd']:.4f}")
    print(f"  Vendi        : {metrics['rec_vendi']:.4f}")
    print(f"  Pressure MSE : {metrics['rec_pressure_mse']:.4f}")
    print(f"  cd MSE       : {metrics['rec_cd_mse']:.6f}")
    print(f"  cl MSE       : {metrics['rec_cl_mse']:.6f}")
    print(f"  Pressure slice MSE (s0..s8): "
          + " ".join(f"{v:.4f}" for v in rec_slice_mse_mean))
    print("── Reconstruction by flow regime ───────────────────────────")
    machs = np.array([fc["mach"] for fc in flow_conditions])
    regimes = [
        ("Subsonic   (Mach < 0.8)", machs < 0.8),
        ("Transonic  (0.8-1.0)   ", (machs >= 0.8) & (machs < 1.0)),
        ("Supersonic (Mach >= 1.0)", machs >= 1.0),
    ]
    rec_arr = pass_metrics[0]  # already unwrapped above
    rec_airfoils_np = rec_airfoils_t.numpy()
    gt_airfoils_np  = gt_airfoils_t.numpy()
    rec_press_np    = rec_pressures_t.numpy()
    gt_press_np     = gt_pressures_t.numpy()
    for label, mask in regimes:
        if mask.sum() == 0:
            print(f"  {label}: no wings")
            continue
        s_mse = float(((rec_airfoils_np[mask] - gt_airfoils_np[mask]) ** 2).mean())
        p_mse = float(((rec_press_np[mask]    - gt_press_np[mask])    ** 2).mean())
        print(f"  {label}: n={mask.sum():3d}  shape MSE={s_mse:.2e}  pressure MSE={p_mse:.4f}")
    print("=" * 65)

    results_path = os.path.join(save_dir, f"eval_{timestamp}.txt")
    with open(results_path, "w") as f:
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test wings: {n_test}\n")
        f.write(f"x0 seed: {args.seed} ({'matched' if args.seed == 0 else 'shuffled'})\n\n")
        f.write("── Generation ───────────────────────────────────────────\n")
        for k in gen_keys:
            f.write(f"{k:16s}: {metrics[k]:.6f}\n")
        f.write("Pressure slice MSE:\n")
        for s, v in enumerate(slice_mse_mean):
            f.write(f"  slice {s}: {v:.6f}\n")
        f.write("\n── Reconstruction ───────────────────────────────────────\n")
        for k in rec_keys:
            f.write(f"{k:20s}: {metrics[k]:.6f}\n")
        f.write("Pressure slice MSE:\n")
        for s, v in enumerate(rec_slice_mse_mean):
            f.write(f"  slice {s}: {v:.6f}\n")
        f.write("\n── Reconstruction by flow regime ────────────────────────\n")
        for label, mask in regimes:
            if mask.sum() == 0:
                f.write(f"  {label}: no wings\n")
                continue
            s_mse = float(((rec_airfoils_np[mask] - gt_airfoils_np[mask]) ** 2).mean())
            p_mse = float(((rec_press_np[mask]    - gt_press_np[mask])    ** 2).mean())
            f.write(f"  {label}: n={mask.sum():3d}  shape MSE={s_mse:.6f}  pressure MSE={p_mse:.6f}\n")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
