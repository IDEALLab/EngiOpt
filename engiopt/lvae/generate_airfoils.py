"""
Generation script for the LAE.
"""

import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'

import numpy as np
import torch

from .train_lvae import Config, load_bae, build_encoder, build_decoder, build_sampler
from engiopt.lvae.lvae import LAE_AoAInit
from engiopt.lvae.plotting import wing_3D_shape_plot, plot_generated_pressure


def generate_airfoils(
    lae_model: LAE_AoAInit,
    bae_model,
    num_samples: int = 10,
    device: str = 'cpu',
    params=None,
    x0_encoded: list = None,
):
    """Generate optimized wings by decoding (c, x0) pairs with z=0.

    Generation is purely conditional: z is always zero and diversity comes
    entirely from varying the inputs.  Each sample uses a different initial
    wing (x0) cycled from the provided pool, so ``num_samples`` distinct
    (c, x0) pairs produce ``num_samples`` distinct optimized wings.

    Args:
        lae_model:   Trained LAE model.
        bae_model:   Frozen BezierAutoencoder.
        num_samples: Number of (c, x0) pairs to decode.
        device:      Target device.
        params:      Target aerodynamic conditions [mach, reynolds, cl_target,
                     area_case_ratio].  Can be a single vector (broadcast to
                     all samples) or a [num_samples, 4] tensor for per-sample
                     conditions.  Defaults to [0.5, 5e6, 0.8, 1.0].
        x0_encoded:  List of pre-encoded BAE latents, each [latent_channels, L].
                     Cycled across samples so len(x0_encoded) can be smaller
                     than num_samples.  If None, all initial wings are loaded
                     from the training dataset automatically.

    Returns:
        decoded_airfoils: [num_samples, 9, 2, 192] coordinate tensor.
        rescaled_alphas:  [num_samples, 1] angle-of-attack tensor.
        pressure_pred:    [num_samples, 9, 192] Cp field (de-normalised).
        perf_pred:        [num_samples, 2] performance [cd, cl] (de-normalised).
    """
    # Build the raw (un-normalised) condition tensor [num_samples, 4]
    if params is None:
        params = [0.5, 5e6, 0.8, 1.0]
        print("No target conditions provided — using defaults: "
              "mach=0.5, reynolds=5e6, cl_target=0.8, area_ratio=1.0")
    if not isinstance(params, torch.Tensor):
        params = torch.tensor(params, dtype=torch.float32)
    if params.ndim == 1:
        params = params.unsqueeze(0).repeat(num_samples, 1)
    params = params.to(device).float()
    params_scaled = lae_model.scaler_params.transform(params).to(device).float()
    print(f"Target conditions (raw):    {params[0].cpu().numpy()}")
    print(f"Target conditions (scaled): {params_scaled[0].cpu().numpy()}")

    # Build x0 pool — encode all initial wings if not pre-computed.
    if x0_encoded is None:
        from engibench.problems.wings3D.v0 import Wings3D
        problem   = Wings3D(seed=0)
        all_train = list(problem.dataset["train"])
        init_items = [item for item in all_train if item["initial"] == 1]
        print(f"Encoding {len(init_items)} initial wings as x0 pool...")
        x0_encoded = []
        with torch.no_grad():
            for item in init_items:
                coords = torch.tensor(item["coords"], dtype=torch.float32)
                x_root = coords[0].permute(1, 0).unsqueeze(0).to(device)  # [1, 2, 192]
                z = bae_model.encode(x_root, return_z=True, z_ae_mode=True)
                x0_encoded.append(z.squeeze(0).cpu())

    print(f"x0 pool size: {len(x0_encoded)} initial wings  "
          f"(cycling across {num_samples} samples)")

    # Cycle through the x0 pool to build the batch.
    x0 = torch.stack(
        [x0_encoded[i % len(x0_encoded)] for i in range(num_samples)]
    ).to(device)  # [num_samples, latent_channels, L]

    # Encode x0 replicated across all 9 slices as the starting geometry.
    # This gives each sample a z that reflects its own x0, so different initial
    # wings produce different z values and therefore different decoded outputs.
    lae_model.encoder.eval()
    lae_model.decoder.eval()
    with torch.no_grad():
        # Treat x0 as the geometry at every span position: [B, latent_ch, L] → [B, 9, latent_ch, L]
        z_opt_init = x0.unsqueeze(1).repeat(1, 9, 1, 1)
        z = lae_model.encoder(z_opt_init, params_scaled, x0)
        z = lae_model._apply_mask(z)
        gen_z, gen_alpha, gen_eta_y, pressure_pred_norm, perf_pred_norm = lae_model.decoder(
            z, params_scaled, x0
        )

    print(f"gen_z shape:         {gen_z.shape}")
    print(f"gen_alpha shape:     {gen_alpha.shape}")
    print(f"gen_eta_y shape:     {gen_eta_y.shape}")
    print(f"pressure_pred shape: {pressure_pred_norm.shape}")
    print(f"perf_pred shape:     {perf_pred_norm.shape}")

    # De-normalise predictions — always pass tensors so outputs are tensors too.
    rescaled_alphas = lae_model.scaler_aoas.inverse_transform(gen_alpha.cpu())

    if lae_model.scaler_pressures is not None:
        pressure_pred = lae_model.scaler_pressures.inverse_transform(pressure_pred_norm.cpu())
    else:
        pressure_pred = pressure_pred_norm.cpu()

    # Always use the decoder's own perf head for generation — it is tethered to
    # the actual decoded geometry.  The MLP regressor was fit on z values from
    # final optimized wings and extrapolates badly when z comes from initial wings.
    if lae_model.scaler_perfs is not None:
        perf_pred = lae_model.scaler_perfs.inverse_transform(perf_pred_norm.cpu())
    else:
        perf_pred = perf_pred_norm.cpu()
    print("[Perf] Using decoder perf head.")

    # Decode geometry latents via BAE
    print("Decoding airfoils via BAE...")
    decoded_slices = []
    with torch.no_grad():
        for s in range(gen_z.shape[1]):
            z_s   = gen_z[:, s, :, :]
            dec_s = bae_model.decode_z(
                z_s, z_ae_mode=True, denormalize_output=False, normalized_data=True
            )[0]
            decoded_slices.append(dec_s.cpu())
    decoded_unshifted = torch.stack(decoded_slices, dim=1)  # [B, n_slices, 2, 192]

    decoded_airfoils = decoded_unshifted.clone()
    decoded_airfoils[:, :, 1, :] += gen_eta_y.cpu()

    print(f"Decoded airfoils shape: {decoded_airfoils.shape}")
    print(f"Alpha values: {rescaled_alphas.squeeze()}")
    print(f"Perf [cd, cl]: {perf_pred}")

    return decoded_airfoils, rescaled_alphas, pressure_pred, perf_pred


def save_airfoils(generated_airfoils, generated_alphas, pressure_pred, perf_pred):
    n = len(generated_airfoils)
    alphas_np = generated_alphas.squeeze().cpu().numpy()
    perf_np   = perf_pred.cpu().numpy()   # [N, 2]

    print("\n" + "=" * 55)
    print("GENERATION RESULTS:")
    print(f"  Generated {n} airfoils")
    print(f"  Airfoil shape : {generated_airfoils.shape}")
    print(f"  Pressure shape: {pressure_pred.shape}")
    print(f"  {'idx':>4}  {'alpha (deg)':>12}  {'cd':>10}  {'cl':>10}")
    for i in range(n):
        a  = float(np.atleast_1d(alphas_np)[i])
        cd = float(perf_np[i, 0])
        cl = float(perf_np[i, 1])
        print(f"  {i:>4}  {a:>12.4f}  {cd:>10.6f}  {cl:>10.6f}")
    print("=" * 55 + "\n")

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_dir  = os.path.join("results", "lvae_generated", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    torch.save({
        "airfoils": generated_airfoils.cpu(),
        "alphas":   generated_alphas.cpu(),
        "pressure": pressure_pred.cpu(),
        "perf":     perf_pred.cpu(),
    }, os.path.join(save_dir, "generated_airfoils.pt"))

    np.savez(
        os.path.join(save_dir, "generated_airfoils.npz"),
        airfoils=generated_airfoils.cpu().numpy(),
        alphas=alphas_np,
        pressure=pressure_pred.cpu().numpy(),
        perf=perf_np,
    )

    # Performance summary CSV
    csv_path = os.path.join(save_dir, "performance_summary.csv")
    with open(csv_path, "w") as f:
        f.write("idx,alpha_deg,cd,cl\n")
        for i in range(n):
            a  = float(np.atleast_1d(alphas_np)[i])
            cd = float(perf_np[i, 0])
            cl = float(perf_np[i, 1])
            f.write(f"{i},{a:.6f},{cd:.8f},{cl:.8f}\n")

    print(f"Saved tensors to {save_dir}/generated_airfoils.{{pt,npz}}")
    print(f"Saved performance summary to {csv_path}")
    return save_dir


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a .pth checkpoint (overrides config default)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of training samples used (needed to fit perf regressor if missing)")
    parser.add_argument("--num_generate", type=int, default=20,
                        help="Number of airfoils to generate (default: 20)")
    # Target aerodynamic conditions
    parser.add_argument("--mach", type=float, default=0.5,
                        help="Target Mach number (default: 0.5)")
    parser.add_argument("--reynolds", type=float, default=5e6,
                        help="Target Reynolds number (default: 5e6)")
    parser.add_argument("--cl_target", type=float, default=0.8,
                        help="Target lift coefficient (default: 0.8)")
    parser.add_argument("--area", type=float, default=1.0,
                        help="Target area/case ratio (default: 1.0)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    cfg = Config()

    print("Loading BAE...")
    bae_model = load_bae(cfg)

    print("Building LAE components...")
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg)
    sampler = build_sampler(cfg)

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

    # --- DEBUG: verify checkpoint state ---

    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Testing LVAE reconstruction on training sample")
    print("=" * 60)
    
    # Load a training sample
    from engibench.problems.wings3D.v0 import Wings3D
    problem = Wings3D(seed=0)
    train_data = list(problem.dataset["train"])

    # --- Check training data cd/cl distribution ---
    final_samples = [item for item in train_data if item.get("final", 0) == 1]
    if final_samples:
        cd_vals = [item["cd_val"] for item in final_samples if "cd_val" in item]
        cl_vals = [item["cl_val"] for item in final_samples if "cl_val" in item]
        if cd_vals:
            import statistics
            print(f"\n[DATA CHECK] Training cd/cl over {len(cd_vals)} final samples:")
            print(f"  cd: min={min(cd_vals):.4f}  max={max(cd_vals):.4f}  mean={statistics.mean(cd_vals):.4f}  std={statistics.stdev(cd_vals):.4f}")
            print(f"  cl: min={min(cl_vals):.4f}  max={max(cl_vals):.4f}  mean={statistics.mean(cl_vals):.4f}  std={statistics.stdev(cl_vals):.4f}")
            print(f"  First 5 cd values: {[round(v, 4) for v in cd_vals[:5]]}")
            print(f"  First 5 cl values: {[round(v, 4) for v in cl_vals[:5]]}\n")
    # --- End data check ---

    # Find a sample that is "final" (optimized) and has valid coords
    sample = None
    for item in train_data:
        if item.get("final", 0) == 1 and "coords" in item:
            sample = item
            break
    
    if sample is None:
        print("WARNING: Could not find a suitable training sample")
    else:
        # Get ground truth geometry for ALL 9 slices
        gt_coords_full = torch.tensor(sample["coords"], dtype=torch.float32)
        print(f"Sample coords shape: {gt_coords_full.shape}")
        
        # Reshape to [9, 2, 192] if needed
        if len(gt_coords_full.shape) == 3:
            if gt_coords_full.shape[1] == 192 and gt_coords_full.shape[2] == 2:
                gt_coords_full = gt_coords_full.permute(0, 2, 1)
        elif len(gt_coords_full.shape) == 4:
            pass
        
        # Add batch dimension -> [1, 9, 2, 192]
        gt_coords_full = gt_coords_full.unsqueeze(0).to(device)
        print(f"GT coords shape for BAE (all slices): {gt_coords_full.shape}")
        
        params = torch.tensor([[0.5, 5e6, 0.8, 1.0]]).to(device).float()
        params_scaled = lae_model.scaler_params.transform(params).to(device).float()
        print(f"Params scaled: {params_scaled}")
        
        # Get x0 (initial airfoil) - also all 9 slices
        init_item = next(item for item in train_data if item.get("initial", 0) == 1)
        coords_init = torch.tensor(init_item["coords"], dtype=torch.float32)
        if len(coords_init.shape) == 3:
            if coords_init.shape[1] == 192 and coords_init.shape[2] == 2:
                coords_init = coords_init.permute(0, 2, 1)
        coords_init = coords_init.unsqueeze(0).to(device)
        print(f"Initial coords shape: {coords_init.shape}")

        with torch.no_grad():
            # Encode initial airfoil - root slice only (x0 is a single slice [1, 3, 30])
            x0_root = bae_model.encode(coords_init[0, 0].unsqueeze(0), return_z=True, z_ae_mode=True)
            print(f"x0_root shape: {x0_root.shape}")

            # Encode target geometry - one slice at a time → [1, n_slices, 3, 30]
            z_gt_slices = []
            for s in range(gt_coords_full.shape[1]):
                slice_coords = gt_coords_full[0, s].unsqueeze(0)  # [1, 2, 192]
                z_slice = bae_model.encode(slice_coords, return_z=True, z_ae_mode=True)
                z_gt_slices.append(z_slice)
            z_gt = torch.stack(z_gt_slices, dim=1)  # [1, n_slices, 3, 30]
            print(f"z_gt shape: {z_gt.shape}")

            # BAE-only roundtrip: encode → decode without LAE, to isolate BAE error
            bae_recon_slices = []
            for s in range(z_gt.shape[1]):
                z_s = z_gt[:, s, :, :]  # [1, 3, 30]
                dec_s = bae_model.decode_z(z_s, z_ae_mode=True, denormalize_output=True, normalized_data=True)[0]
                bae_recon_slices.append(dec_s.cpu())
            bae_recon = torch.stack(bae_recon_slices, dim=1)  # [1, n_slices, 2, 192]
            bae_mse = torch.nn.functional.mse_loss(bae_recon, gt_coords_full.cpu())
            print(f"[BAE-only roundtrip MSE]: {bae_mse.item():.6f}  ← floor for total MSE")

            # Encoder expects z_opt: [B, 9, 3, 30] and x0: [B, 3, 30]
            mu = lae_model.encoder(z_gt, c=params_scaled, x0=x0_root)
            print(f"mu shape: {mu.shape}")
            print(f"mu mean: {mu.mean().item():.4f}, std: {mu.std().item():.4f}")

            z_recon = lae_model._apply_mask(mu)  # zero inactive dims, matching training
            # Decoder expects (z, c, x0); returns (z_opt_pred, alpha, eta_y, pressure, perf)
            z_opt_pred, alpha_recon_norm, eta_recon_norm, pressure_recon_norm, perf_recon_norm = lae_model.decoder(z_recon, params_scaled, x0_root)
            print(f"z_opt_pred shape: {z_opt_pred.shape}")

            # Decode each slice through BAE
            decoded_slices = []
            for s in range(z_opt_pred.shape[1]):
                z_s = z_opt_pred[:, s, :, :]  # [1, 3, 30]
                dec_s = bae_model.decode_z(
                    z_s, z_ae_mode=True, denormalize_output=True, normalized_data=True
                )[0]
                decoded_slices.append(dec_s.cpu())
            coords_recon = torch.stack(decoded_slices, dim=1)  # [1, n_slices, 2, 192]
            print(f"coords_recon shape: {coords_recon.shape}")

            # Compute MSE on all slices
            mse = torch.nn.functional.mse_loss(coords_recon, gt_coords_full.cpu())
            print(f"\n>>> RECONSTRUCTION MSE: {mse.item():.6f} <<<")
            if mse.item() > 0.01:
                print(">>> WARNING: High reconstruction error - LVAE encoder/decoder is NOT preserving geometry!")
            else:
                print(">>> GOOD: LVAE can reconstruct training geometry!")
    
    print("=" * 60 + "\n")

    # --- DIAGNOSTIC: perf head reconstruction on training samples ---
    print("=" * 60)
    print("DIAGNOSTIC: Perf head reconstruction on training samples")
    print("=" * 60)
    final_samples_with_perf = [
        item for item in train_data
        if item.get("final", 0) == 1 and "cd_val" in item and "coords" in item
    ][:10]  # check first 10

    if final_samples_with_perf:
        cd_errors, cl_errors = [], []
        cd_preds, cl_preds = [], []
        cd_gts, cl_gts = [], []

        lae_model.encoder.eval()
        lae_model.decoder.eval()

        with torch.no_grad():
            for item in final_samples_with_perf:
                # Ground truth perf
                cd_gt = float(item["cd_val"])
                cl_gt = float(item["cl_val"])

                # Encode geometry
                coords = torch.tensor(item["coords"], dtype=torch.float32)
                if coords.shape[1] == 192 and coords.shape[2] == 2:
                    coords = coords.permute(0, 2, 1)  # [9, 2, 192]
                coords = coords.unsqueeze(0).to(device)  # [1, 9, 2, 192]

                z_slices = []
                for s in range(coords.shape[1]):
                    z_s = bae_model.encode(coords[0, s].unsqueeze(0), return_z=True, z_ae_mode=True)
                    z_slices.append(z_s)
                z_opt = torch.stack(z_slices, dim=1)  # [1, n_slices, 3, 30]

                # Encode params
                params = torch.tensor(
                    [[item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]],
                    dtype=torch.float32
                ).to(device)
                params_sc = lae_model.scaler_params.transform(params).to(device).float()

                # Get x0 from initial airfoil for this case
                case_num = int(item["case_num"])
                init_item = next(
                    (i for i in train_data if i.get("initial", 0) == 1 and int(i["case_num"]) == case_num),
                    None
                )
                if init_item is None:
                    continue
                init_coords = torch.tensor(init_item["coords"], dtype=torch.float32)
                if init_coords.shape[1] == 192 and init_coords.shape[2] == 2:
                    init_coords = init_coords.permute(0, 2, 1)
                x0_r = bae_model.encode(init_coords[0].unsqueeze(0).to(device), return_z=True, z_ae_mode=True)  # [1, 3, 30]

                # LAE encode → mask → decode
                mu = lae_model.encoder(z_opt, c=params_sc, x0=x0_r)
                z_masked = lae_model._apply_mask(mu)
                _, _, _, _, perf_norm = lae_model.decoder(z_masked, params_sc, x0_r)
                if lae_model.scaler_perfs is None:
                    continue
                perf_denorm = lae_model.scaler_perfs.inverse_transform(perf_norm.cpu())

                cd_pred = float(perf_denorm[0, 0])
                cl_pred = float(perf_denorm[0, 1])

                cd_errors.append(abs(cd_pred - cd_gt))
                cl_errors.append(abs(cl_pred - cl_gt))
                cd_preds.append(cd_pred)
                cl_preds.append(cl_pred)
                cd_gts.append(cd_gt)
                cl_gts.append(cl_gt)

        print(f"  {'idx':>4}  {'cd_gt':>8}  {'cd_pred':>8}  {'cd_err':>8}  {'cl_gt':>8}  {'cl_pred':>8}  {'cl_err':>8}")
        for i in range(len(cd_gts)):
            print(f"  {i:>4}  {cd_gts[i]:>8.4f}  {cd_preds[i]:>8.4f}  {cd_errors[i]:>8.4f}  {cl_gts[i]:>8.4f}  {cl_preds[i]:>8.4f}  {cl_errors[i]:>8.4f}")
        print(f"\n  Mean cd error: {sum(cd_errors)/len(cd_errors):.4f}  |  Mean cl error: {sum(cl_errors)/len(cl_errors):.4f}")
        if sum(cd_errors)/len(cd_errors) < 1.0:
            print("  >>> Perf head reconstructs training samples OK — problem is GMM sampling out-of-distribution")
        else:
            print("  >>> Perf head has HIGH error on training samples — decoder perf head is undertrained")
    print("=" * 60 + "\n")
    # --- End perf diagnostic ---

    print(f"\n[DEBUG] Scalers after load:")
    print(f"  aoas_mean_std:     {lae_model.aoas_mean_std}")
    print(f"  params_mean_std:   {lae_model.params_mean_std}")
    print(f"  pressures_mean_std:{lae_model.pressures_mean_std}")
    print(f"  perfs_mean_std:    {lae_model.perfs_mean_std}")
    print(f"[DEBUG] Training history: {len(lae_model.stats.get('train_loss', []))} epochs recorded")
    if len(lae_model.stats.get('train_loss', [])) > 0:
        tl = lae_model.stats['train_loss']
        print(f"  First loss: {tl[0]:.4f}  |  Last loss: {tl[-1]:.4f}  |  Min loss: {min(tl):.4f}")
    print(f"[DEBUG] Active latent dims: {lae_model.active_latent_mask.sum().item()} / {lae_model.lae_latent_dim}\n")
    # --- END DEBUG ---

    from engiopt.lvae.train_lvae import ensure_perf_regressor_fitted
    ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    reg_path  = os.path.join(os.path.dirname(checkpoint_path), f"{ckpt_stem}_perf_reg.pkl")
    ensure_perf_regressor_fitted(lae_model, cfg, reg_path, n_samples=args.n_samples)

    if len(lae_model.stats.get('train_loss', [])) == 0 or lae_model.stats.get('current_loss', None) == 0:
        print(
            "\nWARNING: The loaded checkpoint appears to be untrained.\n"
            "Please train first:\n"
            "  python -m engiopt.lvae.train_lvae\n"
        )
        sys.exit(1)

    target_params = [args.mach, args.reynolds, args.cl_target, args.area]
    print(f"\nTarget conditions: mach={args.mach}  reynolds={args.reynolds:.2e}  "
          f"cl_target={args.cl_target}  area_ratio={args.area}")

    # Pre-encode all initial wings so generate_airfoils() can cycle through them.
    from engibench.problems.wings3D.v0 import Wings3D as _W3D
    _init_items = [item for item in list(_W3D(seed=0).dataset["train"])
                   if item["initial"] == 1]
    print(f"Pre-encoding {len(_init_items)} initial wings as x0 pool...")
    x0_pool = []
    with torch.no_grad():
        for item in _init_items:
            coords = torch.tensor(item["coords"], dtype=torch.float32)
            x_root = coords[0].permute(1, 0).unsqueeze(0).to(device)
            z = bae_model.encode(x_root, return_z=True, z_ae_mode=True)
            x0_pool.append(z.squeeze(0).cpu())

    num_samples = args.num_generate
    print(f"Generating {num_samples} airfoils (z=0, cycling over {len(x0_pool)} x0s)...")
    generated_airfoils, generated_alphas, pressure_pred, perf_pred = generate_airfoils(
        lae_model, bae_model, num_samples, device,
        params=target_params, x0_encoded=x0_pool,
    )

    save_dir = save_airfoils(generated_airfoils, generated_alphas, pressure_pred, perf_pred)

    n = len(generated_airfoils)
    alphas_np = generated_alphas.squeeze().cpu().numpy()

    # Wing geometry plots
    for idx in range(n):
        fig = plt.figure(figsize=(12, 5))
        ax  = fig.add_subplot(projection='3d')
        wing_3D_shape_plot(generated_airfoils[idx], ax=ax,
                           facecolor='steelblue', alpha=0.7, slice_mode=True)
        alpha_val = float(np.atleast_1d(alphas_np)[idx])
        ax.set_title(f'Generated Wing {idx}  (α={alpha_val:.2f}°)')
        ax.set_xlabel('x')
        ax.set_ylabel('z (span)')
        ax.set_zlabel('y')
        ax.set_xlim(0.0, 1.0)
        ax.set_zlim(-0.3, 0.3)
        plot_path = os.path.join(save_dir, f"generated_wing_{idx:02d}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved wing plot to {plot_path}")

    # Generated pressure plots (Cp over slices — no GT available here)
    for idx in range(n):
        pres_path = os.path.join(save_dir, f"pressure_{idx:02d}.png")
        plot_generated_pressure(pressure_pred[idx], sample_idx=idx, save_path=pres_path)
        print(f"Saved pressure plot to {pres_path}")


if __name__ == "__main__":
    main()
