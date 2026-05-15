import sys
import os
from datetime import datetime, timezone

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Aggressively disable all LaTeX
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['text.latex.preamble'] = ''
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['pdf.use14corefonts'] = True
import numpy as np

from engiopt.ddm.train_ddm_newdata import Config, load_bae, build_unet, build_sampler, _SLICES_PKL, _SCALARS_PKL
from engiopt.ddm.ddm import DDM_AoAInit_3D
from engiopt.ddm.plotting import wing_3D_shape_plot
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset


def test_bae_reconstruction(bae_model, device, save_dir=None, n_wings=20):
    """
    Test BAE reconstruction quality across n_wings diverse test wings.
    Reports per-slice and per-wing MSE spread so we can see if any
    wing type or spanwise position is harder to reconstruct.
    """
    print("\n" + "="*60)
    print(f"TESTING BAE RECONSTRUCTION QUALITY ({n_wings} wings)")
    print("="*60)

    dataset = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=0)
    test_items = [item for item in dataset["test"] if item["final"] == 1]
    # Sample evenly across the test set for diversity
    indices = np.linspace(0, len(test_items) - 1, n_wings, dtype=int)
    selected = [test_items[i] for i in indices]

    # Determine n_slices from first item
    first_coords = np.array(selected[0]["coords"])
    n_slices = first_coords.shape[0]

    # mse_matrix[wing, slice]
    mse_matrix = np.zeros((n_wings, n_slices))

    bae_model.eval()
    with torch.no_grad():
        for wi, item in enumerate(selected):
            coords = torch.tensor(item["coords"], dtype=torch.float32)
            coords_centered = coords.clone()
            for s in range(n_slices):
                coords_centered[s, :, 1] -= coords[s, 0, 1]
                coords_centered[s, :, 0] += (1.0 - coords[s, 0, 0])

            for s in range(n_slices):
                x_s = coords_centered[s].permute(1, 0).unsqueeze(0).to(device)
                z_s = bae_model.encode(x_s, return_z=True, z_ae_mode=False)
                recon = bae_model.decode_z(
                    z_s, z_ae_mode=False, denormalize_output=False, normalized_data=False
                )[0]
                mse_matrix[wi, s] = torch.mean((x_s - recon) ** 2).item()

    # Per-slice summary (averaged over wings)
    print("\nPer-slice MSE  (mean ± std over wings):")
    for s in range(n_slices):
        col = mse_matrix[:, s]
        print(f"  Slice {s}: mean={col.mean():.6f}  std={col.std():.6f}  "
              f"min={col.min():.6f}  max={col.max():.6f}")

    # Per-wing summary (averaged over slices)
    print("\nPer-wing MSE  (mean ± std over slices):")
    for wi in range(n_wings):
        row = mse_matrix[wi, :]
        print(f"  Wing {indices[wi]:3d}: mean={row.mean():.6f}  std={row.std():.6f}  "
              f"min={row.min():.6f}  max={row.max():.6f}")

    overall = mse_matrix.mean()
    worst_wing  = indices[mse_matrix.mean(axis=1).argmax()]
    worst_slice = mse_matrix.mean(axis=0).argmax()
    print(f"\nOverall mean MSE: {overall:.6f}")
    print(f"Hardest wing  (by avg MSE): wing index {worst_wing}")
    print(f"Hardest slice (by avg MSE): slice {worst_slice}")

    if overall > 0.01:
        print("\nWARNING: BAE reconstruction is poor (MSE > 0.01)!")
    elif overall > 0.001:
        print("\nCAUTION: BAE reconstruction is acceptable but could be better.")
    else:
        print("\nBAE reconstruction looks good! (MSE < 0.001)")

    # Save one reconstruction plot per slice for the worst wing
    if save_dir:
        worst_item = test_items[int(worst_wing)]
        coords = torch.tensor(worst_item["coords"], dtype=torch.float32)
        coords_centered = coords.clone()
        for s in range(n_slices):
            coords_centered[s, :, 1] -= coords[s, 0, 1]
            coords_centered[s, :, 0] += (1.0 - coords[s, 0, 0])

        n_cols = min(n_slices, 5)
        n_rows = (n_slices + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten()
        with torch.no_grad():
            for s in range(n_slices):
                x_s = coords_centered[s].permute(1, 0).unsqueeze(0).to(device)
                z_s = bae_model.encode(x_s, return_z=True, z_ae_mode=False)
                recon = bae_model.decode_z(
                    z_s, z_ae_mode=False, denormalize_output=False, normalized_data=False
                )[0]
                mse = mse_matrix[mse_matrix.mean(axis=1).argmax(), s]
                ax = axes[s]
                ax.plot(x_s[0, 0].cpu(), x_s[0, 1].cpu(), 'b-', label='Original', lw=2)
                ax.plot(recon[0, 0].cpu(), recon[0, 1].cpu(), 'r--', label='Recon', lw=1.5)
                ax.set_title(f"Slice {s}  MSE={mse:.6f}")
                ax.set_aspect('equal')
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.15, 0.15)
                ax.grid(True)
                if s == 0:
                    ax.legend()
        for ax in axes[n_slices:]:
            ax.set_visible(False)
        fig.suptitle(f"BAE reconstruction — worst wing (index {worst_wing})", fontsize=13)
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, "bae_recon_worst_wing.png"), dpi=150)
        plt.close()
        print(f"Saved reconstruction plot to {save_dir}/bae_recon_worst_wing.png")

    print("="*60 + "\n")
    return overall

def check_training_latent_distribution(bae_model, device, n_samples=200):
    """
    Encode a sample of training airfoils and report per-slice latent statistics.
    Used to validate clamp range and compare with generated latent distributions.
    Returns: dict mapping slice_idx -> dict(min, max, mean, std) for each latent channel.
    """
    print("\n" + "="*60)
    print("TRAINING LATENT DISTRIBUTION (per slice, per channel)")
    print("="*60)

    dataset = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=0)
    train_items = [item for item in dataset["train"] if item["final"] == 1][:n_samples]

    # Determine n_slices from first item
    first_coords = np.array(train_items[0]["coords"])
    n_slices = first_coords.shape[0]

    # Collect latents per slice
    slice_latents = [[] for _ in range(n_slices)]

    bae_model.eval()
    with torch.no_grad():
        for item in train_items:
            coords = torch.tensor(item["coords"], dtype=torch.float32)  # [w_dim,192,2]
            coords_centered = coords.clone()
            for s in range(n_slices):
                coords_centered[s, :, 1] -= coords[s, 0, 1]
                coords_centered[s, :, 0] += (1.0 - coords[s, 0, 0])
            for s in range(n_slices):
                x_s = coords_centered[s].permute(1, 0).unsqueeze(0).to(device)
                z_s = bae_model.encode(x_s, return_z=True, z_ae_mode=False)[:, :, 1:-1]  # [1,3,30]
                slice_latents[s].append(z_s.squeeze(0).cpu())  # [3,30]

    stats = {}
    for s in range(n_slices):
        z_all = torch.stack(slice_latents[s], dim=0)  # [N,3,30]
        s_min  = z_all.min().item()
        s_max  = z_all.max().item()
        s_mean = z_all.mean().item()
        s_std  = z_all.std().item()
        stats[s] = dict(min=s_min, max=s_max, mean=s_mean, std=s_std, tensor=z_all)
        print(f"  Slice {s}: min={s_min:.3f}  max={s_max:.3f}  "
              f"mean={s_mean:.3f}  std={s_std:.3f}")

    overall = torch.stack([stats[s]['tensor'] for s in range(n_slices)])
    print(f"\n  OVERALL: min={overall.min():.3f}  max={overall.max():.3f}  "
          f"mean={overall.mean():.3f}  std={overall.std():.3f}")
    print(f"\n  Suggested clamp range (mean ± 4*std): "
          f"[{overall.mean()-4*overall.std():.3f}, {overall.mean()+4*overall.std():.3f}]")
    print("="*60 + "\n")
    return stats


def check_per_slice_quality(decoded_airfoils, save_dir):
    """
    Per-sample, per-slice quality checks on decoded airfoils.
    Checks:
      1. x range: should be roughly [0, 1]
      2. y range: should be roughly [-0.15, 0.15]
      3. Upper surface above lower surface (no self-intersection)
      4. Leading edge x < 0.1
    Reports a quality score per slice and flags failures.
    decoded_airfoils: [B, 9, 2, N]
    """
    print("\n" + "="*60)
    print("PER-SLICE QUALITY CHECKS")
    print("="*60)

    B, S, _, N = decoded_airfoils.shape
    arr = decoded_airfoils.cpu().numpy()  # [B,9,2,N]

    slice_fail_counts = np.zeros(S, dtype=int)
    sample_fail_counts = np.zeros(B, dtype=int)

    for i in range(B):
        for s in range(S):
            x = arr[i, s, 0, :]
            y = arr[i, s, 1, :]
            failures = []

            if x.min() < -0.1 or x.max() > 1.1:
                failures.append(f"x_range=[{x.min():.2f},{x.max():.2f}]")
            if y.min() < -0.25 or y.max() > 0.25:
                failures.append(f"y_range=[{y.min():.2f},{y.max():.2f}]")
            # Leading edge should reach x < 0.1
            if x.min() > 0.1:
                failures.append(f"no_leading_edge(x_min={x.min():.2f})")
            # Upper/lower check: first half should have y >= second half (mirrored)
            half = N // 2
            y_upper = y[:half]
            y_lower = y[half:][::-1]
            n_cross = int((y_upper < y_lower).sum())
            if n_cross > half * 0.1:
                failures.append(f"self_intersect({n_cross}/{half}pts)")

            if failures:
                slice_fail_counts[s] += 1
                sample_fail_counts[i] += 1
                print(f"  Sample {i:02d} Slice {s}: FAIL — {', '.join(failures)}")

    print(f"\nSlice failure counts (out of {B} samples):")
    for s in range(S):
        bar = '#' * slice_fail_counts[s]
        print(f"  Slice {s}: {slice_fail_counts[s]:3d}  {bar}")

    print(f"\nSample failure counts (out of {S} slices):")
    for i in range(B):
        bar = '#' * sample_fail_counts[i]
        print(f"  Sample {i:02d}: {sample_fail_counts[i]:3d}  {bar}")

    # Save summary to file
    summary_path = os.path.join(save_dir, "quality_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Slice failure counts:\n")
        for s in range(S):
            f.write(f"  Slice {s}: {slice_fail_counts[s]}/{B}\n")
        f.write("\nSample failure counts:\n")
        for i in range(B):
            f.write(f"  Sample {i:02d}: {sample_fail_counts[i]}/{S}\n")
    print(f"\nSaved quality summary to {summary_path}")
    print("="*60 + "\n")
    return slice_fail_counts, sample_fail_counts


def compare_latent_distributions(train_stats, generated_airfoils, save_dir):
    """
    Compare per-slice latent stats of training data vs generated latents (before decode).
    generated_airfoils: [B, w_dim, 3, 30] — the raw DDM output latents
    """
    print("\n" + "="*60)
    print("TRAINING vs GENERATED LATENT DISTRIBUTION COMPARISON")
    print("="*60)

    gen = generated_airfoils.cpu()  # [B,w_dim,3,30]
    # train_stats only covers the slices encoded from the training dataset;
    # generated latents may have more slices (new dataset). Compare what we have.
    n_slices = min(gen.shape[1], len(train_stats))

    lines = []
    for s in range(n_slices):
        tr = train_stats[s]
        g_s = gen[:, s, :, :]  # [B,3,30]
        g_min  = g_s.min().item()
        g_max  = g_s.max().item()
        g_mean = g_s.mean().item()
        g_std  = g_s.std().item()

        oob_lo = (g_s < tr['min']).float().mean().item() * 100
        oob_hi = (g_s > tr['max']).float().mean().item() * 100

        line = (f"  Slice {s}: "
                f"train=[{tr['min']:.2f},{tr['max']:.2f}] μ={tr['mean']:.3f} σ={tr['std']:.3f}  |  "
                f"gen=[{g_min:.2f},{g_max:.2f}] μ={g_mean:.3f} σ={g_std:.3f}  |  "
                f"OOB: {oob_lo:.1f}% below, {oob_hi:.1f}% above")
        print(line)
        lines.append(line)

    summary_path = os.path.join(save_dir, "latent_distribution_comparison.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved latent comparison to {summary_path}")
    print("="*60 + "\n")


def check_channel0_validity(bae_model, device):
    """
    Debug: check whether training wings that encode to channel 0 < 0.1 are themselves valid.
    Answers: is the 20% OOB a model failure, or does the training data naturally live below 0.1?
    """
    print("\n" + "="*60)
    print("CHANNEL 0 VALIDITY DEBUG")
    print("="*60)

    dataset = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=0)
    train_items = [item for item in dataset["train"] if item["final"] == 1]

    # Determine n_slices from first item
    first_coords = np.array(train_items[0]["coords"])
    n_slices = first_coords.shape[0]

    n_total = 0
    n_below = 0
    n_below_valid = 0
    ch0_all = []

    bae_model.eval()
    with torch.no_grad():
        for item in train_items:
            coords = torch.tensor(item["coords"], dtype=torch.float32)
            coords_centered = coords.clone()
            for s in range(n_slices):
                coords_centered[s, :, 1] -= coords[s, 0, 1]
                coords_centered[s, :, 0] += (1.0 - coords[s, 0, 0])

            wing_latents = []
            for s in range(n_slices):
                x_s = coords_centered[s].permute(1, 0).unsqueeze(0).to(device)
                z_s = bae_model.encode(x_s, return_z=True, z_ae_mode=False)[:, :, 1:-1]
                wing_latents.append(z_s)  # [1, 3, 30]

            ch0_vals = torch.cat([z[:, 0, :] for z in wing_latents])  # [n_slices*30]
            ch0_all.append(ch0_vals.cpu())
            has_below = (ch0_vals < 0.1).any().item()
            n_total += 1

            if has_below:
                n_below += 1
                decoded_slices = []
                for s in range(n_slices):
                    dec = bae_model.decode_z(
                        wing_latents[s], z_ae_mode=True,
                        denormalize_output=False, normalized_data=False
                    )[0]
                    decoded_slices.append(dec.cpu())
                decoded = torch.stack(decoded_slices, dim=0)  # [n_slices, 2, N]
                if is_valid_wing(decoded):
                    n_below_valid += 1

    ch0_all = torch.cat(ch0_all)
    print(f"Channel 0 training range: min={ch0_all.min():.4f}  max={ch0_all.max():.4f}  "
          f"mean={ch0_all.mean():.4f}  std={ch0_all.std():.4f}")
    pct_below = 100 * (ch0_all < 0.1).float().mean().item()
    print(f"Training ch0 values < 0.1: {pct_below:.1f}%")
    print(f"Wings with any ch0 slice < 0.1: {n_below}/{n_total} ({100*n_below/n_total:.1f}%)")
    if n_below > 0:
        print(f"Of those, valid after decode:    {n_below_valid}/{n_below} ({100*n_below_valid/n_below:.1f}%)")
        if n_below_valid == n_below:
            print("  => ch0 < 0.1 decodes fine: clamp [0.1, 2.0] may be too conservative")
        elif n_below_valid == 0:
            print("  => ch0 < 0.1 always invalid: clamp is justified, OOB is a model quality issue")
        else:
            print("  => mixed: some ch0 < 0.1 wings are valid, some not")
    print("="*60 + "\n")


def is_valid_wing(airfoil, max_cross_frac=0.05):
    """
    Returns True if every slice has no significant self-intersections.
    airfoil: [9, 2, N] tensor or numpy array.
    Skips 3 points at each end to avoid false failures at TE/LE where both
    surfaces meet at y≈0. Allows up to 5% of interior points to cross.
    """
    arr = airfoil.cpu().numpy() if isinstance(airfoil, torch.Tensor) else airfoil
    half = arr.shape[2] // 2
    trim = 3
    for s in range(arr.shape[0]):
        y_upper = arr[s, 1, trim:half - trim]
        y_lower = arr[s, 1, half + trim:-trim][::-1]
        n_cross = (y_upper <= y_lower).sum()
        if n_cross > max_cross_frac * len(y_upper):
            return False
    return True


def _decode_latents(generated_airfoils, bae_model):
    """Decode a batch of DDM latents [B,w_dim,3,L] → airfoil coords [B,w_dim,2,N]."""
    n_slices = generated_airfoils.shape[1]
    decoded_slices = []
    with torch.no_grad():
        for s in range(n_slices):
            z_s_raw = generated_airfoils[:, s, :, :]
            z_s = z_s_raw.clone()
            z_s[:, 0, :] = torch.clamp(z_s_raw[:, 0, :], 0.1, 2.0)  # weight channel — clamp to BAE training range
            z_s[:, 1:, :] = torch.clamp(z_s_raw[:, 1:, :], -2.228, 3.117) # CP channels
            dec_s = bae_model.decode_z(
                z_s, z_ae_mode=True, denormalize_output=False, normalized_data=False
            )[0]
            decoded_slices.append(dec_s.cpu())
    return torch.stack(decoded_slices, dim=1)  # [B, w_dim, 2, N]


def generate_airfoils_direct(ddm_model, bae_model, num_samples=10, device='cpu',
                             max_attempts=10):
    """
    Generate num_samples valid (non-self-intersecting) wings.
    Conditions are fixed per requested wing; only the diffusion noise is resampled
    on retries so each wing still targets the same aerodynamic parameters.
    """
    # latent_mean/std may be a per-channel tensor [1,1,3,1] or a legacy scalar float
    latent_mean = ddm_model.latent_mean
    latent_std  = ddm_model.latent_std
    if isinstance(latent_mean, torch.Tensor):
        latent_mean = latent_mean.to(device)
        latent_std  = latent_std.to(device)
        print(f"Using per-channel latent stats:")
        print(f"  mean: {latent_mean.squeeze().tolist()}")
        print(f"  std:  {latent_std.squeeze().tolist()}")
        # For z_inits [B, 3, L] we need [1, 3, 1]
        latent_mean_init = latent_mean[0]
        latent_std_init  = latent_std[0]
    else:
        print(f"Using latent stats: mean={latent_mean:.3f}, std={latent_std:.3f}")
        latent_mean_init = latent_mean
        latent_std_init  = latent_std

    # Sample aerodynamic conditions once — these stay fixed per wing across retries
    mach_list = np.random.uniform(0.4, 0.9, num_samples)
    re_list   = np.random.uniform(1e6, 1e7, num_samples)
    cl_list   = np.random.uniform(0.5, 1.2, num_samples)
    area_list = np.random.uniform(0.75, 1.0, num_samples)
    print(f"Sample params - Mach: {mach_list[0]:.2f}, Re: {re_list[0]:.2e}, "
          f"CL: {cl_list[0]:.2f}, Area: {area_list[0]:.2f}")

    _dataset   = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=0)
    all_train  = list(_dataset["train"])
    initial_items = [item for item in all_train if item["initial"] == 1]

    valid_airfoils = []
    valid_alphas   = []
    valid_latents  = []

    # pending_indices tracks which of the num_samples wings still need a valid sample
    pending_indices = list(range(num_samples))

    for attempt in range(max_attempts):
        if not pending_indices:
            break

        batch_n = len(pending_indices)

        # Build scaled params for pending wings only
        params_raw = torch.tensor([
            [mach_list[i], re_list[i], cl_list[i], area_list[i]]
            for i in pending_indices
        ], dtype=torch.float32).to(device)
        params_scaled = ddm_model.scaler_params.transform(params_raw.cpu().numpy())
        params = torch.tensor(params_scaled, dtype=torch.float32).to(device)

        # Fresh noise for this attempt
        w_dim = ddm_model.unet.w_dim
        noise_x     = torch.randn(batch_n, w_dim, 3, 30).to(device)
        noise_alpha = torch.randn(batch_n, 1).to(device)

        # Encode random init airfoils — use initial (unoptimised) items with y-shift only,
        # matching how precompute_latents builds z_init during training.
        encoded_inits = []
        for _ in range(batch_n):
            init_item = np.random.choice(initial_items)
            init_coords = torch.tensor(init_item["coords"], dtype=torch.float32)
            init_root = init_coords[0].clone()
            init_root[:, 1] -= init_root[0, 1]  # y-shift only, no x-shift (matches training)
            x_init = init_root.permute(1, 0).unsqueeze(0).to(device)
            with torch.no_grad():
                z_init = bae_model.encode(x_init, return_z=True, z_ae_mode=False)[:, :, 1:-1]
            encoded_inits.append(z_init)
        encoded_init = (torch.cat(encoded_inits, dim=0) - latent_mean_init) / latent_std_init

        # Run DDM sampler
        with torch.no_grad():
            gen_normalized, gen_alphas = ddm_model.sampler.sample_airfoil(
                model=ddm_model.unet,
                noise_x=noise_x,
                noise_alpha=noise_alpha,
                c=params,
                x0=encoded_init,
                T=ddm_model.sampler.T - 1,
            )

        gen_denorm = gen_normalized * latent_std + latent_mean

        ch0 = gen_denorm[:, :, 0, :]
        above_clamp = (ch0 > 2.0).float().mean().item() * 100
        below_clamp = (ch0 < 0.1).float().mean().item() * 100
        print(f"  [attempt {attempt + 1}] latent norm=[{gen_normalized.min():.2f},{gen_normalized.max():.2f}]  "
              f"ch0 OOB: {below_clamp:.1f}% below, {above_clamp:.1f}% above")


        decoded    = _decode_latents(gen_denorm, bae_model)

        alphas_rescaled = torch.tensor(
            ddm_model.scaler_aoas.inverse_transform(gen_alphas.cpu().numpy())
        )

        # Filter: keep valid, requeue invalid
        # Also reject diverged samples: latent out of range or alpha out of physical range
        still_pending = []
        for local_idx, global_idx in enumerate(pending_indices):
            wing = decoded[local_idx]
            alpha_val = alphas_rescaled[local_idx].item()
            latent_ok = gen_normalized[local_idx].abs().max().item() < 20.0
            alpha_ok = -20.0 < alpha_val < 30.0  # physical AoA bounds (degrees)
            if is_valid_wing(wing) and latent_ok and alpha_ok:
                valid_airfoils.append(wing)
                valid_alphas.append(alphas_rescaled[local_idx])
                valid_latents.append(gen_denorm[local_idx].cpu())
            else:
                still_pending.append(global_idx)

        n_valid_this = batch_n - len(still_pending)
        print(f"Attempt {attempt + 1}/{max_attempts}: "
              f"{n_valid_this}/{batch_n} valid  —  "
              f"total {len(valid_airfoils)}/{num_samples}")

        pending_indices = still_pending

    if len(valid_airfoils) < num_samples:
        print(f"WARNING: only {len(valid_airfoils)}/{num_samples} valid wings "
              f"after {max_attempts} attempts — returning what we have.")

    n_out = min(len(valid_airfoils), num_samples)
    decoded_airfoils = torch.stack(valid_airfoils[:n_out], dim=0)
    rescaled_alphas  = torch.stack(valid_alphas[:n_out], dim=0).to(device)
    raw_latents      = torch.stack(valid_latents[:n_out], dim=0)

    print(f"\ndecoded_airfoils shape: {decoded_airfoils.shape}")
    print(f"x range: [{decoded_airfoils[:,:,0,:].min():.3f}, {decoded_airfoils[:,:,0,:].max():.3f}]")
    print(f"y range: [{decoded_airfoils[:,:,1,:].min():.3f}, {decoded_airfoils[:,:,1,:].max():.3f}]")
    print(f"Alpha values: {rescaled_alphas.squeeze()}")

    return decoded_airfoils, rescaled_alphas, raw_latents


def generate_from_test_items(ddm_model, bae_model, test_items, initial_by_case, device,
                             max_attempts=5):
    """
    Generate wings conditioned on the actual aerodynamic parameters from test dataset items.
    Each generated wing targets the same Mach/Reynolds/CL/area as its paired ground truth,
    so the comparison is apples-to-apples.

    Returns:
        decoded_airfoils:  [N, 9, 2, 192]
        rescaled_alphas:   [N]
        raw_latents:       [N, 9, 3, 30]
        paired_gt_coords:  list of N np.arrays [9, 192, 2]
        paired_gt_alphas:  list of N floats
    """
    latent_mean = ddm_model.latent_mean
    latent_std  = ddm_model.latent_std
    if isinstance(latent_mean, torch.Tensor):
        latent_mean = latent_mean.to(device)
        latent_std  = latent_std.to(device)
        latent_mean_init = latent_mean[0]
        latent_std_init  = latent_std[0]
    else:
        latent_mean_init = latent_mean
        latent_std_init  = latent_std

    valid_airfoils   = []
    valid_alphas     = []
    valid_latents    = []
    paired_gt_coords = []
    paired_gt_alphas = []

    pending_items = list(test_items)

    for attempt in range(max_attempts):
        if not pending_items:
            break

        batch_n = len(pending_items)

        # Build scaled params from the actual test item conditions
        params_raw = torch.tensor([
            [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]]
            for item in pending_items
        ], dtype=torch.float32).to(device)
        params_scaled = ddm_model.scaler_params.transform(params_raw.cpu().numpy())
        params = torch.tensor(params_scaled, dtype=torch.float32).to(device)

        # Fresh noise for this attempt
        w_dim = ddm_model.unet.w_dim
        noise_x     = torch.randn(batch_n, w_dim, 3, 30).to(device)
        noise_alpha = torch.randn(batch_n, 1).to(device)

        # Encode init from the matching initial (unoptimized) case — same logic as training
        encoded_inits = []
        for item in pending_items:
            case_num = int(item["case_num"])
            if case_num in initial_by_case:
                init_coords = torch.tensor(
                    initial_by_case[case_num]["coords"], dtype=torch.float32
                )
                init_root = init_coords[0].clone()
            else:
                print(f"[WARNING] No initial wing for case {case_num}, using final root slice")
                init_root = torch.tensor(item["coords"], dtype=torch.float32)[0].clone()
            init_root[:, 1] -= init_root[0, 1]  # y-shift only (matches training)
            x_init = init_root.permute(1, 0).unsqueeze(0).to(device)
            with torch.no_grad():
                z_init = bae_model.encode(x_init, return_z=True, z_ae_mode=False)[:, :, 1:-1]
            encoded_inits.append(z_init)
        encoded_init = (torch.cat(encoded_inits, dim=0) - latent_mean_init) / latent_std_init

        # Run DDM sampler
        with torch.no_grad():
            gen_normalized, gen_alphas = ddm_model.sampler.sample_airfoil(
                model=ddm_model.unet,
                noise_x=noise_x,
                noise_alpha=noise_alpha,
                c=params,
                x0=encoded_init,
                T=ddm_model.sampler.T - 1,
            )

        gen_denorm = gen_normalized * latent_std + latent_mean
        ch0 = gen_denorm[:, :, 0, :]
        above_clamp = (ch0 > 2.0).float().mean().item() * 100
        below_clamp = (ch0 < 0.1).float().mean().item() * 100
        print(f"  [attempt {attempt + 1}] latent norm=[{gen_normalized.min():.2f},{gen_normalized.max():.2f}]  "
              f"ch0 OOB: {below_clamp:.1f}% below, {above_clamp:.1f}% above")

        decoded = _decode_latents(gen_denorm, bae_model)
        alphas_rescaled = torch.tensor(
            ddm_model.scaler_aoas.inverse_transform(gen_alphas.cpu().numpy())
        )

        still_pending = []
        for local_idx, item in enumerate(pending_items):
            wing = decoded[local_idx]
            alpha_val = alphas_rescaled[local_idx].item()
            latent_ok = gen_normalized[local_idx].abs().max().item() < 20.0
            alpha_ok  = -20.0 < alpha_val < 30.0
            if is_valid_wing(wing) and latent_ok and alpha_ok:
                valid_airfoils.append(wing)
                valid_alphas.append(alphas_rescaled[local_idx])
                valid_latents.append(gen_denorm[local_idx].cpu())
                paired_gt_coords.append(np.array(item["coords"]))
                paired_gt_alphas.append(float(item["alpha"]))
            else:
                still_pending.append(item)

        n_valid_this = batch_n - len(still_pending)
        print(f"Attempt {attempt + 1}/{max_attempts}: "
              f"{n_valid_this}/{batch_n} valid  —  "
              f"total {len(valid_airfoils)}/{len(test_items)}")
        pending_items = still_pending

    if len(valid_airfoils) < len(test_items):
        print(f"WARNING: only {len(valid_airfoils)}/{len(test_items)} valid wings "
              f"after {max_attempts} attempts — returning what we have.")

    decoded_airfoils = torch.stack(valid_airfoils, dim=0)
    rescaled_alphas  = torch.stack(valid_alphas, dim=0).to(device)
    raw_latents      = torch.stack(valid_latents, dim=0)

    print(f"\ndecoded_airfoils shape: {decoded_airfoils.shape}")
    print(f"x range: [{decoded_airfoils[:,:,0,:].min():.3f}, {decoded_airfoils[:,:,0,:].max():.3f}]")
    print(f"y range: [{decoded_airfoils[:,:,1,:].min():.3f}, {decoded_airfoils[:,:,1,:].max():.3f}]")
    print(f"Alpha values: {rescaled_alphas.squeeze()}")

    return decoded_airfoils, rescaled_alphas, raw_latents, paired_gt_coords, paired_gt_alphas


def save_airfoils_only(generated_airfoils, generated_alphas):
    """Save airfoils without trying to plot"""
    print("\n" + "="*50)
    print("GENERATION RESULTS:")
    print(f"Generated {len(generated_airfoils)} airfoils")
    print(f"Airfoil shape: {generated_airfoils.shape}")
    print(f"Alpha values: {generated_alphas}")
    print("="*50 + "\n")

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("results", "generated", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    pt_path = os.path.join(save_dir, "generated_airfoils.pt")
    npz_path = os.path.join(save_dir, "generated_airfoils.npz")

    torch.save({
        "airfoils": generated_airfoils.cpu(),
        "alphas": generated_alphas.cpu()
    }, pt_path)
    print(f"Saved generated airfoils to {pt_path}")

    np.savez(npz_path,
             airfoils=generated_airfoils.cpu().numpy(),
             alphas=generated_alphas.cpu().numpy())
    print(f"Also saved as {npz_path}")

    # Print first 10 points of root slice (slice 0) of first airfoil
    print("\nFirst airfoil, root slice (slice 0), first 10 points (x, y):")
    first_airfoil = generated_airfoils[0].cpu().numpy()  # [9, 2, 192]
    for i in range(min(10, first_airfoil.shape[2])):
        print(f"  {i}: x={first_airfoil[0, 0, i]:.4f}, y={first_airfoil[0, 1, i]:.4f}")
    return save_dir, timestamp

def save_comparison_plots(generated_airfoil, ground_truth_coords, alpha_gen, alpha_gt, save_dir, wing_idx,
                          conditions=None):
    """
    Plot generated vs ground truth slices side by side.
    generated_airfoil: [w_dim, 2, 192] - centered coordinates (trailing edge at y=0)
    ground_truth_coords: [w_dim, 192, 2] - raw from dataset (trailing edge at original y)
    conditions: optional (mach, reynolds, cl_target, area) tuple shown in the title
    """
    gen_np = generated_airfoil.cpu().numpy()
    gt_np = ground_truth_coords.copy()
    n_slices = gen_np.shape[0]

    n_cols = min(n_slices, 5)
    n_rows = (n_slices + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    # Center the ground truth coordinates to match the generated format
    for s in range(n_slices):
        gt_np[s, :, 1] -= gt_np[s, 0, 1]
        gt_np[s, :, 0] += (1.0 - gt_np[s, 0, 0])

    for s in range(n_slices):
        ax = axes[s]

        # Generated (already centered)
        x_gen = gen_np[s, 0, :]
        y_gen = gen_np[s, 1, :]
        ax.plot(x_gen, y_gen, 'r-', label='Generated', linewidth=1.5)

        # Ground truth (now centered)
        x_gt = gt_np[s, :, 0]
        y_gt = gt_np[s, :, 1]
        ax.plot(x_gt, y_gt, 'b-', label='Ground Truth', linewidth=1.5)

        ax.set_title(f"Slice {s}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.15, 0.15)
        if s == 0:
            ax.legend()
    for ax in axes[n_slices:]:
        ax.set_visible(False)

    cond_str = ""
    if conditions is not None:
        mach, reynolds, cl_target, area = conditions
        cond_str = f"  |  Mach={mach:.2f}  Re={reynolds:.2e}  CL={cl_target:.2f}  Area={area:.3f}"
    fig.suptitle(
        f"Wing {wing_idx} — Generated (α={alpha_gen:.2f}°) vs Ground Truth (α={alpha_gt:.2f}°){cond_str}",
        fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = os.path.join(save_dir, f"comparison_wing_{wing_idx:02d}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {path}")

def save_trailing_edge_comparison(generated_airfoil, ground_truth_coords, alpha_gen, alpha_gt, save_dir, wing_idx):
    """
    Zoomed plot focusing on trailing edge region (x > 0.9)
    """
    gen_np = generated_airfoil.cpu().numpy()
    gt_np = ground_truth_coords.copy()
    n_slices = gen_np.shape[0]

    n_cols = min(n_slices, 5)
    n_rows = (n_slices + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    # Center the ground truth coordinates
    for s in range(n_slices):
        gt_np[s, :, 1] -= gt_np[s, 0, 1]
        gt_np[s, :, 0] += (1.0 - gt_np[s, 0, 0])

    for s in range(n_slices):
        ax = axes[s]
        
        # Focus on trailing edge region (last 20 points)
        x_gen = gen_np[s, 0, -20:]
        y_gen = gen_np[s, 1, -20:]
        x_gt = gt_np[s, -20:, 0]
        y_gt = gt_np[s, -20:, 1]
        
        ax.plot(x_gen, y_gen, 'r-', label='Generated', linewidth=1.5, marker='o', markersize=3)
        ax.plot(x_gt, y_gt, 'b-', label='Ground Truth', linewidth=1.5, marker='x', markersize=3)
        
        ax.set_title(f"Slice {s} - Trailing Edge")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.set_xlim(0.95, 1.05)
        ax.set_ylim(-0.05, 0.05)
        if s == 0:
            ax.legend()
    for ax in axes[n_slices:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Trailing Edge Comparison - Wing {wing_idx} (α_gen={alpha_gen:.2f}°, α_gt={alpha_gt:.2f}°)",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = os.path.join(save_dir, f"trailing_edge_wing_{wing_idx:02d}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved trailing edge comparison to {path}")

def save_slice_plots(generated_airfoil, alpha_value, save_dir, wing_idx):
    """
    Save a grid of 2D slice plots for one generated wing.
    generated_airfoil: [w_dim, 2, 192]
    """
    airfoil_np = generated_airfoil.cpu().numpy()
    n_slices = airfoil_np.shape[0]

    n_cols = min(n_slices, 5)
    n_rows = (n_slices + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for s in range(n_slices):
        ax = axes[s]
        x = airfoil_np[s, 0, :]
        y = airfoil_np[s, 1, :]

        ax.plot(x, y)
        ax.set_title(f"Slice {s}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)

        # Optional: keep axes visually consistent across slices
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.15, 0.15)

    for ax in axes[n_slices:]:
        ax.set_visible(False)

    fig.suptitle(f"Generated Wing {wing_idx} slices (α={alpha_value:.2f}°)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    slice_plot_path = os.path.join(save_dir, f"generated_wing_{wing_idx:02d}_slices.png")
    plt.savefig(slice_plot_path, dpi=150)
    plt.close()
    print(f"Saved 2D slice plot to {slice_plot_path}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    cfg = Config()
    cfg.model_name = "ddm_newdata_v1"
    cfg.save_dir = "results/ddm_newdata"
    
    print("Loading BAE...")
    bae_model = load_bae(cfg)

    # Test BAE reconstruction quality
    print("\n" + "="*60)
    print("TESTING BAE RECONSTRUCTION QUALITY")
    print("="*60)
    
    # Create temp directory for BAE test plots
    bae_test_dir = os.path.join("results", "bae_test", datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S"))
    os.makedirs(bae_test_dir, exist_ok=True)
    
    bae_avg_mse = test_bae_reconstruction(bae_model, device, bae_test_dir)
    
    # If BAE reconstruction is bad, ask if user wants to continue
    if bae_avg_mse > 0.01:
        print(f"\n⚠️  BAE reconstruction MSE = {bae_avg_mse:.6f} (> 0.01)")
        print("   Generated airfoils will likely be poor quality.")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(1)

    # SANITY CHECK - encode then immediately decode one real sample
    _sanity_dataset = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=0)
    item = next(item for item in _sanity_dataset["train"] if item["final"] == 1)

    coords = torch.tensor(item["coords"], dtype=torch.float32)  # [w_dim,192,2]

    # Center each slice by subtracting the trailing edge y-coordinate (first point)
    coords_centered = coords.clone()
    for s in range(coords_centered.shape[0]):
        coords_centered[s, :, 1] -= coords[s, 0, 1]
        coords_centered[s, :, 0] += (1.0 - coords[s, 0, 0])

    with torch.no_grad():
        for s in range(coords_centered.shape[0]):
            x_s = coords_centered[s].permute(1,0).unsqueeze(0).to(device)  # [1,2,192]
            z_s = bae_model.encode(x_s, return_z=True, z_ae_mode=False)

            dec_false = bae_model.decode_z(
                z_s,
                z_ae_mode=False,
                denormalize_output=False,
                normalized_data=False
            )[0]

            dec_true = bae_model.decode_z(
                z_s,
                z_ae_mode=False,
                denormalize_output=False,
                normalized_data=True
            )[0]

            orig_x = coords_centered[s, :, 0].cpu()
            orig_y = coords_centered[s, :, 1].cpu()

            recon_false_x = dec_false[0, 0, :].cpu()
            recon_false_y = dec_false[0, 1, :].cpu()

            recon_true_x = dec_true[0, 0, :].cpu()
            recon_true_y = dec_true[0, 1, :].cpu()

            err_false = max(
                (orig_x - recon_false_x).abs().max().item(),
                (orig_y - recon_false_y).abs().max().item()
            )
            err_true = max(
                (orig_x - recon_true_x).abs().max().item(),
                (orig_y - recon_true_y).abs().max().item()
            )

            print(f"Slice {s} | decode normalized_data=False max err: {err_false:.6f}")
            print(f"Slice {s} | decode normalized_data=True  max err: {err_true:.6f}")
            
    print("Building DDM components...")
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
    
    checkpoint_path = f"{cfg.save_dir}/{cfg.model_name}.pth"
    print(f"Loading checkpoint from {checkpoint_path}...")
    ddm_model.load(checkpoint_path, train_mode=False)

    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Current loss in checkpoint: {ddm_model.stats.get('current_loss')}")
    print(f"Train loss entries: {len(ddm_model.stats.get('train_loss', []))}")


    ddm_model.unet = ddm_model.unet.to(device)
    ddm_model.bae_model = ddm_model.bae_model.to(device)
    bae_model = ddm_model.bae_model

    # Sanity check: make sure we actually loaded a trained model (non-empty loss history)
    if len(ddm_model.stats.get('train_loss', [])) == 0 or ddm_model.stats.get('current_loss', None) == 0:
        print("\nWARNING: The loaded checkpoint appears to be untrained (no training loss history or loss=0).\n" \
              "Please train the model first by running: \n" \
              "  KMP_DUPLICATE_LIB_OK=TRUE python -m engiopt.ddm.train_ddm\n")
        sys.exit(1)
    
    # Check training latent distribution before generating
    train_latent_stats = check_training_latent_distribution(bae_model, device, n_samples=200)
    check_channel0_validity(bae_model, device)

    # Load test set and generate conditioned on the actual GT parameters
    dataset_gt = NewWingsDataset(_SLICES_PKL, _SCALARS_PKL, seed=cfg.seed)
    all_test = list(dataset_gt["test"])
    gt_finals = [item for item in all_test if item["final"] == 1]
    initial_by_case = {item["case_num"]: item for item in all_test if item["initial"] == 1}

    num_samples = 20
    test_items_subset = gt_finals[:num_samples]
    print(f"Generating {len(test_items_subset)} airfoils conditioned on GT parameters...")
    generated_airfoils, generated_alphas, raw_latents, paired_gt_coords, paired_gt_alphas = \
        generate_from_test_items(ddm_model, bae_model, test_items_subset, initial_by_case, device)

    # Just save the results without plotting
    save_dir, _ = save_airfoils_only(generated_airfoils, generated_alphas)

    # Diagnostics
    compare_latent_distributions(train_latent_stats, raw_latents, save_dir)
    check_per_slice_quality(generated_airfoils, save_dir)

    # Plot each generated wing (3D view of all 9 slices) and compare with its paired GT
    for idx in range(len(generated_airfoils)):
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(projection='3d')
        wing_3D_shape_plot(generated_airfoils[idx], ax=ax, facecolor='steelblue', alpha=0.7, slice_mode=True)
        ax.set_title(f'Generated Wing {idx}  (α={generated_alphas[idx].item():.2f}°)')
        ax.set_xlabel('x')
        ax.set_ylabel('z (span)')
        ax.set_zlabel('y')
        plot_path = os.path.join(save_dir, f"generated_wing_{idx:02d}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot to {plot_path}")
        save_slice_plots(
            generated_airfoils[idx],
            generated_alphas[idx].item(),
            save_dir,
            idx
        )

        # Compare with the paired ground truth (same Mach/Re/CL/area)
        gt_item = test_items_subset[idx]
        conditions = (
            float(gt_item["mach"]),
            float(gt_item["reynolds"]),
            float(gt_item["cl_target"]),
            float(gt_item["area_case_ratio"]),
        )
        save_comparison_plots(
            generated_airfoils[idx],
            paired_gt_coords[idx],
            generated_alphas[idx].item(),
            paired_gt_alphas[idx],
            save_dir,
            idx,
            conditions=conditions,
        )
        save_trailing_edge_comparison(
            generated_airfoils[idx],
            paired_gt_coords[idx],
            generated_alphas[idx].item(),
            paired_gt_alphas[idx],
            save_dir,
            idx,
        )

if __name__ == "__main__":
    main()