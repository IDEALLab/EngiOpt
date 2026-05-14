"""
Reconstruction comparison across v1 / v3 / v7 for supervisor meeting.

For each of N_SAMPLES test wings, shows root / mid / tip slices:
  - Top row:    airfoil shape  (GT blue, each model a distinct colour)
  - Bottom row: Cp curve       (GT blue, each model a distinct colour)

Saves one PDF per sample to results/version_comparison/.
Usage:
    python -u compare_versions.py
"""

import os
import types
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from engibench.problems.wings3D.v0 import Wings3D

from engiopt.lvae.lvae import LAE_AoAInit
from engiopt.lvae.train_lvae import Config, build_encoder, build_decoder, build_sampler, load_bae


# ── Helpers ───────────────────────────────────────────────────────────────────
def _linear_out_features(module):
    """Return the output feature count of a Linear or the last Linear in a Sequential."""
    if hasattr(module, 'weight_orig'):
        return module.weight_orig.shape[0]
    if hasattr(module, 'weight'):
        return module.weight.shape[0]
    # Sequential: walk to last child
    children = list(module.children())
    for child in reversed(children):
        result = _linear_out_features(child)
        if result is not None:
            return result
    return None


# ── Compat forward for v1-style decoders (global heads, no per-slice features) ──
def _v1_decoder_forward(self, z, c, x0):
    """Old decoder architecture: all heads operate on a single global h_flat."""
    B = z.shape[0]
    c_emb = self.c_net(c)
    zc    = torch.cat([z, c_emb], dim=-1)
    h     = self.z_proj(zc).reshape(B, -1, self.base_length)
    for block in self.up_blocks:
        h = block(h)
    h_flat = h.reshape(B, -1)                               # [B, flat_size]

    alpha_pred    = self.alpha_head(h_flat)                  # [B, 1]
    perf_pred     = self.perf_head(h_flat)                   # [B, perf_dim]
    z_opt_pred    = self.z_opt_head(h_flat).reshape(
        B, self.w_dim, self.latent_channels, self.latent_length)
    eta_y_pred    = self.eta_y_head(h_flat).reshape(B, self.w_dim, 1)
    pressure_pred = self.pressure_head(h_flat).reshape(
        B, self.w_dim, self.pressure_length)

    return z_opt_pred, alpha_pred, eta_y_pred, pressure_pred, perf_pred

# ── Settings ──────────────────────────────────────────────────────────────────
CHECKPOINTS = {
    "v1": "results/lvae/lae_aoa_init_3d_v1.pth",
    "v3": "results/lvae/lae_aoa_init_3d_v3.pth",
    "v7": "results/lvae/lae_aoa_init_3d_v7.pth",
}
COLORS = {
    "GT": "steelblue",
    "v1": "#e07b39",   # orange
    "v3": "#9c59b6",   # purple
    "v7": "#27ae60",   # green
}
SLICE_INDICES = [0, 4, 8]
SLICE_LABELS  = {0: "root (s0)", 4: "mid (s4)", 8: "tip (s8)"}
N_SAMPLES     = 5    # test wings to plot
SAVE_DIR      = "results/version_comparison"


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, cfg: Config, device: str) -> LAE_AoAInit:
    encoder  = build_encoder(cfg)
    decoder  = build_decoder(cfg)
    sampler  = build_sampler(cfg)
    bae      = load_bae(cfg)
    model    = LAE_AoAInit(
        encoder=encoder, decoder=decoder, sampler=sampler, bae_model=bae,
        lae_latent_dim=cfg.lae_latent_dim,
        params_mean_std=(0, 1), aoas_mean_std=(0, 1),
        name="tmp", opt_lr=cfg.lr,
    )
    print(f"  Loading {checkpoint_path}...")
    model.load(checkpoint_path, train_mode=False)

    # Compatibility shims: older checkpoints were saved before several
    # attributes were added.  Patch in zero / no-op modules so forward() works.

    # --- Encoder shims ---
    if not hasattr(model.encoder, 'span_embed'):
        w_dim  = model.encoder.w_dim
        lat_ch = model.encoder.latent_channels
        model.encoder.span_embed = nn.Embedding(w_dim, lat_ch)
        nn.init.zeros_(model.encoder.span_embed.weight)
        print("  Shim: encoder.span_embed (zero)")
    if not hasattr(model.encoder, 'dropout'):
        model.encoder.dropout = nn.Dropout(p=0.0)
        print("  Shim: encoder.dropout (no-op)")

    # --- Decoder shims ---
    dec = model.decoder
    if not hasattr(dec, 'dropout'):
        dec.dropout = nn.Dropout(p=0.0)
        print("  Shim: decoder.dropout (no-op)")

    # Detect v1-style global heads: z_opt_head output == w_dim * lat_ch * lat_len
    z_opt_out = _linear_out_features(dec.z_opt_head)
    expected_global = dec.w_dim * dec.latent_channels * dec.latent_length
    if z_opt_out == expected_global:
        # Old architecture: patch forward to use global (non-per-slice) heads
        dec.forward = types.MethodType(_v1_decoder_forward, dec)
        print("  Shim: decoder.forward patched for v1 global-head architecture")
    else:
        # Current per-slice architecture: just needs span-embedding shims
        if not hasattr(dec, 'span_pos_emb') or not hasattr(dec, 'span_z_proj') or not hasattr(dec, 'span_head'):
            span_latent_dim = 32  # default used in current __init__
            n_blocks  = len(dec.up_blocks)
            final_ch  = dec.up_blocks[-1][0].weight_orig.shape[0]
            flat_size = final_ch * (dec.base_length * (2 ** (n_blocks - 1)))
            zc_dim    = dec.lae_latent_dim + dec.c_dim_latent
            if not hasattr(dec, 'span_pos_emb'):
                dec.span_pos_emb = nn.Embedding(dec.w_dim, span_latent_dim)
                nn.init.zeros_(dec.span_pos_emb.weight)
            if not hasattr(dec, 'span_z_proj'):
                dec.span_z_proj = nn.Linear(zc_dim, span_latent_dim, bias=False)
                nn.init.zeros_(dec.span_z_proj.weight)
            if not hasattr(dec, 'span_head'):
                dec.span_head = nn.Linear(span_latent_dim, flat_size, bias=False)
                nn.init.zeros_(dec.span_head.weight)
            print(f"  Shim: decoder span embedding (zero, flat_size={flat_size})")

    model.to(device)
    return model


# ── Reconstruction helper ─────────────────────────────────────────────────────
def reconstruct(model: LAE_AoAInit, bae_model, z_opts, encoded_inits, params, device):
    """GT z_opts → encoder → decoder → airfoils + Cp.  Returns tensors on CPU."""
    z_init_cat = torch.cat(encoded_inits, dim=0).to(device)   # [B, latent_ch, L]
    with torch.no_grad():
        mu      = model.encode(z_opts.to(device), params.to(device), z_init_cat)
        z       = model._apply_mask(mu)
        z_pred, alpha_pred, eta_y_pred, press_norm, _ = model.decode(z, params.to(device), z_init_cat)

    slices = []
    for s in range(9):
        dec_s = bae_model.decode_z(z_pred[:, s], z_ae_mode=True,
                                   denormalize_output=False, normalized_data=True)[0]
        slices.append(dec_s)
    coords = torch.stack(slices, dim=1).cpu()     # [B, 9, 2, 192]
    coords[:, :, 1, :] += eta_y_pred.cpu()

    if model.scaler_pressures is not None:
        cp = model.scaler_pressures.inverse_transform(press_norm.cpu())
    else:
        cp = press_norm.cpu()
    return coords, cp


# ── Plot one sample ───────────────────────────────────────────────────────────
def plot_sample(sample_idx, gt_coords, gt_cp, recon_data, save_path):
    """
    gt_coords:  [9, 2, 192]
    gt_cp:      [9, 192]
    recon_data: dict version → (coords [9,2,192], cp [9,192])
    """
    n_slices = len(SLICE_INDICES)
    n_versions = len(recon_data)
    fig, axes = plt.subplots(
        2, n_slices,
        figsize=(5 * n_slices, 7),
        squeeze=False,
    )
    fig.suptitle(f"Reconstruction comparison — test sample {sample_idx}", fontsize=12, y=1.01)

    for col, sl in enumerate(SLICE_INDICES):
        label = SLICE_LABELS[sl]
        ax_shape = axes[0, col]
        ax_cp    = axes[1, col]

        gt_xy = gt_coords[sl].numpy()    # [2, 192]
        gt_c  = gt_cp[sl].numpy()        # [192]
        x_pts = gt_xy[0]
        order = np.argsort(x_pts)

        # ── Shape ────────────────────────────────────────────────────
        ax_shape.scatter(gt_xy[0], gt_xy[1], s=2, c=COLORS["GT"], zorder=3, label="GT")
        for name, (r_coords, _) in recon_data.items():
            r_xy = r_coords[sl].numpy()
            mse  = float(((r_xy - gt_xy) ** 2).mean())
            ax_shape.scatter(r_xy[0], r_xy[1], s=2, c=COLORS[name], zorder=4,
                             label=f"{name} (MSE={mse:.2e})")
        ax_shape.set_title(label, fontsize=9)
        ax_shape.set_aspect("equal")
        ax_shape.set_xticks([]); ax_shape.set_yticks([])
        if col == 0:
            ax_shape.set_ylabel("shape", fontsize=9)
            ax_shape.legend(fontsize=7, markerscale=3, loc="upper right")

        # ── Cp ───────────────────────────────────────────────────────
        ax_cp.plot(x_pts[order], gt_c[order], color=COLORS["GT"], lw=1.5, label="GT")
        for name, (_, r_cp) in recon_data.items():
            r_c  = r_cp[sl].numpy()
            mse  = float(((r_c - gt_c) ** 2).mean())
            ax_cp.plot(x_pts[order], r_c[order], color=COLORS[name], lw=1.2,
                       linestyle="--", label=f"{name} (MSE={mse:.4f})")
        ax_cp.invert_yaxis()
        ax_cp.set_xlabel("x/c", fontsize=9)
        if col == 0:
            ax_cp.set_ylabel("Cp", fontsize=9)
            ax_cp.legend(fontsize=7, loc="lower right")
        ax_cp.set_title(f"Cp — {label}", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cfg    = Config()
    device = cfg.device
    print(f"Using device: {device}")

    # Load all models (they share the same BAE backbone)
    models = {}
    for name, ckpt in CHECKPOINTS.items():
        print(f"\nLoading {name}...")
        models[name] = load_model(ckpt, cfg, device)
    bae_model = list(models.values())[0].bae_model

    # Load test data
    print("\nLoading test dataset...")
    problem    = Wings3D(seed=cfg.seed)
    test_all   = list(problem.dataset["test"])
    test_items = [item for item in test_all if item["final"] == 1][:N_SAMPLES]

    for sample_idx, item in enumerate(test_items):
        print(f"\nProcessing test sample {sample_idx}...")

        coords    = torch.tensor(item["coords"],    dtype=torch.float32)   # [9, 192, 2]
        te_shifts = torch.tensor(item["te_shifts"], dtype=torch.float32)   # [9]

        coords_unshifted = coords.clone()
        coords_unshifted[:, :, 1] -= te_shifts.unsqueeze(1)

        x_opt_slices, z_opt_slices = [], []
        for s in range(9):
            x_s = coords_unshifted[s].permute(1, 0).unsqueeze(0).to(device)
            with torch.no_grad():
                z_s   = bae_model.encode(x_s, return_z=True, z_ae_mode=True)
                dec_s = bae_model.decode_z(z_s, z_ae_mode=True,
                                           denormalize_output=False, normalized_data=True)[0]
            x_opt_slices.append(dec_s.squeeze(0).cpu())
            z_opt_slices.append(z_s.squeeze(0).cpu())

        gt_wing = torch.stack(x_opt_slices)              # [9, 2, 192]
        gt_wing[:, 1, :] += te_shifts.unsqueeze(1)
        z_opts  = torch.stack(z_opt_slices).unsqueeze(0) # [1, 9, latent_ch, L]
        gt_cp   = torch.tensor(np.array(item["coef_pressure"]), dtype=torch.float32)  # [9, 192]

        # Params (use first available model's scaler to normalize)
        first_model = list(models.values())[0]
        params_raw = torch.tensor(
            [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)
        params_scaled = first_model.scaler_params.transform(params_raw)

        # Encoded x0 init
        init_by_case = {it["case_num"]: it for it in test_all if it["initial"] == 1}
        case_num = int(item["case_num"])
        if case_num in init_by_case:
            init_coords = torch.tensor(init_by_case[case_num]["coords"], dtype=torch.float32)
            x_init = init_coords[0].permute(1, 0).unsqueeze(0).to(device)
        else:
            x_init = coords_unshifted[0].permute(1, 0).unsqueeze(0).to(device)
        with torch.no_grad():
            z_init = bae_model.encode(x_init, return_z=True, z_ae_mode=True)

        # Run reconstruction through each model
        recon_data = {}
        for name, model in models.items():
            # Each model has its own scaler, re-scale params per model
            p_raw = torch.tensor(
                [item["mach"], item["reynolds"], item["cl_target"], item["area_case_ratio"]],
                dtype=torch.float32,
            ).unsqueeze(0).to(device)
            p_scaled = model.scaler_params.transform(p_raw)

            r_coords, r_cp = reconstruct(
                model, bae_model, z_opts, [z_init], p_scaled, device
            )
            recon_data[name] = (r_coords[0], r_cp[0])  # squeeze batch dim → [9, 2, 192], [9, 192]

        save_path = os.path.join(SAVE_DIR, f"recon_comparison_sample{sample_idx:02d}.pdf")
        plot_sample(sample_idx, gt_wing, gt_cp, recon_data, save_path)

    print(f"\nDone. Plots saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
