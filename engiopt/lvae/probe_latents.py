"""
Linear probe: is v1's decoder trunk (h_flat) or the head itself the bottleneck for cd/cl?

Two-stage probe
---------------
Stage 1 — probe z (encoder output):
  Does the 5-dim latent z contain enough information?
  If GBR on z ≈ v1 head MSE → z is the ceiling, not the head.

Stage 2 — probe h_flat (decoder trunk output, 512-dim):
  Does h_flat (the tensor all heads read from) contain enough information?
  If GBR on h_flat << v1 head MSE → v1's perf_head is the bottleneck; deepening it helps.
  If GBR on h_flat ≈ v1 head MSE  → trunk is the ceiling; head depth can't rescue you.

h_flat is extracted via a forward hook on the decoder — no changes to model code.

Usage
-----
    python -m engiopt.lvae.probe_latents --checkpoint results/lvae/lae_aoa_init_3d_v1.pth
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from engibench.problems.wings3D.v0 import Wings3D
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

from engiopt.lvae.lvae import LAE_AoAInit
from engiopt.lvae.train_lvae import (
    Config, build_encoder, build_decoder, build_sampler, load_bae,
    precompute_latents, PrecomputedWingsDataset,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def extract_latents(lae_model, loader, device):
    """Encode a DataLoader; return z_masked, h_flat, perfs_raw, pressures_raw.

    h_flat is the 512-dim decoder trunk output captured via a forward hook —
    the exact tensor that all decoder heads (perf, pressure, z_opt) read from.
    """
    lae_model.encoder.eval()
    lae_model.decoder.eval()

    # Hook the last up_block to capture its output (= h before reshape → h_flat).
    # h_flat = h.reshape(B, -1), so we just flatten the captured tensor outside.
    _buf = []
    hook = lae_model.decoder.up_blocks[-1].register_forward_hook(
        lambda m, inp, out: _buf.append(out.detach().cpu())
    )

    zs, h_flats, perfs_raw, pressures_raw = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            z_opt, aoa, params, eta_y, pressure, perf = batch
            z = lae_model.encoder(
                z_opt.to(device),
                params.to(device).float(),
            )
            z_masked = lae_model._apply_mask(z)
            lae_model.decoder(z_masked, params.to(device).float())
            zs.append(z_masked.cpu())
            h = _buf.pop()                        # [B, up_ch[-1], length]
            h_flats.append(h.reshape(h.shape[0], -1))   # [B, flat_size]

            # De-normalise so MSE is in raw space (matches evaluate_lvae.py)
            p_raw = (lae_model.scaler_perfs.inverse_transform(perf)
                     if lae_model.scaler_perfs is not None else perf)
            cp_raw = (lae_model.scaler_pressures.inverse_transform(pressure)
                      if lae_model.scaler_pressures is not None else pressure)
            perfs_raw.append(p_raw)
            pressures_raw.append(cp_raw)

    hook.remove()

    Z    = torch.cat(zs,            dim=0).numpy()   # [N, lae_latent_dim]
    H    = torch.cat(h_flats,       dim=0).numpy()   # [N, flat_size]
    P    = torch.cat(perfs_raw,     dim=0).numpy()   # [N, 2]
    Cp   = torch.cat(pressures_raw, dim=0).numpy()   # [N, 9, 192]
    return Z, H, P, Cp


def mse(pred, gt):
    return float(np.mean((pred - gt) ** 2))


def probe_target(Z_train, y_train, Z_test, y_test, label):
    """Fit Ridge + GBR on (Z_train → y_train) and report test MSE."""
    # Ridge (linear — same expressivity as v1's linear head)
    ridge = Ridge(alpha=1.0)
    ridge.fit(Z_train, y_train)
    pred_ridge_test  = ridge.predict(Z_test)
    pred_ridge_train = ridge.predict(Z_train)
    mse_ridge_test  = mse(pred_ridge_test,  y_test)
    mse_ridge_train = mse(pred_ridge_train, y_train)

    # GradientBoosting (strong nonlinear — true capacity of z)
    if y_train.ndim == 1:
        gbr = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
        gbr.fit(Z_train, y_train)
        pred_gbr_test  = gbr.predict(Z_test)
        pred_gbr_train = gbr.predict(Z_train)
    else:
        gbr = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0),
            n_jobs=-1,
        )
        gbr.fit(Z_train, y_train)
        pred_gbr_test  = gbr.predict(Z_test)
        pred_gbr_train = gbr.predict(Z_train)

    mse_gbr_test  = mse(pred_gbr_test,  y_test)
    mse_gbr_train = mse(pred_gbr_train, y_train)

    print(f"\n  {label}")
    print(f"    Ridge  — train MSE: {mse_ridge_train:.6f}  test MSE: {mse_ridge_test:.6f}")
    print(f"    GBR    — train MSE: {mse_gbr_train:.6f}  test MSE: {mse_gbr_test:.6f}")
    return mse_ridge_test, mse_gbr_test


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/lvae/lae_aoa_init_3d_v1.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = Config()
    device = cfg.device
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # ── 1. Build model and load checkpoint ────────────────────────────────────
    bae_model = load_bae(cfg)
    encoder   = build_encoder(cfg)
    decoder   = build_decoder(cfg)
    sampler   = build_sampler(cfg)

    lae_model = LAE_AoAInit(
        encoder=encoder, decoder=decoder, sampler=sampler, bae_model=bae_model,
        lae_latent_dim=cfg.lae_latent_dim,
        params_mean_std=(0, 1), aoas_mean_std=(0, 1),
        name=cfg.model_name, opt_lr=cfg.lr,
    )
    lae_model.load(args.checkpoint, train_mode=False)
    lae_model.to(device)

    n_active = int(lae_model.active_latent_mask.sum().item())
    print(f"Active latent dims: {n_active}/{lae_model.lae_latent_dim}")

    # ── 2. Build datasets ─────────────────────────────────────────────────────
    problem = Wings3D(seed=cfg.seed)

    all_train = list(problem.dataset["train"])
    train_items = [item for item in all_train if item["final"] == 1]

    all_test  = list(problem.dataset["test"])
    test_items  = [item for item in all_test  if item["final"] == 1]

    print(f"Train: {len(train_items)} | Test: {len(test_items)}")

    print("Encoding train set...")
    z_opts_tr, aoas_tr, params_tr, eta_ys_tr, pressures_tr, perfs_tr = \
        precompute_latents(train_items, lae_model.bae_model, device)

    print("Encoding test set...")
    z_opts_te, aoas_te, params_te, eta_ys_te, pressures_te, perfs_te = \
        precompute_latents(test_items, lae_model.bae_model, device)

    def make_loader(z_opts, aoas, params, eta_ys, pressures, perfs):
        ds = PrecomputedWingsDataset(
            z_opts, aoas, params, eta_ys, pressures, perfs,
            scaler_params=lae_model.scaler_params,
            scaler_aoas=lae_model.scaler_aoas,
            scaler_pressures=lae_model.scaler_pressures,
            scaler_perfs=lae_model.scaler_perfs,
        )
        return DataLoader(ds, batch_size=256, shuffle=False)

    loader_train = make_loader(z_opts_tr, aoas_tr, params_tr, eta_ys_tr, pressures_tr, perfs_tr)
    loader_test  = make_loader(z_opts_te, aoas_te, params_te, eta_ys_te, pressures_te, perfs_te)

    # ── 3. Extract latents + h_flat ───────────────────────────────────────────
    print("Extracting train latents...")
    Z_tr, H_tr, P_tr, Cp_tr = extract_latents(lae_model, loader_train, device)
    print("Extracting test latents...")
    Z_te, H_te, P_te, Cp_te = extract_latents(lae_model, loader_test,  device)

    mask = lae_model.active_latent_mask.cpu().numpy()
    Z_tr_active = Z_tr[:, mask]
    Z_te_active = Z_te[:, mask]

    print(f"\nz (active): {Z_tr_active.shape}  h_flat: {H_tr.shape}")

    # ── 4. Probe cd / cl from z (encoder output) ──────────────────────────────
    print("\n" + "=" * 65)
    print("STAGE 1 — probe z (5 active dims, encoder output)")
    print("=" * 65)
    print("Tells us: is the ceiling set by z, or by the decoder heads?")

    probe_target(Z_tr_active, P_tr[:, 0], Z_te_active, P_te[:, 0], "cd  from z")
    probe_target(Z_tr_active, P_tr[:, 1], Z_te_active, P_te[:, 1], "cl  from z")

    # ── 5. Probe cd / cl from h_flat (decoder trunk output) ───────────────────
    print("\n" + "=" * 65)
    print("STAGE 2 — probe h_flat (decoder trunk output, before heads)")
    print("=" * 65)
    print("Key comparison: GBR-on-h_flat vs v1 perf_head test MSE (0.417 for cd).")
    print("If GBR-on-h_flat << 0.417 → perf_head is the bottleneck.")
    print("If GBR-on-h_flat ≈ 0.417  → trunk is the ceiling; deeper head won't help.")

    probe_target(H_tr, P_tr[:, 0], H_te, P_te[:, 0], "cd  from h_flat")
    probe_target(H_tr, P_tr[:, 1], H_te, P_te[:, 1], "cl  from h_flat")

    # ── 6. Pressure from h_flat (Ridge — GBR on 1728 outputs not practical) ───
    Cp_tr_flat = Cp_tr.reshape(len(Cp_tr), -1)
    Cp_te_flat = Cp_te.reshape(len(Cp_te), -1)
    print("\n  pressure (9×192 flattened, Ridge only):")
    for features, label in [(Z_tr_active, "z      "), (H_tr, "h_flat ")]:
        feat_te = Z_te_active if label.strip() == "z" else H_te
        r = Ridge(alpha=1.0)
        r.fit(features, Cp_tr_flat)
        pred = r.predict(feat_te)
        print(f"    {label} — test MSE: {mse(pred, Cp_te_flat):.6f}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
