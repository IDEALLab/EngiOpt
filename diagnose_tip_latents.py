"""
Diagnostic: BAE latent ch0 distribution at tip slices vs twist magnitude.

Each dataset item is one case (one wing), with coords [15, 192, 2] covering
all 15 span stations from root to tip.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/cluster/home/adelbeke/EngiOpt")

from engiopt.bezier_ae.bezier_ae import BezierAutoencoder
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

BAE_CKPT = "bezier_ae_best.pt"
DATA_DIR  = "Wing_TL/data/processed"
N_CTRL    = 32
N_DATA    = 192
DEVICE    = "cpu"
CH0_LO, CH0_HI = 0.05, 2.10

# ── load BAE ───────────────────────────────────────────────────────────────
print("Loading BAE...")
bae = BezierAutoencoder(n_control_points=N_CTRL, n_data_points=N_DATA,
                        batch_size=64, auto_batch=True).to(DEVICE)
ckpt = torch.load(BAE_CKPT, map_location=DEVICE, weights_only=False)
bae.load_state_dict(ckpt["model_state_dict"])
bae.eval()
for p in bae.parameters():
    p.requires_grad_(False)
print("BAE loaded.")

# ── load dataset (final iterations only, all splits) ──────────────────────
print("Loading dataset...")
ds = NewWingsDataset(
    slices_pkl=f"{DATA_DIR}/new_dataset_slices.pkl",
    scalars_pkl=f"{DATA_DIR}/new_dataset_scalars.pkl",
)
# Keep only final-optimised cases (training targets), deduplicate by case_num.
seen = set()
finals = []
for split in ["train", "val", "test"]:
    for item in ds[split]:
        if item["final"] == 1 and item["case_num"] not in seen:
            seen.add(item["case_num"])
            finals.append(item)

print(f"  {len(finals)} unique final-iteration cases")

# ── encode all cases ───────────────────────────────────────────────────────
# coords: [15, 192, 2] → encode all 15 slices at once → z: [15, 3, L]
print("Encoding BAE latents...")
all_ch0   = []   # [n_cases, 15, L]
all_twist = []   # tip twist value (scalar) per case  [index 18 in geo_params = twist_value6]

with torch.no_grad():
    for i, item in enumerate(finals):
        coords = torch.tensor(item["coords"], dtype=torch.float32)  # [15, 192, 2]
        z = bae.encode(coords, z_ae_mode=True)                       # [15, 3, L]
        all_ch0.append(z[:, 0, :].cpu().numpy())                     # [15, L]
        # geo_params layout: [sweep, base_span, span_taper, thick_taper,
        #   chord_0..7 (8), twist_0..6 (7), thick_taper_0..7 (8), dihedral_0..6 (7)]
        # twist_value6 is at index 4+8+6 = 18
        all_twist.append(float(item["geo_params"][18]))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(finals)}")

all_ch0   = np.array(all_ch0)    # [N, 15, L]
all_twist = np.array(all_twist)  # [N]
N, n_span, L = all_ch0.shape
print(f"Encoded: {N} cases × {n_span} span stations × {L} latent dims\n")

# ── verify twist index is correct ─────────────────────────────────────────
# Quick sanity check: tip twist should span roughly [-10, +20]
print(f"Twist range check: min={all_twist.min():.2f}  max={all_twist.max():.2f}  "
      f"(expected ~[-10, +20])")

# ── Analysis 1: ch0 per span station ──────────────────────────────────────
print("\n" + "="*68)
print("ch0 PER SPAN STATION  (station 0=root → 14=tip)")
print(f"{'Stn':>4}  {'min':>6}  {'p5':>6}  {'p50':>6}  {'p95':>6}  {'max':>6}  "
      f"{'%>1.8':>6}  {'%>1.9':>6}  {'%>2.0':>6}")
print("-"*68)
for s in range(n_span):
    vals = all_ch0[:, s, :].flatten()
    p5, p50, p95 = np.percentile(vals, [5, 50, 95])
    pct = lambda t: 100.0 * (vals > t).sum() / len(vals)
    print(f"  {s:2d}   {vals.min():+6.3f}  {p5:+6.3f}  {p50:+6.3f}  {p95:+6.3f}  {vals.max():+6.3f}"
          f"  {pct(1.8):6.2f}  {pct(1.9):6.2f}  {pct(2.0):6.2f}")

# ── Analysis 2: tip stations detail ───────────────────────────────────────
print("\n" + "="*68)
print("TIP STATIONS (12, 13, 14) — absolute counts above cliff")
print("="*68)
for s in [12, 13, 14]:
    vals = all_ch0[:, s, :].flatten()
    total = len(vals)
    print(f"\n  Station {s}  ({total} values = {N} cases × {L} ctrl pts)")
    for thresh in [1.7, 1.8, 1.9, 2.0, CH0_HI]:
        n_above = (vals > thresh).sum()
        print(f"    ch0 > {thresh:.2f}: {n_above:6d} / {total}  = {100*n_above/total:5.2f}%")
    # How many *cases* have at least one ch0 > 1.9 at this station?
    per_case_max = all_ch0[:, s, :].max(axis=1)
    n_cases_above = (per_case_max > 1.9).sum()
    print(f"    cases with max(ch0) > 1.9: {n_cases_above}/{N} = {100*n_cases_above/N:.1f}%")

# ── Analysis 3: correlation with twist magnitude ───────────────────────────
print("\n" + "="*68)
print("CORRELATION: |twist_tip| vs tip-station ch0")
print("="*68)
abs_twist    = np.abs(all_twist)
tip_ch0_max  = all_ch0[:, 14, :].max(axis=1)
tip_ch0_mean = all_ch0[:, 14, :].mean(axis=1)
corr_max  = np.corrcoef(abs_twist, tip_ch0_max)[0, 1]
corr_mean = np.corrcoef(abs_twist, tip_ch0_mean)[0, 1]
print(f"  corr(|twist_tip|, tip ch0 MAX):  {corr_max:+.4f}")
print(f"  corr(|twist_tip|, tip ch0 MEAN): {corr_mean:+.4f}")

print(f"\n  |twist| bin       n_cases  mean(tip ch0 max)  frac(max ch0 > 1.9)")
bins = np.linspace(0, abs_twist.max(), 7)
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (abs_twist >= lo) & (abs_twist < hi)
    if mask.sum() == 0:
        continue
    ch0_max_bin = tip_ch0_max[mask]
    frac = (ch0_max_bin > 1.9).mean()
    print(f"    [{lo:4.1f}°, {hi:4.1f}°)   {mask.sum():5d}     {ch0_max_bin.mean():.4f}            {frac:.3f}")

# ── Analysis 4: sign of twist ─────────────────────────────────────────────
print("\n" + "="*68)
print("SIGN OF TWIST: does direction matter?")
print("="*68)
for sign, label in [(+1, "positive twist (tip pitched up, wash-out)"),
                    (-1, "negative twist (tip pitched down, wash-in)")]:
    mask = (all_twist * sign) > 3.0
    if mask.sum() == 0:
        continue
    ch0_tip = all_ch0[mask, 14, :].flatten()
    frac19  = (ch0_tip > 1.9).mean()
    print(f"  {label}")
    print(f"    n_cases={mask.sum()},  frac(ch0>1.9)={frac19:.3f},  max ch0={ch0_tip.max():.4f}")

print("\nDone.")
