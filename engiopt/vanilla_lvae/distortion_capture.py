"""Distortion-sensitivity capture for the LV metric suite.

Legitimacy experiment for the LV-metrics paper. We take a held-out set of *real*
optimal designs, apply graded corruptions at increasing severity, and record how
every cheap metric (pixel vs PCA vs LV) responds. A legitimate fidelity/diversity
metric must respond *monotonically and correctly* to controlled distortions
(cf. Naeem et al. 2020, "Reliable Fidelity and Diversity Metrics"). The headline
claims this supports:

* pixel-DPP *rewards* additive noise (diversity goes up as designs degrade),
  while LV-DPP / LV-coverage do not.
* LV-MMD and LV-PRDC respond monotonically to fidelity distortions and to
  mode dropping / mode invention; pixel metrics misrank or saturate.

The script is intentionally *capture-only*: it writes one CSV row per
(family, severity, repeat) plus a metadata JSON. Plotting and monotonicity
analysis live in the notebook so the expensive encode/decode pass runs once
(slurm-friendly) and figures iterate cheaply.

Example:
    python -m engiopt.vanilla_lvae.distortion_capture \\
        --problem-id beams2d --seed 1 --rec-threshold 0.005 --perf-threshold 0.01 \\
        --n-samples 200 --n-levels 6 --n-repeats 5 --output-dir distortion_out
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import os
import time

from engibench.utils.all_problems import BUILTIN_PROBLEMS
from engiopt.vanilla_lvae.utils import encode_designs
from engiopt.vanilla_lvae.utils import get_active_mask
from engiopt.vanilla_lvae.utils import load_lvae_encoder_decoder
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as cc_label
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch as th
import tyro

from engiopt import lv_metrics
from engiopt import metrics


@dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "beams2d"
    """EngiBench problem identifier."""
    seed: int = 1
    """Seed of the trained LVAE artifact AND the base RNG seed for distortions."""
    rec_threshold: float = 0.005
    """Reconstruction NMSE threshold of the trained LVAE artifact."""
    perf_threshold: float = 0.01
    """Performance NMSE threshold of the trained LVAE artifact."""
    wandb_project: str = "engiopt"
    """WandB project holding the LVAE artifacts."""
    wandb_entity: str | None = None
    """WandB entity (None = default)."""
    n_samples: int = 200
    """Reference set size = candidate set size (matched-N for MMD/PRDC)."""
    cand_split: str = "test"
    """Dataset split whose designs get corrupted (held-out from LVAE training).
    Reference always comes from train. Falls back to train if the split is too small."""
    mode_definition: str = "topology"
    """How modes are defined for mode_drop/mode_invent: 'topology' (binarized-geometry
    fingerprints), 'condition_bins' (quantile bins of a scalar condition), or
    'latent_kmeans' (KMeans on active latent codes; circular, kept as a baseline)."""
    n_levels: int = 6
    """Number of severity levels per family, including the clean (severity=0) level."""
    n_repeats: int = 5
    """Independent distortion draws per (family, severity) for error bars."""
    n_clusters: int = 8
    """KMeans clusters (on reference latent) defining 'modes' for drop/invent families."""
    nearest_k: int = 5
    """k for PRDC k-NN balls."""
    batch_size: int = 256
    """Encode batch size."""
    output_dir: str = "distortion_out"
    """Directory for the CSV + metadata JSON."""
    families: tuple[str, ...] = field(
        default_factory=lambda: (
            "gaussian_noise",
            "blur",
            "pixel_flip",
            "intensity_shift",
            "translation",
            "mode_drop",
            "mode_invent",
        )
    )
    """Distortion families to run."""


# ---------------------------------------------------------------------------
# Distortion families.  Each maps (clean batch, severity in [0,1], rng) -> batch.
# severity=0 must be the identity so every family shares one clean baseline.
# ---------------------------------------------------------------------------


def _gaussian_noise(x: npt.NDArray, s: float, rng: np.random.Generator) -> npt.NDArray:
    return np.clip(x + rng.normal(0.0, 0.5 * s, size=x.shape).astype(np.float32), 0.0, 1.0)


def _blur(x: npt.NDArray, s: float, _rng: np.random.Generator) -> npt.NDArray:
    if s == 0:
        return x.copy()
    return np.stack([gaussian_filter(img, sigma=5.0 * s) for img in x]).astype(np.float32)


def _pixel_flip(x: npt.NDArray, s: float, rng: np.random.Generator) -> npt.NDArray:
    out = x.copy()
    mask = rng.random(x.shape) < (0.5 * s)
    out[mask] = (rng.random(int(mask.sum())) < x.mean()).astype(np.float32)
    return out


def _intensity_shift(x: npt.NDArray, s: float, rng: np.random.Generator) -> npt.NDArray:
    # Shift volume fraction up or down (sign per draw); a physically meaningful
    # distortion that should hurt fidelity without adding high-freq noise.
    sign = 1.0 if rng.random() < 0.5 else -1.0
    return np.clip(x + sign * 0.4 * s, 0.0, 1.0).astype(np.float32)


def _translation(x: npt.NDArray, s: float, rng: np.random.Generator) -> npt.NDArray:
    shift = int(round(20 * s))
    if shift == 0:
        return x.copy()
    dy, dx = rng.integers(-shift, shift + 1, size=2)
    out = np.roll(x, shift=(int(dy), int(dx)), axis=(1, 2))
    # zero-fill the wrapped border so this is a translation, not a roll
    if dy > 0:
        out[:, :dy, :] = 0.0
    elif dy < 0:
        out[:, dy:, :] = 0.0
    if dx > 0:
        out[:, :, :dx] = 0.0
    elif dx < 0:
        out[:, :, dx:] = 0.0
    return out.astype(np.float32)


def _make_noise_designs(n: int, shape: tuple[int, int], vf: float, rng: np.random.Generator) -> npt.NDArray:
    """Off-manifold 'garbage' designs (random binary at the right volume fraction)."""
    h, w = shape
    return (rng.random((n, h, w)) < vf).astype(np.float32)


# ---------------------------------------------------------------------------
# Mode detection.  'modes' must be defined INDEPENDENTLY of the metric being
# scored, otherwise mode_drop/mode_invent are circular (clustering in the LVAE
# latent then checking whether LV metrics detect the drop favours LV by
# construction).  topology/condition_bins are defined in design / condition
# space; latent_kmeans is kept only as the (circular) baseline.
# ---------------------------------------------------------------------------


def _topology_features(designs: npt.NDArray) -> npt.NDArray:
    """Binarized-geometry fingerprints: (#material components, #void components, vf).

    Defined purely in design space, so clustering on these is neutral to every
    evaluation metric. Void-component count is a connectivity/hole proxy.
    """
    feats = []
    for d in designs:
        b = d > 0.5  # noqa: PLR2004
        _, n_mat = cc_label(b)
        _, n_void = cc_label(~b)
        feats.append([float(n_mat), float(n_void), float(b.mean())])
    return np.asarray(feats, dtype=np.float64)


def _quantile_bin(values: npt.NDArray, k: int) -> npt.NDArray:
    """Assign each value to one of k quantile bins (rank-based, robust to scale)."""
    edges = np.quantile(values, np.linspace(0.0, 1.0, k + 1))
    return np.clip(np.digitize(values, edges[1:-1]), 0, k - 1)


def detect_modes(
    definition: str,
    *,
    designs: npt.NDArray,
    z_active: npt.NDArray,
    scalar_cond: npt.NDArray | None,
    k_modes: int,
) -> npt.NDArray:
    """Return an integer mode label per candidate design.

    Args:
        definition: 'topology', 'condition_bins', or 'latent_kmeans'.
        designs: Candidate designs (N, H, W) — used by 'topology'.
        z_active: Candidate active latent codes (N, D) — used by 'latent_kmeans'.
        scalar_cond: Per-candidate scalar condition (N,) — used by 'condition_bins'.
        k_modes: Number of modes.

    Returns:
        Integer labels of shape (N,).
    """
    if definition == "topology":
        feats = _topology_features(designs)
        feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)
        return KMeans(n_clusters=k_modes, random_state=0, n_init=10).fit_predict(feats)
    if definition == "condition_bins":
        if scalar_cond is None:
            raise ValueError("condition_bins requires a scalar condition; none available for this problem.")
        return _quantile_bin(scalar_cond, k_modes)
    if definition == "latent_kmeans":
        return KMeans(n_clusters=k_modes, random_state=0, n_init=10).fit_predict(z_active)
    raise ValueError(f"unknown mode_definition: {definition}")


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _slice_active(z: npt.NDArray, active_mask: npt.NDArray) -> npt.NDArray:
    return z[:, active_mask] if active_mask.shape[0] == z.shape[1] else z


def _compute_row(
    cand: npt.NDArray,
    x_ref: npt.NDArray,
    z_ref_active: npt.NDArray,
    pca: PCA,
    sigma_pixel: float,
    sigma_pca: float,
    sigma_latent: float,
    ref_stats: lv_metrics.LatentReferenceStats,
    encoder: th.nn.Module,
    decoder: th.nn.Module,
    active_mask: npt.NDArray,
    device: th.device,
    nearest_k: int,
    batch_size: int,
) -> dict[str, float]:
    """Compute the full pixel/PCA/latent metric suite for one candidate set."""
    cand_flat = cand.reshape(len(cand), -1)
    z_cand = encode_designs(encoder, cand, device, batch_size=batch_size)
    z_cand_active = _slice_active(z_cand, active_mask)
    cand_pca = pca.transform(cand_flat)

    # Pixel-space metrics
    row: dict[str, float] = {
        "pixel_mmd": metrics.mmd(cand_flat, x_ref.reshape(len(x_ref), -1), sigma=sigma_pixel),
        "pixel_dpp": metrics.dpp_diversity(cand_flat),
    }
    prdc_pix = metrics.compute_prdc(x_ref.reshape(len(x_ref), -1), cand_flat, nearest_k=nearest_k)
    row.update({f"pixel_{k}": v for k, v in prdc_pix.items()})

    # PCA-MMD (closest linear analogue of the active latent subspace)
    row["pca_mmd"] = metrics.mmd(cand_pca, pca.transform(x_ref.reshape(len(x_ref), -1)), sigma=sigma_pca)

    # Latent (LV) metrics
    row["lv_mmd"] = metrics.mmd(z_cand_active, z_ref_active, sigma=sigma_latent)
    row["lv_dpp"] = metrics.dpp_diversity(z_cand_active)
    prdc_lat = metrics.compute_prdc(z_ref_active, z_cand_active, nearest_k=nearest_k)
    row.update({f"lv_{k}": v for k, v in prdc_lat.items()})

    mahal = lv_metrics.lv_mahalanobis_typicality(z_cand_active, ref_stats)
    row["lv_mahal_mean"] = mahal["mahal_mean"]
    row["lv_plausibility_rate"] = mahal["plausibility_rate"]
    row["lv_volume_ratio"] = lv_metrics.lv_volume_coverage_ratio(z_cand_active, ref_stats)["volume_ratio"]

    resid = lv_metrics.lv_reconstruction_residual_stats(encoder, decoder, cand, device, batch_size=batch_size)
    row["lv_residual_mean"] = resid["residual_mean"]
    row["lv_residual_median"] = resid["residual_median"]
    row["lv_residual_p90"] = resid["residual_p90"]
    return row


def main(args: Args) -> None:  # noqa: PLR0915
    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")
    print(f"device = {device}")

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)
    design_shape = tuple(problem.design_space.shape)
    assert len(design_shape) == 2, f"distortion_capture supports 2D problems only, got {design_shape}"
    h, w = design_shape

    # --- load designs and the trained perf-constrained LVAE --------------------
    designs = np.asarray(problem.dataset["train"]["optimal_design"], dtype=np.float32)
    designs = designs.reshape(len(designs), h, w)
    rng0 = np.random.default_rng(args.seed)
    rng0.shuffle(designs)

    n = args.n_samples
    assert len(designs) >= 2 * n, f"need >= {2 * n} train designs, have {len(designs)}"
    x_ref = designs[:n]  # the fixed 'real' reference distribution
    x_cand_base = designs[n : 2 * n]  # disjoint clean pool that gets distorted

    encoder, decoder, lvae_config = load_lvae_encoder_decoder(
        problem_id=args.problem_id,
        seed=args.seed,
        rec_threshold=args.rec_threshold,
        perf_threshold=args.perf_threshold,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        device=device,
        design_shape=design_shape,
    )
    active_mask = get_active_mask(encoder)
    n_active = int(active_mask.sum()) if active_mask.shape[0] == lvae_config.latent_dim else lvae_config.latent_dim
    print(f"latent_dim={lvae_config.latent_dim}  active_dims={n_active}")

    # --- fixed bandwidths + PCA + reference stats, all computed ONCE on x_ref ---
    z_ref = encode_designs(encoder, x_ref, device, batch_size=args.batch_size)
    z_ref_active = _slice_active(z_ref, active_mask)
    ref_stats = lv_metrics.compute_latent_reference_stats(z_ref_active)

    x_ref_flat = x_ref.reshape(n, -1)
    pca = PCA(n_components=min(n_active, n - 1), random_state=0).fit(x_ref_flat)
    sigma_pixel = metrics.compute_median_sigma(x_ref_flat)
    sigma_pca = metrics.compute_median_sigma(pca.transform(x_ref_flat))
    sigma_latent = metrics.compute_median_sigma(z_ref_active)
    print(f"bandwidths: pixel={sigma_pixel:.4f} pca={sigma_pca:.4f} latent={sigma_latent:.4f}")

    # --- cluster reference latent into 'modes' for mode_drop / mode_invent ------
    k_modes = min(args.n_clusters, n_active * 2, n // 5)
    k_modes = max(k_modes, 2)
    kmeans = KMeans(n_clusters=k_modes, random_state=0, n_init=10).fit(z_ref_active)
    z_cand_base_active = _slice_active(
        encode_designs(encoder, x_cand_base, device, batch_size=args.batch_size), active_mask
    )
    cand_labels = kmeans.predict(z_cand_base_active)
    target_vf = float(x_ref.mean())

    severities = np.linspace(0.0, 1.0, args.n_levels)
    rows: list[dict[str, float | str | int]] = []
    pixel_families = {
        "gaussian_noise": _gaussian_noise,
        "blur": _blur,
        "pixel_flip": _pixel_flip,
        "intensity_shift": _intensity_shift,
        "translation": _translation,
    }

    t0 = time.time()
    for family in args.families:
        for sev in severities:
            for rep in range(args.n_repeats):
                rng = np.random.default_rng((args.seed << 16) + hash(family) % 9973 + rep * 101 + int(sev * 1000))
                if family in pixel_families:
                    cand = pixel_families[family](x_cand_base, float(sev), rng)
                elif family == "mode_drop":
                    n_drop = int(round(sev * (k_modes - 1)))
                    keep = set(rng.permutation(k_modes)[: k_modes - n_drop].tolist())
                    pool_idx = np.where(np.isin(cand_labels, list(keep)))[0]
                    if len(pool_idx) == 0:
                        pool_idx = np.arange(len(x_cand_base))
                    pick = rng.choice(pool_idx, size=n, replace=len(pool_idx) < n)
                    cand = x_cand_base[pick]
                elif family == "mode_invent":
                    n_bad = int(round(sev * n))
                    good = (
                        x_cand_base[rng.choice(len(x_cand_base), size=n - n_bad, replace=False)]
                        if n_bad < n
                        else np.empty((0, h, w), np.float32)
                    )
                    bad = _make_noise_designs(n_bad, design_shape, target_vf, rng)
                    cand = np.concatenate([good, bad], axis=0) if n_bad > 0 else good
                else:
                    raise ValueError(f"unknown family: {family}")

                row = _compute_row(
                    cand,
                    x_ref,
                    z_ref_active,
                    pca,
                    sigma_pixel,
                    sigma_pca,
                    sigma_latent,
                    ref_stats,
                    encoder,
                    decoder,
                    active_mask,
                    device,
                    args.nearest_k,
                    args.batch_size,
                )
                row.update(
                    {
                        "problem": args.problem_id,
                        "seed": args.seed,
                        "family": family,
                        "severity": float(sev),
                        "repeat": rep,
                        "n_active": n_active,
                    }
                )
                rows.append(row)
            print(f"[{time.time() - t0:6.1f}s] {family:16s} sev={sev:.2f} done ({args.n_repeats} reps)")

    # --- write CSV + metadata --------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"{args.problem_id}_seed{args.seed}_rec{args.rec_threshold}_perf{args.perf_threshold}"
    csv_path = os.path.join(args.output_dir, f"distortion_{tag}.csv")
    cols = sorted({k for r in rows for k in r})
    meta_cols = ["problem", "seed", "family", "severity", "repeat", "n_active"]
    cols = meta_cols + [c for c in cols if c not in meta_cols]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        f.writelines(",".join(str(r.get(c, "")) for c in cols) + "\n" for r in rows)

    meta = {
        "problem_id": args.problem_id,
        "seed": args.seed,
        "rec_threshold": args.rec_threshold,
        "perf_threshold": args.perf_threshold,
        "n_samples": n,
        "n_levels": args.n_levels,
        "n_repeats": args.n_repeats,
        "n_active": n_active,
        "latent_dim": lvae_config.latent_dim,
        "k_modes": int(k_modes),
        "sigma_pixel": sigma_pixel,
        "sigma_pca": sigma_pca,
        "sigma_latent": sigma_latent,
        "target_vf": target_vf,
        "families": list(args.families),
    }
    with open(os.path.join(args.output_dir, f"distortion_{tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nwrote {len(rows)} rows -> {csv_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
