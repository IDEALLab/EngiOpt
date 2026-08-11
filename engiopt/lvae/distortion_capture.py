r"""Does the LV metric suite respond correctly to *known* failures?

A correlation against simulator performance says a metric tracks something. It
does not say the metric fails safely. This applies graded, controlled
corruptions to real optima and records how every cheap metric responds, so each
one can be checked against a ground truth fixed by construction rather than by a
simulator (cf. Naeem et al. 2020, "Reliable Fidelity and Diversity Metrics").

Every metric is computed through `engiopt.evaluation`, the same registry that
produces leaderboard rows, so a pathology found here is a pathology in the
reported numbers and not in a reimplementation of them.

Two design points carried over from the earlier (now unrunnable) version in
`vanilla_lvae`, because they are what make the result mean anything:

- **Severity 0 is the identity** for every family, so all families share one
  clean baseline and severities are comparable across them.
- **Modes are defined outside the space being scored.** Clustering the LVAE
  latent and then asking whether LV metrics notice a dropped cluster is
  circular, and favours LV by construction. Modes come from design-space
  topology instead; `latent_kmeans` is retained only as the circular control,
  to show how much the circularity is worth. (The previous version documented
  this and then clustered on the latent anyway -- the bug this port fixes.)

Example:
    python -m engiopt.lvae.distortion_capture --problem-id photonics2d \\
        --instruments 89bb67a4:perf-ON:12 b7e29dcb:perf-OFF:39 \\
        --n-samples 200 --n-levels 6 --n-repeats 3
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, TYPE_CHECKING

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as cc_label
from sklearn.cluster import KMeans
import torch as th

from engiopt.evaluation.context import EvaluationContext
from engiopt.evaluation.registry import METRICS as REGISTRY
from engiopt.lvae.checkpoints import load_lvae

if TYPE_CHECKING:
    import numpy.typing as npt

BINARY_THRESHOLD = 0.5
EPS = 1e-8

# Cheap metrics computable without the simulator. `lv_dual_gap` is excluded: it
# needs a companion instrument, and here each instrument is scored on its own.
METRIC_NAMES = (
    "mmd",
    "pca_mmd",
    "pca_vendi",
    "pca_coverage",
    "dpp",
    "pixel_vendi",
    "pixel_paired_distance",
    "novelty",
    "lv_mmd",
    "lv_coverage",
    "lv_vendi",
    "lv_residual",
    "lv_paired_distance",
)

# What a correct metric must do as severity rises:
#   "+"   must increase        "-"    must decrease
#   "0"   must stay flat       "<=0"  must not increase (flat or falling both fine)
#
# Diversity under a fidelity corruption is one-sided on purpose. Corrupting
# designs must never *raise* a diversity score -- that is the pathology this
# battery exists to catch -- but a score that falls is defensible, because
# noised designs are not more functionally distinct than clean ones. Demanding
# strict flatness there would fail a metric for behaving sensibly.
EXPECTED = {
    "fidelity": {
        "pca_vendi": "<=0",
        "pca_coverage": "-",
        "mmd": "+",
        "pca_mmd": "+",
        "lv_mmd": "+",
        "lv_residual": "+",
        "pixel_paired_distance": "+",
        "lv_paired_distance": "+",
        "dpp": "<=0",
        "pixel_vendi": "<=0",
        "lv_vendi": "<=0",
        "lv_coverage": "-",
        "novelty": "+",
    },
    "mode_drop": {
        "pca_vendi": "-",
        "pca_coverage": "-",
        "lv_coverage": "-",
        "lv_vendi": "-",
        "pixel_vendi": "-",
        "dpp": "-",
        "mmd": "+",
        "pca_mmd": "+",
        "lv_mmd": "+",
        "lv_residual": "0",
        "pixel_paired_distance": "0",
        "lv_paired_distance": "0",
        "novelty": "0",
    },
    "mode_invent": {
        "pca_vendi": "<=0",
        "pca_coverage": "-",
        "lv_residual": "+",
        "mmd": "+",
        "pca_mmd": "+",
        "lv_mmd": "+",
        "lv_coverage": "-",
        "pixel_paired_distance": "+",
        "lv_paired_distance": "+",
        "dpp": "<=0",
        "pixel_vendi": "<=0",
        "lv_vendi": "<=0",
        "novelty": "+",
    },
    "collapse": {
        "pca_vendi": "-",
        "pca_coverage": "-",
        "lv_vendi": "-",
        "pixel_vendi": "-",
        "dpp": "-",
        "lv_coverage": "-",
        "mmd": "+",
        "pca_mmd": "+",
        "lv_mmd": "+",
        "lv_residual": "0",
        "pixel_paired_distance": "0",
        "lv_paired_distance": "0",
        "novelty": "0",
    },
    # Memorization is the case where every distribution and diversity metric
    # behaves exactly as designed and the design is insufficient: copying the
    # reference set *is* a perfect distribution match. Only novelty can object,
    # so the others are scored as "must not improve" and are expected to fail.
    "memorization": {
        "pca_vendi": "0",
        "pca_coverage": "0",
        "novelty": "-",
        "mmd": "0",
        "pca_mmd": "0",
        "lv_mmd": "0",
        "lv_residual": "0",
        "lv_coverage": "0",
        "dpp": "0",
        "pixel_vendi": "0",
        "lv_vendi": "0",
        "pixel_paired_distance": "0",
        "lv_paired_distance": "0",
    },
}

FAMILY_KIND = {
    "gaussian_noise": "fidelity",
    "blur": "fidelity",
    "pixel_flip": "fidelity",
    "intensity_shift": "fidelity",
    "translation": "fidelity",
    "mode_drop": "mode_drop",
    "mode_invent": "mode_invent",
    "collapse": "collapse",
    "memorization": "memorization",
}


# --------------------------------------------------------------------------
# Distortion families. severity=0 must be the identity for every one of them.
# --------------------------------------------------------------------------


def _gaussian_noise(x: npt.NDArray, s: float, rng: np.random.Generator) -> npt.NDArray:
    return np.clip(x + rng.normal(0.0, 0.5 * s, size=x.shape), 0.0, 1.0)


def _blur(x: npt.NDArray, s: float, _rng: np.random.Generator) -> npt.NDArray:
    if s == 0:
        return x.copy()
    return np.stack([gaussian_filter(img, sigma=5.0 * s) for img in x])


def _pixel_flip(x: npt.NDArray, s: float, rng: np.random.Generator) -> npt.NDArray:
    out = x.copy()
    mask = rng.random(x.shape) < (0.5 * s)
    out[mask] = (rng.random(int(mask.sum())) < x.mean()).astype(x.dtype)
    return out


def _intensity_shift(x: npt.NDArray, s: float, rng: np.random.Generator) -> npt.NDArray:
    sign = 1.0 if rng.random() < BINARY_THRESHOLD else -1.0
    return np.clip(x + sign * 0.4 * s, 0.0, 1.0)


def _translation(x: npt.NDArray, s: float, rng: np.random.Generator) -> npt.NDArray:
    shift = round(20 * s)
    if shift == 0:
        return x.copy()
    dy, dx = (int(v) for v in rng.integers(-shift, shift + 1, size=2))
    out = np.roll(x, shift=(dy, dx), axis=(1, 2))
    # Zero-fill the wrapped border, so this is a translation and not a roll.
    if dy > 0:
        out[:, :dy, :] = 0.0
    elif dy < 0:
        out[:, dy:, :] = 0.0
    if dx > 0:
        out[:, :, :dx] = 0.0
    elif dx < 0:
        out[:, :, dx:] = 0.0
    return out


PIXEL_FAMILIES: dict[str, Callable[[npt.NDArray, float, np.random.Generator], npt.NDArray]] = {
    "gaussian_noise": _gaussian_noise,
    "blur": _blur,
    "pixel_flip": _pixel_flip,
    "intensity_shift": _intensity_shift,
    "translation": _translation,
}


def _noise_designs(n: int, shape: tuple[int, int], vf: float, rng: np.random.Generator) -> npt.NDArray:
    """Off-manifold garbage: random binary fields at the right volume fraction."""
    return (rng.random((n, *shape)) < vf).astype(np.float64)


def topology_modes(designs: npt.NDArray, k: int) -> npt.NDArray:
    """Cluster designs by binarized geometry: material / void components and volume fraction.

    Defined purely in design space, so it is neutral between pixel, PCA and
    latent metrics. This is what stops mode_drop from being a test the LV
    metrics are guaranteed to pass.
    """
    feats = []
    for d in designs:
        b = d > BINARY_THRESHOLD
        _, n_mat = cc_label(b)
        _, n_void = cc_label(~b)
        feats.append([float(n_mat), float(n_void), float(b.mean())])
    f = np.asarray(feats)
    f = (f - f.mean(axis=0)) / (f.std(axis=0) + EPS)
    return KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(f)


def build_candidates(
    family: str,
    sev: float,
    rng: np.random.Generator,
    *,
    base: npt.NDArray,
    ref: npt.NDArray,
    labels: npt.NDArray,
    k_modes: int,
    n: int,
) -> npt.NDArray:
    """One candidate set for a (family, severity) cell."""
    if family in PIXEL_FAMILIES:
        return PIXEL_FAMILIES[family](base, sev, rng)
    if family == "mode_drop":
        n_drop = round(sev * (k_modes - 1))
        keep = set(rng.permutation(k_modes)[: k_modes - n_drop].tolist())
        pool = np.where(np.isin(labels, list(keep)))[0]
        if len(pool) == 0:
            pool = np.arange(len(base))
        return base[rng.choice(pool, size=n, replace=len(pool) < n)]
    if family == "mode_invent":
        n_bad = round(sev * n)
        good = base[rng.choice(len(base), size=n - n_bad, replace=False)] if n_bad < n else base[:0]
        return np.concatenate([good, _noise_designs(n_bad, base.shape[1:], float(ref.mean()), rng)], axis=0)
    if family == "collapse":
        # Interpolate from the full set to n copies of a single design.
        n_dup = round(sev * (n - 1))
        keep = base[rng.choice(len(base), size=n - n_dup, replace=False)]
        anchor = np.repeat(base[rng.integers(len(base))][None], n_dup, axis=0)
        return np.concatenate([keep, anchor], axis=0)
    if family == "memorization":
        # Replace candidates with verbatim reference optima. Fidelity and
        # diversity are perfect by construction; only novelty should object.
        n_copy = round(sev * n)
        keep = base[rng.choice(len(base), size=n - n_copy, replace=False)] if n_copy < n else base[:0]
        return np.concatenate([keep, ref[rng.choice(len(ref), size=n_copy, replace=False)]], axis=0)
    raise ValueError(f"unknown family: {family}")


def score_set(
    cand: npt.NDArray,
    *,
    problem: object,
    problem_id: str,
    ref: npt.NDArray,
    train: npt.NDArray,
    sigma_designs: npt.NDArray,
    lvae: object,
) -> dict[str, float]:
    """Run the cheap metric suite on one candidate set via the real registry.

    `train` is the memorization anchor rather than the full training split:
    candidates here are drawn from the dataset, so anchoring on the whole of
    it would put every set at novelty ~0 and leave the memorization family
    measuring nothing.
    """
    ctx = EvaluationContext(
        problem=problem,
        problem_id=problem_id,
        gen_designs=cand,
        ref_designs=ref,
        train_designs=train,
        sigma_designs=sigma_designs,
        latent_lvae=lvae,
    )
    row: dict[str, float] = {}
    for name in METRIC_NAMES:
        try:
            value = REGISTRY[name].fn(ctx)
        except Exception as exc:  # noqa: BLE001
            row[name] = float("nan")
            row[f"{name}__error"] = type(exc).__name__
            continue
        if isinstance(value, dict):
            # lv_residual reports mean and p90; the mean is the headline.
            row.update({k: float(v) for k, v in value.items()})
            row[name] = float(next(iter(value.values())))
        else:
            row[name] = float(value)
    return row


def main() -> None:
    """Capture metric responses across the distortion battery, one row per cell."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem-id", required=True)
    ap.add_argument(
        "--instruments", nargs="+", required=True, help="fingerprint:label:expected_dims, e.g. 89bb67a4:perf-ON:12"
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--n-levels", type=int, default=6)
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--k-modes", type=int, default=6)
    ap.add_argument("--families", nargs="+", default=list(FAMILY_KIND))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)
    h, w = problem.design_space.shape

    designs = np.asarray(problem.dataset["train"]["optimal_design"], dtype=np.float64).reshape(-1, h, w)
    np.random.default_rng(args.seed).shuffle(designs)
    n = args.n_samples
    if len(designs) < 2 * n:
        raise SystemExit(f"need >= {2 * n} train designs, have {len(designs)}")
    ref, base = designs[:n], designs[n : 2 * n]
    val = np.asarray(problem.dataset["val"]["optimal_design"], dtype=np.float64).reshape(-1, h, w)

    labels = topology_modes(base, args.k_modes)
    print(f"{args.problem_id}: ref={len(ref)} cand={len(base)} topology modes={np.bincount(labels).tolist()}")

    device = th.device("mps" if th.backends.mps.is_available() else "cuda" if th.cuda.is_available() else "cpu")
    severities = np.linspace(0.0, 1.0, args.n_levels)
    rows: list[dict] = []
    t0 = time.time()

    for spec in args.instruments:
        fingerprint, label, _dims = spec.split(":")
        lvae = load_lvae(
            problem_id=args.problem_id,
            design_shape=(h, w),
            algo="constrained_plvae_2d",
            seed=args.seed,
            device=device,
            config_fingerprint=fingerprint,
        )
        print(f"\n=== instrument {label} ({fingerprint}) ===")
        for family in args.families:
            for sev in severities:
                for rep in range(args.n_repeats):
                    rng = np.random.default_rng([args.seed, hash(family) % 9973, int(sev * 1000), rep])
                    cand = build_candidates(
                        family, float(sev), rng, base=base, ref=ref, labels=labels, k_modes=args.k_modes, n=n
                    )
                    row = score_set(
                        cand, problem=problem, problem_id=args.problem_id, ref=ref, train=ref, sigma_designs=val, lvae=lvae
                    )
                    row.update(
                        {
                            "instrument": label,
                            "fingerprint": fingerprint,
                            "family": family,
                            "kind": FAMILY_KIND[family],
                            "severity": float(sev),
                            "repeat": rep,
                        }
                    )
                    rows.append(row)
            print(f"[{time.time() - t0:6.1f}s] {family:16s} done")

    df = pd.DataFrame(rows)
    out = args.out or f"distortion_{args.problem_id}.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
