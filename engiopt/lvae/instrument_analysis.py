"""Score the Phase 1 ladder's instruments on the task criterion, not on reconstruction.

Handoff 3.6 settles that reconstruction thresholds only *generate* candidate
instruments; selection happens afterwards. This applies the two criteria it names:

- **Residual-performance correlation (3.2).** Does latent distance track
  |delta performance| once the conditions are regressed out? Raw performance is
  contaminated by condition encoding -- on beams2d that inflates rho from 0.30 to
  0.88 and means almost nothing. photonics2d and heatconduction2d have 44% and
  60% design-attributable variance, so this is the first time the test runs where
  there is something to detect.
- **Participation ratio (3.4).** PR of the latent variance spectrum, which needs
  no pruning threshold and is defined even on runs that never converged.

pixel L2 and PCA are carried as the baselines each instrument has to beat. If a
latent does not beat pixel space, the autoencoder contributed nothing.
"""

from __future__ import annotations

import argparse
import json
import warnings

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
import torch as th

from engiopt.lvae.checkpoints import load_lvae_encoder
from engiopt.lvae.encode import encode_designs
from engiopt.transforms import get_performance_target
from engiopt.transforms import get_scalar_condition_keys

warnings.filterwarnings("ignore", category=UserWarning)

N_PAIRS = 40_000
PR_EPS = 1e-12


def participation_ratio(z: np.ndarray) -> float:
    """PR of the latent variance spectrum: (sum v)^2 / sum(v^2).

    Equals the number of dimensions carrying variance when the spectrum is flat,
    and ->1 when a single dimension dominates. No threshold, no pruning mask.
    """
    v = z.var(axis=0)
    total = v.sum()
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    # Normalise before squaring. A collapsed arm can carry variances spanning
    # 1e100, where (sum v)^2 and sum(v^2) both overflow float64 and the ratio
    # comes back below 1 -- which PR can never be.
    w = v / total
    return float(1.0 / (np.square(w).sum() + PR_EPS))


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman correlation of x and y with z partialled out.

    Rank-transform all three, then correlate the residuals of x~z and y~z. The
    control z is the pairwise condition distance: without it, two designs at
    distant conditions look both far apart in latent space and different in
    performance for reasons that have nothing to do with the manifold.
    """
    rx, ry, rz = (stats.rankdata(a) for a in (x, y, z))
    zc = np.column_stack([np.ones_like(rz), rz])
    beta_x, *_ = np.linalg.lstsq(zc, rx, rcond=None)
    beta_y, *_ = np.linalg.lstsq(zc, ry, rcond=None)
    # Applied as an explicit affine rather than `zc @ beta`: NumPy 2.0 on Apple
    # Accelerate emits spurious divide-by-zero/overflow warnings for this matmul
    # shape. Verified bit-identical to the matmul, and to the closed-form
    # partial-Spearman expression, before switching.
    ex = rx - (beta_x[0] + beta_x[1] * rz)
    ey = ry - (beta_y[0] + beta_y[1] * rz)
    return float(np.corrcoef(ex, ey)[0, 1])


def residual_performance(conds: np.ndarray, perf: np.ndarray, *, shuffle: bool = True) -> np.ndarray:
    """Performance with the part predictable from the conditions removed.

    Uses out-of-fold predictions so the residual is not shrunk by the model
    having already seen the point it is subtracting from.
    """
    perf = np.asarray(perf, dtype=np.float64)
    # Shuffled folds: these datasets are grid samples ordered by condition, so
    # contiguous folds would hold out a whole region and understate what the
    # conditions explain.
    cv = KFold(n_splits=5, shuffle=shuffle, random_state=0 if shuffle else None)
    resid = np.empty_like(perf)
    for j in range(perf.shape[1]):
        pred = cross_val_predict(HistGradientBoostingRegressor(random_state=0), conds, perf[:, j], cv=cv)
        resid[:, j] = perf[:, j] - pred
    return resid


def score(
    name: str,
    feats: np.ndarray,
    pairs: tuple[np.ndarray, np.ndarray],
    d_cond: np.ndarray,
    d_resid: np.ndarray,
    d_raw: np.ndarray,
) -> dict:
    """Score one representation: its spread, and how well it tracks performance."""
    i, j = pairs
    f = np.asarray(feats, dtype=np.float64).reshape(len(feats), -1)
    d_feat = np.linalg.norm(f[i] - f[j], axis=1)
    return {
        "representation": name,
        "dims": f.shape[1],
        "PR": participation_ratio(f),
        "rho_residual": partial_spearman(d_feat, d_resid, d_cond),
        "rho_raw": partial_spearman(d_feat, d_raw, d_cond),
    }


def load_dataset(problem: object, problem_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Designs, the conditions that actually vary, and the performance targets.

    Constant condition columns are dropped: they explain nothing, and they make
    the pairwise condition distance that the partial correlation controls for
    degenerate.
    """
    ds = problem.dataset["train"].with_format("torch")
    designs = ds["optimal_design"][:].numpy().astype(np.float64)
    cond_keys = get_scalar_condition_keys(problem, problem.dataset["train"])
    conds = np.column_stack([ds[k][:].numpy() for k in cond_keys]).astype(np.float64)
    perf = get_performance_target(problem, ds).numpy().astype(np.float64)
    varying = conds.std(axis=0) > 0
    print(f"{problem_id}: {len(designs)} designs, {perf.shape[1]} objective(s)")
    print(f"  varying conditions: {[k for k, v in zip(cond_keys, varying, strict=True) if v]}")
    return designs, conds[:, varying], perf


def report(rows: list[dict], problem_id: str, out: str | None) -> None:
    """Print the ranked table and persist it next to the run."""
    df = pd.DataFrame(rows).sort_values("rho_residual", ascending=False)
    pd.set_option("display.width", 200)
    print(f"\n=== {problem_id}: latent distance vs |delta performance|, conditions controlled ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    path = out or f"instrument_scores_{problem_id}.csv"
    df.to_csv(path, index=False)
    print(f"\nwrote {path}")


def main() -> None:
    """Score every candidate instrument for one problem against the baselines."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem-id", required=True)
    ap.add_argument("--checkpoints", default="ladder_checkpoints.json")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=1)
    design_shape = problem.design_space.shape
    designs, conds_v, perf = load_dataset(problem, args.problem_id)
    n = len(designs)

    # Both fold schemes, because on a grid-sampled dataset they bracket the
    # truth rather than agree: shuffled folds leak (a held-out point's grid
    # neighbours stay in train, so the conditions look more predictive than they
    # are), contiguous folds hold out whole condition regions and force
    # extrapolation (so they look less predictive). Report the interval.
    resid = residual_performance(conds_v, perf, shuffle=True)
    resid_block = residual_performance(conds_v, perf, shuffle=False)
    r2 = 1 - resid.var(axis=0) / perf.var(axis=0)
    r2_block = 1 - resid_block.var(axis=0) / perf.var(axis=0)
    print(f"  R^2(conditions -> performance): shuffled {np.round(r2, 4)} | blocked {np.round(r2_block, 4)}")
    print(f"  design-attributable variance: {np.round(100 * (1 - r2), 1)}% .. {np.round(100 * (1 - r2_block), 1)}%")

    i = rng.integers(0, n, N_PAIRS)
    j = rng.integers(0, n, N_PAIRS)
    keep = i != j
    i, j = i[keep], j[keep]
    cs = (conds_v - conds_v.mean(0)) / (conds_v.std(0) + PR_EPS)
    d_cond = np.linalg.norm(cs[i] - cs[j], axis=1)
    d_resid = np.abs(resid[i] - resid[j]).sum(axis=1)
    d_raw = np.abs(perf[i] - perf[j]).sum(axis=1)

    rows = [score("pixel L2", designs, (i, j), d_cond, d_resid, d_raw)]
    flat = designs.reshape(n, -1)
    rows.extend(
        score(f"PCA-{k}", PCA(n_components=k, svd_solver="full").fit_transform(flat), (i, j), d_cond, d_resid, d_raw)
        for k in (6, 20)
    )

    with open(args.checkpoints) as fh:
        entries = {k: v for k, v in json.load(fh).items() if k.startswith(args.problem_id + "|")}
    device = th.device("mps" if th.backends.mps.is_available() else "cuda" if th.cuda.is_available() else "cpu")
    for key, pkgs in sorted(entries.items()):
        pkg = next((p for p in pkgs if p["seed"] == args.seed), pkgs[0])
        fingerprint = pkg["path"].split("/")[1].removeprefix("cfg_")
        _, mode, thr = key.split("|")
        try:
            enc, _cfg, _res = load_lvae_encoder(
                problem_id=args.problem_id,
                design_shape=design_shape,
                algo="constrained_plvae_2d",
                seed=pkg["seed"],
                device=device,
                config_fingerprint=fingerprint,
                revision=pkg["rev"],
            )
            z = encode_designs(enc, designs, device, 256)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {key}: {type(exc).__name__}: {exc}")
            continue
        # Dead dimensions carry no variance and only dilute the distance.
        active = z.var(axis=0) > 1e-8  # noqa: PLR2004
        rows.append(score(f"LV perf-{mode} thr={thr} ({fingerprint})", z[:, active], (i, j), d_cond, d_resid, d_raw))
        print(f"  scored {key}: {active.sum()} active dims")

    report(rows, args.problem_id, args.out)


if __name__ == "__main__":
    main()
