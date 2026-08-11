r"""Choose a problem's canonical latent instrument by a stated rule, not by eye.

Every latent metric is only as comparable as the autoencoder it is measured in,
so which instrument a spec pins is a load-bearing choice a paper has to defend.
This applies one rule uniformly across problems and prints the evidence for each
candidate, so the pin is justified from a table rather than a preference.

The rule, in order. Each stage is a gate rather than a weighted score, because a
candidate that fails any of them is not an instrument at all:

1. **It compressed.** `n_active < latent_dim`. An arm whose satisfaction gate
   never opened is an unregularised autoencoder wearing the name. Those are kept
   as evidence about threshold calibration, but they can never be the pin.
2. **It is seed-stable.** Across-seed coefficient of variation of `n_active`
   below `--max-cv`. An instrument whose width depends on the seed produces
   columns that are not comparable even to themselves.
3. **Held-out reconstruction is not much worse than the best candidate's**
   (`--max-val-penalty`). Validation NMSE is never gated on during *training* --
   the val split is 40 designs on heat, and gating would make `n_active` less
   stable -- but at selection time it is the trust diagnostic, and paying 2x val
   error for three fewer dimensions is a bad trade.
4. **Among survivors, maximise the task criterion**: partial Spearman of latent
   distance against *residual* performance, conditions partialled out. That is
   what the instrument is for, and it is the only stage that looks at
   performance at all.

Intrinsic dimension is printed but deliberately not gated: it is bounded by the
count of varying conditions, and landing near that bound is corroboration rather
than a requirement.

Example:
    python -m engiopt.lvae.select_instrument --problem-id photonics2d \
        --candidates ladder_checkpoints.json
"""

from __future__ import annotations

import argparse
import json
import statistics

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd
import torch as th

from engiopt.lvae.checkpoints import load_lvae_encoder
from engiopt.lvae.encode import encode_designs
from engiopt.lvae.encode import get_active_mask
from engiopt.lvae.instrument_analysis import load_dataset
from engiopt.lvae.instrument_analysis import partial_spearman
from engiopt.lvae.instrument_analysis import residual_performance

N_PAIRS = 40_000
EPS = 1e-12
DEAD_VARIANCE = 1e-8


def training_stats(problem_id: str, projects: list[str]) -> dict[str, dict]:
    """Per-config val NMSE and across-seed dimension spread, read from W&B.

    Read rather than recomputed: the training run already measured both, and
    re-deriving val NMSE here would use a different protocol than the one the
    recorded number came from.
    """
    import wandb

    api = wandb.Api()
    stats: dict[str, dict] = {}
    for project in projects:
        try:
            runs = api.runs(f"engibench/{project}")
        except Exception as exc:  # noqa: BLE001 - a project that does not exist is simply skipped
            print(f"  no project {project}: {type(exc).__name__}")
            continue
        for run in runs:
            summary = run.summary
            if summary.get("val_nmse_rec") is None or run.name.split("__")[0] != problem_id:
                continue
            path = str(summary.get("hf_config_package_path", ""))
            fingerprint = path.split("/")[1].removeprefix("cfg_") if "/" in path else None
            if not fingerprint:
                continue
            entry = stats.setdefault(fingerprint, {"dims": [], "val": []})
            entry["dims"].append(int(summary.get("active_dims", summary.get("vol_active", -1))))
            entry["val"].append(float(summary["val_nmse_rec"]))
    return stats


def score_candidates(
    problem_id: str, candidates: dict, shape: tuple, designs: np.ndarray, pairs: tuple, stats: dict
) -> pd.DataFrame:
    """Measure each candidate: how wide, how stable, and how well it tracks performance."""
    i, j, d_resid, d_cond = pairs
    device = th.device("mps" if th.backends.mps.is_available() else "cuda" if th.cuda.is_available() else "cpu")
    rows = []
    for label, pkgs in sorted(candidates.items()):
        pkg = pkgs[0]
        fingerprint = pkg["path"].split("/")[1].removeprefix("cfg_")
        try:
            encoder, config, _ = load_lvae_encoder(
                problem_id=problem_id,
                design_shape=shape,
                algo="constrained_plvae_2d",
                seed=pkg["seed"],
                device=device,
                config_fingerprint=fingerprint,
            )
            codes = encode_designs(encoder, designs, device, 256)
            n_active = int(get_active_mask(encoder).sum())
            latent_dim = int(config.latent_dim)
        except Exception as exc:  # noqa: BLE001 - an unloadable candidate is simply not a candidate
            print(f"  [skip] {label}: {type(exc).__name__}")
            continue

        features = codes[:, codes.var(axis=0) > DEAD_VARIANCE]
        rho = partial_spearman(np.linalg.norm(features[i] - features[j], axis=1), d_resid, d_cond)

        seen = stats.get(fingerprint, {})
        dims = seen.get("dims") or [n_active]
        mean_dims = statistics.mean(dims)
        cv = statistics.pstdev(dims) / mean_dims if len(dims) > 1 and mean_dims else 0.0
        rows.append(
            {
                "candidate": label.split("|", 1)[1],
                "cfg": fingerprint,
                "n_active": n_active,
                "latent_dim": latent_dim,
                "compressed": n_active < latent_dim,
                "seed_cv": cv,
                "val_nmse": statistics.median(seen["val"]) if seen.get("val") else float("nan"),
                "rho_residual": rho,
            }
        )
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, problem_id: str, varying: int, max_cv: float, max_val: float) -> pd.DataFrame:
    """Apply the gates in order and print the evidence behind the pin."""
    df["val_ratio"] = df["val_nmse"] / df["val_nmse"].min(skipna=True)
    df["gate1_compressed"] = df["compressed"]
    df["gate2_stable"] = df["seed_cv"] <= max_cv
    df["gate3_val_ok"] = df["val_ratio"] <= max_val
    df["eligible"] = df["gate1_compressed"] & df["gate2_stable"] & df["gate3_val_ok"]

    pd.set_option("display.width", 220)
    print(f"\n=== {problem_id}: instrument candidates ===")
    cols = [
        "candidate",
        "cfg",
        "n_active",
        "seed_cv",
        "val_nmse",
        "val_ratio",
        "rho_residual",
        "gate1_compressed",
        "gate2_stable",
        "gate3_val_ok",
        "eligible",
    ]
    print(df[cols].sort_values("rho_residual", ascending=False).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nintrinsic dimension is bounded by {varying} varying conditions (context, not a gate)")

    eligible = df[df["eligible"]]
    if eligible.empty:
        print("\nNO ELIGIBLE INSTRUMENT: every candidate fails a gate. Do not pin one.")
    else:
        pick = eligible.sort_values("rho_residual", ascending=False).iloc[0]
        print(
            f"\nPIN -> {pick['cfg']} ({pick['candidate']}): {pick['n_active']} dims, "
            f"val {pick['val_nmse']:.4g} ({pick['val_ratio']:.2f}x best), "
            f"seed cv {pick['seed_cv']:.2f}, rho_residual {pick['rho_residual']:+.3f}"
        )
    return df


def main() -> None:
    """Print the candidate table and the pin the rule selects."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem-id", required=True)
    ap.add_argument("--candidates", required=True, help="JSON of {label: [{seed, path, rev}]}")
    ap.add_argument("--max-cv", type=float, default=0.25, help="Max across-seed CV of n_active.")
    ap.add_argument("--max-val-penalty", type=float, default=1.25, help="Max val NMSE vs the best candidate's.")
    ap.add_argument(
        "--wandb-projects",
        nargs="*",
        default=["engiopt-lvladder", "engiopt-lvmatch", "engiopt-heatlip", "engiopt"],
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=1)
    designs, conds, perf = load_dataset(problem, args.problem_id)
    resid = residual_performance(conds, perf)

    rng = np.random.default_rng(0)
    i = rng.integers(0, len(designs), N_PAIRS)
    j = rng.integers(0, len(designs), N_PAIRS)
    keep = i != j
    i, j = i[keep], j[keep]
    scaled = (conds - conds.mean(0)) / (conds.std(0) + EPS)
    pairs = (i, j, np.abs(resid[i] - resid[j]).sum(axis=1), np.linalg.norm(scaled[i] - scaled[j], axis=1))

    with open(args.candidates) as fh:
        candidates = {k: v for k, v in json.load(fh).items() if k.startswith(args.problem_id + "|")}

    stats = training_stats(args.problem_id, args.wandb_projects)
    df = score_candidates(args.problem_id, candidates, problem.design_space.shape, designs, pairs, stats)
    if df.empty:
        raise SystemExit("no candidates could be loaded")

    df = report(df, args.problem_id, int((conds.std(axis=0) > 0).sum()), args.max_cv, args.max_val_penalty)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
