"""Fetch evaluation metrics from WandB training runs and save to a flat CSV.

For each (model, problem, seed) combination, this script:
1. Looks up the training run via its model artifact
2. Reads all eval/* summary keys (base metrics + per-threshold LV/PCA metrics)
3. Emits one row per (run × LVAE threshold combination)

The output CSV has consistent named columns and can be loaded with pd.read_csv()
directly in the analysis notebook.

Usage:
    python engiopt/fetch_wandb_metrics.py --output-csv all_metrics_wandb.csv
    python engiopt/fetch_wandb_metrics.py --entity engibench --project engiopt
    python engiopt/fetch_wandb_metrics.py --models cgan_cnn_2d vqgan --seeds 1 2 3
"""

from __future__ import annotations

import dataclasses
import itertools
import re
from typing import Any

import pandas as pd
import tyro

import wandb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROBLEMS = ["beams2d", "heatconduction2d", "photonics2d"]

MODELS = ["cgan_cnn_2d", "diffusion_2d_cond", "gan_cnn_2d", "vqgan"]

# Artifact name suffix used to look up each model's training run.
# The full artifact path is: {entity}/{project}/{problem_id}_{suffix}:seed_{seed}
ARTIFACT_SUFFIX: dict[str, str] = {
    "cgan_cnn_2d": "cgan_cnn_2d_generator",
    "diffusion_2d_cond": "diffusion_2d_cond_model",
    "gan_cnn_2d": "gan_cnn_2d_generator",
    "vqgan": "vqgan_transformer",
}

# LVAE threshold combinations per problem — must match what the evaluate scripts used.
LVAE_THRESHOLDS: dict[str, dict[str, list[float]]] = {
    "beams2d": {"rec": [0.0005, 0.001, 0.0025], "perf": [0.001, 1000.0]},
    "heatconduction2d": {"rec": [0.01, 0.025, 0.05], "perf": [0.005, 1000.0]},
    "photonics2d": {"rec": [0.1, 0.15, 0.2], "perf": [0.005, 1000.0]},
}

# Base eval keys in WandB summary → output column name
BASE_EVAL_MAP: dict[str, str] = {
    "eval/iog": "iog",
    "eval/cog": "cog",
    "eval/fog": "fog",
    "eval/mmd": "mmd",
    "eval/dpp": "dpp",
    "eval/mmd_sigma": "mmd_sigma",
    "eval/cond_mmd": "cond_mmd",
    "eval/cond_perf_mmd": "cond_perf_mmd",
}

# Per-threshold LV sub-keys (relative to eval/lv_rec{r}_perf{p}_lvae{s}/)
LV_SUBKEYS: dict[str, str] = {
    "lv_mmd": "lv_mmd",
    "lv_dpp": "lv_dpp",
    "lv_sigma": "lv_sigma",
    "n_active_dims": "lvae_n_active_dims",
    "lv_cond_mmd": "lv_cond_mmd",
    "lv_cond_perf_mmd": "lv_cond_perf_mmd",
}

# Per-threshold PCA sub-keys
PCA_SUBKEYS: dict[str, str] = {
    "pca_mmd": "pca_mmd",
    "pca_dpp": "pca_dpp",
    "pca_sigma": "pca_sigma",
    "pca_cond_mmd": "pca_cond_mmd",
    "pca_cond_perf_mmd": "pca_cond_perf_mmd",
}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Args:
    """Arguments for fetching WandB metrics."""

    entity: str = "engibench"
    """WandB entity (org/user)."""
    project: str = "engiopt"
    """WandB project."""
    models: list[str] = dataclasses.field(default_factory=lambda: list(MODELS))
    """Model IDs to fetch. Defaults to all four models."""
    problems: list[str] = dataclasses.field(default_factory=lambda: list(PROBLEMS))
    """Problem IDs to fetch. Defaults to all three problems."""
    seeds: list[int] = dataclasses.field(default_factory=lambda: list(range(1, 11)))
    """Generator seeds to fetch."""
    lvae_seed: int = 1
    """LVAE seed used during evaluation."""
    n_samples: int = 50
    """Number of samples (recorded in output, not used for fetching)."""
    output_csv: str = "all_metrics_wandb.csv"
    """Output CSV path."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_run(
    api: wandb.Api, entity: str, project: str, problem_id: str, model_id: str, seed: int
) -> wandb.apis.public.Run | None:
    """Look up the training run for a given (model, problem, seed) via its artifact."""
    suffix = ARTIFACT_SUFFIX.get(model_id)
    if suffix is None:
        return None
    artifact_path = f"{entity}/{project}/{problem_id}_{suffix}:seed_{seed}"
    try:
        artifact = api.artifact(artifact_path, type="model")
        run = artifact.logged_by()
        return run
    except Exception as e:
        print(f"    Artifact not found: {artifact_path} ({e})")
        return None


def _safe(summary: dict, key: str) -> Any:
    """Return summary[key] or None if missing/nan."""
    val = summary.get(key)
    if val is None:
        return None
    try:
        f = float(val)
        return None if (f != f) else f  # NaN check
    except (TypeError, ValueError):
        return None


def _find_lvae_seeds(summary: dict) -> set[int]:
    """Scan summary keys to find which LVAE seeds were logged."""
    seeds = set()
    for key in summary:
        m = re.match(r"eval/lv_rec[\d.]+_perf[\d.]+_lvae(\d+)/", key)
        if m:
            seeds.add(int(m.group(1)))
    return seeds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def fetch_metrics(args: Args) -> pd.DataFrame:
    api = wandb.Api()
    rows: list[dict[str, Any]] = []

    total = len(args.models) * len(args.problems) * len(args.seeds)
    done = 0

    for model_id, problem_id, seed in itertools.product(args.models, args.problems, args.seeds):
        done += 1
        print(f"[{done}/{total}] {model_id} / {problem_id} / seed={seed}")

        run = _get_run(api, args.entity, args.project, problem_id, model_id, seed)
        if run is None:
            print("    No run found, skipping.")
            continue

        summary = dict(run.summary)

        # Skip runs with no eval metrics at all
        if _safe(summary, "eval/cog") is None and _safe(summary, "eval/mmd") is None:
            print("    No eval metrics in summary, skipping.")
            continue

        # Base row (shared across threshold combos)
        base: dict[str, Any] = {
            "problem_id": problem_id,
            "model_id": model_id,
            "seed": seed,
            "n_samples": args.n_samples,
            "lvae_seed": args.lvae_seed,
        }
        for wkey, col in BASE_EVAL_MAP.items():
            base[col] = _safe(summary, wkey)

        # Config extras (viol is not logged to WandB; keep as None)
        base["viol"] = None

        # Determine LVAE threshold combos from the problem config
        pt = LVAE_THRESHOLDS.get(problem_id, {})
        combos = list(itertools.product(pt.get("rec", []), pt.get("perf", [])))

        if not combos:
            # No threshold info — emit one base row with NaN LV metrics
            row = dict(base)
            row.update({"rec_threshold": None, "perf_threshold": None, "has_perf": None})
            for col in list(LV_SUBKEYS.values()) + list(PCA_SUBKEYS.values()):
                row[col] = None
            rows.append(row)
            continue

        for rec, perf in combos:
            prefix = f"eval/lv_rec{rec}_perf{perf}_lvae{args.lvae_seed}"
            row = dict(base)
            row["rec_threshold"] = rec
            row["perf_threshold"] = perf
            row["has_perf"] = perf < 100  # noqa: PLR2004

            for subkey, col in LV_SUBKEYS.items():
                row[col] = _safe(summary, f"{prefix}/{subkey}")
            for subkey, col in PCA_SUBKEYS.items():
                row[col] = _safe(summary, f"{prefix}/{subkey}")

            lv_present = any(row[col] is not None for col in LV_SUBKEYS.values())
            print(
                f"    rec={rec}, perf={perf}: lv_mmd={row.get('lv_mmd')}, "
                f"pca_mmd={row.get('pca_mmd')}, {'OK' if lv_present else 'no LV data'}"
            )
            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    args = tyro.cli(Args)
    print(f"Fetching from {args.entity}/{args.project}")
    print(f"Models: {args.models}")
    print(f"Problems: {args.problems}")
    print(f"Seeds: {args.seeds}")
    print()

    df = fetch_metrics(args)
    df.to_csv(args.output_csv, index=False)
    print(f"\nWrote {len(df)} rows to {args.output_csv}")
    print(f"LV metrics present: {df['lv_mmd'].notna().sum()} / {len(df)} rows")
    print(f"PCA metrics present: {df['pca_mmd'].notna().sum()} / {len(df)} rows")
    print(f"cond_mmd present:   {df['cond_mmd'].notna().sum()} / {len(df)} rows")
    print("\nRows per problem × model:")
    print(df.groupby(["problem_id", "model_id"]).size().unstack(fill_value=0).to_string())
