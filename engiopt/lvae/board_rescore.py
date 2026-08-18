r"""Do the cheap metrics rank *real* generators the way the simulator does?

The distortion audit fixes ground truth by construction, which is what makes it
decisive about failure modes -- and also what makes it insufficient. A metric
that handles synthetic corruption correctly still has to reproduce the
model-selection decision a designer would make after hours of simulation.

This rescores an existing physics board's *cheap* columns and correlates each
against the simulator columns already on it. The expensive columns (IOG, COG,
FOG) are reused rather than recomputed: they cost hours, and nothing here
changes them.

Its first use is to check whether the pixel-space metrics recover once the MMD
bandwidth is calibrated rather than fixed at the spec's 10.0. On photonics2d the
fixed bandwidth left pixel MMD unable to separate real optima from random
fields, so any earlier "latent beats pixel" margin was partly a bandwidth
artifact and has to be re-measured before it is claimed.

Example:
    python -m engiopt.lvae.board_rescore --problem-id beams2d \
        --board workshops/idetc26/tools/ground_truth_16.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd
from scipy import stats

from engiopt.evaluation import Evaluator

CHEAP = [
    "mmd",
    "pca_mmd",
    "pca_vendi",
    "pca_coverage",
    "dpp",
    "dpp_geometric",
    "pixel_vendi",
    "novelty",
    "lv_mmd",
    "lv_coverage",
    "lv_vendi",
    "lv_residual",
    # All three spaces for the paired distance. `lv_paired_distance` is the
    # strongest cheap predictor on the boards so far, and scoring it without its
    # pixel and PCA counterparts cannot separate "the latent space is the right
    # place to measure" from "pairing against the optimum for the same condition
    # is a good idea anywhere".
    "lv_paired_distance",
    "pca_paired_distance",
    "pixel_paired_distance",
    # The rung between PCA and the instrument: a least-volume space trained
    # without the performance constraint. Its whole purpose is to sit on the
    # board beside its lv_ twin, so the ladder can say how much of the gain is
    # "a learned compressed space" and how much is "performance-awareness".
    "lvoff_mmd",
    "lvoff_coverage",
    "lvoff_vendi",
    "lvoff_paired_distance",
    # The units the lv_*/lvoff_*/pca_* columns are in; the PCA control is fitted
    # to the instrument's active width, so a board without these cannot be
    # checked, and the two learned rungs need not share a width either.
    "lv_active_dims",
    "lvoff_active_dims",
    "pca_dims",
]
PHYSICS = ["iog", "cog", "fog", "iog_median", "cog_median", "fog_median"]
"""Mean and median optimality gaps. The per-design gap is unbounded above, so a
mean over ~50 samples is set by its worst member -- on the beams2d board a model
reports mean IOG 1.5e8 while finishing at FOG -2.2, which is one unrecoverable
starting design rather than a worse model. Rank correlations are unaffected;
any statement about magnitude needs the medians."""


def main() -> None:
    """Rescore a board's cheap metrics and correlate them against the simulator."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem-id", default="beams2d")
    ap.add_argument("--board", required=True, help="CSV carrying keys and the physics columns.")
    ap.add_argument("--spec", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    board = pd.read_csv(args.board)
    from engiopt.baselines.base import DatasetGenerator
    from engiopt.utils.all_generators import BUILTIN_GENERATORS

    evaluator = Evaluator.for_problem(args.problem_id, spec=args.spec)

    # Restartable for the same reason `physics_board` is. Rescoring resamples
    # every generator, and on a 200-row board with 46 diffusion packages that is
    # hours -- long enough that a job dying at the end used to cost the whole
    # run. Rows land in a sidecar as they finish; the final CSV is written from
    # it plus whatever this pass adds.
    partial = Path(f"{args.out or f'board_rescored_{args.problem_id}.csv'}.partial")
    prior = pd.read_csv(partial) if partial.exists() else pd.DataFrame(columns=["key"])
    done = set(prior["key"])
    if done:
        print(f"resuming: {len(done)} rows already scored in {partial}")

    rows = []
    for key in board["key"]:
        if key in done:
            continue
        # Board keys are "{algo}/{config_fingerprint}/s{seed}"; "default" means
        # the package published at the training script's own defaults.
        algo, fingerprint, seed_part = key.split("/")
        started = time.perf_counter()
        cls = BUILTIN_GENERATORS[algo]
        try:
            # Dataset-fitted baselines (kNN, ridge) have no checkpoint: they are
            # built with `from_problem`, and `build` raises by design. Dispatching
            # them through `from_pretrained` is what made them 404 -- they were
            # never meant to be on the Hub, so no publish is needed.
            if isinstance(cls, type) and issubclass(cls, DatasetGenerator):
                generator = cls.from_problem(evaluator.problem, problem_id=args.problem_id, seed=int(seed_part.lstrip("s")))
            else:
                generator = cls.from_pretrained(
                    evaluator.problem,
                    problem_id=args.problem_id,
                    seed=int(seed_part.lstrip("s")),
                    model_source="hf",
                    config_fingerprint=None if fingerprint == "default" else fingerprint,
                )
            generator.seed = args.seed
            scores = evaluator.score(generator, only=CHEAP)
        except Exception as exc:  # noqa: BLE001 - one bad package must not end the board
            print(f"  {key}: FAILED {type(exc).__name__}: {str(exc)[:100]}")
            continue
        row = {"key": key, **{m: scores.get(m) for m in CHEAP}}
        rows.append(row)
        pd.DataFrame([row]).to_csv(partial, mode="a", header=not partial.exists(), index=False)
        print(f"  {key:32s} {time.perf_counter() - started:5.1f}s  mmd={scores.get('mmd'):.4f}")

    fresh = pd.concat([prior, pd.DataFrame(rows)], ignore_index=True) if rows else prior
    merged = fresh.merge(board[["key", "algo", *PHYSICS]], on="key", how="inner")
    out = args.out or f"board_rescored_{args.problem_id}.csv"
    merged.to_csv(out, index=False)
    print(f"\nwrote {len(merged)} rows -> {out}")

    report(merged)


def report(merged: pd.DataFrame) -> None:
    """Correlate each cheap column against the simulator columns and print the pairings."""
    print(f"\n=== Spearman(cheap metric, simulator) over n={len(merged)} real generators ===")
    print("(recomputed cheap columns, calibrated bandwidth; physics columns reused from the board)")
    table = []
    for metric in CHEAP:
        if metric not in merged or merged[metric].notna().sum() < 3:  # noqa: PLR2004
            continue
        entry = {"metric": metric}
        for target in PHYSICS:
            sub = merged[[metric, target]].dropna()
            entry[target] = stats.spearmanr(sub[metric], sub[target]).statistic if len(sub) > 2 else float("nan")  # noqa: PLR2004
        table.append(entry)
    result = pd.DataFrame(table).set_index("metric")
    pd.set_option("display.width", 200)
    print(result.to_string(float_format=lambda v: f"{v:+.3f}"))

    # The paper's claim is a *pairing*: the same question asked in two spaces.
    print("\n=== same question, different space (|rho| vs IOG) ===")
    for pixel, latent in (
        ("pixel_vendi", "lv_vendi"),
        ("dpp", "lv_vendi"),
        ("mmd", "lv_mmd"),
        ("pca_vendi", "lv_vendi"),
        ("pca_mmd", "lv_mmd"),
        ("pca_coverage", "lv_coverage"),
        ("pixel_paired_distance", "lv_paired_distance"),
        ("pca_paired_distance", "lv_paired_distance"),
    ):
        if pixel in result.index and latent in result.index:
            a, b = abs(result.loc[pixel, "iog"]), abs(result.loc[latent, "iog"])
            gain = b / a if a else float("inf")
            print(f"  {pixel:14s} {a:.3f}  ->  {latent:14s} {b:.3f}   gain {gain:.2f}x")

    ladder(result)


LADDER = {
    "does it match the data?": ("mmd", "pca_mmd", "lvoff_mmd", "lv_mmd"),
    "did it cover the modes?": (None, "pca_coverage", "lvoff_coverage", "lv_coverage"),
    "how many distinct designs?": ("pixel_vendi", "pca_vendi", "lvoff_vendi", "lv_vendi"),
    "did it answer the condition?": (
        "pixel_paired_distance",
        "pca_paired_distance",
        "lvoff_paired_distance",
        "lv_paired_distance",
    ),
}
"""The four rungs, per question: pixels, a matched linear subspace, a learned
least-volume space without the performance constraint, and the instrument.

The third rung is the one that decides what the method claims. Pixels to PCA
measures dimensionality reduction; PCA to recon-only measures nonlinearity; and
recon-only to the instrument measures the performance constraint alone -- the
only step no cheaper representation can reproduce."""


def ladder(result: pd.DataFrame, target: str = "iog_median") -> None:
    """Print the ablation ladder: the same question, climbing four spaces.

    Args:
        result: Spearman table indexed by metric, columns per physics target.
        target: Which physics column to read the ladder against. Defaults to the
            median, since the mean optimality gap is set by its worst design.
    """
    if target not in result.columns:
        return
    print(f"\n=== the ladder: |rho| vs {target}, same question in four spaces ===")
    header = f"  {'question':30s} {'pixel':>8s} {'PCA':>8s} {'LV recon':>9s} {'LV perf':>8s}   {'constraint':>10s}"
    print(header)
    for question, rungs in LADDER.items():
        cells = []
        for metric in rungs:
            value = abs(result.loc[metric, target]) if metric in result.index else float("nan")
            cells.append(f"{value:8.3f}" if value == value else f"{'--':>8s}")  # noqa: PLR0124 - NaN check
        off, on = cells[2].strip(), cells[3].strip()
        try:
            delta = f"{float(on) - float(off):+10.3f}"
        except ValueError:
            delta = f"{'--':>10s}"
        print(f"  {question:30s} {cells[0]} {cells[1]} {cells[2]:>9s} {cells[3]}   {delta}")
    print("\n  'constraint' is LV perf minus LV recon -- the gain attributable to the")
    print("  performance constraint alone, both arms being learned, compressed and nonlinear.")


if __name__ == "__main__":
    main()
