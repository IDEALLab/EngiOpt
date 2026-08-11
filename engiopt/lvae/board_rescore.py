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
    "pixel_vendi",
    "novelty",
    "lv_mmd",
    "lv_coverage",
    "lv_vendi",
    "lv_residual",
    "lv_paired_distance",
]
PHYSICS = ["iog", "cog", "fog"]


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

    rows = []
    for key in board["key"]:
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
        rows.append({"key": key, **{m: scores.get(m) for m in CHEAP}})
        print(f"  {key:32s} {time.perf_counter() - started:5.1f}s  mmd={scores.get('mmd'):.4f}")

    fresh = pd.DataFrame(rows)
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
    ):
        if pixel in result.index and latent in result.index:
            a, b = abs(result.loc[pixel, "iog"]), abs(result.loc[latent, "iog"])
            gain = b / a if a else float("inf")
            print(f"  {pixel:14s} {a:.3f}  ->  {latent:14s} {b:.3f}   gain {gain:.2f}x")


if __name__ == "__main__":
    main()
