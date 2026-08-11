r"""Evaluate every published checkpoint and store the numbers with the checkpoint.

The IDETC workshop plan's standing commitment is "produce a large pool, curate
late": train more models than needed, evaluate everything, publish checkpoints
*and* metrics, then pick the bank from real numbers. That only works if a
package's numbers travel with it, so this writes a `metrics.json` next to the
weights rather than into a CSV somebody has to find later.

Two things follow from the metrics living in the package:

- **Curation needs no weight download.** Choosing a workshop bank, or a paper's
  canonical instrument, becomes a scan over metadata.
- **A number can never be orphaned from the thing it describes.** The recorded
  spec, revision and instrument identify exactly what produced it, which is the
  failure the whole leaderboard design is built to avoid.

Expensive metrics are opt-in (`--expensive`) because they cost roughly half an
hour per model on photonics2d against seconds for the cheap suite. They are also
instrument-independent, so a board computed once survives every retrained
instrument, and the cheap columns can be rescored against a new instrument
without touching them.

Shardable for SLURM arrays: `--shard k/N` takes every Nth package, so a pool of
several hundred spreads across an array without any coordination.

Example:
    python -m engiopt.evaluation.evaluate_pool --problem-id beams2d \
        --spec beams2d/v2 --shard 0/12 --publish --out pool_beams2d.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import pandas as pd

from engiopt.evaluation import Evaluator
from engiopt.evaluation.physics_board import discover

CHEAP = [
    "mmd",
    "pca_mmd",
    "pca_vendi",
    "pca_coverage",
    "dpp",
    "pixel_vendi",
    "pixel_paired_distance",
    "novelty",
    "cond_err",
    "viol",
    "gen_seconds",
    "params",
    "lv_mmd",
    "lv_coverage",
    "lv_vendi",
    "lv_residual",
    "lv_paired_distance",
    "lv_dual_gap",
]
PHYSICS = ["iog", "cog", "fog"]


def publish_metrics(repo: str, package_path: str, payload: dict[str, Any]) -> str | None:
    """Write `metrics.json` into an existing checkpoint package.

    Adds a file beside weights that are already public; it never rewrites or
    deletes them.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "metrics.json"
        local.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=f"{package_path}/metrics.json",
                repo_id=repo,
                commit_message=f"metrics for {package_path}",
            )
        except Exception as exc:  # noqa: BLE001 - a failed upload must not lose the local row
            return f"{type(exc).__name__}: {str(exc)[:80]}"
    return None


def main() -> None:
    """Score a shard of the pool and record the results locally and on the Hub."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem-id", required=True)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--algos", nargs="*", default=None)
    ap.add_argument("--expensive", action="store_true", help="Also run iog/cog/fog (~30 min/model on photonics).")
    ap.add_argument("--publish", action="store_true", help="Write metrics.json into each HF package.")
    ap.add_argument("--shard", default=None, help="'k/N': take every Nth package starting at k.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    done = set(pd.read_csv(out)["key"]) if out.exists() else set()
    entries = discover(args.problem_id, args.algos)
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        entries = entries[k::n]
        print(f"shard {k}/{n}")
    entries = [e for e in entries if e[0] not in done]
    if args.limit:
        entries = entries[: args.limit]

    wanted = CHEAP + (PHYSICS if args.expensive else [])
    print(f"{len(entries)} packages to score ({len(done)} already done), {len(wanted)} metrics")

    from engiopt.baselines.base import DatasetGenerator
    from engiopt.utils.all_generators import BUILTIN_GENERATORS

    evaluator = Evaluator.for_problem(args.problem_id, spec=args.spec)

    for index, (key, algo, fingerprint, seed) in enumerate(entries, start=1):
        started = time.perf_counter()
        cls = BUILTIN_GENERATORS[algo]
        try:
            if isinstance(cls, type) and issubclass(cls, DatasetGenerator):
                generator = cls.from_problem(evaluator.problem, problem_id=args.problem_id, seed=seed)
            else:
                generator = cls.from_pretrained(
                    evaluator.problem,
                    problem_id=args.problem_id,
                    seed=seed,
                    model_source="hf",
                    config_fingerprint=fingerprint,
                )
            generator.seed = args.seed
            scores = evaluator.score(generator, only=wanted, include_expensive=args.expensive)
        except Exception as exc:  # noqa: BLE001 - one bad package must not end a multi-hour sweep
            print(f"  [{index}/{len(entries)}] {key}: FAILED {type(exc).__name__}: {str(exc)[:100]}", flush=True)
            continue

        clean = {k: v for k, v in scores.items() if not isinstance(v, (list, dict))}
        row = {"key": key, "algo": algo, "seed": seed, "spec": args.spec or "", **clean}
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)

        note = ""
        if args.publish and fingerprint:
            payload = {
                "problem_id": args.problem_id,
                "spec": args.spec or "",
                "sampling_seed": args.seed,
                "expensive": args.expensive,
                "metrics": clean,
            }
            repo = f"IDEALLab/engiopt-{algo.replace('_', '-')}"
            err = publish_metrics(repo, f"{args.problem_id}/cfg_{fingerprint}/seed_{seed}", payload)
            note = "  [published]" if err is None else f"  [publish failed: {err}]"

        print(f"  [{index}/{len(entries)}] {key:38s} {time.perf_counter() - started:7.1f}s{note}", flush=True)

    print(f"\npool -> {out}")


if __name__ == "__main__":
    main()
