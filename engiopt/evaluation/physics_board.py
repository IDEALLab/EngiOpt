r"""Score generators on the simulator-backed metrics, restartably.

The paper's stated question is whether a cheap metric reproduces the
model-selection decision a designer would make after running simulation-based
COG for hours. Answering it needs the expensive side of that comparison, and so
far it exists only for beams2d -- the one problem the R^2(conditions ->
performance) diagnostic calls a negative control. This produces the same board
for any problem.

**The expensive columns are instrument-independent.** IOG, COG and FOG depend on
the generated designs and the optimizer, not on any autoencoder, so a board
computed once stays valid while latent instruments are retrained and the cheap
columns are rescored against them. That is why this writes the physics columns
alone and leaves the cheap ones to `board_rescore`.

Restartable by design: one row is appended per model as it finishes, and a
rerun skips keys already present. A simulator sweep that dies at hour two must
not start over.

Example:
    python -m engiopt.evaluation.physics_board --problem-id photonics2d \
        --limit 2 --out photonics_physics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from engiopt.evaluation import Evaluator

PHYSICS = ["iog", "cog", "fog", "iog_median", "cog_median", "fog_median"]
"""Mean and median optimality gaps. The per-design gap is unbounded above, so a
mean over ~50 samples is set by its worst member -- on the beams2d board a model
reports mean IOG 1.5e8 while finishing at FOG -2.2, which is one unrecoverable
starting design rather than a worse model. Rank correlations are unaffected;
any statement about magnitude needs the medians."""


def discover(problem_id: str, algos: list[str] | None) -> list[tuple[str, str, str | None, int]]:
    """Published packages for a problem as (key, algo, config_fingerprint, seed)."""
    from huggingface_hub import HfApi

    from engiopt.utils.all_generators import BUILTIN_GENERATORS

    api = HfApi()
    found: list[tuple[str, str, str | None, int]] = []
    for algo in algos or sorted(BUILTIN_GENERATORS):
        repo = f"IDEALLab/engiopt-{algo.replace('_', '-')}"
        try:
            files = api.list_repo_files(repo)
        except Exception as exc:  # noqa: BLE001 - a family with no repo for this problem is simply absent
            print(f"  no repo for {algo}: {type(exc).__name__}")
            continue
        seen = set()
        for f in files:
            parts = f.split("/")
            if len(parts) < 3 or parts[0] != problem_id:  # noqa: PLR2004
                continue
            cfg, seed_dir = parts[1], parts[2]
            if not seed_dir.startswith("seed_"):
                continue
            fingerprint = cfg.removeprefix("cfg_") if cfg.startswith("cfg_") else None
            seed = int(seed_dir.removeprefix("seed_"))
            key = f"{algo}/{fingerprint or 'default'}/s{seed}"
            if key in seen:
                continue
            seen.add(key)
            found.append((key, algo, fingerprint, seed))
    return sorted(found)


def main() -> None:
    """Score each discovered package on the physics metrics, appending as it goes."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem-id", required=True)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--algos", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Score at most this many packages (timing probe).")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    done = set(pd.read_csv(out)["key"]) if out.exists() else set()
    entries = [e for e in discover(args.problem_id, args.algos) if e[0] not in done]
    if args.limit:
        entries = entries[: args.limit]
    print(f"{len(entries)} packages to score ({len(done)} already done)")

    from engiopt.utils.all_generators import BUILTIN_GENERATORS

    evaluator = Evaluator.for_problem(args.problem_id, spec=args.spec)

    for index, (key, algo, fingerprint, seed) in enumerate(entries, start=1):
        started = time.perf_counter()
        try:
            generator = BUILTIN_GENERATORS[algo].from_pretrained(
                evaluator.problem,
                problem_id=args.problem_id,
                seed=seed,
                model_source="hf",
                config_fingerprint=fingerprint,
            )
            generator.seed = args.seed
            scores = evaluator.score(generator, only=PHYSICS, include_expensive=True)
        except Exception as exc:  # noqa: BLE001 - one bad package must not end a multi-hour sweep
            print(f"  [{index}/{len(entries)}] {key}: FAILED {type(exc).__name__}: {str(exc)[:110]}", flush=True)
            continue

        row = {"key": key, "algo": algo, "seed": seed, **{m: scores.get(m) for m in PHYSICS}}
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
        elapsed = time.perf_counter() - started
        print(
            f"  [{index}/{len(entries)}] {key:36s} {elapsed:7.1f}s  " + "  ".join(f"{m}={scores.get(m)}" for m in PHYSICS),
            flush=True,
        )

    print(f"\nboard -> {out}")


if __name__ == "__main__":
    main()
