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
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd

from engiopt.evaluation import Evaluator

ALIASES = {
    "beams2d": ("beams2d", "beams"),
    "heatconduction2d": ("heatconduction2d", "heat"),
    "photonics2d": ("photonics2d", "photonics", "phot"),
}
"""Substrings identifying a problem in a board *filename*.

A board row is keyed `algo/fingerprint/sN`, which carries no problem id -- the
same key names a different model on every problem. Successive sweeps also named
their output three different ways, so reading a directory means matching a
substring and saying which files were taken.
"""


def problem_of(path: Path, frame: pd.DataFrame) -> str | None:
    """Which problem a board file describes, preferring its data over its name."""
    if "problem_id" in frame.columns:
        found = set(frame["problem_id"].dropna().unique())
        return str(next(iter(found))) if len(found) == 1 else None
    name = path.name.lower()
    matched = [pid for pid, aliases in ALIASES.items() if any(alias in name for alias in aliases)]
    return matched[0] if len(matched) == 1 else None


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


def published_physics(
    problem_id: str, algo: str, fingerprint: str | None, seed: int, *, required: list[str] | None = None
) -> dict[str, float] | None:
    """The physics already attached to a package on the Hub, or None.

    The counterpart to `publish`. A sweep that cannot see what it has already
    paid for will pay for it again -- and on photonics2d that is >1.7 h per
    package, so "score everything and let the CSV dedupe" is not a strategy that
    survives a second run from a fresh directory.

    Args:
        problem_id: Problem the package was evaluated on.
        algo: Model family, selecting the repo.
        fingerprint: Configuration, or None for the canonical package.
        seed: Training seed.
        required: Columns that must all be present for the row to count.
            Defaults to the mean gaps; pass `PHYSICS` to demand the medians too.

    Returns:
        The physics columns found, or None when the package has none, is
        missing them, or has no published metrics at all.
    """
    from huggingface_hub import hf_hub_download

    needed = required if required is not None else ["iog", "cog", "fog"]
    path = f"{problem_id}/" + (f"cfg_{fingerprint}/" if fingerprint else "") + f"seed_{seed}/metrics.json"
    try:
        with open(hf_hub_download(f"IDEALLab/engiopt-{algo.replace('_', '-')}", path)) as handle:
            payload = json.load(handle)
    except Exception:  # noqa: BLE001 - no published metrics yet is the normal case, not an error
        return None
    # Both shapes are in the wild; see `publish`.
    metrics = payload["metrics"] if isinstance(payload.get("metrics"), dict) else payload
    found = {m: metrics[m] for m in PHYSICS if m in metrics and metrics[m] == metrics[m]}
    return found if all(m in found for m in needed) else None


def publish(
    row: dict[str, Any],
    *,
    problem_id: str,
    algo: str,
    fingerprint: str | None,
    seed: int,
    spec: str | None,
    hf_entity: str,
    hf_repo_prefix: str,
) -> None:
    """Attach one scored row to its checkpoint package on the Hub.

    A CSV in a home directory is not where hours of optimizer time should live:
    it is overwritten, it is not versioned, and nobody downloading the checkpoint
    can see it. This puts the numbers beside the weights they describe.

    **Writes the wrapper shape `evaluate_pool` established**, not a flat dict.
    Two functions publish to `metrics.json` and they disagree:
    `checkpoint_store.publish_checkpoint_metrics` writes whatever dict it is
    handed, verbatim, while `evaluate_pool.publish_metrics` nests the columns
    under a `metrics` key beside `problem_id` / `spec` / `sampling_seed` /
    `expensive`. Every file published so far uses the nested shape, and every
    reader looks for `["metrics"]` -- so writing flat produced a file that
    uploaded cleanly and was invisible to everything that reads it.

    Merges rather than replaces, because a package's `metrics.json` may already
    hold the cheap columns from an earlier evaluation and this run only computed
    the physics.
    """
    from engiopt.evaluation.evaluate_pool import publish_metrics

    package_path = f"{problem_id}/" + (f"cfg_{fingerprint}/" if fingerprint else "") + f"seed_{seed}"
    existing = _existing_metrics(f"{hf_entity}/{hf_repo_prefix}-{algo.replace('_', '-')}", package_path)

    payload = {
        "problem_id": problem_id,
        "spec": spec or "",
        "sampling_seed": seed,
        "expensive": True,
        "metrics": {
            **existing,
            **{metric: row[metric] for metric in PHYSICS if row.get(metric) is not None},
            "algo_id": algo,
            "config_fingerprint": fingerprint,
            "problem_id": problem_id,
            "seed": seed,
        },
    }
    error = publish_metrics(f"{hf_entity}/{hf_repo_prefix}-{algo.replace('_', '-')}", package_path, payload)
    if error:
        print(f"      publish FAILED ({error}); the row is still in the CSV", flush=True)
    else:
        print(f"      published -> {package_path}/metrics.json", flush=True)


def _existing_metrics(repo: str, package_path: str) -> dict[str, Any]:
    """Columns already published for a package, from either file shape.

    Tolerates the flat shape as well as the nested one, so a package written by
    the other publisher is merged rather than silently discarded.
    """
    from huggingface_hub import hf_hub_download

    try:
        with open(hf_hub_download(repo, f"{package_path}/metrics.json", force_download=True)) as handle:
            payload = json.load(handle)
    except Exception:  # noqa: BLE001 - no published metrics yet is the normal case
        return {}
    nested = payload.get("metrics")
    if isinstance(nested, dict):
        return nested
    return {key: value for key, value in payload.items() if key not in {"spec", "sampling_seed", "expensive"}}


def publish_from(source: Path, problem_id: str, spec: str | None, hf_entity: str, hf_repo_prefix: str) -> None:
    """Push rows already scored in board CSVs, without recomputing anything.

    The backfill path. Sweeps that ran before `--publish` existed left their
    results only as CSVs; this puts them where the checkpoints are so nothing
    depends on a file in somebody's home directory.
    """
    files = sorted(source.glob("*.csv")) if source.is_dir() else [source]
    published = 0
    for path in files:
        frame = pd.read_csv(path)
        if "key" not in frame.columns or problem_of(path, frame) != problem_id:
            continue
        for row in frame.to_dict("records"):
            algo, fingerprint, seed_part = str(row["key"]).split("/")
            if not all(row.get(m) == row.get(m) for m in ("iog", "cog", "fog")):
                continue
            publish(
                row,
                problem_id=problem_id,
                algo=algo,
                fingerprint=None if fingerprint == "default" else fingerprint,
                seed=int(seed_part.removeprefix("s")),
                spec=spec,
                hf_entity=hf_entity,
                hf_repo_prefix=hf_repo_prefix,
            )
            published += 1
    print(f"published {published} rows for {problem_id}")


def select(entries: list[tuple[str, str, str | None, int]], args: Any) -> list[tuple[str, str, str | None, int]]:
    """Narrow a shard's packages to the ones actually worth scoring.

    Two filters, both about not paying twice. `--only` restricts to named
    packages, for filling gaps without re-running a pool. `--skip-published`
    drops anything whose checkpoint already carries these metrics on the Hub,
    which is the filter that makes a repeat sweep cheap: the `--out` CSV only
    protects a run that resumes in the same directory, and photonics2d is >1.7 h
    per package to rediscover that.
    """
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {e[0] for e in entries}
        entries = [entry for entry in entries if entry[0] in wanted]
        if unknown:
            print(f"  [warning] --only names {sorted(unknown)}, not in this shard's slice of the pool")

    if args.skip_published:
        required = list(PHYSICS) if args.require_medians else None
        keep = [
            entry
            for entry in entries
            if published_physics(args.problem_id, entry[1], entry[2], entry[3], required=required) is None
        ]
        print(f"  {len(entries) - len(keep)} package(s) already published on the Hub; skipping them")
        entries = keep

    return entries


def build_parser() -> argparse.ArgumentParser:
    """The command line. Lifted out of `main` so the flags can grow without it."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem-id", required=True)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--algos", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Score at most this many packages (timing probe).")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-shards", type=int, default=1, help="Split the pool this many ways for a job array.")
    ap.add_argument("--shard", type=int, default=0, help="Which shard this task scores, in [0, num_shards).")
    ap.add_argument(
        "--publish",
        action="store_true",
        help="Also write each row into its checkpoint package on the Hub, beside the weights it describes.",
    )
    ap.add_argument("--hf-entity", default="IDEALLab")
    ap.add_argument("--hf-repo-prefix", default="engiopt")
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Score only these keys (`algo/fingerprint/sN`). For filling gaps without re-running a pool.",
    )
    ap.add_argument(
        "--skip-published",
        action="store_true",
        help="Drop packages whose checkpoint already carries these metrics on the Hub.",
    )
    ap.add_argument(
        "--require-medians",
        action="store_true",
        help="With --skip-published, treat a package carrying only the means as still needing a run.",
    )
    ap.add_argument(
        "--publish-from",
        type=Path,
        default=None,
        help="Publish rows already in these board CSVs and exit, scoring nothing.",
    )
    return ap


def score_all(entries: list[tuple[str, str, str | None, int]], evaluator: Any, out: Path, args: Any) -> None:
    """Score each package, appending as it goes so a killed task loses one row.

    Every result is written to the CSV *before* it is published, so an upload
    failure costs a push and never the hours that produced the number.
    """
    from engiopt.utils.all_generators import BUILTIN_GENERATORS

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
        if args.publish:
            publish(
                row,
                problem_id=args.problem_id,
                algo=algo,
                fingerprint=fingerprint,
                seed=seed,
                spec=evaluator.spec.qualified_name if hasattr(evaluator.spec, "qualified_name") else args.spec,
                hf_entity=args.hf_entity,
                hf_repo_prefix=args.hf_repo_prefix,
            )
        elapsed = time.perf_counter() - started
        print(
            f"  [{index}/{len(entries)}] {key:36s} {elapsed:7.1f}s  " + "  ".join(f"{m}={scores.get(m)}" for m in PHYSICS),
            flush=True,
        )

    print(f"\nboard -> {out}")


def main() -> None:
    """Score each discovered package on the physics metrics, appending as it goes."""
    parser = build_parser()
    args = parser.parse_args()

    if args.publish_from:
        publish_from(args.publish_from, args.problem_id, args.spec, args.hf_entity, args.hf_repo_prefix)
        return

    if not 0 <= args.shard < args.num_shards:
        parser.error(f"--shard {args.shard} is outside [0, {args.num_shards})")

    out = Path(args.out)
    done = set(pd.read_csv(out)["key"]) if out.exists() else set()
    # Shard on position in the sorted discovery, so a task's slice is a
    # deterministic function of the pool alone. Striding rather than slicing
    # keeps the expensive families spread across tasks: `discover` sorts by key,
    # so a contiguous slice would hand one task every diffusion package.
    entries = select(
        [e for e in discover(args.problem_id, args.algos)[args.shard :: args.num_shards] if e[0] not in done], args
    )
    if args.limit:
        entries = entries[: args.limit]

    shard_note = f" [shard {args.shard}/{args.num_shards}]" if args.num_shards > 1 else ""
    print(f"{len(entries)} packages to score ({len(done)} already done){shard_note}")

    evaluator = Evaluator.for_problem(args.problem_id, spec=args.spec)
    score_all(entries, evaluator, out, args)


if __name__ == "__main__":
    main()
