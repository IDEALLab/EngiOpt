"""One evaluation entry point for every model.

Replaces the per-model `evaluate_*.py` scripts, which differed only in how they
loaded and called their generator -- now the `Generator` contract's job.

    # score three models on one problem, cheap metrics only
    python -m engiopt.evaluate --problem-id beams2d --generators cgan_cnn_2d gan_cnn_2d vqgan

    # the whole zoo, several seeds, including simulator-backed metrics
    python -m engiopt.evaluate --problem-id beams2d --generators all --seeds 1 2 3 --include-expensive

    # what can I run?
    python -m engiopt.evaluate --list-generators --list-metrics
"""

from __future__ import annotations

import dataclasses
import sys
from typing import TYPE_CHECKING

import pandas as pd
import tyro

from engiopt.checkpoint_store import publish_checkpoint_metrics
from engiopt.core import config_path_parts
from engiopt.evaluation.evaluator import Evaluator
from engiopt.evaluation.leaderboard import already_evaluated
from engiopt.evaluation.leaderboard import append_rows
from engiopt.evaluation.leaderboard import disagreement
from engiopt.evaluation.leaderboard import load_from_hub
from engiopt.evaluation.leaderboard import push_to_hub
from engiopt.evaluation.registry import METRICS
from engiopt.evaluation.submission import FLAG_IGNORES_CONDITIONS
from engiopt.evaluation.submission import FLAG_MEMORIZED
from engiopt.evaluation.submission import FLAG_UNVERIFIED
from engiopt.evaluation.submission import integrity_flags
from engiopt.utils.all_generators import BUILTIN_GENERATORS
from engiopt.utils.all_generators import generators_for

if TYPE_CHECKING:
    from engiopt.core import Generator


@dataclasses.dataclass
class Args:
    """Command-line arguments for evaluating one or more generators."""

    problem_id: str = "beams2d"
    """EngiBench problem to evaluate on."""
    generators: tuple[str, ...] = ("all",)
    """Generator names, or `all` for every registered model that fits the problem."""
    seeds: tuple[int, ...] = (1,)
    """Checkpoint seeds to load for each generator."""
    config_fingerprints: tuple[str, ...] = ()
    """Specific hyperparameter configurations to evaluate, by fingerprint.

    Empty means the canonical default-hyperparameter checkpoint -- what a bare
    model name refers to. Pass several to score a sweep.

    Scope an entry to one model with `algo:fingerprint`. A fingerprint hashes
    one algorithm's hyperparameters, so evaluating several models against a flat
    list tries every fingerprint against every model and reports the mismatches
    as load failures. `--generators cgan_cnn_2d gan_cnn_2d --config-fingerprints
    cgan_cnn_2d:023dd1fb gan_cnn_2d:06d9a9a1` asks for exactly the two that
    exist."""
    spec: str | None = None
    """Eval spec reference, e.g. `beams2d/v1`. Defaults to `<problem_id>/v1`."""
    metrics: tuple[str, ...] = ()
    """Metric names; defaults to the spec's list."""
    include_expensive: bool = False
    """Run simulator-backed metrics (COG/IOG/FOG, feasibility). Slow."""
    output_csv: str = "leaderboard_{problem_id}.csv"
    """Where to append results locally; may include `{problem_id}`."""
    push_to: str | None = None
    """HuggingFace dataset repo to publish into, e.g. `IDEALLab/engiopt-leaderboard`.

    The published board is downloaded and merged, so publishing one model never
    recomputes or overwrites anyone else's rows."""
    skip_existing: bool = False
    """Skip generator/seed pairs the published board already has rows for."""
    attach_metrics: bool = False
    """Also write each model's scores into its own HF checkpoint package.

    Makes a checkpoint self-describing: whoever downloads it sees how it scored
    without needing the leaderboard."""
    model_source: str = "auto"
    """Checkpoint backend: `auto`, `hf`, or `local`."""
    local_model_dir: str | None = None
    """Directory holding an unpacked checkpoint package, for `--model-source local`.

    The package layout is the one `save_checkpoint_package` writes: the weight
    files plus `run_config.json` and `metadata.json`. `{algo}` and `{seed}` are
    substituted, so one directory template can serve a multi-model run."""
    hf_entity: str = "IDEALLab"
    """HF org/user holding the checkpoints."""
    hf_repo_prefix: str = "engiopt"
    """HF repo prefix for model-family repositories."""
    on_error: str = "skip"
    """`skip` past models that fail to load, or `raise` to stop."""
    show_disagreement: bool = True
    """Print each metric's ranking side by side after evaluating."""
    list_generators: bool = False
    """List registered generators and exit."""
    check_availability: bool = False
    """With `--list-generators`, also report which have published checkpoints.

    One Hub request per generator. Being registered means an adapter exists in
    this repository; being available additionally means somebody published
    weights, and only the second lets you evaluate anything."""
    list_metrics: bool = False
    """List registered metrics and exit."""


def _print_generators(args: Args) -> None:
    """Print every registered generator, and say which ones can actually be loaded.

    Registered and available are different things, and the gap is wide enough to
    mislead: an adapter is code in this repository, while a usable model also
    needs published weights. Listing all fourteen as though they were
    interchangeable sends people to `--generators all` and a wall of load
    failures that look like bugs.
    """
    print(f"{len(BUILTIN_GENERATORS)} generators registered:\n")
    published = _published_packages(args)
    for name, generator in sorted(BUILTIN_GENERATORS.items()):
        kinds = "/".join(generator.design_kinds)
        conditioning = "conditional" if generator.conditional else "unconditional"
        availability = "" if published is None else f"  {_availability_label(published.get(name))}"
        print(f"  {name:<20} {kinds:<10} {conditioning:<15}{availability}")
    if published is not None:
        missing = sorted(name for name, count in published.items() if not count)
        if missing:
            print(
                f"\n{len(missing)} generator(s) have no published {args.problem_id} checkpoint under "
                f"{args.hf_entity}: {', '.join(missing)}.\n"
                "They can be trained and evaluated, but `--generators all` will report them as load "
                "failures until someone publishes weights. W&B is not a checkpoint source."
            )
    else:
        print(
            f"\nPass --check-availability to also query {args.hf_entity} for which of these have "
            f"published {args.problem_id} checkpoints."
        )


def _published_packages(args: Args) -> dict[str, int] | None:
    """How many packages each generator has for this problem, or None if not asked.

    Behind a flag because it is one Hub request per generator, and `-h`-adjacent
    commands should not depend on the network.
    """
    if not args.check_availability:
        return None
    from engiopt.checkpoint_store import build_hf_repo_id
    from engiopt.checkpoint_store import list_packages

    counts: dict[str, int] = {}
    for name in BUILTIN_GENERATORS:
        repo = build_hf_repo_id(args.hf_entity, args.hf_repo_prefix, name)
        try:
            counts[name] = len(list_packages(repo, args.problem_id))
        # A repo that does not exist is the answer, not an error.
        except Exception:  # noqa: BLE001
            counts[name] = 0
    return counts


def _availability_label(count: int | None) -> str:
    """One-word availability, with the package count when there is one."""
    if not count:
        return "no published checkpoints"
    return f"{count} package(s)"


def _print_metrics() -> None:
    """Print every registered metric grouped by the question it answers."""
    print(f"{len(METRICS)} metrics registered:\n")
    for family in sorted({spec.family for spec in METRICS.values()}):
        print(f"  [{family}]")
        for spec in METRICS.select(family=family):
            direction = {True: "higher better", False: "lower better", None: "diagnostic"}[spec.higher_is_better]
            print(f"    {spec.name:<10} {spec.cost:<10} {direction:<14} {spec.description}")


def _resolve_generator_names(requested: tuple[str, ...], problem_id: str) -> list[str]:
    """Expand `all` to every registered generator compatible with the problem."""
    if "all" not in requested:
        return list(requested)
    from engibench.utils.all_problems import BUILTIN_PROBLEMS

    return sorted(generators_for(BUILTIN_PROBLEMS[problem_id]()))


def _fingerprints_for(requested: tuple[str, ...], algo: str) -> tuple[str | None, ...]:
    """Which configurations to try for one algorithm.

    A fingerprint hashes one algorithm's hyperparameters, so it is meaningless
    against a different algorithm: evaluating two models against the union of
    both their fingerprints asks for packages that were never going to exist and
    buries the run in load failures that are not failures. `algo:fingerprint`
    scopes an entry to its owner; a bare fingerprint still applies to every
    algorithm, which is what a single-model run wants.

    A model that no entry is scoped to falls back to its canonical checkpoint
    rather than to nothing. Returning an empty tuple would drop it before the
    load is even attempted -- no row, no error, no message -- so
    `--generators gan_cnn_2d vqgan --config-fingerprints gan_cnn_2d:6293adb3`
    would quietly score one model and never mention the other. A leaderboard
    silently missing an entrant is worse than one reporting a load failure.

    Args:
        requested: Raw `--config-fingerprints` values.
        algo: The algorithm being loaded.

    Returns:
        Fingerprints to try, or `(None,)` meaning the canonical checkpoint.
    """
    if not requested:
        return (None,)
    selected = [
        entry.split(":", 1)[1] if ":" in entry else entry
        for entry in requested
        if ":" not in entry or entry.split(":", 1)[0] == algo
    ]
    return tuple(selected) or (None,)


def _load_generators(args: Args, evaluator: Evaluator) -> list[Generator]:
    """Load every requested generator/seed pair, reporting those that fail."""
    generators: list[Generator] = []
    for name in _resolve_generator_names(args.generators, args.problem_id):
        generator_cls = BUILTIN_GENERATORS[name]
        for seed in args.seeds:
            for fingerprint in _fingerprints_for(args.config_fingerprints, name):
                try:
                    generators.append(
                        generator_cls.from_pretrained(
                            evaluator.problem,
                            problem_id=args.problem_id,
                            seed=seed,
                            device=evaluator.device,
                            model_source=args.model_source,  # type: ignore[arg-type]
                            hf_entity=args.hf_entity,
                            hf_repo_prefix=args.hf_repo_prefix,
                            local_model_dir=_local_dir_for(args, name, seed),
                            config_fingerprint=fingerprint,
                        )
                    )
                # A missing checkpoint should not abort a whole sweep.
                except Exception as exc:  # noqa: PERF203
                    if args.on_error != "skip":
                        raise
                    label = f"{name} seed {seed}" + (f" cfg {fingerprint}" if fingerprint else "")
                    print(f"  could not load {label}: {exc}")
    return generators


def _local_dir_for(args: Args, algo: str, seed: int) -> str | None:
    """Fill `{algo}` / `{seed}` into `--local-model-dir`, so one template serves a run."""
    if args.local_model_dir is None:
        return None
    return args.local_model_dir.format(algo=algo, seed=seed, problem_id=args.problem_id)


def _drop_already_published(args: Args, evaluator: Evaluator, generators: list[Generator]) -> list[Generator]:
    """Drop generators the published board already holds rows for, under `--skip-existing`."""
    if not (args.push_to and args.skip_existing):
        return generators
    published = load_from_hub(args.push_to)
    remaining = [
        generator
        for generator in generators
        if not already_evaluated(
            published,
            problem_id=args.problem_id,
            algo_id=generator.algo_id,
            config_fingerprint=generator.config_fingerprint,
            seed=generator.seed,
            spec_version=evaluator.spec.version,
            checkpoint_hash=generator.checkpoint_hash,
        )
    ]
    print(f"{len(remaining)} generator(s) left after skipping already-published rows.")
    return remaining


def _attach_metrics_to_checkpoints(args: Args, board: pd.DataFrame) -> None:
    """Write each row's scores into the checkpoint package it describes."""
    for row in board.to_dict("records"):
        fingerprint = row.get("config_fingerprint")
        try:
            path = publish_checkpoint_metrics(
                hf_entity=args.hf_entity,
                hf_repo_prefix=args.hf_repo_prefix,
                problem_id=row["problem_id"],
                algo=row["algo_id"],
                seed=int(row["seed"]),
                metrics={k: v for k, v in row.items() if pd.notna(v)},
                extra_path_parts=config_path_parts(fingerprint),
            )
            print(f"  metrics -> {row['algo_id']}: {path}")
        except Exception as exc:  # noqa: BLE001 - one failure must not lose the rest
            print(f"  could not attach metrics for {row['algo_id']}: {exc}")


def main(args: Args) -> int:
    """Evaluate the requested generators and append the results to a CSV.

    Returns:
        A process exit status: non-zero when nothing could be evaluated, so a
        batch run that loaded no checkpoints does not look like a clean sweep.
        A partially successful run still succeeds -- one bad checkpoint must not
        discard the rest of the results.
    """
    if args.list_generators:
        _print_generators(args)
    if args.list_metrics:
        _print_metrics()
    if args.list_generators or args.list_metrics:
        return 0

    spec = args.spec or f"{args.problem_id}/v1"
    evaluator = Evaluator.for_problem(args.problem_id, spec=spec)
    print(f"Problem {args.problem_id} | spec {evaluator.spec.version} | n={evaluator.spec.n_samples}")

    generators = _load_generators(args, evaluator)
    loaded = len(generators)
    generators = _drop_already_published(args, evaluator, generators)
    if not generators:
        # Having skipped everything already on the board is a finished job;
        # having loaded nothing at all is a failed one.
        if loaded:
            print("Everything requested is already published; nothing to evaluate.")
            return 0
        print("No generators could be loaded; nothing was evaluated.")
        return 1

    board = evaluator.leaderboard(
        generators,
        only=list(args.metrics) or None,
        include_expensive=args.include_expensive,
        on_error=args.on_error,
    )
    if board.empty:
        print("Every generator failed to evaluate; no rows produced.")
        return 1
    destination = append_rows(board, args.output_csv.format(problem_id=args.problem_id))
    print(f"\n{board.to_string(index=False)}\n")
    print(f"Wrote {len(board)} rows to {destination}")

    _publish(args, evaluator, board)
    return 0


def _publish(args: Args, evaluator: Evaluator, board: pd.DataFrame) -> None:
    """Push the results wherever the flags asked, then print the ranking comparison."""
    if args.push_to:
        merged = push_to_hub(board, args.push_to, eval_spec=evaluator.spec)
        print(f"Published {len(board)} row(s) to {args.push_to}; board now holds {len(merged)} rows.")
        print(
            "These rows are unverified. They will not be ranked until a runner re-fetches the "
            f"checkpoints and reproduces the scores: `python -m engiopt.verify --board {args.push_to}`."
        )

    if args.attach_metrics:
        _attach_metrics_to_checkpoints(args, board)

    _print_integrity_warnings(board)

    if args.show_disagreement and len(board) > 1:
        ranked = [m for m in board.columns if m in METRICS and METRICS[m].higher_is_better is not None]
        if len(ranked) > 1:
            # Freshly computed rows are unverified by construction, so the
            # eligibility filter would empty this table. It is a local preview
            # of one run, not the public ordering.
            print("\nRankings by metric (1 = best) -- where these disagree is the interesting part:\n")
            print(disagreement(board, ranked, eligible_only=False).to_string())


def _print_integrity_warnings(board: pd.DataFrame) -> None:
    """Say so, locally and immediately, when a row will not be rankable.

    Better here than on the board: the submitter finds out while they can still
    do something about it, rather than after publishing and wondering why their
    model never appears in the ordering.
    """
    for row in board.to_dict("records"):
        flags = [flag for flag in integrity_flags(row) if flag != FLAG_UNVERIFIED]
        if not flags:
            continue
        label = f"{row.get('algo_id')} seed {row.get('seed')}"
        if FLAG_MEMORIZED in flags:
            print(
                f"\n  [{label}] copy_rate={row.get('copy_rate'):.2f}: most of this batch reproduces designs "
                "from the dataset rather than generating them. Its distribution and performance scores "
                "measure retrieval, and it will be published but not ranked."
            )
        if FLAG_IGNORES_CONDITIONS in flags:
            print(
                f"\n  [{label}] cond_sens=0: output did not change at all when the conditions were shuffled, "
                "though this model declares itself conditional. Its conditions are most likely not reaching "
                "the network. It will be published but not ranked."
            )


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
