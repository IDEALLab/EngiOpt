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
    model name refers to. Pass several to score a sweep."""
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
    list_metrics: bool = False
    """List registered metrics and exit."""


def _print_generators() -> None:
    """Print every registered generator, including any that failed to import."""
    print(f"{len(BUILTIN_GENERATORS)} generators registered:\n")
    for name, generator in sorted(BUILTIN_GENERATORS.items()):
        kinds = "/".join(generator.design_kinds)
        conditioning = "conditional" if generator.conditional else "unconditional"
        print(f"  {name:<20} {kinds:<10} {conditioning}")


def _print_metrics() -> None:
    """Print every registered metric grouped by the question it answers.

    Metrics needing something the spec must supply are marked, so the ones that
    are not simply available are visible before a run rather than after it.
    """
    width = max(len(spec.name) for spec in METRICS.values())
    needs_setup = [spec for spec in METRICS.values() if spec.requires]

    print(f"{len(METRICS)} metrics registered:\n")
    for family in sorted({spec.family for spec in METRICS.values()}):
        print(f"  [{family}]")
        for spec in METRICS.select(family=family):
            direction = {True: "higher", False: "lower", None: "diagnostic"}[spec.higher_is_better]
            mark = "*" if spec.requires else " "
            print(f"   {mark}{spec.name:<{width}}  {spec.cost:<9} {direction:<10} {spec.description}")

    if needs_setup:
        print(
            f"\n  * needs an eval spec that pins a latent instrument ({len(needs_setup)} of {len(METRICS)}). "
            "\n    Everything unmarked runs on any problem with no extra setup."
        )


def _resolve_generator_names(requested: tuple[str, ...], problem_id: str) -> list[str]:
    """Expand `all` to every registered generator compatible with the problem."""
    if "all" not in requested:
        return list(requested)
    from engibench.utils.all_problems import BUILTIN_PROBLEMS

    return sorted(generators_for(BUILTIN_PROBLEMS[problem_id]()))


def _load_generators(args: Args, evaluator: Evaluator) -> list[Generator]:
    """Load every requested generator/seed pair, reporting those that fail."""
    generators: list[Generator] = []
    fingerprints: tuple[str | None, ...] = args.config_fingerprints or (None,)
    for name in _resolve_generator_names(args.generators, args.problem_id):
        generator_cls = BUILTIN_GENERATORS[name]
        for seed in args.seeds:
            for fingerprint in fingerprints:
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
        _print_generators()
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

    _publish(args, board)
    return 0


def _publish(args: Args, board: pd.DataFrame) -> None:
    """Push the results wherever the flags asked, then print the ranking comparison."""
    if args.push_to:
        merged = push_to_hub(board, args.push_to)
        print(f"Published {len(board)} row(s) to {args.push_to}; board now holds {len(merged)} rows.")

    if args.attach_metrics:
        _attach_metrics_to_checkpoints(args, board)

    if args.show_disagreement and len(board) > 1:
        ranked = [m for m in board.columns if m in METRICS and METRICS[m].higher_is_better is not None]
        if len(ranked) > 1:
            print("\nRankings by metric (1 = best) -- where these disagree is the interesting part:\n")
            print(disagreement(board, ranked).to_string())


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
