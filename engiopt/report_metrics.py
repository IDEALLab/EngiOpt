"""Aggregate evaluation shards into paper-ready CSVs and optional W&B tables."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import tyro

from engiopt.reporting import DEFAULT_METRICS
from engiopt.reporting import add_display_name_column
from engiopt.reporting import load_metric_shards
from engiopt.reporting import summarize_metrics
from engiopt.reporting import upload_report_to_wandb


@dataclasses.dataclass
class Args:
    """Arguments for summarizing evaluation shards."""

    shard_dir: str
    """Directory containing evaluation shard CSVs."""
    output_dir: str
    """Directory where merged and summary CSVs should be written."""
    problem_id: str | None = None
    """Optional problem filter such as beams2d."""
    latest_only: bool = True
    """Use only the latest row from each shard. Recommended for rerun-safe summaries."""
    metrics: tuple[str, ...] = DEFAULT_METRICS
    """Metrics to aggregate."""
    wandb_project: str = "engiopt"
    """Weights & Biases project for summary uploads."""
    wandb_entity: str | None = None
    """Weights & Biases entity for summary uploads."""
    upload_wandb: bool = False
    """Upload the raw and summary tables to Weights & Biases."""
    run_name: str | None = None
    """Optional W&B run name for the summary upload."""
    artifact_name: str | None = None
    """Optional artifact name for the uploaded evaluation results."""


if __name__ == "__main__":
    args = tyro.cli(Args)

    raw_df = load_metric_shards(
        shard_dir=args.shard_dir,
        problem_id=args.problem_id,
        latest_only=args.latest_only,
    )
    summary_long_df, summary_wide_df = summarize_metrics(raw_df, metrics=args.metrics)
    raw_df = add_display_name_column(raw_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problem_slug = args.problem_id or "all_problems"
    freshness_slug = "latest" if args.latest_only else "all_rows"
    raw_csv_path = output_dir / f"{problem_slug}_{freshness_slug}_raw_metrics.csv"
    summary_long_csv_path = output_dir / f"{problem_slug}_{freshness_slug}_summary_long.csv"
    summary_wide_csv_path = output_dir / f"{problem_slug}_{freshness_slug}_summary_wide.csv"

    raw_df.to_csv(raw_csv_path, index=False)
    summary_long_df.to_csv(summary_long_csv_path, index=False)
    summary_wide_df.to_csv(summary_wide_csv_path, index=False)

    print(f"Wrote raw metrics to {raw_csv_path}")
    print(f"Wrote long summary to {summary_long_csv_path}")
    print(f"Wrote wide summary to {summary_wide_csv_path}")

    if args.upload_wandb:
        run_name = args.run_name or f"{problem_slug}_{freshness_slug}_evaluation_summary"
        artifact_name = args.artifact_name or f"{problem_slug}_{freshness_slug}_evaluation_summary"
        artifact_ref = upload_report_to_wandb(
            raw_df=raw_df,
            summary_long_df=summary_long_df,
            summary_wide_df=summary_wide_df,
            raw_csv_path=raw_csv_path,
            summary_long_csv_path=summary_long_csv_path,
            summary_wide_csv_path=summary_wide_csv_path,
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=run_name,
            artifact_name=artifact_name,
        )
        print(f"Uploaded W&B artifact {artifact_ref}")
