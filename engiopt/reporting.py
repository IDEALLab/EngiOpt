"""Utilities for writing and summarizing evaluation metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import wandb

DEFAULT_METRICS = ("cog", "mmd", "dpp", "viol", "iog", "fog")


def build_display_name(row: Mapping[str, Any]) -> str:
    """Build a readable label for report tables."""
    parts: list[str] = []

    def append_int_part(label: str, value: Any) -> None:
        if value is not None and not pd.isna(value):
            parts.append(f"{label}={int(value)}")

    problem_id = row.get("problem_id")
    if pd.notna(problem_id):
        parts.append(str(problem_id))

    model_id = row.get("model_id")
    if pd.notna(model_id):
        parts.append(str(model_id))

    seed = row.get("seed")
    append_int_part("seed", seed)

    rank = row.get("selection_rank")
    if pd.isna(rank):
        rank = row.get("rank")
    append_int_part("rank", rank)

    epoch = row.get("selection_candidate_epoch")
    if pd.isna(epoch):
        epoch = row.get("epoch")
    append_int_part("epoch", epoch)

    method = row.get("method")
    if pd.notna(method):
        parts.append(f"solver={method}")

    integration_steps = row.get("integration_steps")
    append_int_part("steps", integration_steps)

    return " | ".join(parts)


def add_display_name_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the frame with a leading display_name column."""
    if "display_name" in frame.columns:
        return frame.copy()
    result = frame.copy()
    result.insert(0, "display_name", result.apply(build_display_name, axis=1))
    return result


def write_metrics_csv(records: list[dict], output_path: str, append_output: bool = False) -> Path:
    """Write evaluation rows to CSV, overwriting by default."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    if append_output and output.exists():
        df.to_csv(output, mode="a", header=False, index=False)
    else:
        df.to_csv(output, mode="w", header=True, index=False)

    return output


def load_metric_shards(
    shard_dir: str,
    problem_id: str | None = None,
    latest_only: bool = True,
) -> pd.DataFrame:
    """Load metric shard CSVs from a directory."""
    base = Path(shard_dir)
    if not base.exists():
        raise FileNotFoundError(f"Metric shard directory not found: {base}")

    pattern = "metrics_*.csv" if problem_id is None else f"metrics_*_{problem_id}_seed*.csv"
    frames: list[pd.DataFrame] = []

    for path in sorted(base.glob(pattern)):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        if latest_only:
            frame = frame.tail(1).copy()
        else:
            frame = frame.copy()
        frame["source_file"] = path.name
        frames.append(frame)

    if not frames:
        raise ValueError(f"No metric shards matched pattern '{pattern}' in {base}")

    return pd.concat(frames, ignore_index=True, sort=False)


def summarize_metrics(
    raw_df: pd.DataFrame,
    metrics: Iterable[str] = DEFAULT_METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long-form and wide-form metric summaries."""
    metric_list = [metric for metric in metrics if metric in raw_df.columns]
    if not metric_list:
        raise ValueError("No requested metrics found in raw results")

    group_cols = [col for col in ("problem_id", "model_id", "integration_steps") if col in raw_df.columns]
    if not group_cols:
        raise ValueError("No grouping columns found in raw results")

    long_df = (
        raw_df.melt(
            id_vars=[
                col
                for col in ("problem_id", "model_id", "integration_steps", "seed", "source_file")
                if col in raw_df.columns
            ],
            value_vars=metric_list,
            var_name="metric",
            value_name="value",
        )
        .groupby(group_cols + ["metric"], as_index=False)
        .agg(
            mean=("value", "mean"),
            median=("value", "median"),
            std=("value", "std"),
            n_seeds=("value", "count"),
        )
    )

    wide_df = long_df.pivot(index=group_cols, columns="metric", values=["mean", "median", "std"])
    wide_df.columns = [f"{metric}_{stat}" for stat, metric in wide_df.columns]
    wide_df = wide_df.reset_index()
    return long_df, wide_df


def upload_report_to_wandb(
    raw_df: pd.DataFrame,
    summary_long_df: pd.DataFrame,
    summary_wide_df: pd.DataFrame,
    raw_csv_path: Path,
    summary_long_csv_path: Path,
    summary_wide_csv_path: Path,
    project: str,
    entity: str | None,
    run_name: str,
    artifact_name: str,
    job_type: str = "evaluation-summary",
) -> str:
    """Upload raw and summary results to Weights & Biases."""
    raw_table_df = add_display_name_column(raw_df)
    summary_long_table_df = add_display_name_column(summary_long_df)
    summary_wide_table_df = add_display_name_column(summary_wide_df)

    run = wandb.init(
        project=project,
        entity=entity,
        job_type=job_type,
        name=run_name,
        config={
            "raw_csv": str(raw_csv_path),
            "summary_long_csv": str(summary_long_csv_path),
            "summary_wide_csv": str(summary_wide_csv_path),
        },
    )
    if run is None:
        raise RuntimeError("Failed to initialize Weights & Biases run")

    run.log(
        {
            "evaluation/raw_table": wandb.Table(dataframe=raw_table_df),
            "evaluation/summary_table": wandb.Table(dataframe=summary_wide_table_df),
            "evaluation/summary_long_table": wandb.Table(dataframe=summary_long_table_df),
        }
    )

    scalar_payload: dict[str, float] = {}
    for row in summary_long_df.itertuples(index=False):
        problem_id = str(getattr(row, "problem_id"))
        model_id = str(getattr(row, "model_id"))
        integration_steps = getattr(row, "integration_steps", None)
        step_suffix = (
            f"/steps{int(integration_steps)}"
            if integration_steps is not None and pd.notna(integration_steps)
            else ""
        )
        metric = str(getattr(row, "metric"))
        mean = getattr(row, "mean")
        median = getattr(row, "median")
        std = getattr(row, "std")
        if pd.notna(mean):
            scalar_payload[f"evaluation/{problem_id}/{model_id}{step_suffix}/{metric}_mean"] = float(mean)
        if pd.notna(median):
            scalar_payload[f"evaluation/{problem_id}/{model_id}{step_suffix}/{metric}_median"] = float(median)
        if pd.notna(std):
            scalar_payload[f"evaluation/{problem_id}/{model_id}{step_suffix}/{metric}_std"] = float(std)
    if scalar_payload:
        run.log(scalar_payload)

    artifact = wandb.Artifact(name=artifact_name, type="evaluation-results")
    artifact.add_file(str(raw_csv_path), name=raw_csv_path.name)
    artifact.add_file(str(summary_long_csv_path), name=summary_long_csv_path.name)
    artifact.add_file(str(summary_wide_csv_path), name=summary_wide_csv_path.name)
    run.log_artifact(artifact)
    artifact.wait()
    run.finish()
    return artifact.name
