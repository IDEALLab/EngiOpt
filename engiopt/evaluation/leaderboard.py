"""Persisting and publishing leaderboard results.

Results are rows keyed by `ROW_KEY`. Locally that is a CSV; published, it is a
HuggingFace dataset repo holding the same CSV, so a local sweep and the public
board are literally the same table.

Everything here merges rather than replaces. Adding one model means evaluating
one model and pushing its rows -- the existing board is downloaded, the new rows
are merged on `ROW_KEY`, and the result is uploaded. Nothing else is recomputed.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, TYPE_CHECKING

import pandas as pd

from engiopt.evaluation.evaluator import order_columns

if TYPE_CHECKING:
    from engiopt.evaluation.registry import MetricRegistry

ROW_KEY = ["problem_id", "algo_id", "config_fingerprint", "seed", "spec_version"]
"""Columns that uniquely identify a leaderboard row.

`config_fingerprint` is what separates two hyperparameter settings of the same
algorithm. Without it a 50-config sweep would collapse onto one row per seed,
silently keeping only whichever ran last.
"""

LEADERBOARD_FILE = "leaderboard.csv"
"""Filename used inside a published HuggingFace dataset repo."""


def append_rows(frame: pd.DataFrame, path: str | Path, *, deduplicate: bool = True) -> Path:
    """Append rows to a CSV leaderboard, keeping the newest row per key.

    Args:
        frame: New rows.
        path: Destination CSV. Created with a header if absent.
        deduplicate: Keep only the most recent row per `ROW_KEY`. Re-evaluating
            a model then supersedes its old row instead of duplicating it.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(destination) if destination.exists() else pd.DataFrame()
    combined = merge_rows(existing, frame) if deduplicate else order_columns(pd.concat([existing, frame]))
    combined.to_csv(destination, index=False)
    return destination


def rank(
    frame: pd.DataFrame,
    metric: str,
    *,
    registry: MetricRegistry | None = None,
    aggregate: str = "median",
) -> pd.DataFrame:
    """Rank models by one metric, aggregating across seeds.

    Args:
        frame: Leaderboard rows.
        metric: Column to rank by.
        registry: Registry used to look up the metric's direction.
        aggregate: How to combine seeds. Defaults to `median`, because
            simulation-backed objectives are heavy-tailed and a single blown-up
            run otherwise decides the ranking.

    Returns:
        One row per `(problem_id, algo_id)` sorted best-first, with a `rank`
        column and the seed count that produced each value.
    """
    from engiopt.evaluation.registry import METRICS

    spec = (registry or METRICS)[metric]
    if spec.higher_is_better is None:
        raise ValueError(f"Metric {metric!r} has no ranking direction; it is diagnostic only.")

    grouped = (
        frame.groupby(["problem_id", "algo_id"], as_index=False)
        .agg(value=(metric, aggregate), n_seeds=(metric, "count"))
        .sort_values("value", ascending=not spec.higher_is_better)
        .reset_index(drop=True)
    )
    grouped["rank"] = grouped.index + 1
    return grouped.rename(columns={"value": f"{metric}_{aggregate}"})


def disagreement(
    frame: pd.DataFrame,
    metrics: list[str],
    *,
    registry: MetricRegistry | None = None,
    aggregate: str = "median",
) -> pd.DataFrame:
    """Show each metric's ranking side by side, to expose where they disagree.

    Whether metrics agree on the winner is the question a leaderboard is
    supposed to answer, so this is a first-class view rather than a notebook
    one-off.

    Returns:
        A table indexed by model, one column of ranks per metric.
    """
    boards = []
    for metric in metrics:
        board = rank(frame, metric, registry=registry, aggregate=aggregate)
        boards.append(board.set_index(["problem_id", "algo_id"])["rank"].rename(metric))
    return pd.concat(boards, axis=1).sort_values(metrics[0])


def merge_rows(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Combine two result tables, letting newer rows supersede older ones.

    Rows are matched on `ROW_KEY`, so re-evaluating a checkpoint replaces its
    entry while a different hyperparameter setting, seed, or spec version adds
    one.
    """
    combined = pd.concat([existing, new_rows], ignore_index=True) if len(existing) else new_rows
    key = [col for col in ROW_KEY if col in combined.columns]
    if key:
        combined = combined.drop_duplicates(subset=key, keep="last")
    return order_columns(combined.reset_index(drop=True))


def load_from_hub(repo_id: str, *, token: str | None = None) -> pd.DataFrame:
    """Download the published leaderboard, or an empty frame if none exists yet."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id=repo_id, filename=LEADERBOARD_FILE, repo_type="dataset", token=token)
    except Exception:  # noqa: BLE001 - a missing or private board is not an error here
        return pd.DataFrame()
    return pd.read_csv(path)


def push_to_hub(
    new_rows: pd.DataFrame,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
    commit_message: str | None = None,
) -> pd.DataFrame:
    """Merge `new_rows` into the published leaderboard and upload the result.

    The existing board is downloaded first and merged on `ROW_KEY`, so adding a
    model never requires re-running anyone else's evaluation.

    Args:
        new_rows: Rows to publish, as returned by `Evaluator.leaderboard`.
        repo_id: HuggingFace dataset repo, e.g. `"IDEALLab/engiopt-leaderboard"`.
        token: HF token; falls back to the ambient login.
        private: Create the repo private if it does not exist yet.
        commit_message: Defaults to a summary of what was added.

    Returns:
        The full merged leaderboard as uploaded.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    merged = merge_rows(load_from_hub(repo_id, token=token), new_rows)

    models = ", ".join(sorted(new_rows["algo_id"].unique())) if "algo_id" in new_rows else "results"
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / LEADERBOARD_FILE
        merged.to_csv(local, index=False)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=LEADERBOARD_FILE,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message or f"Add {len(new_rows)} row(s): {models}",
        )
    return merged


def already_evaluated(board: pd.DataFrame, **key: Any) -> bool:
    """Whether the board already holds a row for the given `ROW_KEY` values.

    Lets a sweep skip work it has already done instead of recomputing it.
    """
    if board.empty:
        return False
    mask = pd.Series(data=True, index=board.index)
    for column, value in key.items():
        if column in board.columns:
            mask &= board[column] == value
    return bool(mask.any())
