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

RANK_PARTITION = ["problem_id", "spec_version"]
"""Within which group a rank of 1 means "best".

Two models are only comparable if they were scored on the same problem under the
same protocol, so ranks restart per partition rather than running across the
whole table.
"""

ENTRY_KEY = ["problem_id", "algo_id", "config_fingerprint", "spec_version"]
"""What a *ranked entry* is: one configuration of one algorithm under one spec.

`ROW_KEY` minus `seed`, because seeds are what a ranking aggregates over.
Everything else must match: averaging one configuration's score with another's,
or a score under `v1` with a score under `v2`, produces a number that describes
no model and no protocol.
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
        One row per `ENTRY_KEY` sorted best-first, with a `rank` column and the
        seed count that produced each value. Ranks restart at 1 within each
        `(problem_id, spec_version)`: a beams2d score and a photonics2d score
        measure different things, so placing them in one ordering would invent a
        comparison that does not exist.
    """
    from engiopt.evaluation.registry import METRICS

    spec = (registry or METRICS)[metric]
    if spec.higher_is_better is None:
        raise ValueError(f"Metric {metric!r} has no ranking direction; it is diagnostic only.")

    grouped = (
        frame.groupby(entry_key(frame), as_index=False)
        .agg(value=(metric, aggregate), n_seeds=(metric, "count"))
        .sort_values("value", ascending=not spec.higher_is_better)
        .reset_index(drop=True)
    )
    partition = [col for col in RANK_PARTITION if col in grouped.columns]
    grouped["rank"] = grouped.groupby(partition).cumcount() + 1 if partition else grouped.index + 1
    return grouped.rename(columns={"value": f"{metric}_{aggregate}"})


def entry_key(frame: pd.DataFrame) -> list[str]:
    """The `ENTRY_KEY` columns present in `frame`, for grouping.

    Raises:
        ValueError: If none of them are, since grouping would then silently
            pool every result in the table into one number.
    """
    key = [col for col in ENTRY_KEY if col in frame.columns]
    if not key:
        raise ValueError(f"Leaderboard is missing all of {ENTRY_KEY}; cannot tell entries apart.")
    return key


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
        A table indexed by `ENTRY_KEY`, one column of ranks per metric.
    """
    key = entry_key(frame)
    boards = []
    for metric in metrics:
        board = rank(frame, metric, registry=registry, aggregate=aggregate)
        boards.append(board.set_index(key)["rank"].rename(metric))
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
    """Download the published leaderboard, or an empty frame if none exists yet.

    Raises:
        HfHubHTTPError: On any failure other than "no board published yet".
            Treating an auth or network error as an empty board would let the
            next `push_to_hub` upload only the new rows, deleting everyone
            else's.
    """
    return _load_from_hub(repo_id, token=token)[0]


def _load_from_hub(repo_id: str, *, token: str | None = None) -> tuple[pd.DataFrame, str | None]:
    """Download the board together with the repo commit it was read at.

    The two "not found" cases are different and must stay different:

    - **No repo yet.** Nothing exists, so there is no revision either, and the
      first publish creates it.
    - **Repo exists, no `leaderboard.csv` yet.** The repo already has a head
      commit, and returning it matters: it becomes the `parent_commit` of the
      first publish, so two jobs racing to publish first cannot overwrite each
      other.

    A missing repo is also what the Hub reports for a private repo the caller
    cannot see, so that case is separated by status code rather than by
    exception type -- treating a 401 as an empty board would let the next
    publish delete everything.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub import HfApi
    from huggingface_hub.errors import EntryNotFoundError
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        revision = HfApi(token=token).repo_info(repo_id=repo_id, repo_type="dataset").sha
    except RepositoryNotFoundError as exc:
        if _status_code(exc) in {401, 403}:
            raise
        return pd.DataFrame(), None

    try:
        path = hf_hub_download(
            repo_id=repo_id, filename=LEADERBOARD_FILE, repo_type="dataset", revision=revision, token=token
        )
    # The repo is there but empty; keep its revision for the first publish.
    except EntryNotFoundError:
        return pd.DataFrame(), revision
    return pd.read_csv(path), revision


def _status_code(exc: Exception) -> int | None:
    """HTTP status behind a `huggingface_hub` error, when it carries one."""
    return getattr(getattr(exc, "response", None), "status_code", None)


def push_to_hub(
    new_rows: pd.DataFrame,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
    commit_message: str | None = None,
    max_attempts: int = 3,
) -> pd.DataFrame:
    """Merge `new_rows` into the published leaderboard and upload the result.

    The existing board is downloaded first and merged on `ROW_KEY`, so adding a
    model never requires re-running anyone else's evaluation.

    The upload is conditional on the revision the board was read at. If another
    job published in between, the commit is rejected rather than overwriting it,
    and the read-merge-upload cycle is retried against the newer board.

    Args:
        new_rows: Rows to publish, as returned by `Evaluator.leaderboard`.
        repo_id: HuggingFace dataset repo, e.g. `"IDEALLab/engiopt-leaderboard"`.
        token: HF token; falls back to the ambient login.
        private: Create the repo private if it does not exist yet.
        commit_message: Defaults to a summary of what was added.
        max_attempts: How many times to retry after losing a race.

    Returns:
        The full merged leaderboard as uploaded.

    Raises:
        RuntimeError: If every attempt lost the race to a concurrent publisher.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    models = ", ".join(sorted(new_rows["algo_id"].unique())) if "algo_id" in new_rows else "results"

    for attempt in range(max_attempts):
        existing, revision = _load_from_hub(repo_id, token=token)
        merged = merge_rows(existing, new_rows)
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / LEADERBOARD_FILE
            merged.to_csv(local, index=False)
            try:
                api.upload_file(
                    path_or_fileobj=str(local),
                    path_in_repo=LEADERBOARD_FILE,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=commit_message or f"Add {len(new_rows)} row(s): {models}",
                    parent_commit=revision,
                )
            except HfHubHTTPError as exc:
                if not _is_stale_commit(exc) or attempt == max_attempts - 1:
                    raise
                print(f"[leaderboard] {repo_id} moved under us; re-merging (attempt {attempt + 2}/{max_attempts}).")
                continue
        return merged
    raise RuntimeError(f"Could not publish to {repo_id} in {max_attempts} attempts: it kept changing underneath.")


def _is_stale_commit(exc: Exception) -> bool:
    """Whether an upload failed because `parent_commit` was no longer the head."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status == 412 or "parent_commit" in str(exc).lower()  # noqa: PLR2004 - HTTP 412 Precondition Failed


def already_evaluated(board: pd.DataFrame, **key: Any) -> bool:
    """Whether the board already holds a row for exactly this model.

    Lets a sweep skip work it has already done instead of recomputing it.

    Pass `checkpoint_hash` and the skip becomes about the *weights*, not just
    the name: re-training the same configuration and seed produces different
    weights under the same identity, and those deserve a fresh score rather
    than inheriting the old row's. A row predating the hash column has no value
    to compare, so it is treated as a match and still skipped.
    """
    if board.empty:
        return False
    mask = pd.Series(data=True, index=board.index)
    for column, value in key.items():
        if column not in board.columns:
            continue
        if column == "checkpoint_hash":
            # Older rows carry no hash; do not force them to be re-evaluated.
            mask &= board[column].isna() | (board[column] == value)
        else:
            mask &= board[column] == value
    return bool(mask.any())
