"""What a public leaderboard accepts, and what it accepts but does not rank.

A board that anyone may write to needs two different gates, and conflating them
is the usual way these go wrong:

- **Admission.** A row that cannot be checked by anybody is not a weak result,
  it is not a result. Without a fetchable checkpoint address there is nothing to
  re-run, so the number is an assertion and the table is a comment section.
  Rows that fail admission are refused at publish time.
- **Integrity flags.** A row that *is* checkable but whose score does not mean
  what it appears to -- a retrieval system, an unconditional model on a
  conditional problem, a single cherry-picked seed. These are published and
  visible, and left out of the ranking. Hiding them would throw away the most
  useful thing the board knows.

Deleting a submission is a moderation action; declining to rank one is a
statement about what the number measures. Only the second scales.
"""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from engiopt.evaluation.spec import EvalSpec

CHECKPOINT_ADDRESS_COLUMNS = ("checkpoint_repo", "checkpoint_path", "checkpoint_revision", "checkpoint_hash")
"""What it takes to fetch the exact weights a row was scored on.

Repo and path say where to look, revision pins the commit so a later upload
cannot change what the row refers to, and the hash detects it if one does
anyway.
"""

IDENTITY_COLUMNS = ("problem_id", "algo_id", "config_fingerprint", "seed", "spec_version")
"""What it takes to know which model, and under which protocol, a row describes."""

FLAG_MEMORIZED = "memorized"
"""The batch is largely reproductions of designs the model could have seen."""

FLAG_UNVERIFIED = "unverified"
"""No runner has re-fetched these weights and reproduced these numbers."""

FLAG_IGNORES_CONDITIONS = "ignores_conditions"
"""Output did not move when the conditions did, on a problem where they vary."""

DISQUALIFYING_FLAGS = frozenset({FLAG_MEMORIZED, FLAG_IGNORES_CONDITIONS})
"""Flags that keep a row out of the ranking while leaving it on the board.

`unverified` is not among them only because ranking already requires a
verification stamp, so listing it would be saying the same thing twice.
"""


def is_flagged(flags: Any) -> bool:
    """Whether a row's `flags` cell carries anything disqualifying."""
    if _is_blank(flags):
        return False
    return bool(DISQUALIFYING_FLAGS.intersection(part.strip() for part in str(flags).split(",")))


def admission_problems(row: dict[str, Any]) -> list[str]:
    """Why this row cannot join a public board at all, or an empty list.

    Deliberately short. Everything here is about whether the row can be
    *checked*, never about whether it scored well -- a board that refuses bad
    results is a board that only publishes good ones.
    """
    problems = [f"missing {column}" for column in IDENTITY_COLUMNS if _is_blank(row.get(column))]
    missing_address = [column for column in CHECKPOINT_ADDRESS_COLUMNS if _is_blank(row.get(column))]
    if missing_address:
        problems.append(
            f"no fetchable checkpoint ({', '.join(missing_address)} not set) -- nobody, including you, "
            "could re-run this row. Train with --checkpoint-backend hf and evaluate the published package "
            "rather than a local directory."
        )
    if not _is_blank(row.get("verified")) and bool(row.get("verified")):
        problems.append(
            "verified=True was set by the submitter. Verification records that somebody else re-ran these "
            "weights; it is stamped by `python -m engiopt.verify` and cannot be self-asserted."
        )
    return problems


def integrity_flags(row: dict[str, Any], spec: EvalSpec | None = None) -> list[str]:
    """Which integrity checks this row trips.

    Row-level only. Whether an entry covers the seeds the spec requires is a
    property of the *entry*, which is several rows, so it is applied during
    ranking instead -- see `leaderboard.rank`.
    """
    flags: list[str] = []
    copy_rate = as_metric_value(row.get("copy_rate"))
    max_copy_rate = spec.max_copy_rate if spec is not None else 0.5
    if copy_rate is not None and copy_rate > max_copy_rate:
        flags.append(FLAG_MEMORIZED)
    if _declares_conditioning_it_does_not_use(row):
        flags.append(FLAG_IGNORES_CONDITIONS)
    if not bool(row.get("verified")):
        flags.append(FLAG_UNVERIFIED)
    return flags


def _declares_conditioning_it_does_not_use(row: dict[str, Any]) -> bool:
    """Whether a model claiming to be conditional produced identical output for different briefs.

    No tolerance to choose here, and deliberately so: the comparison holds the
    latent draw fixed, so a model that genuinely reads its conditions cannot
    return a bit-identical batch under a derangement of them. Exactly zero is
    the signature of the conditions never reaching the network.

    An honestly unconditional model is not flagged. The contract permits those,
    and they are handed conditions only so that this can be measured.
    """
    sensitivity = as_metric_value(row.get("cond_sens"))
    if sensitivity is None or sensitivity > 0:
        return False
    from engiopt.utils.all_generators import BUILTIN_GENERATORS

    generator = BUILTIN_GENERATORS.get(str(row.get("algo_id")))
    return generator is not None and generator.conditional


def prepare_submission(frame: pd.DataFrame, spec: EvalSpec | None = None) -> pd.DataFrame:
    """Check a table of new rows and stamp their flags, ready to publish.

    Args:
        frame: Rows as produced by `Evaluator.leaderboard`.
        spec: The spec they were scored under, for its gate thresholds.

    Returns:
        The same rows with `verified` forced False and `flags` filled in.

    Raises:
        SubmissionRejectedError: If any row fails admission. All of them are reported
            at once, since fixing them one round-trip at a time is miserable.
    """
    if frame.empty:
        return frame
    prepared = frame.copy()
    if "verified" not in prepared.columns:
        prepared["verified"] = False

    rejected: dict[int, list[str]] = {}
    for position, record in enumerate(prepared.to_dict("records")):
        problems = admission_problems(record)
        if problems:
            rejected[position] = problems
    if rejected:
        raise SubmissionRejectedError(rejected)

    # Publishing never carries a verification stamp: the point of the column is
    # that somebody other than the scorer put it there.
    prepared["verified"] = False
    prepared["verified_by"] = None
    prepared["verified_at"] = None
    prepared["flags"] = [",".join(integrity_flags(record, spec)) for record in prepared.to_dict("records")]
    return prepared


class SubmissionRejectedError(ValueError):
    """Raised when rows cannot join a public board, listing every reason found."""

    def __init__(self, problems_by_row: dict[int, list[str]]) -> None:
        self.problems_by_row = problems_by_row
        lines = [f"{len(problems_by_row)} row(s) cannot be published:"]
        lines.extend(f"  row {position}: {'; '.join(problems)}" for position, problems in sorted(problems_by_row.items()))
        super().__init__("\n".join(lines))


def as_metric_value(value: Any) -> float | None:
    """A leaderboard cell as a usable number, or None when it carries no measurement.

    NaN counts as no measurement, which matters in both places this is used: a
    metric that did not run must not be flagged as a zero, and it must not be
    reported as a claim the verifier failed to reproduce. `cond_sens` is NaN on
    an unconditional problem, and reading that as 0.0 would accuse every model
    on it of ignoring conditions it was never given.
    """
    if _is_blank(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _is_blank(value: Any) -> bool:
    """Whether a cell carries no usable value, covering None, NaN, and empty strings."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    # Arrays and other non-scalars are never blank.
    except (TypeError, ValueError):
        return False
