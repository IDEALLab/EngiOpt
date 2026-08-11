"""Re-running a published row from its checkpoint address.

The board cannot trust the numbers it is given. Whoever computed a row also
chose what to report, and no amount of schema design changes that -- the defence
against a fabricated score is not a stricter column, it is running the model
again.

So a row is a *claim plus an address*. Verification follows the address, fetches
the weights at the revision the row named, and scores them itself. What it
publishes is **its own numbers**, not a verdict on the submitted ones. That
distinction matters more than it looks: sampling from the same seed on different
hardware genuinely produces different designs, because CUDA and CPU draw
different values from the same generator state. A pass/fail comparison would
therefore have to carry a tolerance loose enough to wave through real fudging.
Re-scoring instead makes the runner's result the result, and demotes the
submitted numbers to a claim that is reported as corroborated or not.

Nothing here is privileged. The runner is simply the first auditor, and because
every row carries a full address, anyone can run the same command and get the
same answer::

    python -m engiopt.verify --board IDEALLab/engiopt-leaderboard --problem-id beams2d
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import datetime as dt
from typing import Any, Literal, TYPE_CHECKING

import pandas as pd

from engiopt.checkpoint_store import resolve_checkpoint_reference
from engiopt.core import condition_keys_for
from engiopt.core import condition_stats_for
from engiopt.core import pick_device
from engiopt.evaluation.evaluator import Evaluator
from engiopt.evaluation.leaderboard import ROW_KEY
from engiopt.evaluation.registry import METRICS
from engiopt.evaluation.submission import as_metric_value
from engiopt.evaluation.submission import integrity_flags
from engiopt.utils.all_generators import BUILTIN_GENERATORS

if TYPE_CHECKING:
    import torch as th

    from engiopt.core import Generator

VerificationStatus = Literal["verified", "weights_moved", "unresolvable", "unknown_generator"]
"""Outcome of trying to reproduce one row.

- `verified` -- the weights were fetched and re-scored. The row now carries the
  runner's numbers.
- `weights_moved` -- the package at that address no longer hashes to what the
  row recorded. Whatever is there now is not what earned the score, so the row
  is not verified and must not be silently re-scored into place either.
- `unresolvable` -- the address did not lead to a loadable package.
- `unknown_generator` -- no registered adapter can rebuild this model, so
  nothing here can run it.
"""

CORROBORATION_TOLERANCE = 0.05
"""Relative gap above which a submitted value is reported as uncorroborated.

Advisory only -- it decides what the note says, never whether the row is
verified, because the published numbers are the runner's own either way. Set
loosely on purpose: this is the scale of honest hardware-to-hardware variation,
so anything inside it is not evidence of anything.
"""


@dataclass
class VerificationResult:
    """What happened when one row was re-run."""

    key: dict[str, Any]
    status: VerificationStatus
    detail: str = ""
    row: dict[str, Any] | None = None
    """The re-scored row, present only when `status` is `verified`."""
    uncorroborated: dict[str, tuple[float, float]] = field(default_factory=dict)
    """Metric -> (submitted, recomputed) for values that did not survive re-scoring."""

    @property
    def ok(self) -> bool:
        """Whether this row was successfully reproduced."""
        return self.status == "verified"


def verify_row(
    row: dict[str, Any],
    *,
    verifier: str,
    evaluator: Evaluator | None = None,
    device: th.device | None = None,
    include_expensive: bool = False,
) -> VerificationResult:
    """Fetch a row's checkpoint at its recorded revision and score it again.

    Args:
        row: A published leaderboard row.
        verifier: Who is running this, recorded in `verified_by`. A verification
            is only worth what its author is, so it is signed rather than
            anonymous.
        evaluator: Reuse an evaluator when sweeping a whole board; one is built
            from the row's `problem_id` and `spec_version` otherwise.
        device: Torch device; auto-selected when omitted.
        include_expensive: Re-run the simulator-backed metrics too. Off by
            default, so a full board can be checked for fabricated *cheap*
            metrics without an optimizer budget.

    Returns:
        A `VerificationResult` whose `row` carries the runner's own numbers.
    """
    key = {column: row.get(column) for column in ROW_KEY}
    algo_id = str(row.get("algo_id"))
    generator_cls = BUILTIN_GENERATORS.get(algo_id)
    if generator_cls is None:
        return VerificationResult(
            key=key,
            status="unknown_generator",
            detail=(
                f"No registered adapter named {algo_id!r}. A model has to be in `engiopt/generators/` "
                "for anyone to be able to rebuild it; submit the adapter before submitting the scores."
            ),
        )

    try:
        generator = _load_at_recorded_revision(row, generator_cls, device=device)
    except Exception as exc:  # noqa: BLE001 - any failure to fetch is the same verdict to the caller
        return VerificationResult(key=key, status="unresolvable", detail=str(exc))

    recorded_hash = row.get("checkpoint_hash")
    if recorded_hash and generator.checkpoint_hash != recorded_hash:
        return VerificationResult(
            key=key,
            status="weights_moved",
            detail=(
                f"The package at {row.get('checkpoint_repo')}/{row.get('checkpoint_path')} now hashes to "
                f"{generator.checkpoint_hash}, but the row was scored on {recorded_hash}. The weights behind "
                "this score no longer exist at that address; re-submit the current ones under their own row."
            ),
        )

    evaluator = evaluator or Evaluator.for_problem(
        str(row.get("problem_id")), spec=f"{row.get('problem_id')}/{row.get('spec_version')}", device=device
    )
    rescored = evaluator.score(generator, include_expensive=include_expensive)
    rescored.update(
        {
            "verified": True,
            "verified_by": verifier,
            "verified_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
    )
    uncorroborated = _uncorroborated(row, rescored)
    rescored["flags"] = ",".join(integrity_flags(rescored, evaluator.spec))
    return VerificationResult(
        key=key,
        status="verified",
        detail=_corroboration_note(uncorroborated),
        row=rescored,
        uncorroborated=uncorroborated,
    )


def _load_at_recorded_revision(
    row: dict[str, Any], generator_cls: type[Generator], *, device: th.device | None
) -> Generator:
    """Rebuild the model from the exact package a row names.

    Deliberately not `from_pretrained`: that resolves by problem/algo/seed
    convention and would silently follow the *current* canonical checkpoint. A
    verification has to read the address written down at scoring time --
    including the revision, so that a later upload to the same path cannot
    launder itself into an old row's score.

    Raises:
        ValueError: If the row carries no usable checkpoint address.
    """
    repo = row.get("checkpoint_repo")
    path = row.get("checkpoint_path")
    if not repo or not path:
        raise ValueError("Row carries no checkpoint address, so there is nothing to fetch.")

    resolved = resolve_checkpoint_reference(
        model_source="hf",
        model_ref=f"hf://{repo}/{path}",
        required_files=list(generator_cls.checkpoint_files),
        revision=row.get("checkpoint_revision") or None,
    )
    from engibench.utils.all_problems import BUILTIN_PROBLEMS

    problem_id = str(row.get("problem_id"))
    problem = BUILTIN_PROBLEMS[problem_id]()
    device = device or pick_device()
    return generator_cls.build(
        resolved,
        problem=problem,
        device=device,
        problem_id=problem_id,
        seed=int(row["seed"]),
        run_config=resolved.run_config,
        condition_keys=condition_keys_for(problem, resolved),
        condition_stats=condition_stats_for(resolved),
        checkpoint_revision=resolved.revision,
        checkpoint_hash=resolved.content_hash,
        checkpoint_repo=resolved.repo_id,
        checkpoint_path=resolved.package_path,
    )


def _uncorroborated(submitted: dict[str, Any], recomputed: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Metrics whose submitted value the re-run did not reproduce.

    Compared relative to the recomputed magnitude, so a gap means the same thing
    for `mmd` near 0.01 as for `cog` in the thousands. Values the runner did not
    recompute -- expensive metrics on a cheap pass -- are skipped rather than
    counted as disagreement.
    """
    differences: dict[str, tuple[float, float]] = {}
    for name in METRICS.columns():
        if name not in submitted or name not in recomputed:
            continue
        claimed, actual = as_metric_value(submitted[name]), as_metric_value(recomputed[name])
        if claimed is None or actual is None:
            continue
        scale = max(abs(actual), abs(claimed), 1e-12)
        if abs(claimed - actual) / scale > CORROBORATION_TOLERANCE:
            differences[name] = (claimed, actual)
    return differences


def _corroboration_note(uncorroborated: dict[str, tuple[float, float]]) -> str:
    """One line saying whether the submitted numbers survived the re-run."""
    if not uncorroborated:
        return "re-scored; submitted values corroborated"
    reported = ", ".join(
        f"{name} claimed {claimed:.4g} vs {actual:.4g}" for name, (claimed, actual) in sorted(uncorroborated.items())
    )
    return f"re-scored; submitted values NOT reproduced ({reported})"


def verify_board(
    board: pd.DataFrame,
    *,
    verifier: str,
    problem_id: str | None = None,
    only_unverified: bool = True,
    include_expensive: bool = False,
    device: th.device | None = None,
) -> tuple[pd.DataFrame, list[VerificationResult]]:
    """Re-run every eligible row of a board, returning the verified replacements.

    Evaluators are built once per `(problem_id, spec_version)` and reused, since
    resolving a spec draws its conditions and reference designs.

    Args:
        board: The published leaderboard.
        verifier: Recorded in `verified_by` on every row this stamps.
        problem_id: Restrict to one problem.
        only_unverified: Skip rows already carrying a verification stamp.
        include_expensive: Re-run simulator-backed metrics as well.
        device: Torch device; auto-selected when omitted.

    Returns:
        The re-scored rows (ready to merge into the board), and one
        `VerificationResult` per row attempted, including the failures.
    """
    if board.empty:
        return board, []
    candidates = board
    if problem_id is not None:
        candidates = candidates[candidates["problem_id"] == problem_id]
    if only_unverified and "verified" in candidates.columns:
        candidates = candidates[~candidates["verified"].fillna(value=False).astype(bool)]

    evaluators: dict[tuple[str, str], Evaluator] = {}
    results: list[VerificationResult] = []
    verified_rows: list[dict[str, Any]] = []
    for record in candidates.to_dict("records"):
        cache_key = (str(record.get("problem_id")), str(record.get("spec_version")))
        if cache_key not in evaluators:
            try:
                evaluators[cache_key] = Evaluator.for_problem(
                    cache_key[0], spec=f"{cache_key[0]}/{cache_key[1]}", device=device
                )
            # A spec that will not resolve fails every row under it, not the run.
            except Exception as exc:  # noqa: BLE001
                results.append(
                    VerificationResult(
                        key={column: record.get(column) for column in ROW_KEY},
                        status="unresolvable",
                        detail=f"could not load spec {cache_key[0]}/{cache_key[1]}: {exc}",
                    )
                )
                continue
        result = verify_row(
            record,
            verifier=verifier,
            evaluator=evaluators[cache_key],
            device=device,
            include_expensive=include_expensive,
        )
        results.append(result)
        if result.row is not None:
            verified_rows.append(result.row)
    return pd.DataFrame(verified_rows), results
