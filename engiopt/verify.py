"""Re-run published leaderboard rows from their checkpoint addresses.

Verification is what a public board has instead of trust. Every row names the
repo, path, revision, and content hash of the weights it was scored on, so
anyone can fetch them and score them again::

    # check what the public board claims, without publishing anything
    python -m engiopt.verify --board IDEALLab/engiopt-leaderboard --problem-id beams2d

    # the official runner: re-score and stamp the rows it reproduced
    python -m engiopt.verify --board IDEALLab/engiopt-leaderboard --verifier IDEALLab-ci --publish

    # include the simulator-backed metrics, which the cheap pass leaves alone
    python -m engiopt.verify --board IDEALLab/engiopt-leaderboard --include-expensive --publish

Only `--publish` writes, and it writes the *runner's own* numbers. Running this
without it is a read-only audit that anybody can perform, which is the property
that keeps the official runner honest too.
"""

from __future__ import annotations

import dataclasses
import sys

import pandas as pd
import tyro

from engiopt.evaluation.leaderboard import load_from_hub
from engiopt.evaluation.leaderboard import push_to_hub
from engiopt.evaluation.verify import VerificationResult
from engiopt.evaluation.verify import verify_board


@dataclasses.dataclass
class Args:
    """Command-line arguments for verifying a published leaderboard."""

    board: str = "IDEALLab/engiopt-leaderboard"
    """HuggingFace dataset repo holding the leaderboard, or a local CSV path."""
    verifier: str = ""
    """Who is running this, recorded in `verified_by`.

    Required with `--publish`: an unsigned verification says only that some
    machine somewhere agreed, which is not a thing a reader can weigh."""
    problem_id: str | None = None
    """Restrict to one problem."""
    include_expensive: bool = False
    """Re-run the simulator-backed metrics too. Slow, and usually a separate pass."""
    recheck_verified: bool = False
    """Also re-run rows that already carry a stamp, e.g. after a metric changed."""
    publish: bool = False
    """Write the re-scored rows back to the board. Off by default: auditing is
    the common case, and it should never need write access."""
    output_csv: str | None = None
    """Also write the re-scored rows here."""


def _load_board(reference: str) -> pd.DataFrame:
    """Read the board from a HuggingFace dataset repo or a local CSV."""
    if reference.endswith(".csv"):
        return pd.read_csv(reference)
    return load_from_hub(reference)


def _report(results: list[VerificationResult]) -> None:
    """Print one line per row attempted, grouped so the failures are readable."""
    by_status: dict[str, list[VerificationResult]] = {}
    for result in results:
        by_status.setdefault(result.status, []).append(result)

    for status in sorted(by_status):
        rows = by_status[status]
        print(f"\n{status} ({len(rows)}):")
        for result in rows:
            label = f"{result.key.get('algo_id')} cfg {result.key.get('config_fingerprint')} seed {result.key.get('seed')}"
            print(f"  {label}: {result.detail}")


def main(args: Args) -> int:
    """Verify a board, optionally publishing the re-scored rows.

    Returns:
        A process exit status. Non-zero when a row could not be reproduced, so
        an unattended runner surfaces a moved checkpoint or a missing adapter
        rather than reporting a clean sweep over rows it silently skipped.
    """
    if args.publish and not args.verifier:
        print("--publish needs --verifier: a verification records who performed it.")
        return 2

    board = _load_board(args.board)
    if board.empty:
        print(f"{args.board} holds no rows; nothing to verify.")
        return 0

    verified, results = verify_board(
        board,
        verifier=args.verifier or "unsigned-audit",
        problem_id=args.problem_id,
        only_unverified=not args.recheck_verified,
        include_expensive=args.include_expensive,
    )
    if not results:
        print("No rows were eligible for verification.")
        return 0

    _report(results)
    reproduced = sum(1 for result in results if result.ok)
    uncorroborated = [result for result in results if result.ok and result.uncorroborated]
    print(f"\n{reproduced}/{len(results)} row(s) reproduced.")
    if uncorroborated:
        print(
            f"{len(uncorroborated)} reproduced row(s) did not match the submitted numbers. "
            "The board now carries the re-scored values."
        )

    if args.output_csv and not verified.empty:
        verified.to_csv(args.output_csv, index=False)
        print(f"Wrote {len(verified)} re-scored row(s) to {args.output_csv}")

    if args.publish and not verified.empty:
        # The verification runner is the one publisher whose `verified=True` is
        # not a self-assertion, so it is also the one that bypasses admission.
        merged = push_to_hub(verified, args.board, check_admission=False)
        print(f"Published {len(verified)} verified row(s); board now holds {len(merged)} rows.")

    return 0 if reproduced == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
