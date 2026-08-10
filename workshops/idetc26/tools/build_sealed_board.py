"""Compute the physics board for a bank and seal it into the repository.

Run before the session, never during it. The result is committed encrypted, with
its plaintext SHA256 committed beside it, so a participant can verify after the
reveal that the answer was fixed before they made theirs.

    python workshops/idetc26/tools/build_sealed_board.py --problem-id beams2d \
        --passphrase "different metrics, different stories"

Cost, measured rather than guessed: `beams2d` runs one optimization and two
simulations per sample at about 3.4 s/sample on a laptop CPU, so a 50-sample
board over an eight-model bank is roughly twenty minutes. It is CPU-bound
topology optimization -- a GPU does not help it, and that is worth knowing
before anyone budgets cluster time for it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from engiopt.evaluation import Evaluator
from engiopt.workshops.idetc26.bank import ModelBank
from engiopt.workshops.idetc26.config import WorkshopConfig
from engiopt.workshops.idetc26.seal import seal

SEALED_DIR = Path(__file__).resolve().parents[1] / "sealed"


def build(problem_id: str, *, seed: int = 1) -> pd.DataFrame:
    """Score every bank member on the expensive metrics.

    Args:
        problem_id: Which problem's bank to score.
        seed: Sampling seed the board is computed at.

    Returns:
        One row per bank member, keyed by the member's stable key rather than
        by its anonymous letter -- letters are permuted per team, so a board
        keyed by them would only be readable by one team.
    """
    config = WorkshopConfig.load(problem_id)
    evaluator = Evaluator.for_problem(problem_id, spec=config.spec)
    bank = ModelBank.assemble(config, evaluator.problem, team="")

    rows = []
    for member in bank:
        started = time.perf_counter()
        generator = member.load()
        generator.seed = seed
        scores = evaluator.score(generator, only=list(config.expensive_metrics), include_expensive=True)
        rows.append({"key": member.key, "algo": member.identity, **{m: scores[m] for m in config.expensive_metrics}})
        print(
            f"  {member.key:24s} {time.perf_counter() - started:6.1f}s  "
            + " ".join(f"{m}={scores[m]:.3f}" for m in config.expensive_metrics)
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Build and seal one problem's physics board."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-id", default="beams2d")
    parser.add_argument("--passphrase", required=True, help="Announced aloud at the reveal.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--plaintext-out", type=Path, default=None, help="Also write the board unencrypted, for the facilitator."
    )
    args = parser.parse_args()

    frame = build(args.problem_id, seed=args.seed)
    csv = frame.to_csv(index=False)

    destination = SEALED_DIR / f"{args.problem_id}_physics.csv.enc"
    digest = seal(csv, args.passphrase, destination)
    print(f"\nSealed {len(frame)} rows to {destination}")
    print(f"Published SHA256: {digest}")

    if args.plaintext_out:
        args.plaintext_out.parent.mkdir(parents=True, exist_ok=True)
        args.plaintext_out.write_text(csv)
        print(f"Facilitator copy (DO NOT COMMIT): {args.plaintext_out}")


if __name__ == "__main__":
    main()
