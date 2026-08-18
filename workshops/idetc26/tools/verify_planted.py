r"""Does each planted construction actually top the column it was built to top?

A planted model earns its place in the line-up by being ranked first by a column
a paper would report, and last by one that costs more or that nobody reports. If
it tops nothing, it is a row of numbers nobody will argue about -- and a bank
whose members all agree is an afternoon of metric discussion with no disagreement
in it.

So this is the curation gate, and it is meant to be run *before* the line-up is
frozen. It scores the whole bank on every cheap column and answers three
questions per construction:

    places      which columns put it in the top few, of any model in the bank
    beaten in   which put it in the bottom half
    predicted   how many of its declared `wins` it actually placed in

**Placing, not winning.** The first cut of this gate demanded rank 1 and called
three working constructions dead weight for landing second and third -- a
suspect that is second on the column you were going to report is every bit as
entitled to be picked as one that is first, and rank 1 on any given column
mostly reflects which real checkpoints happen to be in the bank that day.

What both halves together check is the only thing that matters: some affordable
column says pick me, and some other column says don't. A model that wins
everything is as useless here as one that wins nothing.

**Physics is not included.** The expensive half is sealed and rebuilt
separately; this measures whether the trap is set, not whether it was worth
setting. Run `build_sealed_board.py` afterwards to see the second half.

Cheap -- cached designs where they exist, no simulator, but it does read the
dataset and load the pinned encoder. **Run it on the cluster.**

Example:
    python workshops/idetc26/tools/verify_planted.py --problem-id beams2d
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from engiopt.workshops.idetc26.case import Case

if TYPE_CHECKING:
    import pandas as pd

PLACES = 3
"""How near the top a construction has to land to be worth ranking.

Three because that is roughly what a reader treats as "the good ones" in a
results table, and because rank 1 on a bank this size is as much a fact about
which checkpoints are in it as about the construction.
"""


def report(problem_id: str, *, n_samples: int | None) -> tuple[bool, pd.DataFrame]:
    """Score one problem's bank and judge each planted member.

    Args:
        problem_id: Which workshop problem to measure.
        n_samples: Score on this many of the spec's conditions, or all of them.

    Returns:
        `(ok, board)`. `ok` is True when every planted member tops at least one
        column and loses at least one, which is the whole of what makes it worth
        ranking. The board is returned rather than recomputed for the caller's
        CSV: scoring a bank twice is minutes of encoder time for a file write.
    """
    case = Case.open(problem_id)
    planted = [member.label for member in case.bank if member.kind == "planted"]
    board = case.evaluate(n_samples=n_samples, show_cli=False)
    if not planted:
        print(f"{problem_id}: no planted members in the line-up; nothing to verify.")
        return True, board

    ranks = case.rank(board)

    print(f"\n===== {problem_id}: {len(planted)} planted of {len(case.bank)} suspects =====")
    print(
        f"columns ranked: {len(ranks.columns)}   (dropped as constant or directionless: "
        f"{len(board.columns) - len(ranks.columns)})"
    )

    # A line-up nobody can argue about is the failure mode this whole file
    # exists to catch, so it gets reported whether or not anything is planted.
    winners = {column: ranks[column].idxmin() for column in ranks.columns}
    distinct = sorted(set(winners.values()))
    print(f"distinct rank-1 models across columns: {len(distinct)} -> {distinct}")

    ok = True
    for label in planted:
        member = case.bank[label]
        placed = [(column, int(ranks[column][label])) for column in ranks.columns if ranks[column][label] <= PLACES]
        beaten = [
            (column, int(ranks[column][label])) for column in ranks.columns if ranks[column][label] > len(case.bank) / 2
        ]
        predicted = [column for column in member.wins if column in ranks.columns and ranks[column][label] <= PLACES]
        missed = [column for column in member.wins if column in ranks.columns and ranks[column][label] > PLACES]

        print(f"\n  {label}")
        print(f"    places      {placed or 'NOWHERE -- this member is dead weight'}")
        print(f"    beaten in   {sorted(beaten, key=lambda pair: -pair[1])[:4] or 'nothing'}")
        print(
            f"    predicted   {len(predicted)}/{len([c for c in member.wins if c in ranks.columns])} "
            f"declared wins placed{'; missed ' + str(missed) if missed else ''}"
        )
        for column in member.wins:
            if column not in ranks.columns:
                print(f"    [note] declared win {column!r} is not a ranked column here")

        if not placed or not beaten:
            ok = False

    print(f"\n{problem_id}: {'OK' if ok else 'NOT READY -- see the members with no contradiction above'}")
    return ok, board


def main() -> None:
    """Run the curation gate over one problem or all of them."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem-id", default="beams2d")
    parser.add_argument("--n-samples", type=int, default=None, help="Score on fewer conditions, for a quick look.")
    parser.add_argument("--csv", default=None, help="Write the full cheap board here.")
    args = parser.parse_args()

    ok, board = report(args.problem_id, n_samples=args.n_samples)
    if args.csv:
        board.to_csv(args.csv)
        print(f"board written to {args.csv}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
