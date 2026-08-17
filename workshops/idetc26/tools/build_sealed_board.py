"""Assemble the physics board for a line-up and seal it into the repository.

Run before the session, never during it. The result is committed encrypted, with
its plaintext SHA256 committed beside it, so a participant can verify after the
reveal that the answer was fixed before they made theirs.

**Harvest first, compute last.** `iog`/`cog`/`fog` for most of these packages
have already been paid for -- by the full-pool sweeps (`physics_board`, which
writes `physboard_<problem>_shard*.csv` on the cluster) and by
`engiopt.evaluate --publish`, which writes a `metrics.json` into each checkpoint
package on the Hub. Recomputing them costs hours and would produce the same
numbers, so this reads them:

    # 1. rsync the cluster boards down, if there are any
    rsync -av euler:projects/EngiOpt-eval/'physboard_*_shard*.csv' /tmp/boards/

    # 2. see what is covered without running anything
    python workshops/idetc26/tools/build_sealed_board.py --problem-id beams2d
        --from-boards /tmp/boards --dry-run

    # 3. seal it
    python workshops/idetc26/tools/build_sealed_board.py --problem-id beams2d
        --from-boards /tmp/boards --passphrase "..."

Anything still missing is computed with `Case.evaluate`, which is the same path
the notebook uses -- so a harvested row and a computed row mean the same thing.
That costs one optimization and two simulations per design: about 3.4 s on
beams2d, 3.0 s on heatconduction2d and **36 s on photonics2d**, times 50 designs
per model. Use `--compute-missing` deliberately, and on a machine you are
willing to lose for a few hours.

Every row is stamped with the spec the *config* names, not the spec the source
recorded. `beams2d/v2` was renamed to `v1` with its conditions unchanged -- the
`condition_digest` is identical -- so harvested rows carry the old label and
would otherwise seal a board claiming to be something the repository no longer
contains.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from engiopt.evaluation.physics_board import problem_of
from engiopt.workshops.idetc26.config import WorkshopConfig
from engiopt.workshops.idetc26.seal import seal

SEALED_DIR = Path(__file__).resolve().parents[2] / "engiopt" / "workshops" / "idetc26" / "sealed"
"""Inside the package, because Colab installs the package and never sees this checkout."""

PHYSICS = ("iog", "cog", "fog")
"""The mean optimality gaps. A row without all three is not a usable row."""

PHYSICS_MEDIAN = ("iog_median", "cog_median", "fog_median")
"""The median counterparts, taken when the source has them.

Worth having wherever they exist. The per-design gap is unbounded above, so a
mean over 50 designs is set by its worst member: on beams2d two models report
mean IOG above 1e9 while finishing at an ordinary FOG, which is one
unrecoverable starting design rather than a worse model. The cluster boards
carry both; the published `metrics.json` predates the median metrics and carries
only the means.
"""

ALL_PHYSICS = (*PHYSICS, *PHYSICS_MEDIAN)


def board_key(algo: str, fingerprint: str | None, seed: int) -> str:
    """The identifier `physics_board` files a row under.

    It is not the identifier the bank uses -- `physics_board` writes
    `algo/fingerprint/sN` and the bank writes `algo#N[cfg=...]` -- so joining
    the two means translating, and doing it in one named function means the two
    formats cannot quietly drift apart in three places.
    """
    return f"{algo}/{fingerprint or 'default'}/s{seed}"


def from_boards(directory: Path, problem_id: str) -> dict[str, dict[str, float]]:
    """Read every board CSV under `directory` that describes `problem_id`.

    Shards are separate files by design -- tasks never contend for one -- so a
    complete board is their concatenation. A key appearing twice keeps the first
    row and says so, since two shards scoring one package means the sharding was
    wrong and the numbers may not agree.

    **Filtering by problem is load-bearing.** `physics_board` files rows under
    `algo/fingerprint/sN`, which carries no problem id: the same key names a
    different model on every problem. Reading a directory indiscriminately would
    let one problem's optimality gaps answer for another's, silently, with no
    shape error to catch it. Where a file carries a `problem_id` column that
    decides; otherwise the filename does.
    """
    rows: dict[str, dict[str, float]] = {}
    sources: dict[str, str] = {}
    conflicts: dict[str, list[str]] = {}
    candidates = sorted(directory.glob("*.csv")) if directory.is_dir() else [directory]
    taken: list[str] = []
    for path in candidates:
        frame = pd.read_csv(path)
        if "key" not in frame.columns:
            continue
        found = problem_of(path, frame)
        if found != problem_id:
            if found is None:
                print(f"  [skip] {path.name}: cannot tell which problem it describes")
            continue
        taken.append(path.name)
        for row in frame.to_dict("records"):
            key = str(row["key"])
            present = {metric: row[metric] for metric in ALL_PHYSICS if metric in row and row[metric] == row[metric]}
            if not all(metric in present for metric in PHYSICS):
                continue
            if key in rows:
                # **Never merge columns across files.** Two sweeps of the same
                # package do not agree: sampling is seeded but not reproducible
                # across hardware, and the optimizer amplifies a small design
                # difference into a large gap -- on beams2d, 262 of 263 shared
                # keys differ, one of them flipping sign. Taking the means from
                # one sweep and the medians from another would produce a row
                # that describes no evaluation that ever happened, and nothing
                # downstream could detect it. Keep the first complete row and
                # say that a second exists.
                if any(rows[key].get(m) != present.get(m) for m in PHYSICS):
                    conflicts.setdefault(key, []).append(path.name)
                continue
            rows[key] = present
            sources[key] = path.name
    print(f"  read {len(rows)} scored {problem_id} packages from {len(taken)} board file(s)")
    _report_conflicts(conflicts, sources)
    return rows


def _report_conflicts(conflicts: dict[str, list[str]], sources: dict[str, str]) -> None:
    """Say when more than one sweep scored the same package differently."""
    if not conflicts:
        return
    print(
        f"  [conflict] {len(conflicts)} package(s) are scored differently by more than one sweep; "
        "kept the first and ignored the rest. Physics is not reproducible across hardware, so these "
        "are separate evaluations rather than a corruption -- but only one of them can be sealed."
    )
    for key, others in list(conflicts.items())[:3]:
        print(f"      {key}: kept {sources[key]}, also in {', '.join(sorted(set(others)))}")


def from_hub(problem_id: str, algo: str, fingerprint: str | None, seed: int) -> dict[str, float] | None:
    """Physics from a package's published `metrics.json`, or None if it has none."""
    from huggingface_hub import hf_hub_download

    from engiopt.checkpoint_store import build_hf_repo_id

    path = f"{problem_id}/" + (f"cfg_{fingerprint}/" if fingerprint else "") + f"seed_{seed}/metrics.json"
    try:
        payload = json.load(open(hf_hub_download(build_hf_repo_id("IDEALLab", "engiopt", algo), path)))  # noqa: SIM115
    except Exception:  # noqa: BLE001 - a package with no published metrics is the normal case, not an error
        return None
    metrics = payload.get("metrics", {})
    found = {m: metrics[m] for m in ALL_PHYSICS if m in metrics and metrics[m] == metrics[m]}
    return found if all(m in found for m in PHYSICS) else None


def gather(config: WorkshopConfig, boards: dict[str, dict[str, float]], *, use_hub: bool) -> pd.DataFrame:
    """One row per line-up member, from whichever source already has it.

    Args:
        config: The problem's workshop configuration.
        boards: Rows harvested from cluster board CSVs, by board key.
        use_hub: Also look in each package's published `metrics.json`.

    Returns:
        A frame with `key`, `algo`, `spec`, `source` and the physics columns.
        Members nothing covers appear with the physics columns missing.
    """
    from engiopt.workshops.idetc26.bank import _key_for

    rows = []
    for entry in config.bank:
        algo, seed = entry["algo"], int(entry.get("seed", 1))
        fingerprint = entry.get("config_fingerprint")
        options = {"cfg": fingerprint} if fingerprint else _baseline_options(entry)
        name = entry.get("name", algo)

        found, source = boards.get(board_key(algo, fingerprint, seed)), "cluster board"
        if found is None and use_hub:
            found, source = from_hub(config.problem_id, algo, fingerprint, seed), "hub metrics.json"
        if found is None:
            source = "MISSING"

        rows.append(
            {
                "key": _key_for(algo, seed, options),
                "algo": name,
                # The spec the config names, never the one the source recorded.
                "spec": config.spec,
                "source": source,
                **(found or {}),
            }
        )
    return pd.DataFrame(rows)


def _baseline_options(entry: dict[str, Any]) -> dict[str, Any]:
    """Constructor options a dataset-fitted baseline carries in its cache key."""
    declared = {
        "kind",
        "algo",
        "name",
        "seed",
        "optional",
        "summary",
        "model_source",
        "local_model_dir",
        "config_fingerprint",
        "train_minutes",
    }
    return {key: value for key, value in entry.items() if key not in declared}


def compute_missing(frame: pd.DataFrame, problem_id: str) -> pd.DataFrame:
    """Run the simulator for the rows nothing had, using the notebook's own path.

    Goes through `Case.evaluate` rather than a second scoring loop, so a computed
    row and a row a participant produces in the session are the same measurement
    made the same way.
    """
    from engiopt.workshops.idetc26 import Case

    missing = frame.index[frame["source"] == "MISSING"]
    if missing.empty:
        return frame

    case = Case.open(problem_id)
    by_key = {member.key: member.label for member in case.bank}
    labels = [by_key[key] for key in frame.loc[missing, "key"] if key in by_key]
    if not labels:
        print("  nothing missing is in the assembled line-up; leaving those rows blank")
        return frame

    print(f"  computing physics for {len(labels)}: {', '.join(labels)}")
    scored = case.evaluate(list(ALL_PHYSICS), models=labels, confirm=True, show_cli=False)
    label_to_key = {label: key for key, label in by_key.items()}
    for label, row in scored.iterrows():
        position = frame.index[frame["key"] == label_to_key[str(label)]]
        for metric in ALL_PHYSICS:
            if metric in row:
                frame.loc[position, metric] = row[metric]
        frame.loc[position, "source"] = "computed here"
    return frame


def main() -> None:
    """Assemble and seal one problem's physics board."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem-id", default="beams2d")
    parser.add_argument("--passphrase", help="Announced aloud at the reveal. Not needed with --dry-run.")
    parser.add_argument("--from-boards", type=Path, default=None, help="Directory of physboard_*.csv from the cluster.")
    parser.add_argument("--no-hub", action="store_true", help="Do not read published metrics.json.")
    parser.add_argument("--compute-missing", action="store_true", help="Run the simulator for uncovered members.")
    parser.add_argument("--dry-run", action="store_true", help="Report coverage and write nothing.")
    parser.add_argument("--plaintext-out", type=Path, default=None, help="Also write the board unencrypted.")
    args = parser.parse_args()

    config = WorkshopConfig.load(args.problem_id)
    print(f"{args.problem_id}: {len(config.bank)} members, spec {config.spec}")

    boards = from_boards(args.from_boards, args.problem_id) if args.from_boards else {}
    frame = gather(config, boards, use_hub=not args.no_hub)
    if args.compute_missing:
        frame = compute_missing(frame, args.problem_id)

    covered = frame["source"] != "MISSING"
    print(f"\n{covered.sum()}/{len(frame)} members have physics:\n")
    for row in frame.to_dict("records"):
        shown = [m for m in ALL_PHYSICS if m in row and row[m] == row[m]]
        values = "  ".join(f"{m}={row[m]:.4g}" for m in shown)
        print(f"  {row['algo']:<24}{row['source']:<18}{values if row['source'] != 'MISSING' else ''}")

    if not covered.all():
        print(
            f"\n{(~covered).sum()} member(s) uncovered. Rerun with --compute-missing (and the time to spare), "
            "or point --from-boards at the cluster CSVs for this problem."
        )
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return
    if not args.passphrase:
        parser.error("--passphrase is required unless --dry-run")
    if not covered.all():
        parser.error("refusing to seal a partial board; `case.physics()` would report blanks as results")

    csv = frame.to_csv(index=False)
    destination = SEALED_DIR / f"{args.problem_id}_physics.csv.enc"
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = seal(csv, args.passphrase, destination)
    print(f"\nSealed {len(frame)} rows to {destination}")
    print(f"Published SHA256: {digest}")

    if args.plaintext_out:
        args.plaintext_out.parent.mkdir(parents=True, exist_ok=True)
        args.plaintext_out.write_text(csv)
        print(f"Facilitator copy (DO NOT COMMIT): {args.plaintext_out}")


if __name__ == "__main__":
    main()
