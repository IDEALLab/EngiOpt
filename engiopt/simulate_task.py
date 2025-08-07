"""Distribute engibench simulations across a Slurm job array.

and append results to a shared CSV **exactly once per design.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import sys

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--array-size", type=int, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--problem-id", default="heatconduction2d")
    p.add_argument("--outfile", required=True, help="Shared CSV written on the parallel filesystem")
    return p.parse_args()


def catalogue(problem_id: str) -> list[tuple[str, int]]:
    """Flatten train/val/test into one ordered list of (split, index)."""
    prob = BUILTIN_PROBLEMS[problem_id]()
    data = prob.dataset
    return [(split, idx) for split in ("train", "val", "test") for idx in range(len(data[split]["optimal_design"]))]


def slice_for_task(cat: list, task_id: int, array_size: int) -> list:
    jobs_per_task = math.ceil(len(cat) / array_size)
    start = task_id * jobs_per_task
    end = min(start + jobs_per_task, len(cat))
    return cat[start:end]


def load_done(outfile: str) -> set[tuple[str, int]]:
    """Return the {(split, index)} already present in the CSV."""
    if not os.path.exists(outfile):
        return set()
    df = pd.read_csv(outfile, usecols=["split", "index"])
    return set(zip(df["split"], df["index"]))


def append_rows(outfile: str, rows: list[dict]):
    """Append with a POSIX advisory lock (works on Lustre/BeeGFS/GPFS)."""
    if not rows:
        return
    header = rows[0].keys()
    with open(outfile, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        writer = csv.DictWriter(f, fieldnames=header)
        if f.tell() == 0:  # first writer
            writer.writeheader()
        writer.writerows(rows)
        fcntl.flock(f, fcntl.LOCK_UN)


def main():
    args = parse_args()
    cat = catalogue(args.problem_id)
    work = slice_for_task(cat, args.task_id, args.array_size)
    if not work:
        sys.exit(0)

    # Skip designs already simulated (resume-safe)
    already = load_done(args.outfile)

    prob = BUILTIN_PROBLEMS[args.problem_id]()  # fresh instance per task
    data = prob.dataset
    rows = []

    for split, idx in work:
        if (split, idx) in already:
            continue
        prob.reset(seed=args.seed)
        design = np.array(data[split]["optimal_design"][idx])
        cfg = data[split].select_columns(prob.conditions_keys)[idx]
        result = prob.simulate(design, config=cfg)
        rows.append(
            {
                "split": split,
                "index": idx,
                "result_json": json.dumps(result, separators=(",", ":")),
            }
        )

    append_rows(args.outfile, rows)


if __name__ == "__main__":
    main()
