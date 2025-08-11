"""Simulate EngiBench problem."""

from __future__ import annotations

import dataclasses
import os
from typing import Literal

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd
import tyro


@dataclasses.dataclass
class Args:
    """Command-line arguments for a single-seed CGAN 2D evaluation."""

    problem_id: str = "heatconduction2d"
    """Problem identifier (e.g. beams2d)."""
    seed: int = 1
    """Random seed for reproducibility."""
    split: Literal["train", "test", "val"] = "test"
    """train, test, or val."""
    output_csv: str = "{problem_id}_simulate.csv"
    """Output CSV path template; may include {problem_id}."""


if __name__ == "__main__":
    args = tyro.cli(Args)

    seed = args.seed
    problem = BUILTIN_PROBLEMS[args.problem_id]()

    # Load the dataset
    data = problem.dataset[args.split]

    results_list = []
    for idx, design in enumerate(data["optimal_design"]):
        problem.reset(seed=seed)
        result = problem.simulate(np.array(design))
        results_list.append({"result": result[0], "split": args.split, "index": idx})

    # Create DataFrame with each dict field as a column
    metrics_df = pd.DataFrame(results_list)
    out_path = args.output_csv.format(problem_id=args.problem_id)
    write_header = not os.path.exists(out_path)
    metrics_df.to_csv(out_path, mode="a", header=write_header, index=False)
