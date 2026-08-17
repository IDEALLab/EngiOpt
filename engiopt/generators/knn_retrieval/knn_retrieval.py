"""k-nearest-neighbour retrieval baseline for conditional design generation.

Habibi et al., *When Is it Actually Worth Learning Inverse Design?*
(J. Mech. Des. 148(6):061704, 2026; first presented at IDETC-CIE 2023) compared
inverse-design model families on a topology-optimization warm-start task and
found that k-nearest neighbours and random forests **outperform deconvolutional
networks when training data is limited**, once the cost of generating that data
is counted. On this benchmark that makes kNN a method to beat, not a straw man,
and it belongs on the leaderboard under the same contract as everything else.

"Training" here is fitting the per-condition distance scaling and deciding `k`.
That takes about a second, which is the point: the interesting column for this
model is not quality but cost, and a leaderboard that reports one without the
other cannot express what that paper measured.

The checkpoint is deliberately self-contained -- it carries the training
conditions *and designs* rather than a pointer to the dataset -- because for this
model the training set **is** the parameters. A kNN package weighs what its
training data weighs, and putting that number next to a GAN's parameter count is
a more honest comparison than pretending the data is free.

Usage:
    python engiopt/generators/knn_retrieval/knn_retrieval.py --problem-id beams2d --save-model
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import torch as th
import tyro

from engiopt.checkpoint_store import save_checkpoint_package
from engiopt.core import checkpoint_identity
from engiopt.transforms import condition_keys


@dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "beams2d"
    """Problem identifier."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this algorithm."""

    seed: int = 1
    """Random seed. Affects only the training subsample, since the fit is exact."""
    save_model: bool = False
    """Save the fitted model to disk and to the checkpoint backend."""
    checkpoint_backend: Literal["hf", "none"] = "none"
    """Durable checkpoint backend."""
    hf_entity: str = "IDEALLab"
    """HF org/user holding the checkpoint repos."""
    hf_repo_prefix: str = "engiopt"
    """Prefix of the per-model-family HF repo."""

    k: int = 1
    """Neighbours averaged per query, defaulting to pure retrieval.

    `k = 1` is the canonical configuration: it can only
    return designs already in the training set."""
    distance_weighted: bool = False
    """Weight neighbours by inverse distance rather than averaging them equally."""
    match_volume: bool = True
    """Rescale the retrieved design to the requested volume fraction.

    The post-hoc feasibility step any practitioner would apply. Off, the model
    inherits whatever volume fraction its neighbours happened to have, which is
    a different (and worse) method rather than a purer one."""
    max_train: int = 0
    """Cap on training designs kept, for the data-size axis Habibi et al. sweep.
    Zero keeps the whole split."""
    store_designs_fp16: bool = True
    """Store retrieved designs at half precision, halving a package that is
    otherwise the size of the dataset."""


def fit(args: Args) -> dict[str, object]:
    """Fit the retrieval index: condition scaling, and the data it searches.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The checkpoint payload.
    """
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    keys = condition_keys(problem)
    train = problem.dataset["train"]

    designs = np.asarray(train["optimal_design"], dtype=np.float32)
    conditions = np.stack([np.asarray(train[key], dtype=np.float64) for key in keys], axis=1)

    if args.max_train and args.max_train < len(designs):
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(designs), args.max_train, replace=False)
        designs, conditions = designs[keep], conditions[keep]

    # Standardize per condition, so a variable measured in thousands does not
    # dominate the neighbour search over one measured in hundredths.
    scale = conditions.std(axis=0)
    scale[scale == 0] = 1.0

    print(f"Fitted on {len(designs)} designs, {len(keys)} conditions, k={args.k}.")
    return {
        "conditions": th.from_numpy(conditions),
        "designs": th.from_numpy(designs.astype(np.float16 if args.store_designs_fp16 else np.float32)),
        "scale": th.from_numpy(scale),
        "condition_keys": list(keys),
        "k": args.k,
        "distance_weighted": args.distance_weighted,
        "match_volume": args.match_volume,
    }


if __name__ == "__main__":
    args = tyro.cli(Args)
    payload = fit(args)

    if args.save_model:
        th.save(payload, "knn.pth")
        size_mb = os.path.getsize("knn.pth") / 1e6
        print(f"knn.pth is {size_mb:.1f} MB -- this model's training set is its parameters.")

        if args.checkpoint_backend == "hf":
            save_checkpoint_package(
                checkpoint_backend="hf",
                hf_entity=args.hf_entity,
                hf_repo_prefix=args.hf_repo_prefix,
                hf_private=False,
                problem_id=args.problem_id,
                algo=args.algo,
                seed=args.seed,
                checkpoint_files={"knn.pth": "knn.pth"},
                run_config=vars(args),
                **checkpoint_identity(args),
                primary_files=["knn.pth"],
                condition_keys=list(payload["condition_keys"]),  # type: ignore[arg-type]
            )
