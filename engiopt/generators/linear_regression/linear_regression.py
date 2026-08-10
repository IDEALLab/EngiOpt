"""Ridge-regression baseline: the cheapest thing that could possibly work.

A polynomial expansion of the conditions, mapped to pixels by a closed-form
ridge solve. No latent variable, no iterations, no GPU -- and therefore exactly
one design per condition, which is the honest trade the diversity columns exist
to price.

Its job on the leaderboard is to make the other rows interpretable. A generative
model that cannot beat a linear map from four numbers to five thousand pixels
has not earned its training budget, and until this row exists nobody can tell.
It is also the natural companion to the kNN baseline in Habibi et al.,
*When Is it Actually Worth Learning Inverse Design?* (J. Mech. Des. 148(6):061704,
2026): both ask what the expensive model is actually buying you.

Unlike kNN, this one has genuine learned parameters, and there are few of them.
The three checkpoint sizes side by side -- a linear model's weight matrix, a
kNN's training set, a VQGAN's network -- are a better answer to "what is a
model?" than any definition.

Usage:
    python engiopt/generators/linear_regression/linear_regression.py \
        --problem-id beams2d --degree 2 --save-model
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
    """Random seed. Affects only the training subsample; the solve is exact."""
    save_model: bool = False
    """Save the fitted model to disk and to the checkpoint backend."""
    checkpoint_backend: Literal["hf", "none"] = "none"
    """Durable checkpoint backend."""
    hf_entity: str = "IDEALLab"
    """HF org/user holding the checkpoint repos."""
    hf_repo_prefix: str = "engiopt"
    """Prefix of the per-model-family HF repo."""

    degree: int = 2
    """Polynomial degree of the condition features. 1 is a plain linear map."""
    ridge: float = 1e-2
    """L2 penalty. Large values pull every prediction toward the mean design."""
    match_volume: bool = False
    """Rescale predictions to the requested volume fraction.

    Off by default, unlike kNN: the regression is *supposed* to predict the
    volume fraction from the conditions, so switching this on hides whether it
    learned to."""
    max_train: int = 0
    """Cap on training designs used. Zero uses the whole split."""


def polynomial_features(conditions: np.ndarray, degree: int) -> np.ndarray:
    """Bias term followed by each condition raised to powers `1..degree`."""
    columns = [np.ones((len(conditions), 1))]
    columns.extend(conditions**power for power in range(1, degree + 1))
    return np.hstack(columns)


def fit(args: Args) -> dict[str, object]:
    """Solve the ridge problem in closed form.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The checkpoint payload.
    """
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    keys = condition_keys(problem)
    train = problem.dataset["train"]

    designs = np.asarray(train["optimal_design"], dtype=np.float64)
    conditions = np.stack([np.asarray(train[key], dtype=np.float64) for key in keys], axis=1)

    if args.max_train and args.max_train < len(designs):
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(designs), args.max_train, replace=False)
        designs, conditions = designs[keep], conditions[keep]

    design_shape = designs.shape[1:]
    features = polynomial_features(conditions, args.degree)
    targets = designs.reshape(len(designs), -1)

    gram = features.T @ features + args.ridge * np.eye(features.shape[1])
    weights = np.linalg.solve(gram, features.T @ targets)

    residual = targets - features @ weights
    r2 = 1.0 - residual.var() / targets.var()
    print(f"Fitted {weights.shape[0]} x {weights.shape[1]} weights on {len(designs)} designs. Train R^2 = {r2:.4f}")

    return {
        "weights": th.from_numpy(weights.astype(np.float32)),
        "design_shape": list(design_shape),
        "condition_keys": list(keys),
        "degree": args.degree,
        "ridge": args.ridge,
        "match_volume": args.match_volume,
        "train_r2": float(r2),
    }


if __name__ == "__main__":
    args = tyro.cli(Args)
    payload = fit(args)

    if args.save_model:
        th.save(payload, "linear_regression.pth")
        print(f"linear_regression.pth is {os.path.getsize('linear_regression.pth') / 1e6:.3f} MB.")

        if args.checkpoint_backend == "hf":
            save_checkpoint_package(
                checkpoint_backend="hf",
                hf_entity=args.hf_entity,
                hf_repo_prefix=args.hf_repo_prefix,
                hf_private=False,
                problem_id=args.problem_id,
                algo=args.algo,
                seed=args.seed,
                checkpoint_files={"linear_regression.pth": "linear_regression.pth"},
                run_config=vars(args),
                **checkpoint_identity(args),
                primary_files=["linear_regression.pth"],
                condition_keys=list(payload["condition_keys"]),  # type: ignore[arg-type]
            )
