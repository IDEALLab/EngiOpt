"""Promote a swept configuration to the canonical checkpoint path.

`{problem_id}/seed_{seed}` is what a bare model name resolves to -- what
`from_pretrained(problem, seed=1)` and `python -m engiopt.evaluate --seeds 1`
read. Only a run using the training script's default hyperparameters writes it,
which means a hyperparameter sweep never does: every arm varies something, so a
sweep can publish hundreds of packages and still leave that path empty.

This is the escape hatch. Pick the arm that should be the default and promote it,
instead of spending GPU hours retraining a configuration you already have::

    python -m engiopt.promote_checkpoint --algo cgan_cnn_2d --problem-id beams2d \
        --seed 1 --config-fingerprint 023dd1fb

To see what is available first::

    python -m engiopt.promote_checkpoint --algo cgan_cnn_2d --problem-id beams2d --list
"""

from __future__ import annotations

from dataclasses import dataclass
import sys

import tyro

from engiopt.checkpoint_store import build_hf_repo_id
from engiopt.checkpoint_store import list_packages
from engiopt.checkpoint_store import promote_to_canonical
from engiopt.core import DEFAULT_HF_ENTITY
from engiopt.core import DEFAULT_HF_REPO_PREFIX


@dataclass
class Args:
    """Command-line arguments for promoting a checkpoint."""

    algo: str
    """Model family, e.g. `cgan_cnn_2d`. Selects the HF repo."""
    problem_id: str
    """EngiBench problem the checkpoint was trained on."""
    seed: int = 1
    """Training seed. The promoted package keeps it."""
    config_fingerprint: str | None = None
    """Which configuration to promote. Required unless `--list` is passed."""
    list: bool = False
    """Print the packages this repo holds for the problem, then exit."""
    hf_entity: str = DEFAULT_HF_ENTITY
    """HF org/user holding the checkpoint repos."""
    hf_repo_prefix: str = DEFAULT_HF_REPO_PREFIX
    """Prefix of the per-model-family repo."""


def main(args: Args) -> int:
    """Promote one configuration, or list what is available.

    Returns:
        Process exit status: 0 on success, 1 when there is nothing to promote.
    """
    repo_id = build_hf_repo_id(args.hf_entity, args.hf_repo_prefix, args.algo)
    available = list_packages(repo_id, args.problem_id)

    if args.list or args.config_fingerprint is None:
        if not available:
            print(f"No packages for {args.problem_id!r} in {repo_id}.")
            return 1
        print(f"{len(available)} package(s) in {repo_id}:")
        for path in available:
            print(f"  {path}")
        if not args.list:
            print("\nPass --config-fingerprint to promote one of these to the canonical path.")
            return 1
        return 0

    canonical = promote_to_canonical(
        hf_entity=args.hf_entity,
        hf_repo_prefix=args.hf_repo_prefix,
        problem_id=args.problem_id,
        algo=args.algo,
        seed=args.seed,
        config_fingerprint=args.config_fingerprint,
    )
    print(f"Promoted cfg_{args.config_fingerprint} to {repo_id}/{canonical}")
    print(f"`--generators {args.algo} --seeds {args.seed}` now resolves.")
    return 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
