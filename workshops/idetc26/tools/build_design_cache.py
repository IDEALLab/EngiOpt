"""Sample every bank member once, so that nobody in the room has to.

Run before the session, never during it. Sampling the spec's conditions from a
whole bank is the workshop's largest avoidable cost -- minutes of it is one
diffusion model on a CPU runtime -- and it produces the same designs for every
team, every time. Precomputing it turns the notebook's opening from a ten-minute
wait into an instant one, and leaves the costs that are worth feeling (the
simulator, and a fresh sample from a model you deliberately chose to time) fully
intact.

    python workshops/idetc26/tools/build_design_cache.py --problem-id beams2d

Writes into the package (`engiopt/workshops/idetc26/cache/`) so the entries ship
with `pip install git+...` and are there in Colab, which never sees this
repository. `--into repo` writes to the checkout instead, for a cache you do not
intend to commit.

This is a **cluster or workstation job**, not something to run on a laptop while
doing anything else: it loads every checkpoint in the bank, including a 1.8 GB
VQGAN, and samples from each of them.
"""

from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING

from engiopt.evaluation import Evaluator
from engiopt.workshops.idetc26.bank import ModelBank
from engiopt.workshops.idetc26.config import WorkshopConfig
from engiopt.workshops.idetc26.designs import DesignStore
from engiopt.workshops.idetc26.designs import PACKAGE_CACHE
from engiopt.workshops.idetc26.designs import REPO_CACHE

if TYPE_CHECKING:
    from pathlib import Path


def build(problem_id: str, seeds: tuple[int, ...], *, into: Path, overwrite: bool = False) -> None:
    """Sample every bank member at every seed and write the designs to `into`.

    Args:
        problem_id: Which problem's bank to sample.
        seeds: Sampling seeds to cover. The session only ever reads seed 1 --
            `evaluate` offers no seed argument -- so the default is just that.
            More are only worth building if you are studying sampling noise.
        into: Cache root to write to.
        overwrite: Resample entries that are already there.
    """
    config = WorkshopConfig.load(problem_id)
    evaluator = Evaluator.for_problem(problem_id, spec=config.spec)
    bank = ModelBank.assemble(config, evaluator.problem)
    store = DesignStore(
        problem_id,
        spec_version=evaluator.spec.version,
        condition_digest=evaluator.spec.condition_digest,
        roots=(into,),
        write_root=into,
    )

    print(f"{problem_id}: {len(bank)} models x {len(seeds)} seeds -> {into}")
    for member in bank:
        for seed in seeds:
            if not overwrite and store.holds(member.key, seed):
                print(f"  {member.key:28s} seed {seed}  already cached")
                continue
            started = time.perf_counter()
            generator = member.load()
            generator.seed = seed
            context = evaluator.context_for(generator)
            path = store.store(
                member.key,
                seed,
                context.gen_designs,
                sample_seconds=context.sample_seconds,
                model_params=context.model_params,
            )
            size = path.stat().st_size / 1e6
            print(f"  {member.key:28s} seed {seed}  {time.perf_counter() - started:7.1f}s  {size:5.1f} MB  {path.name}")

    total = sum(entry.stat().st_size for entry in into.rglob("*.npz")) / 1e6
    print(f"\nCache is {total:.1f} MB. Commit it if you want the workshop to open instantly in Colab.")


def main() -> None:
    """Build one problem's design cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-id", default="beams2d")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--into",
        choices=("package", "repo"),
        default="package",
        help="`package` ships with pip and reaches Colab; `repo` stays in the checkout.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Resample entries that already exist.")
    args = parser.parse_args()

    build(
        args.problem_id,
        tuple(args.seeds),
        into=PACKAGE_CACHE if args.into == "package" else REPO_CACHE,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
