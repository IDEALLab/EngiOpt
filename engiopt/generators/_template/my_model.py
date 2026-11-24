"""Template training script. Copy this directory, rename this file to match it, and replace the network and the loss.

The script is deliberately small, but it does everything the leaderboard needs
from training:

1. It reads its hyperparameters from `Args`, so they are saved with the checkpoint.
2. It trains on the scalar conditions the problem provides.
3. It times its gradient steps with a `TrainClock`, paused while it logs.
4. It publishes one checkpoint package with `save_checkpoint_package`.

The placeholder model is a small conditional network trained to reproduce the
dataset's designs. It is here to show the plumbing, not to be a good model.

Try it first without publishing anything:

    python engiopt/generators/my_model/my_model.py --problem-id beams2d --n-steps 200 --no-track --save-model --checkpoint-backend local
    python -m engiopt.evaluate --problem-id beams2d --generators my_model --model-source local

Then publish a real run:

    python engiopt/generators/my_model/my_model.py --problem-id beams2d --seed 1 --save-model --hf-entity my-hf-username

See `CONTRIBUTING_A_MODEL.md` for the full walkthrough.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Literal, TYPE_CHECKING

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import torch as th
from torch import nn
import tyro
import wandb

from engiopt.checkpoint_store import save_checkpoint_package
from engiopt.core import checkpoint_identity
from engiopt.core import design_shape_of
from engiopt.core import pick_device
from engiopt.reproducibility import make_dataloader_generator
from engiopt.reproducibility import seed_training
from engiopt.train_clock import TrainClock
from engiopt.transforms import condition_keys

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class Args:
    """Command-line arguments. Every field is saved with the checkpoint as its `run_config`."""

    problem_id: str = "beams2d"
    """Problem identifier."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this algorithm. It must equal the directory name."""

    # Tracking and publishing. These do not change the model, so they are left out of its configuration fingerprint.
    track: bool = True
    """Track the experiment with wandb."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name."""
    hf_entity: str = "IDEALLab"
    """HF org/user where checkpoints are stored. Use your own account."""
    hf_repo_prefix: str = "engiopt"
    """HF repo prefix used for model-family repositories."""
    seed: int = 1
    """Random seed."""
    save_model: bool = False
    """Save the checkpoint package when training ends. Off by default, so a trial run saves nothing."""
    checkpoint_backend: Literal["hf", "local"] = "hf"
    """Where the package goes. `hf` publishes it. `local` writes it to this machine, to try the loader first."""
    sample_interval: int = 100
    """Steps between logged losses."""

    # Algorithm specific. Anything the adapter needs to rebuild the network belongs here.
    n_steps: int = 10_000
    """Number of gradient steps. Steps, not epochs, so the length means the same on every dataset."""
    batch_size: int = 32
    """Size of the batches."""
    lr: float = 1e-4
    """Learning rate."""
    latent_dim: int = 32
    """Dimensionality of the noise input."""


class Generator(nn.Module):
    """Maps noise and conditions to a design. Replace with your own network."""

    def __init__(self, latent_dim: int, n_conds: int, design_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.design_shape = design_shape
        self.model = nn.Sequential(
            nn.Linear(latent_dim + n_conds, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, int(np.prod(design_shape))),
            nn.Sigmoid(),
        )

    def forward(self, z: th.Tensor, conds: th.Tensor) -> th.Tensor:
        """Generate one design per row of noise and conditions."""
        designs = self.model(th.cat((z, conds), dim=-1))
        return designs.view(designs.size(0), *self.design_shape)


def endless(dataloader: th.utils.data.DataLoader) -> Iterator[list[th.Tensor]]:
    """Yield batches forever, starting a new pass over the data whenever one ends."""
    while True:
        yield from dataloader


if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)
    design_shape = design_shape_of(problem)
    # The scalar conditions only. Array-valued and solver-only conditions cannot enter a dense tensor.
    cond_keys = condition_keys(problem)

    if args.track:
        run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), save_code=True, name=run_name)

    seed_training(args.seed)
    device = pick_device()

    generator = Generator(args.latent_dim, len(cond_keys), design_shape).to(device)
    optimizer = th.optim.Adam(generator.parameters(), lr=args.lr)

    training_ds = problem.dataset.with_format("torch", device=device)["train"]
    training_ds = th.utils.data.TensorDataset(
        training_ds["optimal_design"][:].flatten(1), *[training_ds[key][:] for key in cond_keys]
    )
    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=make_dataloader_generator(args.seed),
    )
    batches = endless(dataloader)

    # The clock runs during the gradient steps and is paused for everything else.
    clock = TrainClock(device)
    clock.start()
    for step in range(args.n_steps):
        designs, *conds_columns = next(batches)
        conds = th.stack(conds_columns, dim=1).float()

        # Replace this block with your own training step.
        z = th.randn((designs.size(0), args.latent_dim), device=device)
        loss = nn.functional.mse_loss(generator(z, conds).flatten(1), designs.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.track and step % args.sample_interval == 0:
            clock.pause()
            wandb.log({"loss": loss.item(), "step": step})
            clock.start()
    clock.pause()

    if args.save_model:
        th.save({"generator": generator.state_dict()}, "generator.pth")
        save_checkpoint_package(
            checkpoint_backend=args.checkpoint_backend,
            hf_entity=args.hf_entity,
            hf_repo_prefix=args.hf_repo_prefix,
            hf_private=False,
            problem_id=args.problem_id,
            algo=args.algo,
            seed=args.seed,
            checkpoint_files={"generator.pth": "generator.pth"},
            run_config=vars(args),
            primary_files=["generator.pth"],
            condition_keys=cond_keys,
            metadata=clock.metadata(n_train_steps=args.n_steps),
            **checkpoint_identity(args),
        )

    if args.track:
        wandb.finish()
