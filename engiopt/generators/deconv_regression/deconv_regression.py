"""Deconvolutional regression: the supervised baseline the field keeps re-deriving.

An MLP over the conditions, reshaped into a small feature map and upsampled to
a design by transposed convolutions, trained with mean squared error against the
optimal design. No noise input, no discriminator, no sampling loop -- one
condition in, one design out.

This is the model **Habibi et al.**, *When Is it Actually Worth Learning Inverse
Design?* (J. Mech. Des. 148(6):061704, 2026) compare k-nearest neighbours
against and call a deconvolutional network. The bank already carries the kNN
side of that comparison; without this it carries only half of it.

**It is meant to be beaten in a specific way, and to win in another.** Trained
under a pixel loss, it learns the *conditional mean* design: where several
structures satisfy one brief it returns their average, which is blurred, often
unmanufacturable, and identical on every call. Every diversity column should
collapse. What it should do well is exactly what a generative model is not
obviously better at -- getting close to a plausible design for the conditions
asked, quickly, having cost a few minutes of GPU time.

Deliberately *not* included, because each would hide the finding: no adversarial
term (which would restore sharpness by inventing detail), no noise input (which
would make it a cVAE), and no volume-fraction rescaling by default (which would
zero out `viol` and hide whether the network learned to hit the budget).

Usage:
    python engiopt/generators/deconv_regression/deconv_regression.py \
        --problem-id beams2d --n-epochs 200 --save-model --checkpoint-backend hf
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Literal

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import torch as th
from torch import nn
from torchvision import transforms
import tqdm
import tyro

from engiopt.checkpoint_store import save_checkpoint_package
from engiopt.core import checkpoint_identity
from engiopt.core import pick_device
from engiopt.reproducibility import seed_training
from engiopt.transforms import condition_keys


@dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "beams2d"
    """Problem identifier."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this algorithm."""

    seed: int = 1
    """Random seed for initialization, batching, and the validation split."""
    track: bool = False
    """Track the run with Weights & Biases."""
    wandb_project: str = "engiopt"
    """W&B project name."""
    wandb_entity: str | None = None
    """W&B entity; None uses the default."""
    save_model: bool = False
    """Save the trained model to disk and to the checkpoint backend."""
    checkpoint_backend: Literal["hf", "none"] = "none"
    """Durable checkpoint backend."""
    hf_entity: str = "IDEALLab"
    """HF org/user holding the checkpoint repos."""
    hf_repo_prefix: str = "engiopt"
    """Prefix of the per-model-family HF repo."""

    n_epochs: int = 200
    """Training epochs."""
    batch_size: int = 64
    """Minibatch size."""
    lr: float = 2e-4
    """Adam learning rate."""
    b1: float = 0.5
    """Adam beta1."""
    b2: float = 0.999
    """Adam beta2."""
    hidden: int = 256
    """Width of the MLP that turns conditions into the initial feature map."""
    num_filters: tuple[int, ...] = (256, 128, 64, 32)
    """Channels at each upsampling stage, 7x7 -> 13x13 -> 25x25 -> 50x50 -> 100x100."""
    match_volume: bool = False
    """Rescale predictions to the requested volume fraction.

    Off by default, like the ridge baseline: the network is *supposed* to
    predict the volume fraction from the conditions, and switching this on hides
    whether it learned to."""
    max_train: int = 0
    """Cap on training designs used. Zero uses the whole split."""


class DeconvRegressor(nn.Module):
    """Conditions in, one design out. The cGAN generator with its noise removed.

    The upsampling stack is deliberately the same shape as `cgan_cnn_2d`'s, so a
    comparison between the two prices the *training objective* -- adversarial
    against pixel-wise -- rather than an architecture difference nobody
    controlled for.

    Args:
        n_conds: Number of scalar conditions.
        design_shape: Target design shape.
        hidden: Width of the condition MLP.
        num_filters: Channels at each upsampling stage.
    """

    def __init__(
        self,
        n_conds: int,
        design_shape: tuple[int, int],
        hidden: int = 256,
        num_filters: tuple[int, ...] = (256, 128, 64, 32),
    ) -> None:
        super().__init__()
        self.design_shape = design_shape
        self.n_conds = n_conds

        self.condition_mlp = nn.Sequential(
            nn.Linear(n_conds, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_filters[0] * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.first_channels = num_filters[0]

        self.up_blocks = nn.Sequential(
            # 7x7 -> 13x13
            nn.ConvTranspose2d(num_filters[0], num_filters[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True),
            # 13x13 -> 25x25
            nn.ConvTranspose2d(num_filters[1], num_filters[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True),
            # 25x25 -> 50x50
            nn.ConvTranspose2d(num_filters[2], num_filters[3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[3]),
            nn.ReLU(inplace=True),
            # 50x50 -> 100x100
            nn.ConvTranspose2d(num_filters[3], 1, kernel_size=4, stride=2, padding=1, bias=False),
            # Designs are trained in [0, 1] and denormalized at sampling time,
            # so the output is bounded rather than free.
            nn.Sigmoid(),
        )
        self.resize = transforms.Resize(design_shape, antialias=True)

    def forward(self, conditions: th.Tensor) -> th.Tensor:
        """Map a batch of conditions to a batch of designs."""
        features = self.condition_mlp(conditions).view(-1, self.first_channels, 7, 7)
        return self.resize(self.up_blocks(features)).squeeze(1)


def load_split(problem: Any, split: str, keys: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    """Designs and scalar conditions for one dataset split, or None if absent."""
    if split not in problem.dataset:
        return None
    data = problem.dataset[split]
    designs = np.asarray(data["optimal_design"], dtype=np.float32)
    conditions = np.stack([np.asarray(data[key], dtype=np.float32) for key in keys], axis=1)
    return designs, conditions


def train(args: Args) -> dict[str, object]:
    """Fit the regressor and return its checkpoint payload.

    Returns:
        The payload: weights, the normalization the sampler has to invert, and
        the architecture needed to rebuild the network.
    """
    device = pick_device()
    seed_training(args.seed)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)
    keys = condition_keys(problem)

    designs, conditions = load_split(problem, "train", keys)  # type: ignore[misc]
    if args.max_train and args.max_train < len(designs):
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(designs), args.max_train, replace=False)
        designs, conditions = designs[keep], conditions[keep]

    # Conditions are standardized and designs are put on [0, 1]: an MLP fed
    # volfrac in [0.15, 0.45] beside rmin in [1, 3] spends its first epochs
    # learning the scale difference, and a sigmoid head needs a target it can
    # actually reach. Both transforms are stored, because sampling has to invert
    # exactly what training applied.
    cond_mean, cond_std = conditions.mean(axis=0), conditions.std(axis=0)
    cond_std[cond_std == 0] = 1.0
    design_min, design_max = float(designs.min()), float(designs.max())
    span = max(design_max - design_min, 1e-8)

    x = th.from_numpy((conditions - cond_mean) / cond_std).float()
    y = th.from_numpy((designs - design_min) / span).float()
    loader = th.utils.data.DataLoader(
        th.utils.data.TensorDataset(x, y), batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    validation = load_split(problem, "val", keys)
    if validation is not None:
        val_x = th.from_numpy((validation[1] - cond_mean) / cond_std).float().to(device)
        val_y = th.from_numpy((validation[0] - design_min) / span).float().to(device)

    design_shape = tuple(designs.shape[1:])
    net = DeconvRegressor(len(keys), design_shape, args.hidden, tuple(args.num_filters)).to(device)
    optimizer = th.optim.Adam(net.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    loss_fn = nn.MSELoss()

    if args.track:
        import wandb

        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), save_code=True)

    print(f"{args.problem_id}: {len(designs)} designs, {len(keys)} conditions {keys}, design {design_shape}")
    print(f"{sum(p.numel() for p in net.parameters()):,} parameters on {device}")

    for epoch in tqdm.trange(args.n_epochs, desc="epoch"):
        net.train()
        total = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(net(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_x)
        metrics = {"epoch": epoch, "train_mse": total / len(loader.dataset)}

        if validation is not None:
            net.eval()
            with th.no_grad():
                metrics["val_mse"] = loss_fn(net(val_x), val_y).item()
        if args.track:
            import wandb

            wandb.log(metrics)
        if epoch % 20 == 0 or epoch == args.n_epochs - 1:
            print("  " + "  ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))

    return {
        "model": net.state_dict(),
        "design_shape": list(design_shape),
        "condition_keys": list(keys),
        "cond_mean": th.from_numpy(cond_mean.astype(np.float32)),
        "cond_std": th.from_numpy(cond_std.astype(np.float32)),
        "design_min": design_min,
        "design_max": design_max,
        "hidden": args.hidden,
        "num_filters": list(args.num_filters),
        "match_volume": args.match_volume,
    }


if __name__ == "__main__":
    args = tyro.cli(Args)
    payload = train(args)

    if args.save_model:
        th.save(payload, "deconv_regression.pth")
        print(f"deconv_regression.pth is {os.path.getsize('deconv_regression.pth') / 1e6:.3f} MB.")

        if args.checkpoint_backend == "hf":
            save_checkpoint_package(
                checkpoint_backend="hf",
                hf_entity=args.hf_entity,
                hf_repo_prefix=args.hf_repo_prefix,
                hf_private=False,
                problem_id=args.problem_id,
                algo=args.algo,
                seed=args.seed,
                checkpoint_files={"deconv_regression.pth": "deconv_regression.pth"},
                run_config=vars(args),
                **checkpoint_identity(args),
                primary_files=["deconv_regression.pth"],
                condition_keys=list(payload["condition_keys"]),  # type: ignore[arg-type]
            )
