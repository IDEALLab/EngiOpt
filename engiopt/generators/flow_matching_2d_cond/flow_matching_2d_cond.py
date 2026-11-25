"""Train a conditional velocity field on an EngiBench training split."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, TYPE_CHECKING

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import torch as th
from torch.nn import functional
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import tyro

from engiopt.core import pick_device
from engiopt.generators.flow_matching_2d_cond.model import build_model
from engiopt.reproducibility import make_dataloader_generator
from engiopt.reproducibility import seed_training
from engiopt.transforms import condition_keys

if TYPE_CHECKING:
    from datasets import Dataset


@dataclass
class Args:
    """Training and inference settings stored alongside the trained weights."""

    problem_id: Literal["beams2d", "heatconduction2d"] = "beams2d"
    seed: int = 1
    n_epochs: int = 500
    batch_size: int = 32
    max_batches: int | None = None
    lr: float = 0.0004
    layers_per_block: int = 2
    num_train_timesteps: int = 1000
    method: Literal["euler", "midpoint", "rk4"] = "euler"
    integration_steps: int = 16
    clip_min: float = 0.001
    clip_max: float = 1.0
    device: str = "auto"
    output: Path = Path("flow_matching_checkpoint")


def _training_tensors(split: Dataset, keys: list[str], shape: tuple[int, ...]) -> tuple[th.Tensor, th.Tensor]:
    """Read only the numeric columns, keeping at most one decoded batch alongside the tensors."""
    if not len(split) or not keys:
        raise ValueError("Training requires designs and at least one scalar condition.")
    designs = th.empty((len(split), 1, *shape), dtype=th.float32)
    conditions = th.empty((len(split), 1, len(keys)), dtype=th.float32)
    # The NumPy and Torch formatters in datasets can import torchvision's video
    # decoder even for numeric columns. These two problems need only plain lists.
    rows = split.with_format(None, columns=["optimal_design", *keys])
    start = 0
    for batch in rows.iter(batch_size=128):
        n = len(batch["optimal_design"])
        designs[start : start + n] = th.as_tensor(batch["optimal_design"]).reshape(n, 1, *shape)
        for i, key in enumerate(keys):
            conditions[start : start + n, 0, i] = th.as_tensor(batch[key])
        start += n
    return designs, conditions


def train(args: Args) -> None:
    """Train with uniform continuous times and save the final model as a local package."""
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.clip_min < args.clip_max:
        raise ValueError("clip_min must be smaller than clip_max.")
    if min(args.n_epochs, args.batch_size, args.integration_steps, args.num_train_timesteps) < 1 or (
        args.max_batches is not None and args.max_batches < 1
    ):
        raise ValueError("Epochs, batch size, timesteps and step limits must be positive.")
    seed_training(args.seed)
    device = pick_device() if args.device == "auto" else th.device(args.device)
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    keys = condition_keys(problem)
    shape = problem.design_space.shape
    designs, conditions = _training_tensors(problem.dataset["train"], keys, shape)
    lower, upper = float(designs.min()), float(designs.max())
    designs.sub_(lower).div_(max(upper - lower, 1e-8))
    loader = DataLoader(
        TensorDataset(designs, conditions),
        batch_size=args.batch_size,
        shuffle=True,
        generator=make_dataloader_generator(args.seed),
    )
    net = build_model(shape, len(keys), args.layers_per_block).to(device)
    optimizer = th.optim.AdamW(net.parameters(), lr=args.lr)
    updates = 0
    for _ in range(args.n_epochs):
        for designs_batch, cond in loader:
            target = designs_batch.to(device)
            t = th.rand(len(target), device=device)
            noise = th.randn_like(target)
            fraction = t.view(-1, 1, 1, 1)
            mixed = (1 - fraction) * noise + fraction * target
            prediction = net(mixed, t * max(args.num_train_timesteps - 1, 1), cond.to(device)).sample
            loss = functional.mse_loss(prediction, target - noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            updates += 1
            if updates == args.max_batches:
                break
        if updates == args.max_batches:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        package = Path(tmp) / "package"
        package.mkdir()
        th.save(
            {
                "model": net.state_dict(),
                "design_shape": shape,
                "encoder_hid_dim": len(keys),
                "design_min": lower,
                "design_max": upper,
            },
            package / "model.pth",
        )
        config = asdict(args)
        del config["output"]
        metadata = {
            "condition_keys": keys,
            "primary_files": ["model.pth"],
            "training_updates": updates,
            "problem_id": args.problem_id,
            "seed": args.seed,
            "algo": "flow_matching_2d_cond",
        }
        for filename, data in (("metadata.json", metadata), ("run_config.json", config)):
            (package / filename).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        package.rename(args.output)


if __name__ == "__main__":
    train(tyro.cli(Args))
