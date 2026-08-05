"""Transformations for the data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymnasium import spaces
import numpy as np
import torch as th
import torch.nn.functional as f

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from datasets import Dataset
    from engibench.core import Problem


def get_scalar_condition_keys(problem: Problem, dataset: Dataset, *, drop_constants: bool = False) -> list[str]:
    """Return the condition keys usable as a dense model input.

    `problem.conditions_keys` is the full contract, which is broader than what a
    generator can consume as a `(n, n_conds)` tensor. Two cases are excluded:

    1. Keys absent from the dataset. Some are solver settings rather than
       per-sample conditions -- photonics2d declares `num_elems_x`,
       `num_elems_y`, and `num_optimization_steps`, none of which vary per row.
    2. Array-valued keys. thermoelastic2d encodes boundary conditions as 65x65
       matrices, which cannot be stacked alongside scalars.

    Args:
        problem: An EngiBench problem instance.
        dataset: A dataset split, e.g. `problem.dataset["test"]`.
        drop_constants: Also drop columns with zero standard deviation.

    Returns:
        Condition names, in `conditions_keys` order.
    """
    scalar_keys = [
        key for key in problem.conditions_keys if key in dataset.column_names and np.asarray(dataset[0][key]).ndim == 0
    ]

    if drop_constants and scalar_keys:
        conds = th.stack([th.as_tensor(dataset[c][:]).float() for c in scalar_keys], dim=1)
        std = conds.std(dim=0)
        scalar_keys = [c for i, c in enumerate(scalar_keys) if std[i] > 0]

    return scalar_keys


def condition_keys(problem: Problem, split: str = "train") -> list[str]:
    """The scalar condition columns a generator is conditioned on, in tensor order.

    This is the one definition of "how many conditions does this problem have"
    that training, checkpoint loading, and sampling all share. Using
    `len(problem.conditions_keys)` instead builds a network for columns that
    never reach it: thermoelastic2d declares 7 conditions, of which 4 are 65x65
    boundary matrices, so a generator sized for 7 fails on a 3-column tensor.

    Args:
        problem: An EngiBench problem instance.
        split: Dataset split to inspect; the schema is the same in all of them.

    Returns:
        Condition names, in `conditions_keys` order.
    """
    dataset = problem.dataset
    return get_scalar_condition_keys(problem, dataset[split] if split in dataset else next(iter(dataset.values())))


def get_image_condition_keys(problem: Problem, dataset: Dataset) -> list[str]:
    """Return the array-valued condition keys present in the dataset.

    The complement of `get_scalar_condition_keys`, e.g. thermoelastic2d's 65x65
    boundary matrices. These cannot travel in the dense scalar condition tensor;
    they reach models as a separate image tensor, and the simulator through the
    conditions dataset.
    """
    return [key for key in problem.conditions_keys if key in dataset.column_names and np.asarray(dataset[0][key]).ndim > 0]


def image_condition_keys(problem: Problem, split: str = "train") -> list[str]:
    """The image-valued condition columns a model can be conditioned on, in channel order.

    The image counterpart of `condition_keys`, and the one definition training,
    checkpoint loading, and sampling share. thermoelastic2d's four 65x65
    boundary masks are the motivating case: where the part is held, loaded, and
    cooled is as much a design requirement as its volume budget.

    Args:
        problem: An EngiBench problem instance.
        split: Dataset split to inspect; the schema is the same in all of them.

    Returns:
        Condition names, in `conditions_keys` order.
    """
    dataset = problem.dataset
    return get_image_condition_keys(problem, dataset[split] if split in dataset else next(iter(dataset.values())))


def stack_image_conditions(conditions: Dataset, keys: Sequence[str], device: th.device | None = None) -> th.Tensor | None:
    """Stack image-valued condition columns into one `(n, len(keys), H, W)` tensor.

    Masks are handed over at their native resolution rather than resized to the
    design grid. thermoelastic2d's are 65x65 while its designs are 64x64, because
    the conditions live on finite-element *nodes* and the design on *elements*;
    quietly resampling one onto the other would erase a real distinction. Models
    that need them on the design grid should resize explicitly, e.g. with
    `resize_to`.

    Args:
        conditions: The sampled conditions dataset.
        keys: Image condition names, in the channel order to stack them.
        device: Device for the result; CPU when omitted.

    Returns:
        The stacked masks, or None when `keys` is empty.

    Raises:
        ValueError: If the columns do not all share one spatial shape, since
            there is then no single tensor to stack them into. Read them from
            `ConditionBatch.dataset` instead.
    """
    if not keys:
        return None
    channels = [np.asarray(conditions[key], dtype=np.float32) for key in keys]
    shapes = {channel.shape[1:] for channel in channels}
    if len(shapes) > 1:
        detail = ", ".join(f"{key}={np.asarray(conditions[key]).shape[1:]}" for key in keys)
        raise ValueError(
            f"Image conditions have differing shapes ({detail}), so they cannot stack into one tensor. "
            "Read them individually from the conditions dataset."
        )
    return th.as_tensor(np.stack(channels, axis=1), dtype=th.float32, device=device)


def flatten_dict_factory(problem: Problem, device: th.device) -> Callable:
    """Factory function to create a flatten_dict function."""

    def flatten_dict(x):
        """Convert each design in the batch to a flattened tensor."""
        flattened = []
        for design in x:
            # Move to CPU for numpy conversion, then back to device
            design_cpu = {k: v.cpu().numpy() if isinstance(v, th.Tensor) else v for k, v in design.items()}
            flattened_array = spaces.flatten(problem.design_space, design_cpu)
            flattened.append(th.tensor(flattened_array, device=device))
        return th.stack(flattened)

    return flatten_dict


def resize_to(data: th.Tensor, h: int, w: int, mode: str = "bicubic") -> th.Tensor:
    """Resize 2D data back to any desired (h, w). Data should be a Tensor in the format (B, C, H, W)."""
    low_dim = 3
    if data.ndim == low_dim:
        data = data.unsqueeze(1)  # (B, 1, H, W)
    return f.interpolate(data, size=(h, w), mode=mode)


def normalize(ds: Dataset, condition_names: list[str]) -> tuple[Dataset, th.Tensor, th.Tensor]:
    """Normalize specified condition columns with global mean/std."""
    # stack condition columns into a single tensor (N, C) on CPU
    conds = th.stack([th.as_tensor(ds[c][:]).float() for c in condition_names], dim=1)
    mean = conds.mean(dim=0)
    std = conds.std(dim=0).clamp(min=1e-8)

    # normalize each condition column (HF expects numpy back)
    ds = ds.map(
        lambda batch: {
            c: ((th.as_tensor(batch[c][:]).float() - mean[i]) / std[i]).numpy() for i, c in enumerate(condition_names)
        },
        batched=True,
    )

    return ds, mean, std


def drop_constant(ds: Dataset, condition_names: list[str]) -> tuple[Dataset, list[str]]:
    """Drop constant condition columns (std=0) from dataset."""
    conds = th.stack([th.as_tensor(ds[c][:]).float() for c in condition_names], dim=1)
    std = conds.std(dim=0)

    kept = [c for i, c in enumerate(condition_names) if std[i] > 0]
    dropped = [c for i, c in enumerate(condition_names) if std[i] == 0]

    if dropped:
        print(f"Warning: Dropping constant condition columns (std=0): {dropped}")

    # remove dropped columns from dataset
    ds = ds.remove_columns(dropped)

    return ds, kept
