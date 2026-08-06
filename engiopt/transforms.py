"""Transformations for the data."""

from collections.abc import Callable

from datasets import Dataset
from engibench.core import Problem
from gymnasium import spaces
import numpy as np
import torch as th
import torch.nn.functional as f


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
    boundary matrices. These still reach the simulator through the conditions
    dataset; they simply cannot travel in the dense condition tensor.
    """
    return [key for key in problem.conditions_keys if key in dataset.column_names and np.asarray(dataset[0][key]).ndim > 0]


def get_image_condition_shape(dataset: Dataset, img_keys: list[str]) -> tuple[int, ...]:
    """Return the spatial shape of the first image condition.

    All image conditions on a problem are assumed to share spatial dimensions.

    Args:
        dataset: A dataset split.
        img_keys: Image condition keys, from `get_image_condition_keys`.

    Returns:
        The shape, e.g. `(65, 65)`.
    """
    return tuple(np.asarray(dataset[0][img_keys[0]]).shape)


def rasterize_index_conditions(dataset: Dataset, keys: list[str], grid_shape: tuple[int, int]) -> th.Tensor:
    """Turn sparse node-index conditions into dense binary masks.

    Several problems store boundary conditions as variable-length arrays of flat
    node indices, e.g. `fixed_elements = [23, 24, 25, ...]`. Those cannot travel
    in a dense condition tensor, but they are exactly the information a
    conditional decoder needs, so they are rasterized onto the mesh instead.

    For a design grid of `(H, W)` the node grid is usually `(H + 1, W + 1)` --
    pixel corners rather than pixel centres -- so pass the *node* grid shape or
    indices will unravel onto the wrong rows.

    Args:
        dataset: A dataset split.
        keys: Condition keys holding flat node indices.
        grid_shape: `(H, W)` of the node grid to rasterize onto.

    Returns:
        Float tensor of dense masks, shape `(N, len(keys), H, W)`.
    """
    n_samples = len(dataset[keys[0]])
    height, width = grid_shape
    masks = th.zeros(n_samples, len(keys), height, width)
    for channel, key in enumerate(keys):
        for index in range(n_samples):
            indices = th.as_tensor(np.asarray(dataset[index][key])).long()
            if indices.numel() == 0:
                continue
            rows, cols = indices // width, indices % width
            valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            masks[index, channel, rows[valid], cols[valid]] = 1.0
    return masks


def get_performance_target(problem: Problem, dataset: Dataset) -> th.Tensor:
    """Build the performance target a predictor regresses onto.

    Multi-objective problems return the full objective vector rather than a
    scalarization: the predictor should learn each objective independently, and
    scalarizing here would bake in a weighting the model cannot undo.

    Args:
        problem: The problem, which declares the objectives.
        dataset: A dataset split holding the objective columns.

    Returns:
        Tensor of shape `(N, 1)` for single-objective problems, `(N, n_objs)`
        otherwise.
    """
    obj_keys = [name for name, _ in problem.objectives]
    if len(obj_keys) == 1:
        return th.as_tensor(dataset[obj_keys[0]][:]).float().unsqueeze(-1)
    return th.stack([th.as_tensor(dataset[key][:]).float() for key in obj_keys], dim=-1)


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
