"""Transformations for the data."""

from collections.abc import Callable
import math

from engibench.core import Problem
from gymnasium import spaces
import torch as th
import torch.nn.functional as f


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


def _nearest_power_of_two(x: int) -> int:
    """Round x to the nearest power of 2."""
    lower = 2 ** math.floor(math.log2(x))
    upper = 2 ** math.ceil(math.log2(x))
    return upper if abs(x - upper) < abs(x - lower) else lower


def upsample_nearest(data: th.Tensor, mode: str="bicubic") -> th.Tensor:
    """Upsample 2D data to the nearest 2^n dimensions. Data should be a Tensor in the format (B, C, H, W)."""
    _, _, h, w = data.shape
    target_h = _nearest_power_of_two(h)
    target_w = _nearest_power_of_two(w)
    # If nearest power of two is smaller, multiply it by 2
    if target_h < h:
        target_h *= 2
    if target_w < w:
        target_w *= 2
    return f.interpolate(data, size=(target_h, target_w), mode=mode)


def downsample_nearest(data: th.Tensor, mode: str="bicubic") -> th.Tensor:
    """Downsample 2D data to the nearest 2^n dimensions. Data should be a Tensor in the format (B, C, H, W)."""
    _, _, h, w = data.shape
    target_h = _nearest_power_of_two(h)
    target_w = _nearest_power_of_two(w)
    # If nearest power of two is larger, divide it by 2
    if target_h > h:
        target_h //= 2
    if target_w > w:
        target_w //= 2
    return f.interpolate(data, size=(target_h, target_w), mode=mode)


def resize_to(data: th.Tensor, h: int, w: int, mode: str = "bicubic") -> th.Tensor:
    """Resize 2D data back to any desired (h, w). Data should be a Tensor in the format (B, C, H, W)."""
    return f.interpolate(data, size=(h, w), mode=mode)
