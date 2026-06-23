"""Timing helpers for CUDA-aware evaluation measurements."""

from __future__ import annotations

import time

import torch as th


def synchronize_device(device: str | th.device) -> None:
    """Synchronize CUDA work before reading a wall-clock timer."""
    torch_device = th.device(device)
    if torch_device.type == "cuda":
        th.cuda.synchronize(torch_device)


def generation_timer_start(device: str | th.device) -> float:
    """Start a generation timer after pending CUDA work has completed."""
    synchronize_device(device)
    return time.perf_counter()


def generation_timer_elapsed(device: str | th.device, start: float) -> float:
    """Return elapsed generation time after generated CUDA work has completed."""
    synchronize_device(device)
    return time.perf_counter() - start
