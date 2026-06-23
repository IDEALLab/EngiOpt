"""Timing helpers for CUDA-aware evaluation measurements."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any

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


@dataclass(frozen=True)
class TimingSummary:
    """Summary statistics for repeated generation timing."""

    total_runtime_sec: float
    mean_runtime_sec: float
    median_runtime_sec: float
    std_runtime_sec: float
    min_runtime_sec: float
    max_runtime_sec: float
    timed_repeats: int


def summarize_durations(durations: list[float]) -> TimingSummary:
    """Summarize repeated wall-clock durations deterministically."""
    if not durations:
        raise ValueError("At least one duration is required")

    ordered = sorted(float(value) for value in durations)
    total = sum(ordered)
    count = len(ordered)
    mean = total / count
    mid = count // 2
    median = ordered[mid] if count % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    if count > 1:
        variance = sum((value - mean) ** 2 for value in ordered) / (count - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    return TimingSummary(
        total_runtime_sec=total,
        mean_runtime_sec=mean,
        median_runtime_sec=median,
        std_runtime_sec=std,
        min_runtime_sec=ordered[0],
        max_runtime_sec=ordered[-1],
        timed_repeats=count,
    )


def benchmark_callable(
    fn: Callable[[], Any],
    *,
    device: str | th.device,
    warmup_repeats: int,
    timed_repeats: int,
) -> TimingSummary:
    """Benchmark a callable with warm-up and CUDA synchronization around each timed call."""
    if warmup_repeats < 0:
        raise ValueError("warmup_repeats must be non-negative")
    if timed_repeats <= 0:
        raise ValueError("timed_repeats must be positive")

    with th.no_grad():
        for _ in range(warmup_repeats):
            fn()
        synchronize_device(device)

        durations: list[float] = []
        for _ in range(timed_repeats):
            start = generation_timer_start(device)
            fn()
            durations.append(generation_timer_elapsed(device, start))

    return summarize_durations(durations)
