"""A clock that times the gradient steps of a training loop and nothing else.

A training script does more than train. It logs, renders sample images, scores a
validation batch and uploads a checkpoint. None of that is the cost of the
method, so none of it should be counted. The clock is started before the loop,
paused around anything that is not a gradient step, and started again after:

    clock = TrainClock(device)
    clock.start()
    for step in range(n_steps):
        ...one gradient step...
        if step % log_interval == 0:
            clock.pause()
            ...log, plot, validate...
            clock.start()
    clock.pause()
    save_checkpoint_package(..., metadata=clock.metadata(n_train_steps=n_steps))

The step count is the part of the cost that does not depend on the machine. The
seconds do, which is why the device is recorded beside them.
"""

from __future__ import annotations

import time
from typing import Any

import torch as th


class TrainClock:
    """Accumulates the wall-clock seconds spent between `start` and `pause`."""

    def __init__(self, device: th.device) -> None:
        self.device = device
        self.seconds = 0.0
        self._started_at: float | None = None

    def start(self) -> None:
        """Start, or resume, timing."""
        self._started_at = time.perf_counter()

    def pause(self) -> None:
        """Stop timing and add the elapsed span to `seconds`.

        CUDA and MPS queue work asynchronously, so the span is only the training
        cost once the device has finished what it was given.
        """
        if self._started_at is None:
            raise RuntimeError("TrainClock.pause() was called while the clock was not running.")
        if self.device.type == "cuda":
            th.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            th.mps.synchronize()
        self.seconds += time.perf_counter() - self._started_at
        self._started_at = None

    def metadata(self, *, n_train_steps: int) -> dict[str, Any]:
        """The fields to pass as `metadata=` to `save_checkpoint_package`.

        Args:
            n_train_steps: Passes of the training loop that were run. One pass is
                one step, however many networks it updates.
        """
        if self._started_at is not None:
            raise RuntimeError("Pause the TrainClock before reading its metadata.")
        return {
            "n_train_steps": int(n_train_steps),
            "train_seconds": self.seconds,
            "train_device": device_name(self.device),
        }


def device_name(device: th.device) -> str:
    """The model name of the device training ran on, e.g. the GPU's product name."""
    if device.type == "cuda":
        return str(th.cuda.get_device_name(device))
    return device.type
