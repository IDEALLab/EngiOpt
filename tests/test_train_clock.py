"""The training clock: it counts the spans it is running for and nothing else."""

from __future__ import annotations

import time

import pytest
import torch as th

from engiopt.train_clock import TrainClock

SPAN = 0.05


def test_the_clock_counts_only_the_spans_it_is_running() -> None:
    clock = TrainClock(th.device("cpu"))
    clock.start()
    time.sleep(SPAN)
    clock.pause()
    time.sleep(4 * SPAN)  # logging, validation, an upload: not training
    clock.start()
    time.sleep(SPAN)
    clock.pause()
    assert 2 * SPAN <= clock.seconds < 4 * SPAN, "the paused span must not be counted"


def test_the_metadata_carries_the_count_the_seconds_and_the_device() -> None:
    clock = TrainClock(th.device("cpu"))
    clock.start()
    clock.pause()
    metadata = clock.metadata(n_train_steps=1200)
    assert metadata == {"n_train_steps": 1200, "train_seconds": clock.seconds, "train_device": "cpu"}


def test_a_running_clock_refuses_to_report() -> None:
    """A clock read while still running would report a time that leaves out the last span."""
    clock = TrainClock(th.device("cpu"))
    clock.start()
    with pytest.raises(RuntimeError, match="Pause"):
        clock.metadata(n_train_steps=1)


def test_starting_a_running_clock_is_an_error() -> None:
    """A second start would silently drop the span since the first."""
    clock = TrainClock(th.device("cpu"))
    clock.start()
    with pytest.raises(RuntimeError, match="already running"):
        clock.start()


def test_pausing_a_stopped_clock_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="not running"):
        TrainClock(th.device("cpu")).pause()
