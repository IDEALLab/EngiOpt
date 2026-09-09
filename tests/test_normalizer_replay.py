"""Tests that a checkpoint's preprocessing travels with its weights.

Several 1D and Bezier models scale designs or conditions into `[0, 1]` using
bounds fitted on the training split. `Normalizer` is a plain class rather than an
`nn.Module`, so those bounds never reached the state dict, and loading refitted
them from whatever dataset was current. Same checkpoint, same content hash,
different dataset revision, different outputs -- with nothing to show it happened.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch as th

from engiopt.core import recorded_condition_normalizer
from engiopt.core import recorded_design_normalizer
from engiopt.transforms import load_normalizer_state
from engiopt.transforms import normalizer_state


class _Normalizer:
    """Stand-in with the same surface as the five training scripts' copies."""

    def __init__(self, min_val: th.Tensor, max_val: th.Tensor, eps: float = 1e-7) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.eps = eps

    def normalize(self, x: th.Tensor) -> th.Tensor:
        return (x - self.min_val) / (self.max_val - self.min_val + self.eps)

    def denormalize(self, x: th.Tensor) -> th.Tensor:
        return x * (self.max_val - self.min_val + self.eps) + self.min_val


class _Resolved:
    """Minimal `ResolvedCheckpoint` stand-in carrying only metadata."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata


def test_recorded_bounds_survive_the_dataset_moving() -> None:
    """The property the whole change exists for.

    A checkpoint trained when designs spanned `[0, 10]` must keep decoding
    against `[0, 10]`, even when today's dataset spans `[0, 100]`.
    """
    trained = _Normalizer(th.tensor([0.0]), th.tensor([10.0]))
    recorded = normalizer_state(trained)

    # Later: the dataset revision moved, so loading fits very different bounds.
    refitted = _Normalizer(th.tensor([0.0]), th.tensor([100.0]))
    restored = load_normalizer_state(refitted, recorded, th.device("cpu"))

    sample = th.tensor([[0.5]])
    assert th.allclose(restored.denormalize(sample), trained.denormalize(sample))
    assert not th.allclose(
        restored.denormalize(sample), _Normalizer(th.tensor([0.0]), th.tensor([100.0])).denormalize(sample)
    )


def test_state_round_trips_through_json_shaped_values() -> None:
    """Metadata is JSON, so the recorded form has to be plain lists and floats."""
    normalizer = _Normalizer(th.tensor([1.0, 2.0]), th.tensor([3.0, 4.0]), eps=1e-5)
    state = normalizer_state(normalizer)

    assert state == {"min": [1.0, 2.0], "max": [3.0, 4.0], "eps": 1e-5}

    restored = load_normalizer_state(_Normalizer(th.zeros(2), th.ones(2)), state, th.device("cpu"))
    assert th.allclose(restored.min_val, th.tensor([1.0, 2.0]))
    assert th.allclose(restored.max_val, th.tensor([3.0, 4.0]))
    assert restored.eps == pytest.approx(1e-5)


def test_multidimensional_bounds_keep_their_shape() -> None:
    """Bezier scalar normalizers are not one-dimensional; flattening must be reversed."""
    normalizer = _Normalizer(th.zeros(2, 3), th.ones(2, 3))
    state = normalizer_state(normalizer)

    restored = load_normalizer_state(_Normalizer(th.zeros(2, 3), th.zeros(2, 3)), state, th.device("cpu"))

    assert restored.max_val.shape == (2, 3)


def test_a_checkpoint_without_recorded_bounds_is_left_alone() -> None:
    """Packages predating this must load exactly as before -- fitted from the dataset.

    Those runs really did use the current dataset's bounds, so refitting is the
    faithful reconstruction, not a fallback that quietly changes their meaning.
    """
    fitted = _Normalizer(th.tensor([0.0]), th.tensor([7.0]))

    unchanged = load_normalizer_state(fitted, None, th.device("cpu"))

    assert th.allclose(unchanged.max_val, th.tensor([7.0]))


def test_partial_metadata_is_ignored_rather_than_half_applied() -> None:
    """Half a normalizer is worse than none: it would scale by mismatched bounds."""
    fitted = _Normalizer(th.tensor([0.0]), th.tensor([7.0]))

    unchanged = load_normalizer_state(fitted, {"min": [1.0]}, th.device("cpu"))

    assert th.allclose(unchanged.min_val, th.tensor([0.0]))
    assert th.allclose(unchanged.max_val, th.tensor([7.0]))


# ----------------------------------------------------------------------
# Reading the recorded state back off a checkpoint
# ----------------------------------------------------------------------


def test_recorded_accessors_read_their_own_field() -> None:
    """Condition and design bounds are different numbers and must not be crossed."""
    resolved = _Resolved(
        {
            "condition_normalizer": {"min": [0.0], "max": [1.0]},
            "design_normalizer": {"min": [-5.0], "max": [5.0]},
        }
    )

    assert recorded_condition_normalizer(resolved) == {"min": [0.0], "max": [1.0]}
    assert recorded_design_normalizer(resolved) == {"min": [-5.0], "max": [5.0]}


@pytest.mark.parametrize(
    "metadata",
    [{}, {"design_normalizer": None}, {"design_normalizer": {"min": [0.0]}}, {"design_normalizer": "nonsense"}],
)
def test_unusable_recorded_state_reads_as_absent(metadata: dict[str, Any]) -> None:
    """Anything that is not a complete pair of bounds means "fall back to fitting"."""
    assert recorded_design_normalizer(_Resolved(metadata)) is None


def test_no_checkpoint_reads_as_absent() -> None:
    """Adapters call these before a package exists in some code paths."""
    assert recorded_design_normalizer(None) is None
    assert recorded_condition_normalizer(None) is None
