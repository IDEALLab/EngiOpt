"""Tests for image (field) conditions.

Some problems state part of the requirement as a field rather than a number:
thermoelastic2d gives four 65x65 masks for where the part is held, loaded, and
cooled. These cannot ride in the scalar condition tensor, so they travel in
their own, and these tests pin the contract a model can rely on.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch as th

from engiopt.checkpoint_store import ResolvedCheckpoint
from engiopt.core import ConditionBatch
from engiopt.core import Generator
from engiopt.core import image_condition_keys_for
from engiopt.transforms import stack_image_conditions

# thermoelastic2d's real schema: four 65x65 masks alongside three scalars.
IMAGE_KEYS = ("fixed_elements", "force_elements_x", "force_elements_y", "heatsink_elements")
MASK_SHAPE = (65, 65)


class _Conditions:
    """A stand-in for the sampled-conditions dataset."""

    def __init__(self, columns: dict[str, Any]):
        self._columns = columns
        self.column_names = list(columns)

    def __getitem__(self, name: str) -> Any:
        return self._columns[name]


def _masks(n: int = 4, shape: tuple[int, int] = MASK_SHAPE) -> _Conditions:
    rng = np.random.default_rng(0)
    return _Conditions({key: rng.random((n, *shape)) for key in IMAGE_KEYS})


class _ImageConditionedGenerator(Generator):
    """A model that consumes the masks, as a real image-conditioned model would."""

    algo_id = "image_demo"
    conditional = True
    image_conditional = True

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Any, device: th.device, **base: Any) -> Generator:
        """Unused: these tests construct the generator directly."""
        raise NotImplementedError

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        masks = conditions.require_images(self.algo_id)
        assert masks.shape[1] == self.n_image_conds, f"got {masks.shape[1]} channels, built for {self.n_image_conds}"
        # A real model would resize the masks onto its design grid here.
        return th.zeros((n, *self.design_shape))


def _generator(problem: Any, image_keys: tuple[str, ...] = IMAGE_KEYS) -> _ImageConditionedGenerator:
    return _ImageConditionedGenerator(
        problem=problem,
        problem_id="thermoelastic2d",
        seed=1,
        device=th.device("cpu"),
        condition_keys=("volume_fraction_target",),
        image_condition_keys=image_keys,
    )


# ----------------------------------------------------------------------
# Stacking
# ----------------------------------------------------------------------


def test_masks_stack_into_channels_in_declared_order() -> None:
    """One tensor per sample, one channel per named condition."""
    stacked = stack_image_conditions(_masks(n=6), IMAGE_KEYS)
    assert stacked is not None
    assert stacked.shape == (6, 4, *MASK_SHAPE)


def test_channel_order_follows_the_keys_given() -> None:
    """Channel k must be key k, or a model reads the wrong field."""
    conditions = _masks(n=2)
    stacked = stack_image_conditions(conditions, IMAGE_KEYS)
    assert stacked is not None
    for channel, key in enumerate(IMAGE_KEYS):
        assert np.allclose(stacked[:, channel].numpy(), np.asarray(conditions[key], dtype=np.float32))


def test_no_image_conditions_is_not_an_error() -> None:
    """Most problems have none; they simply get no image tensor."""
    assert stack_image_conditions(_masks(), ()) is None


def test_masks_of_differing_shapes_refuse_to_stack() -> None:
    """There is no one tensor to stack them into, so say so instead of guessing."""
    conditions = _Conditions({"fixed_elements": np.zeros((2, 65, 65)), "heatsink_elements": np.zeros((2, 33, 33))})
    with pytest.raises(ValueError, match="differing shapes"):
        stack_image_conditions(conditions, ("fixed_elements", "heatsink_elements"))


def test_masks_keep_their_native_resolution() -> None:
    """65x65 masks against a 64x64 design: nodes vs elements, not a rounding error.

    Resampling one onto the other in the shared layer would erase a real
    distinction, so the contract hands them over untouched.
    """
    stacked = stack_image_conditions(_masks(n=3), IMAGE_KEYS)
    assert stacked is not None
    assert stacked.shape[2:] == MASK_SHAPE


# ----------------------------------------------------------------------
# The contract a model relies on
# ----------------------------------------------------------------------


def test_an_image_conditioned_model_receives_its_masks(fake_problem: Any) -> None:
    """The end-to-end path: batch in, masks reach `_sample`, designs out."""
    generator = _generator(fake_problem)
    batch = ConditionBatch(
        tensor=th.zeros((4, 1)),
        images=th.zeros((4, 4, *MASK_SHAPE)),
        keys=("volume_fraction_target",),
        image_keys=IMAGE_KEYS,
    )
    assert generator.sample(batch, n=4).shape == (4, *fake_problem.design_space.shape)
    assert generator.n_image_conds == 4


def test_a_model_told_it_is_image_conditioned_says_so_when_given_none(fake_problem: Any) -> None:
    """A clear message beats a None dereference inside the network."""
    generator = _generator(fake_problem)
    with pytest.raises(ValueError, match="image-conditioned"):
        generator.sample(ConditionBatch(tensor=th.zeros((2, 1)), keys=("volume_fraction_target",)), n=2)


def test_masks_from_a_different_schema_are_refused(fake_problem: Any) -> None:
    """Same protection the scalar conditions get: wrong channels, not wrong shape."""
    generator = _generator(fake_problem)
    batch = ConditionBatch(
        tensor=th.zeros((2, 1)),
        images=th.zeros((2, 2, *MASK_SHAPE)),
        keys=("volume_fraction_target",),
        image_keys=("fixed_elements", "something_else"),
    )
    with pytest.raises(ValueError, match="image conditions"):
        generator.sample(batch, n=2)


def test_the_checkpoint_schema_wins_over_the_problem(fake_problem: Any) -> None:
    """An old checkpoint keeps loading when the problem later gains a field condition."""
    resolved = ResolvedCheckpoint(
        source="local",
        root_dir=".",
        files={},
        run_config={},
        metadata={"image_condition_keys": ["fixed_elements", "heatsink_elements"]},
    )
    assert image_condition_keys_for(fake_problem, resolved) == ("fixed_elements", "heatsink_elements")


def test_models_are_image_unconditioned_by_default() -> None:
    """Every existing model must keep working without knowing this exists."""
    from engiopt.utils.all_generators import BUILTIN_GENERATORS

    assert not any(generator.image_conditional for generator in BUILTIN_GENERATORS.values())
