"""Tests for how many conditions a generator is built and sampled for.

`problem.conditions_keys` is the full contract and is broader than what a dense
condition tensor can carry: thermoelastic2d declares seven conditions, four of
which are 65x65 boundary matrices. A network sized from the full list therefore
expects seven columns and receives three. These tests pin the one definition
that training, loading, and sampling all share.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch as th

from engiopt.checkpoint_store import ResolvedCheckpoint
from engiopt.core import condition_keys_for
from engiopt.core import ConditionBatch
from engiopt.core import Generator

# thermoelastic2d's real schema: three scalars among seven declared conditions.
THERMOELASTIC_KEYS = (
    "fixed_elements",
    "force_elements_x",
    "force_elements_y",
    "heatsink_elements",
    "volume_fraction_target",
    "rmin",
    "weight",
)
THERMOELASTIC_SCALARS = ("volume_fraction_target", "rmin", "weight")


class _EchoGenerator(Generator):
    """A generator that returns its condition tensor, so shape errors surface."""

    algo_id = "echo"
    conditional = True

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Any, device: th.device, **base: Any) -> _EchoGenerator:
        """Unused: these tests construct the generator directly."""
        raise NotImplementedError

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        cond = conditions.require_tensor(self.algo_id)
        assert cond.shape[1] == self.n_conds, f"got {cond.shape[1]} condition columns, built for {self.n_conds}"
        return th.zeros((n, *self.design_shape))


def _resolved(metadata: dict[str, Any]) -> ResolvedCheckpoint:
    return ResolvedCheckpoint(source="local", root_dir=".", files={}, run_config={}, metadata=metadata)


def _generator(problem: Any, keys: tuple[str, ...]) -> _EchoGenerator:
    return _EchoGenerator(
        problem=problem, problem_id="thermoelastic2d", seed=1, device=th.device("cpu"), condition_keys=keys
    )


def test_checkpoint_condition_schema_is_preferred_over_the_problem(fake_problem: Any) -> None:
    """A checkpoint states the columns it was trained on, and that wins."""
    resolved = _resolved({"condition_keys": list(THERMOELASTIC_SCALARS)})
    assert condition_keys_for(fake_problem, resolved) == THERMOELASTIC_SCALARS


def test_matrix_valued_conditions_do_not_count_toward_n_conds(fake_problem: Any) -> None:
    """The reviewer's case: 7 declared conditions, 3 columns in the tensor."""
    fake_problem.conditions_keys = list(THERMOELASTIC_KEYS)
    generator = _generator(fake_problem, THERMOELASTIC_SCALARS)
    assert generator.n_conds == len(THERMOELASTIC_SCALARS)
    assert generator.n_conds != len(fake_problem.conditions_keys)


def test_sampling_succeeds_with_the_scalar_condition_schema(fake_problem: Any) -> None:
    """A model built for the scalar columns samples from the scalar columns."""
    fake_problem.conditions_keys = list(THERMOELASTIC_KEYS)
    generator = _generator(fake_problem, THERMOELASTIC_SCALARS)
    batch = ConditionBatch(tensor=th.zeros((4, 3)), keys=THERMOELASTIC_SCALARS)
    assert generator.sample(batch, n=4).shape == (4, *fake_problem.design_space.shape)


def test_conditions_from_a_different_schema_are_refused(fake_problem: Any) -> None:
    """Silently accepting them conditions the design on the wrong numbers."""
    generator = _generator(fake_problem, THERMOELASTIC_SCALARS)
    mismatched = ConditionBatch(tensor=th.zeros((4, 2)), keys=("volfrac", "rmin"))
    with pytest.raises(ValueError, match="was trained on conditions"):
        generator.sample(mismatched, n=4)


def test_unlabelled_conditions_are_accepted(fake_problem: Any) -> None:
    """A bare tensor carries no column names, so there is nothing to contradict."""
    generator = _generator(fake_problem, THERMOELASTIC_SCALARS)
    assert generator.sample(np.zeros((4, 3)), n=4).shape == (4, *fake_problem.design_space.shape)
