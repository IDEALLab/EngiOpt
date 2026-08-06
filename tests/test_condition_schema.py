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


# ----------------------------------------------------------------------
# A model sees the conditions the way its training data looked
# ----------------------------------------------------------------------


class _RecordingGenerator(_EchoGenerator):
    """Captures the tensor `_sample` actually received."""

    algo_id = "recorder"
    received: th.Tensor | None = None

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        type(self).received = conditions.require_tensor(self.algo_id)
        return th.zeros((n, *self.design_shape))


def _recorder(problem: Any, **kwargs: Any) -> _RecordingGenerator:
    type(_RecordingGenerator).received = None
    return _RecordingGenerator(problem=problem, problem_id="beams2d", seed=1, device=th.device("cpu"), **kwargs)


def test_a_model_trained_on_fewer_conditions_gets_them_projected(fake_problem: Any) -> None:
    """VQGAN drops constant columns during training; the evaluator supplies all of them.

    The checkpoint's schema is authoritative, so the extra columns are projected
    away rather than rejected.
    """
    generator = _recorder(fake_problem, condition_keys=("volfrac",))
    batch = ConditionBatch(tensor=th.tensor([[0.3, 2.0], [0.5, 2.0]]), keys=("volfrac", "rmin"))

    generator.sample(batch, n=2)

    assert _RecordingGenerator.received is not None
    assert _RecordingGenerator.received.shape == (2, 1)
    assert _RecordingGenerator.received.flatten().tolist() == pytest.approx([0.3, 0.5])


def test_columns_are_reordered_into_the_trained_order(fake_problem: Any) -> None:
    """Channel k must be the condition the network learned in slot k."""
    generator = _recorder(fake_problem, condition_keys=("rmin", "volfrac"))
    batch = ConditionBatch(tensor=th.tensor([[0.3, 2.0]]), keys=("volfrac", "rmin"))

    generator.sample(batch, n=1)

    assert _RecordingGenerator.received is not None
    assert _RecordingGenerator.received.flatten().tolist() == pytest.approx([2.0, 0.3])


def test_a_condition_the_model_needs_but_did_not_get_is_an_error(fake_problem: Any) -> None:
    """A missing column cannot be reconstructed, so this one still refuses."""
    generator = _recorder(fake_problem, condition_keys=("volfrac", "brand_new"))
    batch = ConditionBatch(tensor=th.tensor([[0.3, 2.0]]), keys=("volfrac", "rmin"))

    with pytest.raises(ValueError, match="brand_new"):
        generator.sample(batch, n=1)


def test_recorded_normalization_is_replayed_not_refitted(fake_problem: Any) -> None:
    """The training split's statistics, not the evaluation sample's.

    Refitting on the evaluation rows hands the network a different scale than it
    was trained on -- wrong designs, no error.
    """
    generator = _recorder(fake_problem, condition_keys=("volfrac",), condition_stats=([0.4], [0.2]))
    batch = ConditionBatch(tensor=th.tensor([[0.6], [0.4]]), keys=("volfrac",))

    generator.sample(batch, n=2)

    # (0.6 - 0.4) / 0.2 = 1.0 ; (0.4 - 0.4) / 0.2 = 0.0 -- and note the evaluation
    # sample's own mean is 0.5, which would have produced a different answer.
    assert _RecordingGenerator.received is not None
    assert _RecordingGenerator.received.flatten().tolist() == pytest.approx([1.0, 0.0])


def test_a_model_without_recorded_statistics_is_left_alone(fake_problem: Any) -> None:
    """Most models feed conditions through unscaled, and must keep doing so."""
    generator = _recorder(fake_problem, condition_keys=("volfrac",))
    generator.sample(ConditionBatch(tensor=th.tensor([[0.6]]), keys=("volfrac",)), n=1)

    assert _RecordingGenerator.received is not None
    assert _RecordingGenerator.received.flatten().tolist() == pytest.approx([0.6])
