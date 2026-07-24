"""Tests for generator discovery and the `Generator` contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch as th

from engiopt.core import ConditionBatch
from engiopt.core import Generator
from engiopt.utils.all_generators import BUILTIN_GENERATORS
from engiopt.utils.all_generators import extract_generator


def test_every_generator_directory_has_an_adapter() -> None:
    """Discovery must find an adapter for every model package."""
    packages = {p.name for p in Path("engiopt/generators").iterdir() if p.is_dir() and not p.name.startswith("_")}
    assert set(BUILTIN_GENERATORS) == packages


def test_algo_ids_match_directory_names() -> None:
    """`algo_id` keys the leaderboard and the HF repo, so it must match the package."""
    for name, generator in BUILTIN_GENERATORS.items():
        assert generator.algo_id == name


def test_template_is_not_registered() -> None:
    """The copy-me template must never appear as a real model."""
    assert "_template" not in BUILTIN_GENERATORS


def test_every_generator_declares_the_files_it_needs() -> None:
    """`checkpoint_files` drives resolution, so it must be non-empty and concrete."""
    for name, generator in BUILTIN_GENERATORS.items():
        assert generator.checkpoint_files, f"{name} declares no checkpoint files"
        assert all(f.endswith(".pth") for f in generator.checkpoint_files), name


def test_extract_generator_rejects_ambiguous_modules() -> None:
    """Two generators in one adapter module would make the registry ambiguous."""

    class _Module:
        First = type("First", (_StubGenerator,), {"algo_id": "a"})
        Second = type("Second", (_StubGenerator,), {"algo_id": "b"})
        __name__ = "fake.adapter"

    with pytest.raises(ValueError, match="Only one generator per adapter"):
        extract_generator(_Module)


class _StubGenerator(Generator):
    """Minimal in-memory generator, standing in for a trained model."""

    algo_id = "stub"
    conditional = True
    design_kinds = ("2d",)

    @classmethod
    def build(cls, resolved: Any, problem: Any, device: Any, **base: Any) -> _StubGenerator:
        raise NotImplementedError

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        # Deliberately the wrong shape and out of range, to prove `sample` fixes both.
        return th.rand(n, int(np.prod(self.design_shape))) * 4.0 - 2.0


def _stub(problem: Any) -> _StubGenerator:
    return _StubGenerator(problem, problem_id="fake", seed=1, device=th.device("cpu"))


def test_sample_normalizes_shape_and_records_cost(fake_problem: Any) -> None:
    """`sample` reshapes to the design shape and times the call."""
    generator = _stub(fake_problem)
    designs = generator.sample(th.zeros(5, 2), n=5)
    assert designs.shape == (5, *fake_problem.design_space.shape)
    assert generator.last_sample_seconds is not None
    assert generator.last_sample_seconds >= 0


def test_sample_applies_output_clip(fake_problem: Any) -> None:
    """`output_clip` bounds designs that a simulator would otherwise reject."""
    generator = _stub(fake_problem)
    generator.output_clip = (0.1, 0.9)
    designs = generator.sample(th.zeros(4, 2), n=4)
    assert designs.min() >= 0.1
    assert designs.max() <= 0.9


def test_sample_infers_n_from_conditions(fake_problem: Any) -> None:
    """Omitting `n` uses the number of supplied conditions."""
    designs = _stub(fake_problem).sample(th.zeros(7, 2))
    assert designs.shape[0] == 7


def test_sample_is_reproducible_for_a_fixed_seed(fake_problem: Any) -> None:
    """Same seed, same designs -- the basis of every reproducibility claim."""
    generator = _stub(fake_problem)
    first = generator.sample(th.zeros(3, 2), n=3, seed=123)
    second = generator.sample(th.zeros(3, 2), n=3, seed=123)
    np.testing.assert_array_equal(first, second)


def test_condition_batch_reports_a_clear_error_when_empty() -> None:
    """A conditional model given nothing should say so by name."""
    with pytest.raises(ValueError, match="my_algo is conditional"):
        ConditionBatch(tensor=None).require_tensor("my_algo")


def test_registry_is_a_plain_mapping_like_builtin_problems() -> None:
    """`BUILTIN_GENERATORS` mirrors `BUILTIN_PROBLEMS`: a dict of name -> class."""
    assert isinstance(BUILTIN_GENERATORS, dict)
    assert all(isinstance(g, type) and issubclass(g, Generator) for g in BUILTIN_GENERATORS.values())
