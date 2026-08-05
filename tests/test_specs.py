"""Tests for the committed evaluation specs.

A spec is the promise that every leaderboard row was produced under identical
conditions. These check the promise is well-formed offline, and -- when the
datasets are reachable -- that every committed spec still reproduces exactly the
conditions it was frozen with.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np
import pytest

from engiopt.evaluation.registry import METRICS
from engiopt.evaluation.spec import _digest
from engiopt.evaluation.spec import EvalSpec
from engiopt.evaluation.spec import SPEC_ROOT

SPEC_PATHS = sorted(SPEC_ROOT.glob("*/*.json"))
SPEC_IDS = [f"{path.parent.name}/{path.stem}" for path in SPEC_PATHS]


class _FakeConditions:
    """A stand-in for the sampled-conditions dataset the digest hashes."""

    def __init__(self, columns: dict[str, Any]):
        self._columns = columns
        self.column_names = list(columns)

    def __getitem__(self, name: str) -> Any:
        return self._columns[name]


# ----------------------------------------------------------------------
# Offline: the committed files are well-formed
# ----------------------------------------------------------------------


@pytest.mark.parametrize("path", SPEC_PATHS, ids=SPEC_IDS)
def test_committed_spec_loads_and_round_trips(path: Any) -> None:
    """A spec must parse into the current dataclass, with no stale fields."""
    spec = EvalSpec.load(str(path))
    assert spec.problem_id == path.parent.name
    assert spec.version == path.stem
    assert dataclasses.asdict(spec) == dataclasses.asdict(EvalSpec(**json.loads(path.read_text())))


@pytest.mark.parametrize("path", SPEC_PATHS, ids=SPEC_IDS)
def test_committed_spec_requests_registered_metrics(path: Any) -> None:
    """A spec asking for a metric nobody registered would fail at evaluation time."""
    for metric in EvalSpec.load(str(path)).metrics:
        assert metric in METRICS, f"{path.name} requests unregistered metric {metric!r}"


@pytest.mark.parametrize("path", SPEC_PATHS, ids=SPEC_IDS)
def test_committed_spec_pins_its_dataset(path: Any) -> None:
    """A package version does not identify a dataset; a revision does.

    Without this, an upstream dataset upload silently changes what every
    published score means.
    """
    spec = EvalSpec.load(str(path))
    assert spec.condition_digest, f"{path.name} was never frozen"
    assert spec.dataset_id, f"{path.name} does not name its dataset"
    assert spec.dataset_revision, f"{path.name} does not pin a dataset revision"


@pytest.mark.parametrize("path", SPEC_PATHS, ids=SPEC_IDS)
def test_multi_objective_specs_declare_a_scalarization(path: Any) -> None:
    """Combining objectives with different units needs a declared rule, not a default."""
    spec = EvalSpec.load(str(path))
    if spec.objective_weights is None and spec.objective_weight_condition is None:
        return
    assert not (spec.objective_weights and spec.objective_weight_condition), (
        f"{path.name} declares two conflicting scalarizations"
    )


# ----------------------------------------------------------------------
# The digest covers everything a score depends on
# ----------------------------------------------------------------------


def test_digest_changes_when_a_condition_value_changes() -> None:
    """Hashing only the row indices would miss an edited condition: row 7 is still row 7."""
    indices = np.array([0, 1, 2])
    designs = np.zeros((3, 4))
    before = _digest(indices, _FakeConditions({"volfrac": [0.3, 0.4, 0.5]}), designs)
    after = _digest(indices, _FakeConditions({"volfrac": [0.3, 0.4, 0.9]}), designs)
    assert before != after


def test_digest_changes_when_a_condition_is_renamed() -> None:
    """The column names are part of the contract, not just the numbers."""
    indices = np.array([0, 1, 2])
    designs = np.zeros((3, 4))
    before = _digest(indices, _FakeConditions({"volfrac": [0.3, 0.4, 0.5]}), designs)
    after = _digest(indices, _FakeConditions({"volume": [0.3, 0.4, 0.5]}), designs)
    assert before != after


def test_digest_is_stable_across_column_order() -> None:
    """Two datasets holding the same conditions must agree, whatever their column order."""
    indices = np.array([0, 1])
    designs = np.zeros((2, 4))
    one = _digest(indices, _FakeConditions({"a": [1.0, 2.0], "b": [3.0, 4.0]}), designs)
    other = _digest(indices, _FakeConditions({"b": [3.0, 4.0], "a": [1.0, 2.0]}), designs)
    assert one == other


def test_digest_covers_array_valued_conditions() -> None:
    """thermoelastic2d's boundary matrices decide the answer as much as the scalars do."""
    indices = np.array([0])
    designs = np.zeros((1, 4))
    before = _digest(indices, _FakeConditions({"fixed_elements": [np.zeros((2, 2))]}), designs)
    after = _digest(indices, _FakeConditions({"fixed_elements": [np.eye(2)]}), designs)
    assert before != after


# ----------------------------------------------------------------------
# Online: every committed spec still resolves
# ----------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.parametrize("path", SPEC_PATHS, ids=SPEC_IDS)
def test_committed_spec_reproduces_its_frozen_conditions(path: Any) -> None:
    """The check the reviewer ran by hand: every spec must resolve, not just parse.

    `resolve` raises if the digest no longer matches, so this fails loudly when
    a dataset moves out from under a published spec.
    """
    from engibench.utils.all_problems import BUILTIN_PROBLEMS

    spec = EvalSpec.load(str(path))
    problem = BUILTIN_PROBLEMS[spec.problem_id]()
    problem.reset(seed=spec.condition_seed)

    resolved = spec.resolve(problem)

    assert len(resolved.ref_designs) == spec.n_samples
    assert resolved.conditions_tensor.shape == (spec.n_samples, len(resolved.condition_keys))
    if spec.volume_condition is not None:
        assert spec.volume_condition in problem.conditions_keys
