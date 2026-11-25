"""Fixed-step flow matching integrates a field at the checkpoint's time scale."""

from types import SimpleNamespace

import pytest
import torch

from engiopt.checkpoint_store import ResolvedCheckpoint
from engiopt.generators.flow_matching_2d_cond.adapter import FlowMatching2DCond
from engiopt.generators.flow_matching_2d_cond.model import integrate


@pytest.mark.parametrize(("method", "expected", "calls"), [("euler", 0.0, 1), ("midpoint", 0.5, 2), ("rk4", 0.5, 4)])
def test_linear_time_field(method, expected, calls):
    times = []

    def field(x, time, conditions):
        times.append(time.item())
        return SimpleNamespace(sample=torch.ones_like(x) * time * conditions)

    result = integrate(field, torch.zeros(1), torch.tensor([0.1]), 1, 11, method)
    assert len(times) == calls
    torch.testing.assert_close(result, torch.tensor([expected]))


@pytest.mark.parametrize(("method", "step_factor"), [("euler", 1.5), ("midpoint", 1.625), ("rk4", 1.6484375)])
def test_state_dependent_field(method, step_factor):
    # For dx/dt = x, each solver takes a different Taylor approximation to exp(dt).
    def field(x, time, conditions):
        return SimpleNamespace(sample=x)

    result = integrate(field, torch.ones(1), torch.zeros(1), 2, 1000, method)
    torch.testing.assert_close(result, torch.tensor([step_factor**2]))


@pytest.mark.parametrize(("steps", "method"), [(0, "euler"), (1, "unsupported")])
def test_invalid_solver(steps, method):
    with pytest.raises(ValueError, match="Invalid flow integration"):
        integrate(None, torch.ones(1), torch.zeros(1), steps, 1000, method)


@pytest.mark.parametrize(
    ("problem_id", "seed", "message"),
    [("beams2d", 2, "Checkpoint seed"), ("heatconduction2d", 1, "Checkpoint problem_id"), ("beams2d", 1, "conditions")],
)
def test_refuse_mismatched_study_package(fake_problem, problem_id, seed, message):
    config = {"seed": 1, "problem_id": "beams2d"}
    package = ResolvedCheckpoint("local", "", {}, config, {"bundle_type": "selected_checkpoint"})
    # Even four names are not interchangeable: this order swaps the first two.
    keys = ("rmin", "volfrac", "forcedist", "overhang_constraint")
    with pytest.raises(ValueError, match=message):
        FlowMatching2DCond.build(
            package, fake_problem, torch.device("cpu"), problem_id=problem_id, seed=seed, condition_keys=keys
        )
