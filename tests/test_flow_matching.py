"""Fixed-step flow matching integrates a field at the checkpoint's time scale."""

from types import SimpleNamespace

import pytest
import torch

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
