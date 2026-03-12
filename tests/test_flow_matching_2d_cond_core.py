"""Tests for the conditional 2D flow-matching helpers."""

from __future__ import annotations

import importlib.util

import pytest

torch = pytest.importorskip("torch")

from engiopt.flow_matching_2d_cond.core import euler_integrate
from engiopt.flow_matching_2d_cond.core import linear_interpolate
from engiopt.flow_matching_2d_cond.core import load_local_checkpoint
from engiopt.flow_matching_2d_cond.core import sample_time_uniform
from engiopt.flow_matching_2d_cond.core import scale_continuous_time


class ConstantVelocityModel:
    """Tiny stand-in model that always predicts unit velocity."""

    def __call__(self, state, timesteps, encoder_hidden_states):  # noqa: ANN001
        del timesteps
        del encoder_hidden_states
        return type("Output", (), {"sample": torch.ones_like(state)})()


def test_sample_time_uniform_stays_in_unit_interval():
    sampled_time = sample_time_uniform(64, torch.device("cpu"))

    assert sampled_time.shape == (64,)
    assert torch.all(sampled_time >= 0.0)
    assert torch.all(sampled_time <= 1.0)


def test_linear_interpolate_matches_bridge_definition():
    noise = torch.zeros((2, 1, 4, 4))
    clean_designs = torch.ones((2, 1, 4, 4))
    time = torch.tensor([0.25, 0.75])

    xt, target_velocity = linear_interpolate(noise, clean_designs, time)

    assert torch.allclose(xt[0], torch.full((1, 4, 4), 0.25))
    assert torch.allclose(xt[1], torch.full((1, 4, 4), 0.75))
    assert torch.allclose(target_velocity, torch.ones_like(clean_designs))


def test_scale_continuous_time_maps_into_embedding_range():
    time = torch.tensor([0.0, 0.5, 1.0])
    scaled = scale_continuous_time(time, num_train_timesteps=1000)

    assert torch.allclose(scaled, torch.tensor([0.0, 499.5, 999.0]))


def test_euler_integrate_matches_constant_velocity_field():
    model = ConstantVelocityModel()
    initial_state = torch.zeros((3, 1, 4, 4))
    encoder_hidden_states = torch.zeros((3, 1, 2))

    final_state = euler_integrate(
        model=model,
        initial_state=initial_state,
        encoder_hidden_states=encoder_hidden_states,
        integration_steps=4,
        num_train_timesteps=1000,
    )

    assert torch.allclose(final_state, torch.ones_like(initial_state), atol=1e-6)


def test_load_local_checkpoint_raises_clear_error(tmp_path):
    missing_path = tmp_path / "missing-model.pth"

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_local_checkpoint(str(missing_path), torch.device("cpu"))


@pytest.mark.skipif(importlib.util.find_spec("diffusers") is None, reason="diffusers is not installed")
def test_unet_forward_matches_input_shape():
    from engiopt.flow_matching_2d_cond.core import build_model

    model = build_model(design_shape=(16, 16), encoder_hid_dim=3, layers_per_block=1)
    input_state = torch.randn((2, 1, 16, 16))
    time = torch.tensor([0.1, 0.9], dtype=torch.float32)
    encoder_hidden_states = torch.randn((2, 1, 3))

    output = model(input_state, scale_continuous_time(time, 1000), encoder_hidden_states).sample

    assert output.shape == input_state.shape
