"""Core utilities for the conditional 2D flow matching baseline."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import is_dataclass
import os
from typing import Any

import torch as th
from torch.nn import functional

DEFAULT_BLOCK_OUT_CHANNELS = (32, 64, 128, 256)
DEFAULT_DOWN_BLOCK_TYPES = ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")
DEFAULT_UP_BLOCK_TYPES = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")


def build_model(
    design_shape: tuple[int, int],
    encoder_hid_dim: int,
    layers_per_block: int,
):
    """Build the conditional UNet backbone used for flow matching."""
    from diffusers import UNet2DConditionModel

    return UNet2DConditionModel(
        sample_size=design_shape,
        in_channels=1,
        out_channels=1,
        cross_attention_dim=64,
        block_out_channels=DEFAULT_BLOCK_OUT_CHANNELS,
        down_block_types=DEFAULT_DOWN_BLOCK_TYPES,
        up_block_types=DEFAULT_UP_BLOCK_TYPES,
        layers_per_block=layers_per_block,
        transformer_layers_per_block=1,
        encoder_hid_dim=encoder_hid_dim,
        only_cross_attention=True,
    )


def sample_time_uniform(batch_size: int, device: th.device) -> th.Tensor:
    """Sample continuous time values in [0, 1)."""
    return th.rand((batch_size,), device=device, dtype=th.float32)


def expand_time(time: th.Tensor, ndim: int) -> th.Tensor:
    """Expand a batch of times so it can broadcast with image tensors."""
    return time.view(-1, *([1] * (ndim - 1)))


def linear_interpolate(noise: th.Tensor, clean_designs: th.Tensor, time: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    """Construct the linear probability path and its target velocity."""
    time_expanded = expand_time(time, clean_designs.ndim)
    xt = (1.0 - time_expanded) * noise + time_expanded * clean_designs
    target_velocity = clean_designs - noise
    return xt, target_velocity


def scale_continuous_time(time: th.Tensor, num_train_timesteps: int) -> th.Tensor:
    """Map continuous [0, 1] time values into the model timestep embedding range."""
    max_timestep = max(num_train_timesteps - 1, 1)
    return time.to(dtype=th.float32) * float(max_timestep)


def predict_velocity(
    model,
    state: th.Tensor,
    time: th.Tensor,
    encoder_hidden_states: th.Tensor,
    num_train_timesteps: int,
) -> th.Tensor:
    """Run the model and return the predicted flow velocity."""
    model_timesteps = scale_continuous_time(time, num_train_timesteps)
    return model(state, model_timesteps, encoder_hidden_states).sample


def compute_flow_matching_loss(
    model,
    clean_designs: th.Tensor,
    encoder_hidden_states: th.Tensor,
    num_train_timesteps: int,
) -> tuple[th.Tensor, th.Tensor]:
    """Sample a flow-matching training pair and compute the MSE objective."""
    time = sample_time_uniform(clean_designs.shape[0], clean_designs.device)
    noise = th.randn_like(clean_designs)
    xt, target_velocity = linear_interpolate(noise, clean_designs, time)
    predicted_velocity = predict_velocity(model, xt, time, encoder_hidden_states, num_train_timesteps)
    loss = functional.mse_loss(predicted_velocity, target_velocity)
    return loss, time


def euler_integrate(
    model,
    initial_state: th.Tensor,
    encoder_hidden_states: th.Tensor,
    integration_steps: int,
    num_train_timesteps: int,
) -> th.Tensor:
    """Integrate the learned velocity field from noise to data with Euler steps."""
    if integration_steps <= 0:
        raise ValueError("integration_steps must be positive")

    state = initial_state
    time_grid = th.linspace(0.0, 1.0, integration_steps + 1, device=state.device, dtype=state.dtype)

    for step in range(integration_steps):
        current_time = th.full((state.shape[0],), float(time_grid[step].item()), device=state.device, dtype=state.dtype)
        dt = time_grid[step + 1] - time_grid[step]
        velocity = predict_velocity(model, state, current_time, encoder_hidden_states, num_train_timesteps)
        state = state + dt * velocity

    return state


def generate_samples(
    model,
    design_shape: tuple[int, int],
    encoder_hidden_states: th.Tensor,
    integration_steps: int,
    num_train_timesteps: int,
    device: th.device,
) -> th.Tensor:
    """Generate designs by integrating from Gaussian noise."""
    initial_state = th.randn((encoder_hidden_states.shape[0], 1, *design_shape), device=device)
    return euler_integrate(model, initial_state, encoder_hidden_states, integration_steps, num_train_timesteps)


def args_to_dict(args: Any) -> dict[str, Any]:
    """Convert a dataclass or mapping-like config into a serializable dictionary."""
    if is_dataclass(args):
        return asdict(args)
    return dict(args)


def load_local_checkpoint(checkpoint_path: str, device: th.device) -> dict[str, Any]:
    """Load a local checkpoint with a clear error for missing files."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return th.load(checkpoint_path, map_location=device)

