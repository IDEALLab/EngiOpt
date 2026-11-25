"""Velocity field and fixed-step solvers used in the EngOpt study."""

from __future__ import annotations

from diffusers import UNet2DConditionModel
import torch as th


def build_model(shape: tuple[int, ...], n_conditions: int, layers: int) -> UNet2DConditionModel:
    """Build the conditional velocity network used by the published checkpoints."""
    return UNet2DConditionModel(
        sample_size=shape,
        in_channels=1,
        out_channels=1,
        cross_attention_dim=64,
        block_out_channels=(32, 64, 128, 256),
        down_block_types=("CrossAttnDownBlock2D",) * 3 + ("DownBlock2D",),
        up_block_types=("UpBlock2D",) + ("CrossAttnUpBlock2D",) * 3,
        layers_per_block=layers,
        transformer_layers_per_block=1,
        encoder_hid_dim=n_conditions,
        only_cross_attention=True,
    )


def integrate(
    net: UNet2DConditionModel, state: th.Tensor, conditions: th.Tensor, steps: int, timesteps: int, method: str
) -> th.Tensor:
    """Integrate from the supplied Gaussian noise to a design over unit time."""
    if steps < 1 or timesteps < 1 or method not in {"euler", "midpoint", "rk4"}:
        raise ValueError(f"Invalid flow integration parameters: {method=}, {steps=}, {timesteps=}")

    def velocity(x: th.Tensor, t: float) -> th.Tensor:
        time = th.full((len(x),), t, device=x.device, dtype=th.float32) * max(timesteps - 1, 1)
        return net(x, time, conditions).sample

    dt = 1.0 / steps
    for i in range(steps):
        t = i * dt
        k1 = velocity(state, t)
        if method == "euler":
            state = state + dt * k1
            continue
        k2 = velocity(state + 0.5 * dt * k1, t + 0.5 * dt)
        if method == "midpoint":
            state = state + dt * k2
            continue
        k3 = velocity(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = velocity(state + dt * k3, t + dt)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return state
