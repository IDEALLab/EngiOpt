"""Focused checks for diffusion training and sampling fixes."""

from __future__ import annotations

import pytest


def test_diffusion_1d_batch_is_normalized_before_forward() -> None:
    """The 1D diffusion model should receive [0, 1] designs before auto-normalization."""
    th = pytest.importorskip("torch")
    diffusion_1d = pytest.importorskip("engiopt.diffusion_1d.diffusion_1d")

    designs = th.tensor([[2.0, 4.0], [6.0, 8.0]])
    normalizer = diffusion_1d.Normalizer(
        min_val=th.tensor([2.0, 4.0]),
        max_val=th.tensor([6.0, 8.0]),
    )

    batch = diffusion_1d._prepare_diffusion_batch(designs, normalizer)  # noqa: SLF001

    expected_shape = (2, 1, 2)
    assert batch.shape == expected_shape
    assert th.allclose(batch.squeeze(1), th.tensor([[0.0, 0.0], [1.0, 1.0]]))


def test_default_linear_schedule_reaches_nearly_pure_noise() -> None:
    """The default noisiest training step should be close to the pure-noise sampling prior."""
    th = pytest.importorskip("torch")
    diffusion_2d = pytest.importorskip("engiopt.diffusion_2d_cond.diffusion_2d_cond")

    betas = diffusion_2d.beta_schedule(
        t=diffusion_2d.Args().num_timesteps,
        start=1e-4,
        end=0.02,
        scale=1.0,
        options={"cosine": False, "exp_biasing": False, "exp_bias_factor": 1},
    )
    terminal_signal = th.sqrt(th.cumprod(1.0 - betas, dim=0)[-1])

    max_terminal_signal = 0.01
    assert terminal_signal.item() < max_terminal_signal


def test_diffusion_step_sample_uses_fresh_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    """The variance term should use random sampling noise, not the predicted epsilon."""
    th = pytest.importorskip("torch")
    diffusion_2d = pytest.importorskip("engiopt.diffusion_2d_cond.diffusion_2d_cond")

    betas = diffusion_2d.beta_schedule(t=4, start=1e-4, end=0.02)
    sampler = diffusion_2d.DiffusionSampler(t=4, betas=betas)
    x_noisy = th.ones(2, 1, 2, 2)
    noise_pred = th.full_like(x_noisy, 0.25)
    t = th.tensor([1, 2], dtype=th.long)

    monkeypatch.setattr(diffusion_2d.th, "randn_like", th.zeros_like)

    actual = sampler.diffusion_step_sample(noise_pred, x_noisy, t)
    betas_t = diffusion_2d.get_index_from_list(sampler.betas, t, x_noisy.shape)
    sqrt_one_minus_alphas_cumprod_t = diffusion_2d.get_index_from_list(
        sampler.sqrt_one_minus_alphas_cumprod,
        t,
        x_noisy.shape,
    )
    sqrt_recip_alphas_t = diffusion_2d.get_index_from_list(sampler.sqrt_recip_alphas, t, x_noisy.shape)
    expected = sqrt_recip_alphas_t * (x_noisy - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)

    assert th.allclose(actual, expected)
