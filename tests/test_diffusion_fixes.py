"""Focused checks for diffusion training and sampling fixes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_diffusion_1d_batch_is_normalized_before_forward() -> None:
    """The 1D diffusion model should receive [0, 1] designs before auto-normalization."""
    th = pytest.importorskip("torch")
    diffusion_1d = pytest.importorskip("engiopt.generators.diffusion_1d.diffusion_1d")

    designs = th.tensor([[2.0, 4.0], [6.0, 8.0]])
    normalizer = diffusion_1d.Normalizer(
        min_val=th.tensor([2.0, 4.0]),
        max_val=th.tensor([6.0, 8.0]),
    )

    batch = diffusion_1d._prepare_diffusion_batch(designs, normalizer)

    expected_shape = (2, 1, 2)
    assert batch.shape == expected_shape
    assert th.allclose(batch.squeeze(1), th.tensor([[0.0, 0.0], [1.0, 1.0]]))


def test_default_linear_schedule_reaches_nearly_pure_noise() -> None:
    """The default noisiest training step should be close to the pure-noise sampling prior."""
    th = pytest.importorskip("torch")
    diffusion_2d = pytest.importorskip("engiopt.generators.diffusion_2d_cond.diffusion_2d_cond")

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


def test_diffusion_2d_normalizer_round_trips_arbitrary_bounds() -> None:
    """2D diffusion should not assume designs are already in [0, 1]."""
    th = pytest.importorskip("torch")
    diffusion_2d = pytest.importorskip("engiopt.generators.diffusion_2d_cond.diffusion_2d_cond")

    designs = th.tensor([[-2.0, 1.0], [4.0, 10.0]])
    design_min = designs.min()
    design_max = designs.max()

    normalized = diffusion_2d.normalize_designs_to_diffusion_range(designs, design_min, design_max)
    restored = diffusion_2d.denormalize_designs_from_diffusion_range(normalized, design_min, design_max)

    assert normalized.min().item() == pytest.approx(-1.0)
    assert normalized.max().item() == pytest.approx(1.0)
    assert th.allclose(restored, designs)


def test_diffusion_2d_uses_problem_design_bounds_when_available() -> None:
    """EngiBench design_space bounds should be the normalization source of truth."""
    th = pytest.importorskip("torch")
    diffusion_2d = pytest.importorskip("engiopt.generators.diffusion_2d_cond.diffusion_2d_cond")

    fallback_designs = th.tensor([[0.2, 0.8]])
    problem = SimpleNamespace(
        design_space=SimpleNamespace(
            low=th.tensor([-2.0, -1.0]).numpy(),
            high=th.tensor([2.0, 3.0]).numpy(),
        ),
    )

    design_min, design_max = diffusion_2d.get_design_bounds(problem, fallback_designs, th.device("cpu"))

    assert th.allclose(design_min, th.tensor([-2.0, -1.0]))
    assert th.allclose(design_max, th.tensor([2.0, 3.0]))


def test_diffusion_step_sample_uses_fresh_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reverse step should clip predicted x0 and use fresh posterior noise."""
    th = pytest.importorskip("torch")
    diffusion_2d = pytest.importorskip("engiopt.generators.diffusion_2d_cond.diffusion_2d_cond")

    betas = diffusion_2d.beta_schedule(t=4, start=1e-4, end=0.02)
    sampler = diffusion_2d.DiffusionSampler(t=4, betas=betas)
    x_noisy = th.ones(2, 1, 2, 2)
    noise_pred = th.full_like(x_noisy, 0.25)
    t = th.tensor([1, 2], dtype=th.long)

    monkeypatch.setattr(diffusion_2d.th, "randn_like", th.zeros_like)

    actual = sampler.diffusion_step_sample(noise_pred, x_noisy, t)
    sqrt_alphas_cumprod_t = diffusion_2d.get_index_from_list(
        sampler.sqrt_alphas_cumprod,
        t,
        x_noisy.shape,
    )
    sqrt_one_minus_alphas_cumprod_t = diffusion_2d.get_index_from_list(
        sampler.sqrt_one_minus_alphas_cumprod,
        t,
        x_noisy.shape,
    )
    pred_original_sample = (x_noisy - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
    pred_original_sample = pred_original_sample.clamp(
        diffusion_2d.DIFFUSION_SAMPLE_MIN,
        diffusion_2d.DIFFUSION_SAMPLE_MAX,
    )
    posterior_mean_coef1_t = diffusion_2d.get_index_from_list(sampler.posterior_mean_coef1, t, x_noisy.shape)
    posterior_mean_coef2_t = diffusion_2d.get_index_from_list(sampler.posterior_mean_coef2, t, x_noisy.shape)
    expected = posterior_mean_coef1_t * pred_original_sample + posterior_mean_coef2_t * x_noisy

    assert th.allclose(actual, expected)
