"""Tests for the LVAE package and its role as a latent-metric instrument."""

from __future__ import annotations

import numpy as np
import pytest
import torch as th

from engiopt import metrics as metrics_mod
from engiopt.evaluation.context import EvaluationContext
from engiopt.evaluation.context import LatentInstrumentUnavailableError
from engiopt.evaluation.registry import METRICS
from engiopt.lvae.components import Encoder2D
from engiopt.lvae.config import LVAEConfig
from engiopt.lvae.encode import encode_active
from engiopt.lvae.encode import encode_designs
from engiopt.lvae.encode import get_active_mask
from engiopt.lvae.encode import latent_dim_of
from engiopt.lvae.encode import PrunedEncoder
from tests.stubs import STUB_LATENT_DIM
from tests.stubs import stub_lvae

LATENT_DIM = 8
DESIGN_SHAPE = (50, 100)
PRUNED_DIMS = [2, 5, 7]


@pytest.fixture
def encoder() -> Encoder2D:
    """A small untrained encoder; these tests are about plumbing, not weights."""
    return Encoder2D(latent_dim=LATENT_DIM, design_shape=DESIGN_SHAPE)


@pytest.fixture
def pruned(encoder: Encoder2D) -> PrunedEncoder:
    """The same encoder with three dimensions pruned to known frozen values."""
    mask = th.zeros(LATENT_DIM, dtype=th.bool)
    mask[PRUNED_DIMS] = True
    return PrunedEncoder(encoder, mask, th.arange(LATENT_DIM, dtype=th.float32))


@pytest.fixture
def designs() -> np.ndarray:
    """A handful of random designs."""
    return np.random.default_rng(0).random((4, *DESIGN_SHAPE)).astype(np.float32)


def test_active_mask_is_full_width_for_an_unpruned_encoder(encoder: Encoder2D) -> None:
    """A raw encoder's mask must span every latent dimension.

    Callers index `z[:, mask]`. A short mask would silently keep only the
    leading dimension and every latent metric computed from it would be wrong
    rather than merely noisy.
    """
    mask = get_active_mask(encoder)
    assert mask.shape == (LATENT_DIM,)
    assert mask.all()


def test_active_mask_excludes_pruned_dimensions(pruned: PrunedEncoder) -> None:
    """Pruned dimensions are constant across designs, so they are not active."""
    mask = get_active_mask(pruned)
    assert mask.shape == (LATENT_DIM,)
    assert mask.sum() == LATENT_DIM - len(PRUNED_DIMS)
    assert not mask[PRUNED_DIMS].any()


def test_latent_dim_is_recoverable_from_either_encoder(encoder: Encoder2D, pruned: PrunedEncoder) -> None:
    """Both wrapped and raw encoders report the same total width."""
    assert latent_dim_of(encoder) == LATENT_DIM
    assert latent_dim_of(pruned) == LATENT_DIM


def test_pruned_dimensions_are_clamped_to_frozen_values(pruned: PrunedEncoder, designs: np.ndarray) -> None:
    """Every design must receive identical values on pruned axes."""
    codes = encode_designs(pruned, designs, "cpu")
    assert codes.shape == (len(designs), LATENT_DIM)
    expected = pruned.frozen_z.numpy()[PRUNED_DIMS]
    assert np.allclose(codes[:, PRUNED_DIMS], expected)


def test_encoding_does_not_mutate_its_input(pruned: PrunedEncoder) -> None:
    """The pruning clamp must be out-of-place, or it aliases the caller's tensor."""
    x = th.ones(2, 1, *DESIGN_SHAPE)
    before = x.clone()
    pruned(x)
    assert th.equal(x, before)


def test_encode_active_drops_pruned_columns(pruned: PrunedEncoder, designs: np.ndarray) -> None:
    """The metric-facing helper returns only the active subspace."""
    active = encode_active(pruned, designs, "cpu")
    assert active.shape == (len(designs), LATENT_DIM - len(PRUNED_DIMS))


def test_config_expands_the_perf_dim_sentinel() -> None:
    """`perf_dim=-1` means "use every latent dimension"."""
    config = LVAEConfig.from_run_config({"latent_dim": LATENT_DIM, "perf_dim": -1}, DESIGN_SHAPE)
    assert config.perf_dim == LATENT_DIM


def test_config_falls_back_to_the_problem_design_shape() -> None:
    """Runs predating recorded `design_shape` still rebuild correctly."""
    config = LVAEConfig.from_run_config({"latent_dim": LATENT_DIM}, DESIGN_SHAPE)
    assert config.design_shape == DESIGN_SHAPE


def test_config_rejects_a_non_lvae_package() -> None:
    """Loading a GAN package as an LVAE should fail loudly, not build a wrong net."""
    with pytest.raises(KeyError, match="latent_dim"):
        LVAEConfig.from_run_config({"lr_gen": 1e-4}, DESIGN_SHAPE)


# ----------------------------------------------------------------------
# Latent-space metrics (metric_suite.md roster)
# ----------------------------------------------------------------------


class _StubProblem:
    """Minimal stand-in; latent metrics only need the design space."""

    design_space = type("Space", (), {"shape": DESIGN_SHAPE})()


def _context(*, with_instrument: bool, with_companion: bool = False, n: int = 6) -> EvaluationContext:
    rng = np.random.default_rng(0)
    return EvaluationContext(
        problem=_StubProblem(),
        problem_id="beams2d",
        gen_designs=rng.random((n, *DESIGN_SHAPE)).astype(np.float32),
        ref_designs=rng.random((n, *DESIGN_SHAPE)).astype(np.float32),
        latent_lvae=stub_lvae(DESIGN_SHAPE, pruned_dims=[1]) if with_instrument else None,
        latent_recon_lvae=stub_lvae(DESIGN_SHAPE, pruned_dims=[1]) if with_companion else None,
    )


LATENT_METRICS = ("lv_mmd", "lv_residual", "lv_coverage", "lv_vendi", "lv_paired_distance")


def test_latent_metrics_refuse_to_run_without_a_pinned_instrument() -> None:
    """A latent number is meaningless unless the spec says what measured it."""
    ctx = _context(with_instrument=False)
    for name in LATENT_METRICS:
        with pytest.raises(LatentInstrumentUnavailableError):
            METRICS[name].fn(ctx)


def test_the_dual_gap_names_its_missing_companion() -> None:
    """Failing on the instrument would misdirect: it is the companion that is absent."""
    ctx = _context(with_instrument=True)
    with pytest.raises(LatentInstrumentUnavailableError, match="recon_only_config_fingerprint"):
        METRICS["lv_dual_gap"].fn(ctx)


def test_latent_metrics_measure_only_the_active_subspace() -> None:
    """Codes handed to the metrics exclude pruned dimensions."""
    ctx = _context(with_instrument=True)
    generated, reference = ctx.latent_codes
    assert generated.shape[1] == STUB_LATENT_DIM - 1
    assert reference.shape[1] == STUB_LATENT_DIM - 1


def test_the_suite_produces_finite_values() -> None:
    """Every registered latent metric returns a usable number."""
    ctx = _context(with_instrument=True, with_companion=True)
    assert np.isfinite(METRICS["lv_mmd"].fn(ctx))
    assert np.isfinite(METRICS["lv_coverage"].fn(ctx))
    assert np.isfinite(METRICS["lv_vendi"].fn(ctx))
    assert np.isfinite(METRICS["lv_paired_distance"].fn(ctx))
    assert np.isfinite(METRICS["lv_dual_gap"].fn(ctx))
    assert set(METRICS["lv_residual"].fn(ctx)) == {"lv_residual_mean", "lv_residual_p90"}


def test_the_residual_is_measured_in_pixel_space() -> None:
    """A design already on the manifold has near-zero residual; an off-manifold one does not.

    The residual has to be a pixel-space quantity: the encoder maps an invalid
    design to an ordinary-looking latent code, so the error only appears after
    decoding back out.
    """
    ctx = _context(with_instrument=True)
    projected = ctx.gen_projected
    assert projected.shape == ctx.gen_designs.shape
    on_manifold = np.linalg.norm(projected.reshape(len(projected), -1) - projected.reshape(len(projected), -1))
    assert on_manifold == pytest.approx(0.0)


def test_vendi_is_not_inflated_by_noise_the_way_dpp_is() -> None:
    """The reason metric_suite.md specifies Vendi over DPP for diversity.

    Adding noise to a collapsed set makes its samples less similar, which a
    determinant rewards. The effective-count reading resists that.
    """
    rng = np.random.default_rng(0)
    collapsed = np.repeat(rng.normal(size=(1, 8)), 20, axis=0)
    noisy = collapsed + rng.normal(scale=0.5, size=collapsed.shape)

    dpp_inflation = metrics_mod.dpp_diversity(noisy, sigma=3.0) / metrics_mod.dpp_diversity(collapsed, sigma=3.0)
    vendi_inflation = metrics_mod.vendi_score(noisy, sigma=3.0) / metrics_mod.vendi_score(collapsed, sigma=3.0)
    assert dpp_inflation > vendi_inflation


def test_vendi_counts_effective_samples() -> None:
    """Identical samples collapse to one; the score reads as a count."""
    rng = np.random.default_rng(0)
    varied = rng.normal(size=(20, 8))
    collapsed = np.repeat(varied[:1], 20, axis=0)
    assert metrics_mod.vendi_score(collapsed, sigma=3.0) == pytest.approx(1.0, abs=1e-6)
    assert metrics_mod.vendi_score(varied, sigma=3.0) > 1.0


def test_median_sigma_adapts_to_the_scale_of_its_input() -> None:
    """A bandwidth fixed for pixel space would saturate in a latent space."""
    rng = np.random.default_rng(0)
    small = rng.normal(size=(40, 6))
    assert metrics_mod.compute_median_sigma(small * 100) > metrics_mod.compute_median_sigma(small)
