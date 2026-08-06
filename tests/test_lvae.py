"""Tests for the LVAE package and its role as a latent-metric instrument."""

from __future__ import annotations

import numpy as np
import pytest
import torch as th

from engiopt.lvae.components import Encoder2D
from engiopt.lvae.config import LVAEConfig
from engiopt.lvae.encode import encode_active
from engiopt.lvae.encode import encode_designs
from engiopt.lvae.encode import get_active_mask
from engiopt.lvae.encode import latent_dim_of
from engiopt.lvae.encode import PrunedEncoder

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
