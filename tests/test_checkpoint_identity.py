"""Tests for how checkpoints are filed by hyperparameter configuration.

Two requirements pull against each other and both must hold:

1. Many hyperparameter settings of one algorithm must coexist on the Hub and be
   evaluated separately.
2. Naming a model with no extra qualification must mean its *default*
   hyperparameters -- a sweep must never redefine that.
"""

from __future__ import annotations

from dataclasses import dataclass

from engiopt.checkpoint_store import build_hf_package_path
from engiopt.core import checkpoint_identity
from engiopt.core import config_fingerprint
from engiopt.core import config_path_parts


@dataclass
class _Args:
    """Stand-in for a training script's Args."""

    latent_dim: int = 32
    lr_gen: float = 1e-4
    seed: int = 1
    wandb_entity: str | None = None
    track: bool = True


def test_default_run_is_marked_as_the_standard() -> None:
    """An untouched Args is the default configuration, so it owns the plain path."""
    identity = checkpoint_identity(_Args())
    assert identity["is_default_config"] is True


def test_tuned_run_is_not_the_standard() -> None:
    """A swept configuration must not claim the canonical path."""
    identity = checkpoint_identity(_Args(latent_dim=128))
    assert identity["is_default_config"] is False


def test_infrastructure_settings_do_not_change_the_configuration() -> None:
    """Seeds and tracking are how a run was operated, not what the model is."""
    assert checkpoint_identity(_Args(seed=7, wandb_entity="someone", track=False))["is_default_config"] is True


def test_each_hyperparameter_setting_gets_its_own_path() -> None:
    """The case that previously overwrote itself: same algo, same seed, different HP."""
    small = config_fingerprint(vars(_Args(latent_dim=32)))
    large = config_fingerprint(vars(_Args(latent_dim=128)))
    assert small != large

    small_path = build_hf_package_path("beams2d", 1, config_path_parts(small))
    large_path = build_hf_package_path("beams2d", 1, config_path_parts(large))
    assert small_path != large_path
    assert small_path == f"beams2d/cfg_{small}/seed_1"


def test_seeds_of_one_configuration_stay_distinct() -> None:
    """Seeds are separate checkpoints, not overwrites of each other."""
    fingerprint = config_fingerprint(vars(_Args()))
    parts = config_path_parts(fingerprint)
    assert build_hf_package_path("beams2d", 1, parts) != build_hf_package_path("beams2d", 2, parts)


def test_bare_model_name_resolves_to_the_canonical_path() -> None:
    """No fingerprint means the default checkpoint, which is what a bare name means."""
    assert config_path_parts(None) is None
    assert build_hf_package_path("beams2d", 1, config_path_parts(None)) == "beams2d/seed_1"


def test_identical_hyperparameters_fingerprint_identically() -> None:
    """Re-running the same configuration must land on the same path, not a new one."""
    assert config_fingerprint(vars(_Args(latent_dim=64))) == config_fingerprint(vars(_Args(latent_dim=64)))
