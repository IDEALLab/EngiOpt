"""Tests for where checkpoints, metrics, and media live.

The split is deliberate: HuggingFace hosts anything that must be reloaded or
compared (weights, run configs, metrics, the leaderboard); Weights & Biases,
when enabled, hosts what you only look at (loss curves, sample images).
"""

from __future__ import annotations

import inspect

from engiopt import checkpoint_store
from engiopt.core import Generator


def test_wandb_is_not_a_checkpoint_backend() -> None:
    """HuggingFace and local directories are the only sources of weights."""
    assert set(checkpoint_store.ModelSource.__args__) == {"auto", "hf", "local"}


def test_wandb_checkpoint_resolvers_are_gone() -> None:
    """The legacy W&B read-fallback was dead weight once HF became the only host."""
    for removed in ("_resolve_wandb_package", "_resolve_wandb_reference", "_build_wandb_artifact_path"):
        assert not hasattr(checkpoint_store, removed), f"{removed} should have been removed"


def test_write_only_run_scoped_path_is_gone() -> None:
    """That path was never readable by any loader, so nothing could use it."""
    assert not hasattr(checkpoint_store, "build_hf_run_package_path")


def test_generators_no_longer_carry_wandb_naming() -> None:
    """The contract should not mention a backend it cannot read from."""
    assert not hasattr(Generator, "wandb_artifact_names")
    assert not hasattr(Generator, "_wandb_artifact_names")


def test_resolve_named_checkpoint_takes_no_wandb_arguments() -> None:
    """Its signature is the clearest statement of what backends exist."""
    params = set(inspect.signature(checkpoint_store.resolve_named_checkpoint).parameters)
    assert not {p for p in params if "wandb" in p}
    assert {"hf_entity", "hf_repo_prefix", "extra_path_parts"} <= params


def test_metrics_are_publishable_next_to_the_weights() -> None:
    """A checkpoint should be self-describing without the leaderboard."""
    assert checkpoint_store.METRICS_FILE == "metrics.json"
    params = set(inspect.signature(checkpoint_store.publish_checkpoint_metrics).parameters)
    assert {"metrics", "problem_id", "algo", "seed", "extra_path_parts"} <= params


def test_wandb_still_receives_a_pointer_to_the_hf_package() -> None:
    """Each side records where the other is, so curves and weights stay linked."""
    source = inspect.getsource(checkpoint_store._log_checkpoint_summary_to_wandb)
    assert "hf_repo_id" in source
    assert "hf_config_package_path" in source
