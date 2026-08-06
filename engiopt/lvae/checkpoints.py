"""Loading trained LVAEs from HuggingFace checkpoint packages.

The predecessor to this module resolved checkpoints through `wandb.Api()`, using
artifact aliases of the form `seed_1_rec0.001_perf0.01`. That required W&B
credentials at evaluation time and encoded hyperparameters in a string that only
the training script knew how to build.

Here, checkpoints resolve through `engiopt.checkpoint_store` like every other
model: the thresholds live in `run_config.json`, and distinct thresholds already
produce distinct `config_fingerprint`s, so the alias scheme is unnecessary.

Nothing in this module imports W&B.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch as th

from engiopt.checkpoint_store import resolve_named_checkpoint
from engiopt.lvae.components import Encoder2D
from engiopt.lvae.config import LVAEConfig
from engiopt.lvae.encode import PrunedEncoder

if TYPE_CHECKING:
    from torch import nn

    from engiopt.checkpoint_store import ModelSource
    from engiopt.checkpoint_store import ResolvedCheckpoint

ENCODER_STATE_KEY = "encoder"
PRUNING_MASK_KEY = "pruning_mask"
PRUNING_FROZEN_Z_KEY = "pruning_frozen_z"

DEFAULT_INSTRUMENT_ALGO = "constrained_plvae_2d"
"""Model family used as the latent-metric instrument unless told otherwise."""


def build_encoder(checkpoint: dict[str, Any], config: LVAEConfig, device: th.device) -> nn.Module:
    """Rebuild an encoder from a loaded checkpoint dict.

    When the checkpoint carries a pruning mask, the encoder is wrapped so pruned
    dimensions are clamped exactly as they were during training. Dropping that
    wrapper silently changes what the latent space means, so its absence is only
    tolerated for checkpoints that genuinely never pruned.

    Args:
        checkpoint: The deserialized `.pth` payload.
        config: Architecture arguments recovered from `run_config.json`.
        device: Device to place the encoder on.

    Returns:
        An eval-mode encoder, wrapped in `PrunedEncoder` when a mask is present.

    Raises:
        KeyError: If the checkpoint holds no encoder state dict.
    """
    if ENCODER_STATE_KEY not in checkpoint:
        raise KeyError(f"checkpoint has no {ENCODER_STATE_KEY!r} state dict; keys are {sorted(checkpoint)}")

    raw = Encoder2D(
        latent_dim=config.latent_dim,
        design_shape=config.design_shape,
        resize_dimensions=config.resize_dimensions,
        whitening=config.whitening,
    )
    raw.load_state_dict(checkpoint[ENCODER_STATE_KEY])

    encoder: nn.Module = raw
    if PRUNING_MASK_KEY in checkpoint and PRUNING_FROZEN_Z_KEY in checkpoint:
        encoder = PrunedEncoder(raw, checkpoint[PRUNING_MASK_KEY], checkpoint[PRUNING_FROZEN_Z_KEY])

    return encoder.to(device).eval()


def load_lvae_encoder(
    *,
    problem_id: str,
    design_shape: tuple[int, int],
    algo: str = DEFAULT_INSTRUMENT_ALGO,
    seed: int = 1,
    weights_filename: str | None = None,
    device: th.device | str = "cpu",
    model_source: ModelSource = "auto",
    hf_entity: str = "IDEALLab",
    hf_repo_prefix: str = "engiopt",
    local_model_dir: str | None = None,
    config_fingerprint: str | None = None,
    revision: str | None = None,
) -> tuple[nn.Module, LVAEConfig, ResolvedCheckpoint]:
    """Load a trained LVAE encoder from its checkpoint package.

    Args:
        problem_id: EngiBench problem the LVAE was trained on.
        design_shape: The problem's design shape, used when the run predates
            `design_shape` being recorded in `run_config`.
        algo: Model family holding the instrument.
        seed: Training seed.
        weights_filename: Weight file inside the package; defaults to
            `{algo_without_2d_suffix}.pth`.
        device: Device to place the encoder on.
        model_source: `auto`, `hf`, or `local`.
        hf_entity: HF org/user holding the checkpoint repos.
        hf_repo_prefix: Prefix of the per-model-family repo.
        local_model_dir: Directory to load from instead of the Hub.
        config_fingerprint: Pins one configuration out of a sweep. Latent metrics
            must pin this: different reconstruction and performance thresholds
            yield different active-subspace widths, so an unpinned instrument
            produces columns that are not comparable across rows.
        revision: Repo commit to read at. `config_fingerprint` pins *which*
            package; this pins *which version* of it. Without it, re-training
            and re-uploading to the same path silently changes every latent
            number already on the leaderboard.

    Returns:
        The encoder, its configuration, and the resolved checkpoint (whose
        `revision` and `content_hash` identify the exact instrument used).
    """
    filename = weights_filename or f"{algo.removesuffix('_2d')}.pth"
    resolved = resolve_named_checkpoint(
        model_source=model_source,
        problem_id=problem_id,
        algo=algo,
        seed=seed,
        hf_entity=hf_entity,
        hf_repo_prefix=hf_repo_prefix,
        required_files=[filename],
        local_model_dir=local_model_dir,
        extra_path_parts=[f"cfg_{config_fingerprint}"] if config_fingerprint else None,
        revision=revision,
    )

    device = th.device(device) if isinstance(device, str) else device
    checkpoint = th.load(resolved.files[filename], map_location=device, weights_only=False)
    config = LVAEConfig.from_run_config(resolved.run_config, design_shape)

    return build_encoder(checkpoint, config, device), config, resolved
