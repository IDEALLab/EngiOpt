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

from dataclasses import dataclass
from dataclasses import field
from functools import cached_property
from typing import Any, TYPE_CHECKING

import numpy as np
import torch as th

from engiopt.checkpoint_store import resolve_named_checkpoint
from engiopt.lvae.components import Encoder2D
from engiopt.lvae.components import TrueSNDecoder2D
from engiopt.lvae.config import LVAEConfig
from engiopt.lvae.encode import decode_designs
from engiopt.lvae.encode import encode_designs
from engiopt.lvae.encode import PrunedEncoder

if TYPE_CHECKING:
    import numpy.typing as npt
    from torch import nn

    from engiopt.checkpoint_store import ModelSource
    from engiopt.checkpoint_store import ResolvedCheckpoint

ENCODER_STATE_KEY = "encoder"
DECODER_STATE_KEY = "decoder"
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


def build_decoder(checkpoint: dict[str, Any], config: LVAEConfig, device: th.device) -> nn.Module:
    """Rebuild a decoder from a loaded checkpoint dict.

    Args:
        checkpoint: The deserialized `.pth` payload.
        config: Architecture arguments recovered from `run_config.json`.
        device: Device to place the decoder on.

    Returns:
        An eval-mode decoder.

    Raises:
        KeyError: If the checkpoint holds no decoder state dict.
    """
    if DECODER_STATE_KEY not in checkpoint:
        raise KeyError(f"checkpoint has no {DECODER_STATE_KEY!r} state dict; keys are {sorted(checkpoint)}")

    decoder = TrueSNDecoder2D(
        latent_dim=config.latent_dim,
        design_shape=config.design_shape,
        lipschitz_scale=config.decoder_lipschitz_scale,
    )
    decoder.load_state_dict(checkpoint[DECODER_STATE_KEY])
    return decoder.to(device).eval()


class DecoderUnavailableError(RuntimeError):
    """Raised when a package's decoder cannot be rebuilt by the current code.

    Kept distinct from a generic load error because the usual cause is an
    architecture change that the published weights predate, and the answer is to
    retrain rather than to retry. The encoder is unaffected, so the metrics that
    only encode stay available and only the ones that must decode fail.
    """

    def __init__(self, package: str, reason: str) -> None:
        super().__init__(
            f"the decoder in {package} cannot be rebuilt by the current code: {reason}\n"
            "Encoder-only latent metrics (lv_mmd, lv_coverage, lv_vendi, lv_paired_distance) still work; "
            "lv_residual and lv_dual_gap need a decoder and will stay unavailable until the instrument is "
            "retrained against the current architecture."
        )


@dataclass
class LoadedLVAE:
    """A trained LVAE, with the decoder built on demand.

    Encoder-only loading is enough for the distribution and diversity metrics.
    Anything that measures a design *against the manifold* -- the projection
    residual, the dual-LVAE gap, encode-decode projection -- has to decode too.

    The decoder is built lazily rather than at load time so that a package whose
    decoder the current code cannot rebuild still serves every metric that only
    needs to encode. Building both eagerly meant one architecture change took
    down the whole latent family, including columns that never decode.

    Attributes:
        encoder: Eval-mode encoder, pruning wrapper included.
        config: Architecture arguments recovered from `run_config.json`.
        resolved: The checkpoint package, whose `revision` and `content_hash`
            identify exactly which instrument produced a number.
    """

    encoder: nn.Module
    config: LVAEConfig
    resolved: ResolvedCheckpoint
    _decoder_factory: Any = field(default=None, repr=False)

    @cached_property
    def decoder(self) -> nn.Module:
        """The decoder, rebuilt on first use.

        Raises:
            DecoderUnavailableError: If the published weights do not match the
                architecture the current code builds.
        """
        if self._decoder_factory is None:
            raise DecoderUnavailableError(str(self.resolved.root_dir), "no decoder was loaded")
        try:
            return self._decoder_factory()
        except RuntimeError as exc:
            raise DecoderUnavailableError(str(self.resolved.root_dir), str(exc).split("\n")[0]) from exc

    @property
    def has_decoder(self) -> bool:
        """Whether the decoder can actually be built, without raising to find out."""
        try:
            _ = self.decoder
        except DecoderUnavailableError:
            return False
        return True

    def project(self, designs: npt.NDArray, batch_size: int = 256) -> npt.NDArray:
        """Project designs onto the learned manifold by encoding then decoding.

        Args:
            designs: Designs of shape `(N, H, W)` or `(N, 1, H, W)`.
            batch_size: Designs per forward pass.

        Returns:
            Reconstructed designs of shape `(N, H, W)`, clamped to `[0, 1]`.
            The decoder is deliberately unbounded during training so its
            Lipschitz bound holds exactly; clamping belongs at inference.
        """
        device = next(self.decoder.parameters()).device
        codes = encode_designs(self.encoder, designs, device, batch_size)
        return np.clip(decode_designs(self.decoder, codes, device, batch_size), 0.0, 1.0)


def load_lvae(**kwargs: Any) -> LoadedLVAE:
    """Load both halves of a trained LVAE.

    Takes the same arguments as `load_lvae_encoder`.

    Returns:
        The encoder, decoder, config, and resolved checkpoint.
    """
    encoder, config, resolved = load_lvae_encoder(**kwargs)
    device = next(encoder.parameters()).device
    filename = resolved.files and next(iter(resolved.files))
    checkpoint = th.load(resolved.files[filename], map_location=device, weights_only=False)
    return LoadedLVAE(
        encoder=encoder,
        config=config,
        resolved=resolved,
        _decoder_factory=lambda: build_decoder(checkpoint, config, device),
    )


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
