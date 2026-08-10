"""`Generator` adapter for the constrained performance-predicting LVAE."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from engiopt.core import ConditionBatch
from engiopt.core import Generator
from engiopt.lvae.generator_base import LVAESamplingMixin

if TYPE_CHECKING:
    from engibench.core import Problem
    import torch as th

    from engiopt.checkpoint_store import ResolvedCheckpoint


class ConstrainedPerfLVAE2D(LVAESamplingMixin, Generator):
    """Constrained performance-predicting least-volume autoencoder.

    This is also the default measuring instrument for latent-space metrics; see
    `engiopt.lvae.checkpoints.DEFAULT_INSTRUMENT_ALGO`.
    """

    algo_id = "constrained_plvae_2d"

    conditional = False
    """The decoder is unconditional unless trained with `--conditional-decoder`."""

    design_kinds = ("2d",)

    checkpoint_files = ("constrained_plvae.pth",)

    primary_state_key = "decoder"

    output_clip: tuple[Any, Any] | None = (1e-3, 1.0)
    """The decoder emits unbounded values so its Lipschitz bound holds exactly."""

    weights_filename = "constrained_plvae.pth"

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> ConstrainedPerfLVAE2D:
        """Rebuild the decoder and fit the latent distribution to sample from.

        Raises:
            NotImplementedError: If the run used a conditional decoder, which
                needs the condition encoder and rasterized image conditions that
                evaluation does not currently reconstruct.
        """
        if resolved.run_config.get("conditional_decoder", False):
            raise NotImplementedError(
                f"{cls.algo_id} checkpoints trained with --conditional-decoder are not yet loadable for "
                "evaluation: sampling would need the condition encoder and rasterized image conditions."
            )
        return cls(**cls.prepare(resolved, problem, device), problem=problem, device=device, **base)

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        """Generate `n` designs; the decoder is unconditional."""
        del conditions
        return self.sample_latent_and_decode(n)
