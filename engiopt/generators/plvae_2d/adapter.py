"""`Generator` adapter for the performance-predicting least-volume autoencoder."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from engiopt.core import ConditionBatch
from engiopt.core import Generator
from engiopt.lvae.generator_base import LVAESamplingMixin

if TYPE_CHECKING:
    from engibench.core import Problem
    import torch as th

    from engiopt.checkpoint_store import ResolvedCheckpoint


class PerfLVAE2D(LVAESamplingMixin, Generator):
    """Least-volume autoencoder whose latent space also predicts performance."""

    algo_id = "plvae_2d"

    conditional = False
    """The predictor may read conditions, but the decoder does not."""

    design_kinds = ("2d",)

    checkpoint_files = ("plvae.pth",)

    primary_state_key = "decoder"

    output_clip: tuple[Any, Any] | None = (1e-3, 1.0)
    """The decoder emits unbounded values so its Lipschitz bound holds exactly."""

    weights_filename = "plvae.pth"

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> PerfLVAE2D:
        """Rebuild the decoder and fit the latent distribution to sample from."""
        return cls(**cls.prepare(resolved, problem, device), **base)

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        """Generate `n` designs; the decoder is unconditional."""
        del conditions
        return self.sample_latent_and_decode(n)
