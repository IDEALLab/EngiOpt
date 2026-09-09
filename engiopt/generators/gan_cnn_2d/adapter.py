"""`Generator` contract for the unconditional GAN with CNN backbone (2D)."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch as th

from engiopt.core import ConditionBatch
from engiopt.core import Generator
from engiopt.generators.gan_cnn_2d.gan_cnn_2d import Generator as GANCNN2DNet

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint


class GANCNN2D(Generator):
    """Unconditional GAN with a CNN generator, for 2D topology problems.

    Conditions are accepted and ignored: the model was never told what to
    optimize for. That is precisely what conditional-adherence metrics exist to
    expose, so the mismatch is left visible rather than papered over.
    """

    algo_id = "gan_cnn_2d"
    conditional = False
    design_kinds = ("2d",)
    checkpoint_files = ("generator.pth",)
    primary_state_key = "generator"
    output_clip = (1e-3, 1.0)

    def __init__(self, net: GANCNN2DNet, latent_dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.net = net
        self.latent_dim = latent_dim

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> GANCNN2D:
        """Load a trained GAN-CNN generator from its checkpoint package."""
        config = resolved.run_config
        net = GANCNN2DNet(latent_dim=config["latent_dim"], design_shape=problem.design_space.shape)
        checkpoint = th.load(resolved.files["generator.pth"], map_location=device, weights_only=True)
        net.load_state_dict(checkpoint[cls.primary_state_key])
        net.eval().to(device)
        return cls(net=net, latent_dim=config["latent_dim"], problem=problem, device=device, **base)

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:  # noqa: ARG002 - unconditional by design
        """Map noise through the generator, ignoring conditions."""
        z = th.randn((n, self.latent_dim, 1, 1), device=self.device, dtype=th.float)
        return self.net(z)
