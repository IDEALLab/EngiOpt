"""`Generator` contract for the unconditional MLP GAN (2D)."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch as th

from engiopt.core import ConditionBatch
from engiopt.core import Generator
from engiopt.generators.gan_2d.gan_2d import Generator as GAN2DNet

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint


class GAN2D(Generator):
    """Unconditional GAN with a fully-connected generator, for 2D problems."""

    algo_id = "gan_2d"
    conditional = False
    design_kinds = ("2d",)
    checkpoint_files = ("generator.pth",)
    primary_state_key = "generator"
    output_clip = (1e-3, 1.0)

    def __init__(self, net: GAN2DNet, latent_dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.net = net
        self.latent_dim = latent_dim

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> GAN2D:
        """Load a trained GAN-2D generator from its checkpoint package."""
        config = resolved.run_config
        net = GAN2DNet(latent_dim=config["latent_dim"], design_shape=problem.design_space.shape).to(device)
        net.load_state_dict(th.load(resolved.files["generator.pth"], map_location=device)[cls.primary_state_key])
        net.eval()
        return cls(net=net, latent_dim=config["latent_dim"], problem=problem, device=device, **base)

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:  # noqa: ARG002 - unconditional by design
        """Map noise through the generator, ignoring conditions."""
        z = th.randn((n, self.latent_dim), device=self.device)
        return self.net(z)
