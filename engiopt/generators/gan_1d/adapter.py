"""`Generator` contract for the unconditional 1D GAN."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch as th

from engiopt.core import ConditionBatch
from engiopt.core import design_shape_of
from engiopt.core import Generator
from engiopt.generators.gan_1d.gan_1d import Generator as GAN1DNet
from engiopt.generators.gan_1d.gan_1d import prepare_data

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint


class GAN1D(Generator):
    """Unconditional GAN over 1D / dict-valued designs such as airfoils."""

    algo_id = "gan_1d"
    conditional = False
    design_kinds = ("1d", "dict")
    checkpoint_files = ("generator.pth",)
    primary_state_key = "generator"

    def __init__(self, net: GAN1DNet, latent_dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.net = net
        self.latent_dim = latent_dim

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> GAN1D:
        """Load a trained GAN-1D generator, rebuilding its design normalizer."""
        config = resolved.run_config
        _, design_normalizer = prepare_data(problem, device)
        net = GAN1DNet(
            latent_dim=config["latent_dim"],
            design_shape=design_shape_of(problem),
            design_normalizer=design_normalizer,
        ).to(device)
        net.load_state_dict(th.load(resolved.files["generator.pth"], map_location=device)[cls.primary_state_key])
        net.eval()
        return cls(net=net, latent_dim=config["latent_dim"], problem=problem, device=device, **base)

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:  # noqa: ARG002 - unconditional by design
        """Map noise through the generator, ignoring conditions."""
        z = th.randn((n, self.latent_dim), device=self.device)
        return self.net(z)
