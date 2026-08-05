"""`Generator` contract for the conditional MLP GAN (2D)."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch as th

from engiopt.core import condition_keys_for
from engiopt.core import ConditionBatch
from engiopt.core import Generator
from engiopt.generators.cgan_2d.cgan_2d import Generator as CGAN2DNet

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint


class CGAN2D(Generator):
    """Conditional GAN with a fully-connected generator, for 2D problems."""

    algo_id = "cgan_2d"
    conditional = True
    design_kinds = ("2d",)
    checkpoint_files = ("generator.pth",)
    primary_state_key = "generator"
    output_clip = (1e-3, 1.0)

    def __init__(self, net: CGAN2DNet, latent_dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.net = net
        self.latent_dim = latent_dim

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> CGAN2D:
        """Load a trained cGAN-2D generator from its checkpoint package."""
        config = resolved.run_config
        net = CGAN2DNet(
            latent_dim=config["latent_dim"],
            n_conds=len(condition_keys_for(problem, resolved)),
            design_shape=problem.design_space.shape,
            generator_output_activation=config.get("generator_output_activation", "tanh"),
        ).to(device)
        net.load_state_dict(th.load(resolved.files["generator.pth"], map_location=device)[cls.primary_state_key])
        net.eval()
        return cls(net=net, latent_dim=config["latent_dim"], problem=problem, device=device, **base)

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        """Map noise plus conditions through the generator."""
        cond = conditions.require_tensor(self.algo_id)
        z = th.randn((n, self.latent_dim), device=self.device)
        return self.net(z, cond)
