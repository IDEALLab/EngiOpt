"""`Generator` contract for the conditional GAN with CNN backbone (2D)."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch as th

from engiopt.core import ConditionBatch
from engiopt.core import Generator
from engiopt.generators.cgan_cnn_2d.cgan_cnn_2d import Generator as CGANCNN2DNet

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint


class CGANCNN2D(Generator):
    """Conditional GAN with a CNN generator, for 2D topology problems."""

    algo_id = "cgan_cnn_2d"
    conditional = True
    design_kinds = ("2d",)
    checkpoint_files = ("generator.pth",)
    primary_state_key = "generator"
    # Several 2D simulators are unstable at exactly zero density.
    output_clip = (1e-3, 1.0)

    def __init__(self, net: CGANCNN2DNet, latent_dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.net = net
        self.latent_dim = latent_dim

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> CGANCNN2D:
        """Load a trained cGAN-CNN generator from its checkpoint package."""
        config = resolved.run_config
        net = CGANCNN2DNet(
            latent_dim=config["latent_dim"],
            n_conds=len(problem.conditions_keys),
            design_shape=problem.design_space.shape,
            generator_output_activation=config.get("generator_output_activation", "tanh"),
        )
        checkpoint = th.load(resolved.files["generator.pth"], map_location=device)
        net.load_state_dict(checkpoint[cls.primary_state_key])
        net.eval().to(device)
        return cls(net=net, latent_dim=config["latent_dim"], problem=problem, device=device, **base)

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        """Map noise plus conditions through the generator."""
        cond = conditions.require_tensor(self.algo_id)
        z = th.randn((n, self.latent_dim, 1, 1), device=self.device, dtype=th.float)
        return self.net(z, cond.reshape(n, self.n_conds, 1, 1))
