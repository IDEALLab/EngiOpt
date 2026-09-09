"""`Generator` contract for the conditional multi-view 3D VAE-GAN."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch as th

from engiopt.core import condition_keys_for
from engiopt.core import ConditionBatch
from engiopt.core import Generator
from engiopt.generators.cgan_cnn_3d.adapter import center_crop_3d
from engiopt.generators.cgan_vae.cgan_vae import Generator3D

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint


class CGANVAE(Generator):
    """Conditional VAE-GAN hybrid with a 3D decoder."""

    algo_id = "cgan_vae"
    conditional = True
    design_kinds = ("3d",)
    checkpoint_files = ("multiview_3d_vaegan.pth",)
    primary_state_key = "generator"
    output_clip = (1e-3, 1.0)

    def __init__(self, net: Generator3D, latent_dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.net = net
        self.latent_dim = latent_dim

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> CGANVAE:
        """Load the trained 3D decoder from its checkpoint package."""
        config = resolved.run_config
        net = Generator3D(
            latent_dim=config["latent_dim"],
            n_conds=len(condition_keys_for(problem, resolved)),
            design_shape=problem.design_space.shape,
        )
        checkpoint = th.load(resolved.files["multiview_3d_vaegan.pth"], map_location=device, weights_only=True)
        net.load_state_dict(checkpoint[cls.primary_state_key])
        net.eval().to(device)
        return cls(net=net, latent_dim=config["latent_dim"], problem=problem, device=device, **base)

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        """Decode noise plus conditions into volumes, then trim the padding."""
        cond = conditions.require_tensor(self.algo_id).reshape(n, self.n_conds, 1, 1, 1)
        z = th.randn((n, self.latent_dim, 1, 1, 1), device=self.device, dtype=th.float)
        volumes = self.net(z, cond).squeeze(1)
        return center_crop_3d(volumes, self.design_shape)
