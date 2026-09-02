"""`Generator` contract for the conditional 3D CNN GAN."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch as th

from engiopt.core import condition_keys_for
from engiopt.core import ConditionBatch
from engiopt.core import Generator
from engiopt.generators.cgan_cnn_3d.cgan_cnn_3d import Generator3D

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint


def center_crop_3d(volumes: th.Tensor, target: tuple[int, ...]) -> th.Tensor:
    """Center-crop `(n, D, H, W)` volumes down to `target`.

    The 3D generators upsample to a power-of-two grid (64) and the problem's
    design space is smaller (51), so the padding introduced for the network's
    convenience is removed symmetrically here.
    """
    cropped = volumes
    for axis, size in enumerate(target, start=1):
        start = (cropped.shape[axis] - size) // 2
        cropped = cropped.narrow(axis, start, size)
    return cropped


class CGANCNN3D(Generator):
    """Conditional GAN with a 3D CNN generator."""

    algo_id = "cgan_cnn_3d"
    conditional = True
    design_kinds = ("3d",)
    checkpoint_files = ("generator_3d.pth",)
    primary_state_key = "generator"
    output_clip = (1e-3, 1.0)

    def __init__(self, net: Generator3D, latent_dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.net = net
        self.latent_dim = latent_dim

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> CGANCNN3D:
        """Load a trained 3D cGAN generator from its checkpoint package."""
        config = resolved.run_config
        net = Generator3D(
            latent_dim=config["latent_dim"],
            n_conds=len(condition_keys_for(problem, resolved)),
            design_shape=problem.design_space.shape,
        )
        net.load_state_dict(
            th.load(resolved.files["generator_3d.pth"], map_location=device, weights_only=True)[cls.primary_state_key]
        )
        net.eval().to(device)
        return cls(net=net, latent_dim=config["latent_dim"], problem=problem, device=device, **base)

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        """Generate volumes from noise plus conditions, then trim the padding."""
        cond = conditions.require_tensor(self.algo_id).reshape(n, self.n_conds, 1, 1, 1)
        z = th.randn((n, self.latent_dim, 1, 1, 1), device=self.device, dtype=th.float)
        volumes = self.net(z, cond).squeeze(1)
        return center_crop_3d(volumes, self.design_shape)
