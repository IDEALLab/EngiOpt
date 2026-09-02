"""`Generator` contract for the conditional 1D GAN."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch as th

from engiopt.core import condition_keys_for
from engiopt.core import ConditionBatch
from engiopt.core import design_shape_of
from engiopt.core import Generator
from engiopt.core import recorded_condition_normalizer
from engiopt.core import recorded_design_normalizer
from engiopt.generators.cgan_1d.cgan_1d import Generator as CGAN1DNet
from engiopt.generators.cgan_1d.cgan_1d import prepare_data
from engiopt.transforms import load_normalizer_state

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint


class CGAN1D(Generator):
    """Conditional GAN over 1D / dict-valued designs such as airfoils.

    The normalizers are part of the model, so the checkpoint records the bounds
    fitted during training and loading replays them. They are still fitted from
    the dataset first, which is what supplies their shape and what checkpoints
    written before the bounds were recorded fall back to.
    """

    algo_id = "cgan_1d"
    conditional = True
    design_kinds = ("1d", "dict")
    checkpoint_files = ("generator.pth",)
    primary_state_key = "generator"

    def __init__(self, net: CGAN1DNet, latent_dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.net = net
        self.latent_dim = latent_dim

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> CGAN1D:
        """Load a trained cGAN-1D generator, rebuilding its normalizers."""
        config = resolved.run_config
        _, conds_normalizer, design_normalizer = prepare_data(problem, device)
        conds_normalizer = load_normalizer_state(conds_normalizer, recorded_condition_normalizer(resolved), device)
        design_normalizer = load_normalizer_state(design_normalizer, recorded_design_normalizer(resolved), device)
        net = CGAN1DNet(
            latent_dim=config["latent_dim"],
            n_conds=len(condition_keys_for(problem, resolved)),
            design_shape=design_shape_of(problem),
            design_normalizer=design_normalizer,
            conds_normalizer=conds_normalizer,
        ).to(device)
        net.load_state_dict(
            th.load(resolved.files["generator.pth"], map_location=device, weights_only=True)[cls.primary_state_key]
        )
        net.eval()
        return cls(net=net, latent_dim=config["latent_dim"], problem=problem, device=device, **base)

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        """Map noise plus conditions through the generator."""
        cond = conditions.require_tensor(self.algo_id)
        z = th.randn((n, self.latent_dim), device=self.device)
        return self.net(z, cond)
