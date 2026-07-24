"""`Generator` contract for the conditional BezierGAN airfoil model."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch as th

from engiopt.core import ConditionBatch
from engiopt.core import Generator
from engiopt.generators.cgan_bezier.cgan_bezier import _EPS
from engiopt.generators.cgan_bezier.cgan_bezier import Generator as CBezierGANNet
from engiopt.generators.cgan_bezier.cgan_bezier import prepare_data
from engiopt.transforms import flatten_dict_factory

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint


class CGANBezier(Generator):
    """Conditional BezierGAN over airfoil designs.

    Like the unconditional BezierGAN it emits Bezier control points plus a
    scalar angle of attack, which are flattened into the single vector the
    evaluator expects. The latent code is uniform and the noise is normal with
    scale 0.5, matching training.
    """

    algo_id = "cgan_bezier"
    conditional = True
    design_kinds = ("dict",)
    checkpoint_files = ("bezier_generator.pth",)
    primary_state_key = "generator"

    def __init__(self, net: CBezierGANNet, latent_dim: int, noise_dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.net = net
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self._flatten = flatten_dict_factory(self.problem, self.device)

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> CGANBezier:
        """Load a trained conditional BezierGAN, rebuilding its normalizers."""
        config = resolved.run_config
        _, conds_normalizer, design_scalars_normalizer, _keys = prepare_data(problem, device)
        net = CBezierGANNet(
            latent_dim=config["latent_dim"],
            noise_dim=config["noise_dim"],
            num_conds=len(problem.conditions_keys),
            n_control_points=config["bezier_control_pts"],
            n_data_points=problem.design_space["coords"].shape[1],
            conds_normalizer=conds_normalizer,
            design_scalars_normalizer=design_scalars_normalizer,
            eps=_EPS,
            scalar_features=1,
        ).to(device)
        net.load_state_dict(th.load(resolved.files["bezier_generator.pth"], map_location=device)[cls.primary_state_key])
        net.eval()
        return cls(
            net=net, latent_dim=config["latent_dim"], noise_dim=config["noise_dim"], problem=problem, device=device, **base
        )

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        """Generate conditioned Bezier airfoils and flatten the dict designs."""
        cond = conditions.require_tensor(self.algo_id)
        latent = th.rand(n, self.latent_dim, device=self.device)
        noise = 0.5 * th.randn(n, self.noise_dim, device=self.device)
        coords, *_rest, alphas = self.net(latent, noise, cond)
        designs = [
            {"coords": coord, "angle_of_attack": alpha[0]}
            for coord, alpha in zip(coords.detach().cpu().numpy(), alphas.detach().cpu().numpy())
        ]
        return self._flatten(designs)
