"""`Generator` contract for the BezierGAN airfoil model."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch as th

from engiopt.core import ConditionBatch
from engiopt.core import Generator
from engiopt.core import recorded_design_normalizer
from engiopt.generators.gan_bezier.gan_bezier import Generator as BezierGANNet
from engiopt.generators.gan_bezier.gan_bezier import prepare_data
from engiopt.transforms import flatten_dict_factory
from engiopt.transforms import load_normalizer_state

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint

_EPS = 1e-7
"""Numerical floor used by the Bezier parameterization, matching training."""


class GANBezier(Generator):
    """BezierGAN over airfoil designs.

    The network emits Bezier control points plus a scalar angle of attack, so a
    design is a dict rather than an array. It is flattened here into the single
    vector the evaluator and metrics expect.

    The latent code is sampled uniformly rather than normally, matching how this
    model was trained and previously evaluated.
    """

    algo_id = "gan_bezier"
    conditional = False
    design_kinds = ("dict",)
    checkpoint_files = ("bezier_generator.pth",)
    primary_state_key = "generator"

    def __init__(self, net: BezierGANNet, latent_dim: int, noise_dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.net = net
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self._flatten = flatten_dict_factory(self.problem, self.device)

    @classmethod
    def build(
        cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, n_prepare_samples: int = 50, **base: Any
    ) -> GANBezier:
        """Load a trained BezierGAN generator, rebuilding its scalar normalizer."""
        config = resolved.run_config
        _, design_scalars_normalizer, _ = prepare_data(problem, n_prepare_samples, device)
        design_scalars_normalizer = load_normalizer_state(
            design_scalars_normalizer, recorded_design_normalizer(resolved), device
        )
        net = BezierGANNet(
            latent_dim=config["latent_dim"],
            noise_dim=config["noise_dim"],
            n_control_points=config["bezier_control_pts"],
            n_data_points=problem.design_space["coords"].shape[1],
            design_scalars_normalizer=design_scalars_normalizer,
            eps=_EPS,
            scalar_features=1,
        ).to(device)
        net.load_state_dict(
            th.load(resolved.files["bezier_generator.pth"], map_location=device, weights_only=True)[cls.primary_state_key]
        )
        net.eval()
        return cls(
            net=net, latent_dim=config["latent_dim"], noise_dim=config["noise_dim"], problem=problem, device=device, **base
        )

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:  # noqa: ARG002 - unconditional by design
        """Generate Bezier airfoils and flatten the dict-valued designs."""
        latent = th.rand(n, self.latent_dim, device=self.device)
        noise = 0.5 * th.randn(n, self.noise_dim, device=self.device)
        coords, *_rest, alphas = self.net(latent, noise)
        designs = [
            {"coords": coord, "angle_of_attack": alpha[0]}
            for coord, alpha in zip(coords.detach().cpu().numpy(), alphas.detach().cpu().numpy())
        ]
        return self._flatten(designs)
