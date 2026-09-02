"""`Generator` contract for the unconditional 1D diffusion model."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from denoising_diffusion_pytorch import GaussianDiffusion1D
from denoising_diffusion_pytorch import Unet1D
import numpy as np
import torch as th

from engiopt.core import ConditionBatch
from engiopt.core import design_shape_of
from engiopt.core import Generator
from engiopt.core import recorded_design_normalizer
from engiopt.generators.diffusion_1d.diffusion_1d import prepare_data
from engiopt.transforms import load_normalizer_state

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint


def _padding_for(length: int) -> int:
    """Padding needed to make a 1D design length divisible by 8, as the UNet requires."""
    return (8 - length % 8) % 8


class Diffusion1D(Generator):
    """Unconditional denoising diffusion over 1D / dict-valued designs.

    The UNet needs a sequence length divisible by 8, so designs are padded for
    generation and the padding is trimmed off again before evaluation.
    """

    algo_id = "diffusion_1d"
    conditional = False
    design_kinds = ("1d", "dict")
    checkpoint_files = ("model.pth",)
    primary_state_key = "model"

    def __init__(self, net: GaussianDiffusion1D, design_normalizer: Any, padding_size: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.net = net
        self.design_normalizer = design_normalizer
        self.padding_size = padding_size

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> Diffusion1D:
        """Rebuild the 1D UNet and its diffusion wrapper from the checkpoint."""
        config = resolved.run_config
        padding_size = _padding_for(design_shape_of(problem)[0])
        padded_shape = (design_shape_of(problem)[0] + padding_size,)
        _, design_normalizer = prepare_data(problem, padding_size, device)
        design_normalizer = load_normalizer_state(design_normalizer, recorded_design_normalizer(resolved), device)
        unet = Unet1D(dim=config["unet_dim"], channels=config["n_channels"]).to(device)
        diffusion = GaussianDiffusion1D(
            unet,
            seq_length=int(np.prod(padded_shape)),
            auto_normalize=config.get("auto_norm", True),
        ).to(device)
        diffusion.load_state_dict(
            th.load(resolved.files["model.pth"], map_location=device, weights_only=True)[cls.primary_state_key]
        )
        diffusion.eval()
        return cls(
            net=diffusion,
            design_normalizer=design_normalizer,
            padding_size=padding_size,
            problem=problem,
            device=device,
            **base,
        )

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:  # noqa: ARG002 - unconditional by design
        """Sample from the diffusion model, denormalize, and trim the UNet padding."""
        designs = self.net.sample(n).squeeze(1)
        designs = self.design_normalizer.denormalize(designs)
        if self.padding_size > 0:
            designs = designs[:, : -self.padding_size]
        return designs
