"""`Generator` contract for the conditional 2D diffusion model."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from diffusers import UNet2DConditionModel
import torch as th

from engiopt.core import condition_keys_for
from engiopt.core import ConditionBatch
from engiopt.core import Generator
from engiopt.generators.diffusion_2d_cond.diffusion_2d_cond import beta_schedule
from engiopt.generators.diffusion_2d_cond.diffusion_2d_cond import denormalize_designs_from_diffusion_range
from engiopt.generators.diffusion_2d_cond.diffusion_2d_cond import DiffusionSampler

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint


class Diffusion2DCond(Generator):
    """Conditional denoising diffusion model over 2D designs.

    Sampling runs the full reverse chain, so this is the slowest generator in the
    zoo by a wide margin -- which is exactly why cost belongs on the leaderboard
    next to quality.
    """

    algo_id = "diffusion_2d_cond"
    conditional = True
    design_kinds = ("2d",)
    checkpoint_files = ("model.pth",)
    primary_state_key = "model"

    def __init__(
        self,
        net: UNet2DConditionModel,
        sampler: DiffusionSampler,
        num_timesteps: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.net = net
        self.sampler = sampler
        self.num_timesteps = num_timesteps
        self.design_min: th.Tensor | None = None
        self.design_max: th.Tensor | None = None

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> Diffusion2DCond:
        """Rebuild the UNet and its noise schedule from the checkpoint package."""
        config = resolved.run_config
        checkpoint = th.load(resolved.files["model.pth"], map_location=device)
        net = UNet2DConditionModel(
            sample_size=problem.design_space.shape,
            in_channels=1,
            out_channels=1,
            cross_attention_dim=64,
            block_out_channels=(32, 64, 128, 256),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            layers_per_block=config["layers_per_block"],
            transformer_layers_per_block=1,
            encoder_hid_dim=len(condition_keys_for(problem, resolved)),
            only_cross_attention=True,
        ).to(device)
        net.load_state_dict(checkpoint[cls.primary_state_key])
        net.eval()
        num_timesteps = config["num_timesteps"]
        betas = beta_schedule(
            t=num_timesteps,
            start=1e-4,
            end=0.02,
            scale=1.0,
            options={
                "cosine": config["noise_schedule"] == "cosine",
                "exp_biasing": config["noise_schedule"] == "exp",
                "exp_bias_factor": 1,
            },
        )
        generator = cls(
            net=net,
            # `.to(device)` is not decoration: the schedule is indexed five
            # times per denoising step, and a host-resident schedule turns each
            # of those into a pipeline stall on CUDA or MPS.
            sampler=DiffusionSampler(num_timesteps, betas).to(device),
            num_timesteps=num_timesteps,
            problem=problem,
            device=device,
            **base,
        )
        if "design_min" in checkpoint and "design_max" in checkpoint:
            generator.design_min = checkpoint["design_min"].to(device)
            generator.design_max = checkpoint["design_max"].to(device)
            generator.output_clip = (
                checkpoint["design_min"].detach().cpu().numpy(),
                checkpoint["design_max"].detach().cpu().numpy(),
            )
        else:
            generator.output_clip = (1e-3, 1.0)
        return generator

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        """Run the reverse diffusion chain from noise, conditioned throughout."""
        cond = conditions.require_tensor(self.algo_id).reshape(n, 1, self.n_conds)
        designs = th.randn((n, 1, *self.design_shape), device=self.device)
        for step in reversed(range(self.num_timesteps)):
            t = th.full((n,), step, device=self.device, dtype=th.long)
            designs = self.sampler.sample_timestep(self.net, designs, t, cond)
        designs = designs.squeeze(1)
        if self.design_min is not None and self.design_max is not None:
            designs = denormalize_designs_from_diffusion_range(designs, self.design_min, self.design_max)
        return designs
