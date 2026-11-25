"""Load and sample conditional flow-matching checkpoints."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch as th

from engiopt.core import condition_keys_for
from engiopt.core import ConditionBatch
from engiopt.core import design_shape_of
from engiopt.core import Generator
from engiopt.generators.flow_matching_2d_cond.model import build_model
from engiopt.generators.flow_matching_2d_cond.model import integrate

if TYPE_CHECKING:
    from diffusers import UNet2DConditionModel
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint


class FlowMatching2DCond(Generator):
    """Conditional 2D design generator with a learned velocity field."""

    algo_id = "flow_matching_2d_cond"
    checkpoint_files = ("model.pth",)
    primary_state_key = "model"
    net: UNet2DConditionModel

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> FlowMatching2DCond:
        """Restore the velocity field and sampling configuration from a package."""
        checkpoint = th.load(resolved.files["model.pth"], map_location=device, weights_only=True)
        config = resolved.run_config
        shape = design_shape_of(problem)
        keys = condition_keys_for(problem, resolved)
        if resolved.metadata.get("bundle_type") == "selected_checkpoint" and not resolved.metadata.get("condition_keys"):
            study_keys = {
                "beams2d": ("volfrac", "rmin", "forcedist", "overhang_constraint"),
                "heatconduction2d": ("volume", "length"),
            }
            if keys != study_keys.get(config["problem_id"]):
                raise ValueError(f"Problem conditions {keys} differ from the published checkpoint schema.")
        if tuple(checkpoint["design_shape"]) != shape or checkpoint["encoder_hid_dim"] != len(keys):
            raise ValueError("Checkpoint design shape or condition count differs from the problem.")
        model = cls(problem, device=device, **base)
        model.net = build_model(shape, len(keys), config["layers_per_block"]).to(device)
        model.net.load_state_dict(checkpoint["model"])
        model.net.eval()
        model.output_clip = (config["clip_min"], config["clip_max"])
        return model

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        cond = conditions.require_tensor(self.algo_id).reshape(n, 1, self.n_conds)
        noise = th.randn((n, 1, *self.design_shape), device=self.device)
        config = self.run_config
        return integrate(
            self.net, noise, cond, config["integration_steps"], config["num_train_timesteps"], config["method"]
        )
