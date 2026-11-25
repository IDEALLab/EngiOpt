"""Load and sample conditional flow-matching checkpoints."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch as th

from engiopt.core import ConditionBatch
from engiopt.core import design_shape_of
from engiopt.core import Generator
from engiopt.generators.flow_matching_2d_cond.model import build_model
from engiopt.generators.flow_matching_2d_cond.model import integrate

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint


class FlowMatching2DCond(Generator):
    """Conditional 2D design generator with a learned velocity field."""

    algo_id = "flow_matching_2d_cond"
    checkpoint_files = ("model.pth",)
    primary_state_key = "model"
    net: th.nn.Module

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> FlowMatching2DCond:
        """Restore the velocity field and sampling configuration from a package."""
        config = resolved.run_config
        model = cls(problem, device=device, **base)
        for key, expected in (("problem_id", model.problem_id), ("seed", model.seed)):
            if config[key] != expected:
                raise ValueError(f"Checkpoint {key}={config[key]!r}, requested {expected!r}.")
        shape = design_shape_of(problem)
        keys = model.condition_keys
        if resolved.metadata.get("bundle_type") == "selected_checkpoint" and not resolved.metadata.get("condition_keys"):
            study_keys = {
                "beams2d": ("volfrac", "rmin", "forcedist", "overhang_constraint"),
                "heatconduction2d": ("volume", "length"),
            }
            if keys != study_keys.get(config["problem_id"]):
                raise ValueError(f"Problem conditions {keys} differ from the published checkpoint schema.")
        checkpoint = th.load(resolved.files["model.pth"], map_location="cpu", weights_only=True)
        if tuple(checkpoint["design_shape"]) != shape or checkpoint["encoder_hid_dim"] != len(keys):
            raise ValueError("Checkpoint design shape or condition count differs from the problem.")
        model.net = build_model(shape, len(keys), config["layers_per_block"])
        model.net.load_state_dict(checkpoint["model"])
        model.net.to(device).eval()
        model.output_clip = (config["clip_min"], config["clip_max"])
        return model

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        cond = conditions.require_tensor(self.algo_id).reshape(n, 1, self.n_conds)
        noise = th.randn((n, 1, *self.design_shape), device=self.device)
        config = self.run_config
        return integrate(
            self.net, noise, cond, config["integration_steps"], config["num_train_timesteps"], config["method"]
        )
