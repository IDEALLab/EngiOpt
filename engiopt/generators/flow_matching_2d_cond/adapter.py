"""Load and sample conditional flow-matching checkpoints."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch as th

from engiopt.checkpoint_store import resolve_checkpoint_reference
from engiopt.core import ConditionBatch
from engiopt.core import DEFAULT_HF_ENTITY
from engiopt.core import DEFAULT_HF_REPO_PREFIX
from engiopt.core import design_shape_of
from engiopt.core import Generator
from engiopt.generators.flow_matching_2d_cond.model import build_model
from engiopt.generators.flow_matching_2d_cond.model import integrate

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ModelSource
    from engiopt.checkpoint_store import ResolvedCheckpoint


class FlowMatching2DCond(Generator):
    """Conditional 2D design generator with a learned velocity field."""

    algo_id = "flow_matching_2d_cond"
    checkpoint_files = ("model.pth",)
    primary_state_key = "model"
    net: th.nn.Module

    @classmethod
    def resolve_checkpoint(
        cls,
        *,
        problem_id: str,
        seed: int,
        model_source: ModelSource = "auto",
        hf_entity: str = DEFAULT_HF_ENTITY,
        hf_repo_prefix: str = DEFAULT_HF_REPO_PREFIX,
        local_model_dir: str | None = None,
        config_fingerprint: str | None = None,
    ) -> ResolvedCheckpoint:
        """The selected study weights predate the canonical EngiOpt package layout."""
        if (
            (hf_entity, hf_repo_prefix) == ("IDEALLab", "engiopt-engopt2026")
            and model_source in {"auto", "hf"}
            and not (model_source == "auto" and local_model_dir is not None)
            and config_fingerprint is None
        ):
            return resolve_checkpoint_reference(
                model_source="hf",
                model_ref=f"hf://IDEALLab/engiopt-engopt2026-flow-matching-2d-cond/{problem_id}/selected/v1/euler_16/seed_{seed}",
                revision="58ae97c4aa59784fbe22ab232b5d196ff088e4c7",
                required_files=list(cls.checkpoint_files),
            )
        return super().resolve_checkpoint(
            problem_id=problem_id,
            seed=seed,
            model_source=model_source,
            hf_entity=hf_entity,
            hf_repo_prefix=hf_repo_prefix,
            local_model_dir=local_model_dir,
            config_fingerprint=config_fingerprint,
        )

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
