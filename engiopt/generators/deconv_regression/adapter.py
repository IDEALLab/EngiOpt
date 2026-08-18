"""`Generator` contract for the deconvolutional regression baseline."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch as th

from engiopt.core import condition_keys_for
from engiopt.core import ConditionBatch
from engiopt.core import Generator
from engiopt.generators.deconv_regression.deconv_regression import DeconvRegressor

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint


class DeconvRegression(Generator):
    """Supervised conditions-to-design regressor -- Habibi et al.'s deconvolutional network.

    Deterministic: one condition in, one design out, the same one every time.
    Trained under a pixel loss it predicts the conditional mean design, so where
    several structures satisfy one brief it returns their average. The diversity
    columns are where that shows up, and it shows up the same way for the ridge
    and retrieval baselines -- what differs between the three is only how much
    capacity was spent getting to a single answer.
    """

    algo_id = "deconv_regression"
    conditional = True
    design_kinds = ("2d",)
    checkpoint_files = ("deconv_regression.pth",)
    primary_state_key = "model"
    # Several 2D simulators are unstable at exactly zero density.
    output_clip = (1e-3, 1.0)

    volume_condition = "volfrac"

    def __init__(
        self,
        net: DeconvRegressor,
        cond_mean: th.Tensor,
        cond_std: th.Tensor,
        design_min: float,
        design_max: float,
        *,
        match_volume: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.net = net
        self.cond_mean = cond_mean
        self.cond_std = cond_std
        self.design_min = design_min
        self.design_max = design_max
        self.match_volume = match_volume

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> DeconvRegression:
        """Rebuild the network and the normalization its weights were fitted under.

        The condition standardization and the design range travel in the
        checkpoint rather than being recomputed here: recomputing them from
        whatever dataset happens to be installed would silently change what the
        weights mean.
        """
        payload = th.load(resolved.files["deconv_regression.pth"], map_location=device, weights_only=False)
        config = resolved.run_config
        net = DeconvRegressor(
            n_conds=len(condition_keys_for(problem, resolved)),
            design_shape=tuple(payload["design_shape"]),
            hidden=int(payload.get("hidden", config.get("hidden", 256))),
            num_filters=tuple(payload.get("num_filters", config.get("num_filters", (256, 128, 64, 32)))),
        )
        net.load_state_dict(payload[cls.primary_state_key])
        net.eval().to(device)
        return cls(
            net=net,
            cond_mean=payload["cond_mean"].to(device),
            cond_std=payload["cond_std"].to(device),
            design_min=float(payload["design_min"]),
            design_max=float(payload["design_max"]),
            match_volume=bool(payload.get("match_volume", config.get("match_volume", False))),
            problem=problem,
            device=device,
            **base,
        )

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        """Evaluate the network at the requested conditions."""
        cond = conditions.require_tensor(self.algo_id)[:n].to(self.device).float()
        normalized = (cond - self.cond_mean) / self.cond_std
        designs = self.net(normalized)
        designs = designs * (self.design_max - self.design_min) + self.design_min

        if not self.match_volume:
            return designs
        keys = conditions.keys or self.condition_keys
        if self.volume_condition not in keys:
            return designs
        from engiopt.generators.knn_retrieval.adapter import match_volume_fraction

        targets = cond[:, keys.index(self.volume_condition)].detach().cpu().numpy()
        rescaled = match_volume_fraction(designs.detach().cpu().numpy(), targets)
        return th.from_numpy(rescaled).to(self.device).float()
