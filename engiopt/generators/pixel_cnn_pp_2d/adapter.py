"""`Generator` contract for the conditional PixelCNN++ model (2D)."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch as th

from engiopt.core import ConditionBatch
from engiopt.core import Generator
from engiopt.generators.pixel_cnn_pp_2d.pixel_cnn_pp_2d import PixelCNNpp
from engiopt.generators.pixel_cnn_pp_2d.pixel_cnn_pp_2d import sample_from_discretized_mix_logistic

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint


class PixelCNNpp2D(Generator):
    """Autoregressive PixelCNN++ over 2D designs.

    Sampling is one forward pass per pixel, so cost scales with the design
    resolution rather than with a fixed step count. On a 50x100 grid that is
    5000 sequential passes per batch, which makes this the natural upper bound
    for the cost axis of the leaderboard.
    """

    algo_id = "pixel_cnn_pp_2d"
    conditional = True
    design_kinds = ("2d",)
    checkpoint_files = ("model.pth",)
    primary_state_key = "model"
    output_clip = (1e-3, 1.0)

    def __init__(self, net: PixelCNNpp, nr_logistic_mix: int, sampling_batch_size: int = 16, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.net = net
        self.nr_logistic_mix = nr_logistic_mix
        self.sampling_batch_size = sampling_batch_size

    @classmethod
    def build(
        cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, sampling_batch_size: int = 16, **base: Any
    ) -> PixelCNNpp2D:
        """Load a trained PixelCNN++ from its checkpoint package."""
        config = resolved.run_config
        net = PixelCNNpp(
            nr_resnet=config["nr_resnet"],
            nr_filters=config["nr_filters"],
            nr_logistic_mix=config["nr_logistic_mix"],
            resnet_nonlinearity=config["resnet_nonlinearity"],
            dropout_p=config["dropout_p"],
            input_channels=1,
            nr_conditions=len(problem.conditions_keys),
        )
        net.load_state_dict(th.load(resolved.files["model.pth"], map_location=device)[cls.primary_state_key])
        net.eval().to(device)
        return cls(
            net=net,
            nr_logistic_mix=config["nr_logistic_mix"],
            sampling_batch_size=sampling_batch_size,
            problem=problem,
            device=device,
            **base,
        )

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        """Fill the design grid one pixel at a time, in memory-bounded batches."""
        cond = conditions.require_tensor(self.algo_id).reshape(n, self.n_conds, 1, 1)
        height, width = self.design_shape[0], self.design_shape[1]
        batches: list[th.Tensor] = []
        for start in range(0, n, self.sampling_batch_size):
            end = min(n, start + self.sampling_batch_size)
            data = th.zeros((end - start, 1, height, width), device=self.device)
            for i in range(height):
                for j in range(width):
                    out = self.net(data, cond[start:end])
                    data[:, :, i, j] = sample_from_discretized_mix_logistic(out, self.nr_logistic_mix).data[:, :, i, j]
            # Move each finished batch off the accelerator so long runs do not
            # accumulate the whole sample set in device memory.
            batches.append(data.cpu())
        return th.cat(batches, dim=0)
