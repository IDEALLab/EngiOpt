"""Template `Generator` adapter -- copy this directory to add a model.

Replace `my_model` throughout, point the imports at your training script, and
you are on the leaderboard. Nothing outside this directory needs to change: the
registry finds your model by walking `engiopt/generators/`.

Checklist:

1. Rename the directory to your `algo_id` (they must match).
2. Set the class attributes below to describe your model.
3. Implement `build` -- rebuild the network from the checkpoint package.
4. Implement `_sample` -- turn conditions into designs.
5. Run: `python -m engiopt.evaluate --problem-id beams2d --generators my_model`

See `CONTRIBUTING_A_MODEL.md` for the full walkthrough.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch as th

from engiopt.core import ConditionBatch
from engiopt.core import Generator

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint


class MyModel(Generator):
    """One-line description of the model, shown by `--list-generators`."""

    algo_id = "my_model"
    """Must equal this directory's name."""

    conditional = True
    """Does `_sample` use the scalar conditions it is given (volume budget, filter radius)?"""

    image_conditional = False
    """Does `_sample` use the *field* conditions -- masks for where a part is
    held, loaded, or cooled? Set True and the evaluator will only pair this model
    with problems that have them. See the image-conditions block in `_sample`."""

    design_kinds = ("2d",)
    """Design spaces this model can serve: any of `1d`, `2d`, `3d`, `dict`."""

    checkpoint_files = ("generator.pth",)
    """Files your training script writes into the checkpoint package."""

    primary_state_key = "generator"
    """Key holding the state dict inside the loaded checkpoint."""

    output_clip: tuple[Any, Any] | None = (1e-3, 1.0)
    """Optional clamp on generated designs; `None` to leave outputs untouched."""

    def __init__(self, net: th.nn.Module, latent_dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.net = net
        self.latent_dim = latent_dim

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> MyModel:
        """Rebuild the trained network from its checkpoint package.

        Fetching the package and choosing the device are already done for you:
        `resolved.run_config` is your training script's `Args` exactly as saved,
        and `resolved.files[name]` is a local path to each checkpoint file.
        """
        config = resolved.run_config

        # Replace this block with your own network, e.g.:
        #
        #     from engiopt.generators.my_model.my_model import Generator as MyNet
        #
        #     net = MyNet(
        #         latent_dim=config["latent_dim"],
        #         # Size the conditioning from the schema the run was trained on --
        #         # not `len(problem.conditions_keys)`, which counts array-valued
        #         # and solver-only conditions your tensor will never carry.
        #         n_conds=len(condition_keys_for(problem, resolved)),
        #         design_shape=design_shape_of(problem),
        #     ).to(device)
        #     state = th.load(resolved.files["generator.pth"], map_location=device)
        #     net.load_state_dict(state[cls.primary_state_key])
        #     net.eval()
        net = cls._build_network(config, problem, device, resolved)

        return cls(net=net, latent_dim=config["latent_dim"], problem=problem, device=device, **base)

    @staticmethod
    def _build_network(
        config: dict[str, Any],
        problem: Problem,
        device: th.device,
        resolved: ResolvedCheckpoint,
    ) -> th.nn.Module:
        """Construct the network and load its weights. Replace this entirely."""
        raise NotImplementedError("Build your network here.")

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        """Draw `n` designs.

        `sample` (the public method) already handles seeding, timing, reshaping
        to `(n, *design_shape)`, and clamping -- so return whatever shape is
        natural for your network.

        Use `conditions.require_tensor(self.algo_id)` for the usual `(n, n_conds)`
        tensor, or `conditions.dataset` if you need the original columns.

        If you set `image_conditional = True`, the field conditions arrive as a
        `(n, n_image_conds, H, W)` tensor at the problem's native resolution --
        65x65 on thermoelastic2d, whose designs are 64x64, because the masks live
        on finite-element nodes rather than elements. Resize to your own grid
        explicitly; the contract will not do it silently::

            masks = conditions.require_images(self.algo_id)  # (n, 4, 65, 65)
            masks = resize_to(masks, *self.design_shape)  # (n, 4, 64, 64)
            return self.net(z, cond, masks)
        """
        cond = conditions.require_tensor(self.algo_id)
        z = th.randn((n, self.latent_dim), device=self.device)
        return self.net(z, cond)
