"""Template `Generator` adapter. Copy this directory to add a model.

This file and `my_model.py` beside it are a working pair: the loader below
rebuilds exactly the network that training script saves. Nothing outside your
directory needs to change, because the registry finds your model by walking
`engiopt/generators/`.

After copying the directory to `engiopt/generators/<your_name>/`, change the
name in three places. They must all agree:

1. Rename `my_model.py` to `<your_name>.py`. The training script names its
   checkpoint repository after its own filename.
2. Set `algo_id` below to `<your_name>`.
3. Point the import below at `engiopt.generators.<your_name>.<your_name>`.

Then replace the network in the training script, and `build` and `_sample`
here to match it.

Check the names agree with:

    python -m pytest tests/test_generator_registry.py -q

See `CONTRIBUTING_A_MODEL.md` for the full walkthrough.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch as th

from engiopt.core import condition_keys_for
from engiopt.core import ConditionBatch
from engiopt.core import design_shape_of
from engiopt.core import Generator

# Change this import to your own package and training script.
from engiopt.generators._template.my_model import Generator as MyNet

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint


class MyModel(Generator):
    """One-line description of the model, shown by `--list-generators`."""

    algo_id = "my_model"
    """Must equal this directory's name."""

    conditional = True
    """Does `_sample` use the conditions it is given?"""

    design_kinds = ("2d",)
    """Design spaces this model can serve: any of `1d`, `2d`, `3d`, `dict`."""

    checkpoint_files = ("generator.pth",)
    """Files your training script writes into the checkpoint package."""

    primary_state_key = "generator"
    """Key holding the state dict inside the loaded checkpoint."""

    output_clip: tuple[Any, Any] | None = (1e-3, 1.0)
    """Optional clamp on generated designs. `None` leaves outputs untouched."""

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
        net = MyNet(
            latent_dim=config["latent_dim"],
            # Size the conditioning from the conditions the run was trained on,
            # not `len(problem.conditions_keys)`, which also counts array-valued
            # and solver-only conditions a dense tensor never carries.
            n_conds=len(condition_keys_for(problem, resolved)),
            design_shape=design_shape_of(problem),
        ).to(device)
        state = th.load(resolved.files["generator.pth"], map_location=device, weights_only=True)
        net.load_state_dict(state[cls.primary_state_key])
        net.eval()
        return cls(net=net, latent_dim=config["latent_dim"], problem=problem, device=device, **base)

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        """Draw `n` designs.

        `sample`, the public method, already handles seeding, timing, reshaping
        to `(n, *design_shape)` and clamping, so return whatever shape is
        natural for your network.

        `conditions.require_tensor(self.algo_id)` is the usual `(n, n_conds)`
        tensor. The base class has already put it in the form your training
        script saw: the same columns in the same order, with any scaling the
        checkpoint recorded applied.
        """
        cond = conditions.require_tensor(self.algo_id)
        z = th.randn((n, self.latent_dim), device=self.device)
        return self.net(z, cond)
