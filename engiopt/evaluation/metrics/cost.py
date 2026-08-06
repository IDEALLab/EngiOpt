"""What a model costs to use.

The `cost` family was declared in the registry from the start and never
populated, which is telling: generation time is already measured on every run
and written to the leaderboard as provenance, then ignored at ranking time.

It should not be ignored. A model that reaches a marginally better optimality
gap by sampling two orders of magnitude slower is a different proposition from
one that matches it in milliseconds, and a board that cannot express that
difference will keep recommending the expensive one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from engiopt.evaluation.registry import register_metric

if TYPE_CHECKING:
    from engiopt.evaluation.context import EvaluationContext


@register_metric(
    "gen_seconds",
    family="cost",
    cost="cheap",
    higher_is_better=False,
    description="Wall-clock seconds to generate the evaluation sample.",
)
def gen_seconds(ctx: EvaluationContext) -> float:
    """How long the model took to produce its designs.

    Already timed by `Generator.sample` and recorded as a provenance column;
    registering it makes it rankable, so cost can enter a comparison rather
    than sitting beside it.

    Wall-clock is device- and load-dependent, so it compares models within one
    evaluation run and not across machines.
    """
    return float("nan") if ctx.sample_seconds is None else float(ctx.sample_seconds)


@register_metric(
    "params",
    family="cost",
    cost="cheap",
    higher_is_better=False,
    description="Number of trainable parameters in the generator.",
)
def params(ctx: EvaluationContext) -> float:
    """Model size, as a hardware-independent companion to `gen_seconds`.

    Timing depends on the machine and what else is running on it; a parameter
    count does not. Neither is a complete account of cost on its own -- a small
    diffusion model can be slower than a large GAN because it samples
    iteratively -- which is why both are reported.
    """
    return float("nan") if ctx.model_params is None else float(ctx.model_params)
