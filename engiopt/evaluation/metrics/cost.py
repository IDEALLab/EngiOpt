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


@register_metric(
    "train_minutes",
    family="cost",
    cost="cheap",
    higher_is_better=False,
    description="Wall-clock minutes to train this model, when the figure is known.",
)
def train_minutes(ctx: EvaluationContext) -> float:
    """What it cost to make the model, as opposed to what it costs to run it.

    The column the field argues about and never reports. Sampling time prices a
    forward pass; this prices the decision to adopt the method at all, and it is
    where a lookup table and a diffusion model differ by three orders of
    magnitude rather than by a few percent of MMD.

    NaN unless somebody supplied it, because no checkpoint package records it
    yet. That gap is the point: the benchmark can tell you a model's parameter
    count to the digit and cannot tell you what it cost to train.
    """
    return float("nan") if ctx.train_minutes is None else float(ctx.train_minutes)
