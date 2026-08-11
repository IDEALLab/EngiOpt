"""Registry of evaluation metrics.

A metric belongs to the *comparison* (generated designs vs a reference set under
a problem), not to a model -- so metrics are registered functions taking an
`EvaluationContext`, rather than methods on a generator.

Registering a metric is one decorated function::

    @register_metric("mmd", family="distribution", cost="cheap", higher_is_better=False)
    def mmd(ctx: EvaluationContext) -> float:
        return engiopt.metrics.mmd(ctx.gen_designs, ctx.ref_designs, sigma=ctx.sigma)

The declaration is what lets the evaluator keep simulation-free metrics strictly
separate from simulator-backed ones, and what lets the leaderboard know which
direction of each column is "better".
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from engiopt.evaluation.context import EvaluationContext

MetricCost = Literal["cheap", "expensive"]
"""`cheap` metrics never invoke a simulator or optimizer; `expensive` ones may."""

MetricFamily = Literal["feasibility", "conditions", "performance", "distribution", "diversity", "memorization", "cost"]
"""The question a metric answers. Also the grouping used by leaderboard views.

`memorization` and `conditions` are integrity families: they do not say how good
a model is, they say whether its other scores mean what they appear to. A
retrieval system and an unconditional model both post excellent `mmd`, and only
these two families distinguish them from a model that earned it.
"""

MetricFn = Callable[["EvaluationContext"], "float | dict[str, float]"]


@dataclass(frozen=True)
class MetricSpec:
    """A registered metric and everything the evaluator needs to know about it."""

    name: str
    fn: MetricFn
    family: MetricFamily
    cost: MetricCost
    higher_is_better: bool | None
    """None for metrics with no intrinsic direction (e.g. a realized volume fraction)."""
    outputs: tuple[str, ...] = ()
    """Column names produced. Defaults to `(name,)` for single-valued metrics."""
    description: str = ""

    @property
    def columns(self) -> tuple[str, ...]:
        """Leaderboard columns this metric fills."""
        return self.outputs or (self.name,)


class MetricRegistry(Mapping[str, MetricSpec]):
    """Name-to-`MetricSpec` mapping populated by the `register_metric` decorator."""

    def __init__(self) -> None:
        self._metrics: dict[str, MetricSpec] = {}

    def __getitem__(self, name: str) -> MetricSpec:
        if name not in self._metrics:
            available = ", ".join(sorted(self._metrics))
            raise KeyError(f"Unknown metric {name!r}. Registered: {available}")
        return self._metrics[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._metrics)

    def __len__(self) -> int:
        return len(self._metrics)

    def add(self, spec: MetricSpec) -> None:
        """Register a metric, rejecting duplicate names and duplicate output columns."""
        if spec.name in self._metrics:
            raise ValueError(f"Metric {spec.name!r} is already registered.")
        taken = {col: owner.name for owner in self._metrics.values() for col in owner.columns}
        for column in spec.columns:
            if column in taken:
                raise ValueError(f"Metric {spec.name!r} emits column {column!r}, already emitted by {taken[column]!r}.")
        self._metrics[spec.name] = spec

    def select(
        self,
        names: list[str] | None = None,
        *,
        cost: MetricCost | None = None,
        family: MetricFamily | None = None,
    ) -> list[MetricSpec]:
        """Return specs filtered by name, cost, and/or family, in registration order."""
        specs = [self[name] for name in names] if names is not None else list(self._metrics.values())
        if cost is not None:
            specs = [spec for spec in specs if spec.cost == cost]
        if family is not None:
            specs = [spec for spec in specs if spec.family == family]
        return specs

    def columns(self, specs: list[MetricSpec] | None = None) -> list[str]:
        """All leaderboard columns for the given specs (default: every metric)."""
        return [col for spec in (specs if specs is not None else self._metrics.values()) for col in spec.columns]


METRICS = MetricRegistry()
"""The global metric registry."""


def register_metric(
    name: str,
    *,
    family: MetricFamily,
    cost: MetricCost,
    higher_is_better: bool | None,
    outputs: tuple[str, ...] = (),
    description: str = "",
    registry: MetricRegistry | None = None,
) -> Callable[[MetricFn], MetricFn]:
    """Register an evaluation metric.

    Args:
        name: Unique metric name, also the default output column.
        family: Which engineering question this answers.
        cost: `cheap` if it never runs a simulator, `expensive` otherwise.
        higher_is_better: Ranking direction, or None if the metric is diagnostic.
        outputs: Column names, when the metric returns a dict of several values.
        description: One-line explanation, surfaced by `engiopt.evaluate --list-metrics`.
        registry: Target registry; defaults to the global one.

    Returns:
        The undecorated function, so it stays directly callable and testable.
    """

    def decorator(fn: MetricFn) -> MetricFn:
        (registry or METRICS).add(
            MetricSpec(
                name=name,
                fn=fn,
                family=family,
                cost=cost,
                higher_is_better=higher_is_better,
                outputs=outputs,
                description=description or (fn.__doc__ or "").strip().split("\n")[0],
            )
        )
        return fn

    return decorator
