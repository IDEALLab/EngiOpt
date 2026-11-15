"""Registry of evaluation metrics.

A metric is a function of an `EvaluationContext` -- the generated designs, the
reference designs they are scored against, and what the spec says about how to
read them -- plus a short declaration of what the number means::

    @register_metric("viol", family="feasibility", cost="cheap", higher_is_better=False)
    def viol(ctx: EvaluationContext) -> float:
        \"\"\"What fraction of the generated designs violate the problem's constraints?\"\"\"
        ...

The docstring's first line is the metric's description: the question it answers,
written so it makes sense without knowing the metric's name. `higher_is_better`
says which way to rank; `None` means the metric is a diagnostic that is read but
never ranked on. `cost` keeps metrics that touch the simulator separate from the
ones that do not.

Where a metric is computed -- pixel space, a PCA of the reference designs, a
learned latent space -- and how one value per design collapses to one number are
not properties of the metric. They are chosen when a board is evaluated; see
`engiopt.evaluation.board.Board`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

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
    """A registered metric and what a reader needs to know to use its column."""

    name: str
    fn: MetricFn
    family: MetricFamily
    cost: MetricCost
    higher_is_better: bool | None
    """Ranking direction. None: a diagnostic, read beside other columns but never ranked on."""
    description: str = ""
    """The question the value answers, in one plain sentence. Defaults to the docstring's first line."""
    outputs: tuple[str, ...] = ()
    """Column names produced. Defaults to `(name,)` for single-valued metrics."""
    pixel_only: bool = False
    """True if the metric needs the actual designs -- a constraint check, a copy corpus --
    and so cannot be asked in a PCA or latent space."""

    @property
    def columns(self) -> tuple[str, ...]:
        """Leaderboard columns this metric fills."""
        return self.outputs or (self.name,)

    @property
    def direction(self) -> str:
        """The ranking direction as a reader sees it."""
        return {True: "higher is better", False: "lower is better", None: "diagnostic"}[self.higher_is_better]


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
        """Register a metric, refusing a second definition under the same name or column."""
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
        """Metrics by name, cost, or family -- every metric if nothing is given."""
        specs = [self[name] for name in names] if names is not None else list(self._metrics.values())
        if cost is not None:
            specs = [spec for spec in specs if spec.cost == cost]
        if family is not None:
            specs = [spec for spec in specs if spec.family == family]
        return specs

    def columns(self, specs: list[MetricSpec] | None = None) -> list[str]:
        """All leaderboard columns for the given specs (default: every metric)."""
        return [col for spec in (specs if specs is not None else self._metrics.values()) for col in spec.columns]

    def explain(self) -> Any:
        """One row per metric: the question it answers, its family, direction and cost.

        Returns:
            A `pandas.DataFrame` indexed by metric name.
        """
        import pandas as pd

        rows = {
            spec.name: {"question": spec.description, "family": spec.family, "direction": spec.direction, "cost": spec.cost}
            for spec in self._metrics.values()
        }
        return pd.DataFrame.from_dict(rows, orient="index")


METRICS = MetricRegistry()
"""The global metric registry."""


def register_metric(
    name: str,
    *,
    family: MetricFamily,
    cost: MetricCost,
    higher_is_better: bool | None,
    description: str = "",
    outputs: tuple[str, ...] = (),
    pixel_only: bool = False,
    registry: MetricRegistry | None = None,
) -> Callable[[MetricFn], MetricFn]:
    """Register an evaluation metric.

    Args:
        name: Unique metric name, also the default output column.
        family: Which engineering question this answers.
        cost: `cheap` if it never runs a simulator, `expensive` otherwise.
        higher_is_better: Ranking direction, or None for a diagnostic.
        description: The question the value answers; defaults to the docstring's first line.
        outputs: Column names, when the metric returns a dict of several values.
        pixel_only: True if the metric needs the actual designs rather than codes in some space.
        registry: Target registry; defaults to the global one.

    Returns:
        The undecorated function, so it stays directly callable and testable.
    """

    def decorator(fn: MetricFn) -> MetricFn:
        (METRICS if registry is None else registry).add(
            MetricSpec(
                name=name,
                fn=fn,
                family=family,
                cost=cost,
                higher_is_better=higher_is_better,
                description=description or (fn.__doc__ or "").strip().split("\n")[0],
                outputs=outputs,
                pixel_only=pixel_only,
            )
        )
        return fn

    return decorator
