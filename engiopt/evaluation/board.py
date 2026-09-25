"""Score several models at once and read the result.

This is the whole user-facing evaluation surface for saved designs::

    board = Board(problem, reference=REFERENCE, train=TRAIN)
    board.evaluate(MODELS, space="pixel", aggregation="mean")
    board.explain()
    board.rank("mmd")

Where a metric is computed and how per-design values collapse to one number are
arguments here, not properties of the metric, so every metric is written once.
For a loaded `Generator` -- which is what the physics metrics and `cond_sens`
need -- use `Evaluator.score`.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Literal, TYPE_CHECKING

import numpy as np
import pandas as pd

from engiopt.evaluation.context import EvaluationContext
from engiopt.evaluation.registry import MetricRegistry
from engiopt.evaluation.registry import METRICS

if TYPE_CHECKING:
    from collections.abc import Mapping

Space = Literal["pixel", "pca"]
"""Where the set-level metrics are computed. Learned latent spaces arrive with the LVAE."""

REFERENCE_ROW = "reference (split-half)"
"""Label of the row scoring half of the reference designs against the other half.

It is what real, correct designs score on every column, measured on this problem
rather than assumed, and it is the row every other row is read against.
"""


@dataclass
class Board:
    """Several models' designs, scored against one set of reference designs.

    Attributes:
        problem: The problem the designs answer; needs `design_space` and
            `check_constraints`, which every EngiBench problem has.
        reference: The withheld optimal designs the models are scored against.
        train: The training designs, for the memorization metrics. Optional.
        sigma: Kernel bandwidth for the distribution metrics.
        frame: The most recent `evaluate` result.
    """

    problem: Any
    reference: Any
    train: Any | None = None
    sigma: float = 10.0
    registry: MetricRegistry = field(default_factory=lambda: METRICS)
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)

    def evaluate(
        self,
        models: Mapping[str, Any],
        *,
        metrics: list[str] | None = None,
        space: Space = "pixel",
        aggregation: Literal["mean", "median"] = "mean",
        pca_dims: int = 8,
        reference_row: bool = True,
    ) -> pd.DataFrame:
        """Score every model on every cheap metric.

        Args:
            models: Model name -> its generated designs, one array per model.
            metrics: Metric names; defaults to every metric that needs no simulator.
            space: `pixel` scores the designs themselves. `pca` projects the
                designs onto the leading principal components of the reference
                designs first, and appends `@pca` to each column; metrics that
                need the actual designs (`viol`, `copy_rate`) are skipped there.
            aggregation: How metrics with one value per design collapse to one number.
            pca_dims: Number of components when `space="pca"`.
            reference_row: Also score one random half of the reference designs
                against the other half, as the row `REFERENCE_ROW`.

        Returns:
            One row per model, one column per metric output. Also kept as `frame`.
        """
        specs = self.registry.select(metrics) if metrics else self.registry.select(cost="cheap")
        if space != "pixel":
            specs = [spec for spec in specs if not spec.pixel_only]
        reference = np.asarray(self.reference)
        project = _pca_projection(reference, pca_dims) if space == "pca" else (lambda x: x)

        def score(generated: Any, against: Any) -> dict[str, float]:
            ctx = EvaluationContext(
                problem=self.problem,
                problem_id=getattr(self.problem, "problem_id", type(self.problem).__name__),
                gen_designs=project(np.asarray(generated)),
                ref_designs=project(np.asarray(against)),
                sigma=self.sigma,
                aggregation=aggregation,
                copy_corpus_fn=(lambda: np.asarray(self.train)) if self.train is not None else None,
            )
            row: dict[str, float] = {}
            for spec in specs:
                value = spec.fn(ctx)
                row.update(value if isinstance(value, dict) else {spec.name: value})
            return row

        rows = {name: score(designs, reference) for name, designs in models.items()}
        if reference_row and len(reference) >= 4:  # noqa: PLR2004 - two halves of at least two designs
            order = np.random.default_rng(0).permutation(len(reference))
            half = len(reference) // 2
            rows[REFERENCE_ROW] = score(reference[order[:half]], reference[order[half:]])

        frame = pd.DataFrame.from_dict(rows, orient="index")
        if space != "pixel":
            frame.columns = [f"{column}@{space}" for column in frame.columns]
        self.frame = frame
        return frame

    def explain(self) -> pd.DataFrame:
        """Read every column of the last evaluation.

        Returns:
            One row per column: the question it answers, which way is better, the
            model it picks (blank for a diagnostic, `tie` when the best is shared),
            and what real designs scored on it.
        """
        models = self.frame.drop(index=REFERENCE_ROW, errors="ignore")
        rows = {}
        for column in self.frame.columns:
            spec = self._spec_for(column)
            if spec is None:
                continue
            picks = ""
            values = models[column].dropna()
            if spec.higher_is_better is not None and not values.empty:
                best = values.min() if spec.higher_is_better is False else values.max()
                winners = list(values.index[values == best])
                picks = winners[0] if len(winners) == 1 else f"tie: {', '.join(winners)}"
            rows[column] = {
                "question": spec.description,
                "direction": spec.direction,
                "picks": picks,
                "real designs score": self.frame.loc[REFERENCE_ROW, column] if REFERENCE_ROW in self.frame.index else None,
            }
        return pd.DataFrame.from_dict(rows, orient="index")

    def rank(self, column: str) -> pd.DataFrame:
        """Models ordered best-first on one column.

        Raises:
            ValueError: If the column is a diagnostic, which is read but not ranked on.
        """
        spec = self._spec_for(column)
        if spec is None:
            raise KeyError(f"{column!r} is not a metric column on this board.")
        if spec.higher_is_better is None:
            raise ValueError(
                f"{column!r} is a diagnostic: it says whether the other columns mean what they look like. "
                f"Read it beside them; do not rank on it."
            )
        return self.frame.sort_values(column, ascending=not spec.higher_is_better)

    def _spec_for(self, column: str) -> Any:
        base = column.split("@", maxsplit=1)[0]
        return next((spec for spec in self.registry.values() if base in spec.columns), None)

    def _repr_html_(self) -> str:
        return self.frame._repr_html_()

    def __repr__(self) -> str:
        return repr(self.frame)


def _pca_projection(reference: np.ndarray, n_components: int) -> Any:
    """Projection onto the leading principal components of the reference designs.

    The control that asks whether a *learned* space was needed: if PCA of the
    data answers the same questions, any rotation of the same width would do.
    """
    flat = np.ascontiguousarray(reference.reshape(len(reference), -1), dtype=np.float64)
    mean = flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat - mean, full_matrices=False)
    components = np.ascontiguousarray(vt[: min(n_components, len(vt))].T)

    def project(designs: np.ndarray) -> np.ndarray:
        centered = np.ascontiguousarray(designs.reshape(len(designs), -1), dtype=np.float64) - mean
        # einsum rather than `@`: on small matrices Apple's BLAS raises spurious FP warnings here.
        return np.einsum("ij,jk->ik", centered, components)

    return project
