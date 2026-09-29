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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Literal

import numpy as np
import pandas as pd

from engiopt.evaluation.context import EvaluationContext
from engiopt.evaluation.registry import MetricRegistry
from engiopt.evaluation.registry import METRICS

SpaceFit = Callable[[np.ndarray, int], Callable[[np.ndarray], np.ndarray]]
"""Given the reference designs and a width, return a function projecting designs into the space."""

SPACES: dict[str, SpaceFit] = {}
"""Where the set-level metrics can be computed, by name. `pixel` and `pca` are built in;
a learned latent space registers itself here with `register_space`."""


def register_space(name: str, fit: SpaceFit) -> None:
    """Make a space available to `Board.evaluate(space=name)`.

    Args:
        name: The space's name; also the column suffix (`mmd@name`).
        fit: `fit(reference_designs, width)` returning `project(designs) -> codes`.
    """
    SPACES[name] = fit


REFERENCE_ROW = "reference (split-half)"
"""Label of the row scoring one random half of the reference designs against the other.

A scale reference, not a competitor: it shows where real, correct designs land on
each column on this problem. It is half the sample size of a model row, and the
set-level metrics move with sample size, so read it as the order of magnitude a
good model should reach rather than as a number to beat. Never ranked.
"""


@dataclass
class Board:
    """Several models' designs, scored against one set of reference designs.

    Attributes:
        problem: The problem the designs answer; needs `design_space` and
            `check_constraints`, which every EngiBench problem has.
        reference: The withheld optimal designs the models are scored against.
        conditions: The conditions each reference design answers, one row per
            design, as anything indexable by condition name and by row. Needed
            by `viol` on problems whose constraint check reads the conditions,
            and by `volume_error`; without them those columns are blank.
        train: The training designs, for the memorization metrics. Optional.
        sigma: Kernel bandwidth for the distribution and diversity metrics. None,
            the default, uses the median pairwise distance of the training designs
            in whichever space is being scored, or of the reference designs when no
            training split was given, so the kernel is neither saturated nor empty
            on a problem it has never seen.
        frame: The most recent `evaluate` result.
        designs: The designs behind each row of `frame`, when the board sampled them itself.
    """

    problem: Any
    reference: Any
    conditions: Any | None = None
    train: Any | None = None
    sigma: float | None = None
    registry: MetricRegistry = field(default_factory=lambda: METRICS)
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    designs: dict[str, Any] = field(default_factory=dict)

    def evaluate(
        self,
        models: Mapping[str, Any],
        *,
        metrics: list[str] | None = None,
        space: str = "pixel",
        aggregation: Literal["mean", "median"] = "mean",
        width: int = 8,
        reference_row: bool = True,
    ) -> pd.DataFrame:
        """Score every model on every cheap metric.

        Args:
            models: Model name -> its generated designs, one array per model.
            metrics: Metric names; defaults to every metric that needs no simulator.
            space: `pixel` scores the designs themselves. Any other name in
                `SPACES` (`pca` is built in) projects every design into that
                space first and appends `@space` to each column; metrics that
                need the actual designs (`viol`, `train_distance`) are skipped there.
            aggregation: How metrics with one value per design collapse to one number.
            width: Dimension of the space, for spaces that have one (`pca`).
            reference_row: Also score one random half of the reference designs
                against the other half, as the row `REFERENCE_ROW`.

        Returns:
            One row per model, one column per metric output. Also kept as `frame`.
        """
        specs = self.registry.select(metrics) if metrics else self.registry.select(cost="cheap")
        if space != "pixel":
            specs = [spec for spec in specs if not spec.pixel_only]
        reference = np.asarray(self.reference)
        if space not in SPACES:
            raise KeyError(f"Unknown space {space!r}. Registered: {', '.join(SPACES)}")
        project = SPACES[space](reference, width)

        def score(generated: Any, against: Any, chosen: list[Any]) -> dict[str, float]:
            ctx = EvaluationContext(
                problem=self.problem,
                problem_id=getattr(self.problem, "problem_id", type(self.problem).__name__),
                gen_designs=project(np.asarray(generated)),
                ref_designs=project(np.asarray(against)),
                conditions=self.conditions if against is reference else None,
                sigma=self.sigma,
                aggregation=aggregation,
                train_designs_fn=(lambda: project(np.asarray(self.train))) if self.train is not None else None,
            )
            row: dict[str, float] = {}
            for spec in chosen:
                value = spec.fn(ctx)
                row.update(value if isinstance(value, dict) else {spec.name: value})
            return row

        rows = {name: score(designs, reference, specs) for name, designs in models.items()}
        if reference_row and len(reference) >= 4:  # noqa: PLR2004 - two halves of at least two designs
            order = np.random.default_rng(0).permutation(len(reference))
            half = len(reference) // 2
            # Two halves of the reference set answer different conditions, so the
            # per-condition metrics have no meaning there and are left blank.
            unpaired = [spec for spec in specs if spec.family != "conditions"]
            rows[REFERENCE_ROW] = score(reference[order[:half]], reference[order[half:]], unpaired)

        frame = pd.DataFrame.from_dict(rows, orient="index")
        if space != "pixel":
            frame.columns = [f"{column}@{space}" for column in frame.columns]
        self.frame = frame
        return frame

    @classmethod
    def from_evaluator(
        cls,
        evaluator: Any,
        models: Mapping[str, Any],
        *,
        designs: Mapping[str, Any] | None = None,
        expensive: bool = False,
        metrics: list[str] | None = None,
    ) -> Board:
        """Score loaded models, and optionally saved designs, under one evaluation spec.

        The evaluator supplies the spec, the reference designs, the conditions and
        the training split, so every row is comparable. Live models get every column;
        `cond_sens` and the cost columns need a model to re-run or inspect, so they
        stay blank for the `designs` rows. The designs each model produced are kept
        on `Board.designs`, so they can be re-scored in another space or saved.

        Args:
            evaluator: An `Evaluator` for the problem.
            models: Label -> loaded `Generator`.
            designs: Label -> one design per spec condition, e.g. a construction
                built from the dataset.
            expensive: Whether to run the simulator-backed metrics.
            metrics: Metric names; defaults to the spec's list.

        Returns:
            A `Board` with one row per model and per designs entry, in that order.
        """
        rows: dict[str, dict[str, float]] = {}
        sampled: dict[str, Any] = {}
        for label, generator in models.items():
            ctx = evaluator.context_for(generator)
            rows[label] = evaluator.score_context(ctx, only=metrics, include_expensive=expensive)
            sampled[label] = ctx.gen_designs
        for label, batch in (designs or {}).items():
            ctx = evaluator.context_for_designs(batch)
            rows[label] = evaluator.score_context(ctx, only=metrics, include_expensive=expensive)
            sampled[label] = ctx.gen_designs
        board = cls(
            problem=evaluator.problem,
            reference=evaluator.resolved.ref_designs,
            conditions=evaluator.resolved.conditions,
            sigma=evaluator.spec.sigma,
            registry=evaluator.registry,
        )
        board.frame = pd.DataFrame.from_dict(rows, orient="index")
        board.designs = sampled
        return board

    @classmethod
    def load(
        cls,
        problem_id: str,
        models: list[str] | Mapping[str, str | None],
        *,
        designs: Mapping[str, Any] | None = None,
        seed: int = 1,
        spec: Any = None,
        expensive: bool = False,
    ) -> Board:
        """Pull published checkpoints by name and score them: the workshop in one call.

        Args:
            problem_id: An EngiBench problem id, e.g. `"beams2d"`.
            models: Generator names, e.g. `["diffusion_2d_cond", "vqgan"]`. A model
                whose default configuration was never trained needs its config
                fingerprint, given as a mapping: `{"cgan_cnn_2d": "825831f6"}`.
            designs: Extra rows of saved designs, one per spec condition; see `from_evaluator`.
            seed: Training seed of the checkpoints to load.
            spec: Spec reference such as `"beams2d/v2"`, or an `EvalSpec`; defaults to the current spec.
            expensive: Whether to run the simulator-backed metrics.

        Returns:
            A `Board` with one row per model, labelled by name.
        """
        from engiopt.evaluation.evaluator import Evaluator
        from engiopt.utils.all_generators import BUILTIN_GENERATORS

        evaluator = Evaluator.for_problem(problem_id, spec=spec)
        requested = models if isinstance(models, Mapping) else dict.fromkeys(models)
        generators = {
            name: BUILTIN_GENERATORS[name].from_pretrained(
                evaluator.problem, problem_id=problem_id, seed=seed, config_fingerprint=fingerprint
            )
            for name, fingerprint in requested.items()
        }
        return cls.from_evaluator(evaluator, generators, designs=designs, expensive=expensive)

    def explain(self) -> pd.DataFrame:
        """Read every column of the last evaluation.

        Returns:
            One row per column: the question it answers, which way is better, the
            model it picks (blank for a diagnostic, `tie` when the best is shared),
            and the split-half reference value when the board carries that row.
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
                "split-half reference": self.frame.loc[REFERENCE_ROW, column]
                if REFERENCE_ROW in self.frame.index
                else None,
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
        models = self.frame.drop(index=REFERENCE_ROW, errors="ignore")
        return models.sort_values(column, ascending=not spec.higher_is_better)

    def _spec_for(self, column: str) -> Any:
        base = column.split("@", maxsplit=1)[0]
        return next((spec for spec in self.registry.values() if base in spec.columns), None)

    def _repr_html_(self) -> str:
        return self.frame._repr_html_()

    def __repr__(self) -> str:
        return repr(self.frame)


def _pixel(reference: np.ndarray, width: int) -> Callable[[np.ndarray], np.ndarray]:  # noqa: ARG001
    """The designs as they are."""
    return lambda designs: designs


def _pca_projection(reference: np.ndarray, n_components: int) -> Callable[[np.ndarray], np.ndarray]:
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


register_space("pixel", _pixel)
register_space("pca", _pca_projection)
