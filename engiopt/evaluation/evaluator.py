"""Scoring generators and assembling leaderboards.

This replaces the per-model `evaluate_*.py` scripts. Those differed only in how
they loaded and called their model -- which is now the `Generator` contract --
so everything else lives here once::

    ev = Evaluator.for_problem("beams2d", spec="beams2d/v1")
    row = ev.score(generator)  # cheap metrics only
    board = ev.leaderboard(zoo)  # a DataFrame, one row per model
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import datetime as dt
import functools
import importlib.metadata
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

from engiopt.core import ConditionBatch
from engiopt.core import pick_device
from engiopt.evaluation.context import EvaluationContext
from engiopt.evaluation.registry import MetricRegistry
from engiopt.evaluation.registry import METRICS
from engiopt.evaluation.registry import MetricSpec
from engiopt.evaluation.spec import EvalSpec
from engiopt.evaluation.spec import ResolvedSpec
from engiopt.evaluation.spec import source_checkout_commit
from engiopt.utils.all_generators import design_kind_of

# Importing the metrics package is what populates the registry.
import engiopt.evaluation.metrics  # noqa: F401  # isort: skip

if TYPE_CHECKING:
    from collections.abc import Callable

    from engibench.core import Problem
    import numpy.typing as npt
    import torch as th

    from engiopt.core import Generator

PROVENANCE_COLUMNS = (
    "problem_id",
    "algo_id",
    "config_fingerprint",
    "seed",
    "spec_version",
    "n_samples",
    "sample_seconds",
    "checkpoint_repo",
    "checkpoint_path",
    "checkpoint_revision",
    "checkpoint_hash",
    "code_version",
    "engibench_version",
    "evaluated_at",
    "verified",
    "verified_by",
    "verified_at",
    "flags",
)
"""Columns identifying *what was measured*, as opposed to metric values.

`config_fingerprint` and `seed` say which configuration produced the row;
`checkpoint_hash` says which *weights* did. Re-training the same configuration
and seed, or changing the training code, yields different weights under the same
name, so without it a row cannot be traced back to the model that earned it.

`checkpoint_repo` and `checkpoint_path` complete that into an *address*. A hash
proves two rows describe the same weights; only the address lets someone else go
and fetch them. Those four columns together are what make a row a reproducible
claim rather than an assertion, and they are the input to `engiopt.verify`.

`verified` is never set by whoever computed the row. It is stamped by a runner
that re-fetched the checkpoint and reproduced the numbers; see
`engiopt.evaluation.verify`. `flags` carries the integrity checks a row tripped,
which is how a retrieval system stays visible in the table while staying out of
the ranking.

`engibench_version` is the EngiBench that *ran* the evaluation, as opposed to the
one the spec was frozen against. A change to `simulate` or `optimize` moves the
optimality gaps without touching any condition name or dataset, so the spec's
digest cannot catch it. It is recorded rather than enforced: refusing to evaluate
unless the sha matches would lock out every contributor whose EngiBench differs
by any commit, and partitioning ranks by it would mean two models could never be
compared unless they were scored on identical builds. Recording it keeps the
difference visible without either cost.
"""


@dataclass
class Evaluator:
    """Scores generators on one problem under one frozen evaluation spec.

    Attributes:
        problem: The EngiBench problem instance.
        problem_id: Its registry key.
        resolved: The spec with its conditions and reference designs drawn.
        device: Device passed to generators when loading and sampling.
        registry: Metric registry to draw from (defaults to the global one).
    """

    problem: Problem
    problem_id: str
    resolved: ResolvedSpec
    device: th.device
    registry: MetricRegistry = field(default_factory=lambda: METRICS)

    @classmethod
    def for_problem(
        cls,
        problem_id: str,
        *,
        spec: str | EvalSpec | None = None,
        device: th.device | None = None,
        registry: MetricRegistry | None = None,
    ) -> Evaluator:
        """Build an evaluator for a problem, loading its spec.

        Args:
            problem_id: EngiBench problem registry key.
            spec: `"<problem_id>/<version>"`, an `EvalSpec`, or None to load
                `"<problem_id>/v1"`.
            device: Torch device; auto-selected when omitted.
            registry: Metric registry override, useful in tests.
        """
        from engibench.utils.all_problems import BUILTIN_PROBLEMS

        problem = BUILTIN_PROBLEMS[problem_id]()
        eval_spec = spec if isinstance(spec, EvalSpec) else EvalSpec.load(spec or f"{problem_id}/v1")
        if eval_spec.problem_id != problem_id:
            raise ValueError(f"Spec is for {eval_spec.problem_id!r}, not {problem_id!r}.")
        device = device or pick_device()
        problem.reset(seed=eval_spec.condition_seed)
        return cls(
            problem=problem,
            problem_id=problem_id,
            resolved=eval_spec.resolve(problem, device=device),
            device=device,
            registry=registry or METRICS,
        )

    @property
    def spec(self) -> EvalSpec:
        """The frozen evaluation contract in force."""
        return self.resolved.spec

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def context_for(self, generator: Generator) -> EvaluationContext:
        """Sample from a generator and wrap the result in an evaluation context."""
        kind = design_kind_of(self.problem)
        if kind not in generator.design_kinds:
            raise ValueError(
                f"{generator.algo_id!r} supports {generator.design_kinds} design spaces, "
                f"but {self.problem_id!r} is {kind!r}."
            )
        designs = self._sample(generator)
        return EvaluationContext(
            problem=self.problem,
            problem_id=self.problem_id,
            gen_designs=designs,
            ref_designs=self.resolved.ref_designs,
            conditions=self.resolved.conditions,
            sigma=self.spec.sigma,
            volfrac_tol=self.spec.volfrac_tol,
            volume_condition=self.spec.volume_condition,
            sample_seconds=generator.last_sample_seconds,
            objective_weights=self.spec.objective_weights,
            objective_weight_condition=self.spec.objective_weight_condition,
            copy_corpus_fn=self.copy_corpus,
            copy_tol=self.spec.copy_tol,
            resample_permuted=self._permuted_sampler(generator),
        )

    def _sample(self, generator: Generator, order: npt.NDArray[Any] | None = None) -> npt.NDArray[Any]:
        """Draw this spec's batch from a generator, optionally reordering the conditions.

        The seed is the generator's own, so re-drawing with a different
        condition order reuses the identical latent sample and isolates the
        model's response to the conditions themselves.
        """
        conditions = self.resolved.conditions
        tensor = self.resolved.conditions_tensor
        if order is not None:
            conditions = conditions.select(order) if conditions is not None else None
            tensor = tensor[order] if tensor is not None else None
        return generator.sample(
            ConditionBatch(tensor=tensor, dataset=conditions, keys=self.resolved.condition_keys),
            n=self.resolved.n_samples,
            seed=getattr(generator, "seed", None),
        )

    def _permuted_sampler(self, generator: Generator) -> Callable[[npt.NDArray[Any]], npt.NDArray[Any]] | None:
        """A callable re-drawing from `generator` under a permutation, or None if meaningless.

        Withheld when the problem supplies no conditions at all: shuffling
        nothing measures nothing, and a metric that reported 0 there would say
        "this model ignores its conditions" about a model that was never given
        any.
        """
        if self.resolved.conditions_tensor is None and self.resolved.conditions is None:
            return None
        return lambda order: self._sample(generator, order)

    @functools.cached_property
    def copy_corpus(self) -> Callable[[], npt.NDArray[Any]]:
        """Designs from the training split that a model on this problem could have memorized.

        Built once per evaluator and shared by every model in a sweep, and
        deferred behind a callable so a run that selects no memorization metric
        never pays for the fetch.
        """

        @functools.cache
        def corpus() -> npt.NDArray[Any]:
            return self._draw_copy_corpus()

        return corpus

    def _draw_copy_corpus(self) -> npt.NDArray[Any]:
        """Subsample the training split's optimal designs, deterministically.

        Drawn with the spec's own `condition_seed`, so the corpus a model is
        checked against is as reproducible as the conditions it is scored on --
        an audit that drew a different corpus could reach a different verdict.
        """
        try:
            train = self.problem.dataset["train"]
            designs = np.asarray(train["optimal_design"])
        # A problem with no training split simply has no wider corpus; the
        # reference designs still are one, and they are the case that matters.
        except (KeyError, TypeError, AttributeError):
            return np.empty((0, 0))
        size = min(self.spec.copy_corpus_size, len(designs))
        rng = np.random.default_rng(self.spec.condition_seed)
        return designs[rng.choice(len(designs), size, replace=False)]

    def score(
        self,
        generator: Generator,
        *,
        only: list[str] | None = None,
        include_expensive: bool = False,
    ) -> dict[str, Any]:
        """Score one generator, returning a single leaderboard row.

        Args:
            generator: A loaded generator.
            only: Metric names to compute; defaults to the spec's metric list.
            include_expensive: Whether to run simulator-backed metrics. Left
                False, no metric can invoke the simulator or optimizer.

        Returns:
            A dict of provenance columns plus one entry per metric column.
        """
        ctx = self.context_for(generator)
        row = self._provenance(generator, ctx)
        row.update(self.score_context(ctx, only=only, include_expensive=include_expensive))
        return row

    def score_context(
        self,
        ctx: EvaluationContext,
        *,
        only: list[str] | None = None,
        include_expensive: bool = False,
    ) -> dict[str, Any]:
        """Run the selected metrics against an existing context."""
        values: dict[str, Any] = {}
        for spec in self._selected(only, include_expensive=include_expensive):
            values.update(_as_columns(spec, spec.fn(ctx)))
        return values

    def _selected(self, only: list[str] | None, *, include_expensive: bool) -> list[MetricSpec]:
        names = list(only) if only is not None else list(self.spec.metrics)
        specs = self.registry.select(names)
        if not include_expensive:
            specs = [spec for spec in specs if spec.cost == "cheap"]
        return specs

    def _provenance(self, generator: Generator, ctx: EvaluationContext) -> dict[str, Any]:
        """The row's identity and audit trail, minus the verification stamp.

        `verified` is deliberately False here and cannot be set from this side.
        The whole point of the column is that it records someone *else* having
        re-fetched these weights and reproduced these numbers, so a value
        written by the process that computed them would mean nothing.
        """
        return {
            "problem_id": self.problem_id,
            "algo_id": generator.algo_id,
            "config_fingerprint": generator.config_fingerprint,
            "seed": getattr(generator, "seed", None),
            "spec_version": self.spec.version,
            "n_samples": ctx.n_samples,
            "sample_seconds": ctx.sample_seconds,
            "checkpoint_repo": getattr(generator, "checkpoint_repo", None),
            "checkpoint_path": getattr(generator, "checkpoint_path", None),
            "checkpoint_revision": getattr(generator, "checkpoint_revision", None),
            "checkpoint_hash": getattr(generator, "checkpoint_hash", None),
            "code_version": code_version(),
            "engibench_version": engibench_version(),
            "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "verified": False,
            "verified_by": None,
            "verified_at": None,
            "flags": "",
        }

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    def leaderboard(
        self,
        generators: list[Generator],
        *,
        only: list[str] | None = None,
        include_expensive: bool = False,
        on_error: str = "raise",
    ) -> pd.DataFrame:
        """Score many generators into one table.

        Args:
            generators: Loaded generators to score.
            only: Metric names; defaults to the spec's list.
            include_expensive: Whether to run simulator-backed metrics.
            on_error: `"raise"`, or `"skip"` to drop models that fail and carry
                on -- useful for a long unattended sweep where one bad
                checkpoint should not lose the rest of the results.

        Returns:
            A DataFrame with provenance columns first, then metric columns.
        """
        rows: list[dict[str, Any]] = []
        for generator in generators:
            try:
                rows.append(self.score(generator, only=only, include_expensive=include_expensive))
            # One unloadable checkpoint must not lose an entire unattended sweep.
            except Exception as exc:  # noqa: PERF203
                if on_error != "skip":
                    raise
                print(f"[leaderboard] skipping {generator.algo_id}: {exc}")
        return order_columns(pd.DataFrame(rows))


@functools.cache
def engibench_version() -> str:
    """Which EngiBench *ran* this evaluation, however it was installed.

    Shares its implementation with the value `EvalSpec.freeze` records, so the
    spec's "frozen against" and the row's "evaluated on" are the same kind of
    string and can be compared directly.
    """
    from engiopt.evaluation.spec import engibench_version as _spec_engibench_version

    return _spec_engibench_version()


@functools.cache
def code_version() -> str:
    """Which EngiOpt produced a result: the installed version, plus a git sha in a checkout.

    Two evaluations of the same checkpoint can differ if the evaluation code
    changed between them, so the row records which code it was.

    The sha is taken with the same guards as EngiBench's, and for the same
    reason: `git -C` searches upward, so an EngiOpt wheel installed into a
    virtualenv inside some *other* repository would otherwise report that
    repository's commit as the code that produced the score. A row claiming a
    commit it did not run is worse than a row claiming none, because verifying
    it means checking out the wrong tree.
    """
    try:
        version = importlib.metadata.version("engiopt")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    sha = source_checkout_commit(Path(__file__).resolve().parent.parent.parent, distribution="engiopt")
    return f"{version}+{sha}" if sha else version


def _as_columns(spec: MetricSpec, value: float | dict[str, float]) -> dict[str, float]:
    """Normalize a metric's return value into named leaderboard columns."""
    if isinstance(value, dict):
        unexpected = set(value) - set(spec.columns)
        if unexpected:
            raise ValueError(f"Metric {spec.name!r} returned undeclared columns: {sorted(unexpected)}")
        return dict(value)
    if len(spec.columns) != 1:
        raise ValueError(f"Metric {spec.name!r} declares {spec.columns} but returned a single value.")
    return {spec.columns[0]: value}


def order_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Put provenance columns first, metric columns after, in a stable order."""
    if frame.empty:
        return frame
    lead = [col for col in PROVENANCE_COLUMNS if col in frame.columns]
    rest = [col for col in frame.columns if col not in lead]
    return frame[lead + rest]
