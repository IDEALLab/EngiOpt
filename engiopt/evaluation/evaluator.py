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
import subprocess
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
from engiopt.utils.all_generators import design_kind_of

# Importing the metrics package is what populates the registry.
import engiopt.evaluation.metrics  # noqa: F401  # isort: skip

if TYPE_CHECKING:
    from engibench.core import Problem
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
    "checkpoint_revision",
    "checkpoint_hash",
    "code_version",
    "evaluated_at",
)
"""Columns identifying *what was measured*, as opposed to metric values.

`config_fingerprint` and `seed` say which configuration produced the row;
`checkpoint_hash` says which *weights* did. Re-training the same configuration
and seed, or changing the training code, yields different weights under the same
name, so without it a row cannot be traced back to the model that earned it.
"""

NOVELTY_ANCHOR_SAMPLES = 1000
"""Training designs sampled as the novelty anchor; see `Evaluator.train_designs`."""


def _count_parameters(generator: Generator) -> int | None:
    """How many numbers this model had to store to make its predictions.

    Counting torch parameters covers the trained models, but it reads `None` for
    anything that is not a network -- which would quietly exempt the retrieval
    and regression baselines from the one column that prices what a model costs
    to keep. A method whose "parameters" are its training set has a size, and
    the comparison is only honest if that size lands on the same axis.

    So a generator may declare its own count via `parameter_count()`; the torch
    walk is the fallback for everything that does not.
    """
    import torch as th

    declared = getattr(generator, "parameter_count", None)
    if callable(declared):
        count = declared()
        if count is not None:
            return int(count)

    modules = [value for value in vars(generator).values() if isinstance(value, th.nn.Module)]
    if not modules:
        return None
    return sum(p.numel() for module in modules for p in module.parameters())


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
        designs = generator.sample(
            ConditionBatch(
                tensor=self.resolved.conditions_tensor,
                dataset=self.resolved.conditions,
                keys=self.resolved.condition_keys,
            ),
            n=self.resolved.n_samples,
            seed=getattr(generator, "seed", None),
        )
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
            latent_lvae=self.latent_lvae,
            latent_recon_lvae=self.latent_recon_lvae,
            sigma_designs=self.sigma_designs,
            train_designs=self.train_designs,
            model_params=_count_parameters(generator),
        )

    def _load_instrument(self, *, config_fingerprint: str | None, seed: int) -> Any:
        """Load one pinned LVAE, both halves."""
        from engiopt.lvae.checkpoints import load_lvae

        instrument = self.spec.latent_instrument
        assert instrument is not None
        return load_lvae(
            problem_id=self.problem_id,
            design_shape=tuple(self.problem.design_space.shape),  # type: ignore[arg-type]
            algo=instrument.algo,
            seed=seed,
            config_fingerprint=config_fingerprint,
            revision=instrument.revision,
            hf_entity=instrument.hf_entity,
            hf_repo_prefix=instrument.hf_repo_prefix,
            device=self.device,
        )

    @functools.cached_property
    def latent_lvae(self) -> Any:
        """The instrument pinned by the spec, loaded once and shared.

        Returns `None` when the spec pins no instrument; latent metrics then
        raise rather than silently substituting a different autoencoder.

        Raises:
            ValueError: If the loaded instrument's active subspace does not
                match the width the spec recorded, which means a different
                model is being used to measure than the one that was pinned.
        """
        instrument = self.spec.latent_instrument
        if instrument is None:
            return None

        from engiopt.lvae.encode import get_active_mask

        lvae = self._load_instrument(config_fingerprint=instrument.config_fingerprint, seed=instrument.seed)

        n_active = int(get_active_mask(lvae.encoder).sum())
        if instrument.expected_n_active is not None and n_active != instrument.expected_n_active:
            raise ValueError(
                f"latent instrument for {self.problem_id!r} reports {n_active} active dimensions but the spec "
                f"pinned {instrument.expected_n_active}. Latent metrics are only comparable across rows when "
                "the instrument is identical; refusing to score against a different one."
            )
        return lvae

    @functools.cached_property
    def latent_recon_lvae(self) -> Any:
        """The reconstruction-only companion, if the spec pins one.

        Only the dual gap needs it, so its absence is not an error until that
        metric is actually selected.
        """
        instrument = self.spec.latent_instrument
        if instrument is None or not instrument.has_recon_only:
            return None
        return self._load_instrument(
            config_fingerprint=instrument.recon_only_config_fingerprint,
            seed=instrument.recon_only_seed if instrument.recon_only_seed is not None else instrument.seed,
        )

    @functools.cached_property
    def train_designs(self) -> Any:
        """Training designs used as the novelty anchor.

        Capped: novelty is a nearest-neighbour distance, and the nearest
        neighbour stops moving long before the whole split is searched, so the
        full set would cost memory for precision nobody reads. Sampling is
        seeded so the anchor is the same for every model on the board.

        Returns `None` when the problem has no training split.
        """
        dataset = getattr(self.problem, "dataset", None)
        if dataset is None or "train" not in dataset:
            return None
        designs = np.asarray(dataset["train"]["optimal_design"])
        if len(designs) <= NOVELTY_ANCHOR_SAMPLES:
            return designs
        rng = np.random.default_rng(self.spec.condition_seed)
        return designs[rng.choice(len(designs), NOVELTY_ANCHOR_SAMPLES, replace=False)]

    @functools.cached_property
    def sigma_designs(self) -> Any:
        """Validation designs used to calibrate the latent kernel bandwidth.

        Returns `None` when the problem has no validation split, in which case
        the context falls back to the reference set and says so.
        """
        dataset = getattr(self.problem, "dataset", None)
        if dataset is None or "val" not in dataset:
            return None
        return np.asarray(dataset["val"]["optimal_design"])

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
        self._check_requirements(specs)
        return specs

    def _check_requirements(self, specs: list[MetricSpec]) -> None:
        """Reject a selection the spec cannot satisfy, before any model is scored.

        Discovering a missing instrument partway through a leaderboard wastes
        everything computed up to that point, and on a batch run the failure
        surfaces far from its cause.

        Raises:
            ValueError: If a selected metric needs something the spec omits.
        """
        instrument = self.spec.latent_instrument
        missing: dict[str, list[str]] = {}

        for spec in specs:
            for requirement in spec.requires:
                if requirement == "latent_instrument" and instrument is None:
                    missing.setdefault("latent_instrument", []).append(spec.name)
                elif requirement.endswith("recon_only_config_fingerprint") and (
                    instrument is None or not instrument.has_recon_only
                ):
                    missing.setdefault("latent_instrument.recon_only_config_fingerprint", []).append(spec.name)

        if missing:
            detail = "; ".join(f"{key} (needed by {', '.join(names)})" for key, names in missing.items())
            raise ValueError(
                f"spec {self.spec.problem_id}/{self.spec.version} does not supply: {detail}. "
                "Pin it in the spec, or drop those metrics from the selection."
            )

    def _provenance(self, generator: Generator, ctx: EvaluationContext) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "algo_id": generator.algo_id,
            "config_fingerprint": generator.config_fingerprint,
            "seed": getattr(generator, "seed", None),
            "spec_version": self.spec.version,
            "n_samples": ctx.n_samples,
            "sample_seconds": ctx.sample_seconds,
            "checkpoint_revision": getattr(generator, "checkpoint_revision", None),
            "checkpoint_hash": getattr(generator, "checkpoint_hash", None),
            "code_version": code_version(),
            "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
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
def code_version() -> str:
    """Which EngiOpt produced a result: the installed version, plus a git sha in a checkout.

    Two evaluations of the same checkpoint can differ if the evaluation code
    changed between them, so the row records which code it was.
    """
    try:
        version = importlib.metadata.version("engiopt")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    try:
        sha = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    # Outside a checkout (an installed wheel, a container) the version is all there is.
    except (OSError, subprocess.SubprocessError):
        return version
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
