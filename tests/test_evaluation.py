"""Tests for the metric registry, evaluation context, and leaderboard.

Metric correctness is checked by properties that must hold for any correct
implementation (MMD of a set against itself is zero; noise is maximally
diverse), rather than by pinning historical numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from engiopt.evaluation.context import EvaluationContext
from engiopt.evaluation.evaluator import _as_columns
from engiopt.evaluation.evaluator import order_columns
from engiopt.evaluation.leaderboard import append_rows
from engiopt.evaluation.leaderboard import disagreement
from engiopt.evaluation.leaderboard import rank
from engiopt.evaluation.leaderboard import ROW_KEY
from engiopt.evaluation.registry import MetricRegistry
from engiopt.evaluation.registry import METRICS
from engiopt.evaluation.registry import MetricSpec
from engiopt.evaluation.registry import register_metric
from engiopt.evaluation.spec import EvalSpec
from tests.stubs import stub_lvae
from tests.stubs import StubConditions

# Importing the metrics package registers the built-ins.
import engiopt.evaluation.metrics  # noqa: F401  # isort: skip


def _context(problem: Any, gen: np.ndarray, ref: np.ndarray, **kwargs: Any) -> EvaluationContext:
    return EvaluationContext(problem=problem, problem_id="fake", gen_designs=gen, ref_designs=ref, **kwargs)


# ----------------------------------------------------------------------
# Metric properties
# ----------------------------------------------------------------------


def test_mmd_of_identical_sets_is_zero(fake_problem: Any) -> None:
    """A generator that reproduced the reference set exactly should score 0."""
    rng = np.random.default_rng(0)
    designs = rng.random((12, *fake_problem.design_space.shape))
    ctx = _context(fake_problem, designs, designs.copy())
    assert METRICS["mmd"].fn(ctx) == pytest.approx(0.0, abs=1e-9)


def test_mmd_is_invariant_to_sample_order(fake_problem: Any) -> None:
    """MMD compares distributions, so shuffling the samples must not change it."""
    rng = np.random.default_rng(1)
    gen = rng.random((10, *fake_problem.design_space.shape))
    ref = rng.random((10, *fake_problem.design_space.shape))
    baseline = METRICS["mmd"].fn(_context(fake_problem, gen, ref))
    shuffled = METRICS["mmd"].fn(_context(fake_problem, rng.permutation(gen), ref))
    assert baseline == pytest.approx(shuffled, rel=1e-9)


def test_mmd_grows_as_distributions_separate(fake_problem: Any) -> None:
    """Shifting the generated set away from the reference must raise MMD."""
    rng = np.random.default_rng(2)
    ref = rng.random((15, *fake_problem.design_space.shape))
    near = METRICS["mmd"].fn(_context(fake_problem, ref + 0.01, ref, sigma=1.0))
    far = METRICS["mmd"].fn(_context(fake_problem, ref + 0.5, ref, sigma=1.0))
    assert far > near


def test_dpp_prefers_varied_designs_over_duplicates(fake_problem: Any) -> None:
    """A collapsed generator must score below a varied one."""
    rng = np.random.default_rng(3)
    varied = rng.random((8, *fake_problem.design_space.shape))
    collapsed = np.repeat(varied[:1], 8, axis=0)
    ref = rng.random((8, *fake_problem.design_space.shape))
    assert METRICS["dpp"].fn(_context(fake_problem, varied, ref)) > METRICS["dpp"].fn(
        _context(fake_problem, collapsed, ref)
    )


# ----------------------------------------------------------------------
# Cheap / expensive separation
# ----------------------------------------------------------------------


def test_builtin_metrics_declare_their_cost() -> None:
    """Only simulator-backed metrics may be marked expensive.

    `viol` is cheap: feasibility describes the design as generated, so it is
    judged by a constraint check rather than by running the optimizer. Latent
    metrics are cheap too -- encoding is a forward pass, not a simulation.

    Asserted as "exactly these are expensive" rather than "exactly these are
    cheap", so adding a cheap metric does not require editing this test while
    still catching anything that quietly gains access to the solver. Adding an
    expensive one is meant to land here: the median gaps are the same
    simulator runs aggregated differently, and this list is where that is
    declared.
    """
    assert {spec.name for spec in METRICS.select(cost="expensive")} == {
        "iog",
        "cog",
        "fog",
        "iog_median",
        "cog_median",
        "fog_median",
    }
    assert {"mmd", "dpp", "viol"} <= {spec.name for spec in METRICS.select(cost="cheap")}


def test_cheap_metrics_never_touch_the_solver(fake_problem: Any) -> None:
    """Running every cheap metric must not call `problem.reset`, which only the solver path does.

    Every dependency a cheap metric may have is supplied here -- instrument,
    companion, conditions -- so each one actually runs. A metric that raised for
    a missing input would pass this test without ever proving it stays off the
    solver.
    """
    rng = np.random.default_rng(4)
    shape = fake_problem.design_space.shape
    ctx = _context(
        fake_problem,
        rng.random((6, *shape)),
        rng.random((6, *shape)),
        conditions=StubConditions(6),
    )
    ctx.latent_lvae = stub_lvae(shape)
    ctx.latent_recon_lvae = stub_lvae(shape)

    ran = 0
    for spec in METRICS.select(cost="cheap"):
        spec.fn(ctx)
        ran += 1

    assert ran == len(list(METRICS.select(cost="cheap")))
    assert fake_problem.reset_calls == 0


# ----------------------------------------------------------------------
# Registry behaviour
# ----------------------------------------------------------------------


def test_registry_rejects_duplicate_names() -> None:
    """Two metrics with one name would make the leaderboard ambiguous."""
    registry = MetricRegistry()
    register_metric("dup", family="diversity", cost="cheap", higher_is_better=True, registry=registry)(lambda ctx: 1.0)
    with pytest.raises(ValueError, match="already registered"):
        register_metric("dup", family="diversity", cost="cheap", higher_is_better=True, registry=registry)(lambda ctx: 2.0)


def test_registry_rejects_colliding_output_columns() -> None:
    """Two metrics writing the same column would silently overwrite each other."""
    registry = MetricRegistry()
    register_metric(
        "first", family="diversity", cost="cheap", higher_is_better=True, outputs=("shared",), registry=registry
    )(lambda ctx: {"shared": 1.0})
    with pytest.raises(ValueError, match="already emitted"):
        register_metric(
            "second", family="diversity", cost="cheap", higher_is_better=True, outputs=("shared",), registry=registry
        )(lambda ctx: {"shared": 2.0})


def test_multi_output_metric_must_declare_its_columns() -> None:
    """Undeclared columns are rejected so the leaderboard schema stays predictable."""
    spec = MetricSpec(
        name="multi",
        fn=lambda ctx: {"a": 1.0},
        family="diversity",
        cost="cheap",
        higher_is_better=True,
        outputs=("a", "b"),
    )
    assert _as_columns(spec, {"a": 1.0, "b": 2.0}) == {"a": 1.0, "b": 2.0}
    with pytest.raises(ValueError, match="undeclared columns"):
        _as_columns(spec, {"a": 1.0, "c": 3.0})


# ----------------------------------------------------------------------
# Leaderboard
# ----------------------------------------------------------------------


def _board() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "problem_id": "p",
                "algo_id": "a",
                "config_fingerprint": "cfg_a",
                "seed": 1,
                "spec_version": "v1",
                "mmd": 0.1,
                "dpp": 5.0,
            },
            {
                "problem_id": "p",
                "algo_id": "a",
                "config_fingerprint": "cfg_a",
                "seed": 2,
                "spec_version": "v1",
                "mmd": 0.3,
                "dpp": 7.0,
            },
            {
                "problem_id": "p",
                "algo_id": "b",
                "config_fingerprint": "cfg_b",
                "seed": 1,
                "spec_version": "v1",
                "mmd": 0.05,
                "dpp": 1.0,
            },
        ]
    )


def test_rank_respects_each_metric_direction() -> None:
    """Lower MMD ranks first; higher DPP ranks first."""
    assert rank(_board(), "mmd").iloc[0]["algo_id"] == "b"
    assert rank(_board(), "dpp").iloc[0]["algo_id"] == "a"


def test_rank_aggregates_seeds_with_the_median_by_default() -> None:
    """Median across seeds, so one blown-up run cannot decide the ranking."""
    ranked = rank(_board(), "mmd").set_index("algo_id")
    assert ranked.loc["a", "mmd_median"] == pytest.approx(0.2)
    assert ranked.loc["a", "n_seeds"] == 2


def test_disagreement_surfaces_conflicting_rankings() -> None:
    """The two metrics crown different winners; the view must show both."""
    table = disagreement(_board(), ["mmd", "dpp"])
    assert table.loc[("p", "b", "cfg_b", "v1"), "mmd"] == 1
    assert table.loc[("p", "a", "cfg_a", "v1"), "dpp"] == 1


def _two_configs_board() -> pd.DataFrame:
    """One algorithm, two hyperparameter settings, both evaluated at seed 1."""
    return pd.DataFrame(
        [
            {"problem_id": "p", "algo_id": "a", "config_fingerprint": "cfg_a", "seed": 1, "spec_version": "v1",
             "mmd": 0.1},
            {"problem_id": "p", "algo_id": "a", "config_fingerprint": "cfg_b", "seed": 1, "spec_version": "v1",
             "mmd": 0.9},
        ]
    )  # fmt: skip


def test_rank_keeps_hyperparameter_configurations_apart() -> None:
    """Two configs are two entries, not one averaged score over a phantom two seeds."""
    ranked = rank(_two_configs_board(), "mmd")
    assert len(ranked) == 2
    assert set(ranked["config_fingerprint"]) == {"cfg_a", "cfg_b"}
    assert list(ranked["n_seeds"]) == [1, 1]
    assert ranked.iloc[0]["mmd_median"] == pytest.approx(0.1)


def test_rank_keeps_spec_versions_apart() -> None:
    """A score under v1 and one under v2 measure different protocols."""
    frame = pd.DataFrame(
        [
            {"problem_id": "p", "algo_id": "a", "config_fingerprint": "cfg", "seed": 1, "spec_version": "v1",
             "mmd": 0.1},
            {"problem_id": "p", "algo_id": "a", "config_fingerprint": "cfg", "seed": 1, "spec_version": "v2",
             "mmd": 0.9},
        ]
    )  # fmt: skip
    ranked = rank(frame, "mmd")
    assert len(ranked) == 2
    assert set(ranked["spec_version"]) == {"v1", "v2"}


def test_disagreement_keeps_hyperparameter_configurations_apart() -> None:
    """The disagreement view groups exactly as `rank` does."""
    table = disagreement(_two_configs_board(), ["mmd"])
    assert len(table) == 2


def test_append_rows_supersedes_a_rerun_instead_of_duplicating(tmp_path: Any) -> None:
    """Re-evaluating a model replaces its row rather than adding a second one."""
    destination = tmp_path / "board.csv"
    append_rows(_board(), destination)
    rerun = _board().iloc[[0]].copy()
    rerun.loc[:, "mmd"] = 0.999
    append_rows(rerun, destination)

    stored = pd.read_csv(destination)
    assert len(stored) == 3
    match = stored.set_index(ROW_KEY).loc[("p", "a", "cfg_a", 1, "v1")]
    assert match["mmd"] == pytest.approx(0.999)


def test_order_columns_puts_provenance_first() -> None:
    """Leaderboard CSVs should read identity-first, then measurements."""
    scrambled = pd.DataFrame([{"mmd": 0.1, "algo_id": "a", "problem_id": "p"}])
    assert list(order_columns(scrambled).columns)[:2] == ["problem_id", "algo_id"]


# ----------------------------------------------------------------------
# Eval spec
# ----------------------------------------------------------------------


def test_spec_round_trips_through_disk(tmp_path: Any) -> None:
    """A spec must reload byte-identically, since it is the published contract."""
    spec = EvalSpec(problem_id="fake", version="v3", n_samples=7, condition_digest="abc123")
    spec.save(root=tmp_path)
    assert EvalSpec.load("fake/v3", root=tmp_path) == spec


def test_missing_spec_explains_how_to_create_one(tmp_path: Any) -> None:
    """The error should tell you the next command, not just that a file is absent."""
    with pytest.raises(FileNotFoundError, match="freeze"):
        EvalSpec.load("nope/v1", root=tmp_path)


def test_committed_specs_are_frozen() -> None:
    """Every committed spec must carry a digest, or it is not pinning anything."""
    from engiopt.evaluation.spec import SPEC_ROOT

    specs = list(SPEC_ROOT.glob("*/*.json"))
    assert specs, "expected at least one committed eval spec"
    for path in specs:
        spec = EvalSpec.load(str(path))
        assert spec.condition_digest, f"{path} is not frozen"
        assert spec.metrics


# ----------------------------------------------------------------------
# Incremental leaderboard updates
# ----------------------------------------------------------------------


def _row(**overrides: Any) -> dict[str, Any]:
    base = {
        "problem_id": "beams2d",
        "algo_id": "cgan_cnn_2d",
        "config_fingerprint": "aaaa1111",
        "seed": 1,
        "spec_version": "v1",
        "mmd": 0.10,
    }
    base.update(overrides)
    return base


def test_different_hyperparameters_get_their_own_rows() -> None:
    """A sweep of one algorithm must not collapse onto a single row.

    This is the case that silently discarded 49 of 50 sweep configs before
    `config_fingerprint` joined the key.
    """
    from engiopt.evaluation.leaderboard import merge_rows

    board = merge_rows(
        pd.DataFrame([_row(config_fingerprint="aaaa1111", mmd=0.10)]),
        pd.DataFrame([_row(config_fingerprint="bbbb2222", mmd=0.42)]),
    )
    assert len(board) == 2
    assert set(board["mmd"]) == {0.10, 0.42}


def test_reevaluating_the_same_checkpoint_supersedes_it() -> None:
    """Same model, same config, same seed, same spec -> one row, newest wins."""
    from engiopt.evaluation.leaderboard import merge_rows

    board = merge_rows(pd.DataFrame([_row(mmd=0.10)]), pd.DataFrame([_row(mmd=0.99)]))
    assert len(board) == 1
    assert board.iloc[0]["mmd"] == pytest.approx(0.99)


def test_adding_a_model_leaves_existing_rows_untouched() -> None:
    """Publishing one model must not disturb anyone else's results."""
    from engiopt.evaluation.leaderboard import merge_rows

    existing = pd.DataFrame([_row(algo_id="cgan_cnn_2d", mmd=0.10), _row(algo_id="vqgan", mmd=0.20)])
    board = merge_rows(existing, pd.DataFrame([_row(algo_id="my_model", mmd=0.05)]))
    assert len(board) == 3
    assert board.set_index("algo_id").loc["cgan_cnn_2d", "mmd"] == pytest.approx(0.10)
    assert board.set_index("algo_id").loc["vqgan", "mmd"] == pytest.approx(0.20)


def test_seeds_are_separate_rows_and_aggregate_at_ranking_time() -> None:
    """The board stores one row per seed; `rank` collapses them."""
    from engiopt.evaluation.leaderboard import merge_rows

    board = merge_rows(
        pd.DataFrame([_row(seed=1, mmd=0.10)]),
        pd.DataFrame([_row(seed=2, mmd=0.30), _row(seed=3, mmd=0.20)]),
    )
    assert len(board) == 3
    ranked = rank(board, "mmd")
    assert ranked.iloc[0]["n_seeds"] == 3
    assert ranked.iloc[0]["mmd_median"] == pytest.approx(0.20)


def test_already_evaluated_detects_published_work() -> None:
    """`--skip-existing` relies on this to avoid recomputing published rows."""
    from engiopt.evaluation.leaderboard import already_evaluated

    board = pd.DataFrame([_row()])
    assert already_evaluated(board, algo_id="cgan_cnn_2d", config_fingerprint="aaaa1111", seed=1)
    assert not already_evaluated(board, algo_id="cgan_cnn_2d", config_fingerprint="bbbb2222", seed=1)
    assert not already_evaluated(board, algo_id="my_model", config_fingerprint="aaaa1111", seed=1)
    assert not already_evaluated(pd.DataFrame(), algo_id="anything")


# ----------------------------------------------------------------------
# Novelty, cost, and the linear baseline
# ----------------------------------------------------------------------


def _novelty_context(problem: Any, gen: np.ndarray, train: np.ndarray, ref: np.ndarray) -> EvaluationContext:
    return _context(problem, gen, ref, train_designs=train, sigma_designs=train[:30], sigma=5.0)


def test_distribution_metrics_reward_memorization_and_novelty_catches_it(fake_problem: Any) -> None:
    """The gap novelty exists to close.

    MMD is minimized by reproducing the training distribution, so a model that
    replays its training set verbatim scores better than one that generalizes.
    PCA-MMD inherits the same blind spot. Only distance-to-training-set falls.
    """
    rng = np.random.default_rng(0)
    shape = fake_problem.design_space.shape
    train = rng.random((60, *shape))
    ref = train[:12] + rng.normal(scale=0.01, size=(12, *shape))

    memorizer = _novelty_context(fake_problem, train[:12].copy(), train, ref)
    honest = _novelty_context(fake_problem, ref + rng.normal(scale=0.05, size=ref.shape), train, ref)

    # Lower MMD is "better", and the memorizer wins it.
    assert METRICS["mmd"].fn(memorizer) < METRICS["mmd"].fn(honest)
    assert METRICS["pca_mmd"].fn(memorizer) < METRICS["pca_mmd"].fn(honest)

    # Novelty is the only column that tells them apart the right way round.
    assert METRICS["novelty"].fn(memorizer) == pytest.approx(0.0, abs=1e-9)
    assert METRICS["novelty"].fn(honest) > METRICS["novelty"].fn(memorizer)


def test_novelty_is_nan_without_a_training_anchor(fake_problem: Any) -> None:
    """Measured against train specifically; without it there is nothing to say."""
    rng = np.random.default_rng(1)
    shape = fake_problem.design_space.shape
    ctx = _context(fake_problem, rng.random((4, *shape)), rng.random((4, *shape)))
    assert np.isnan(METRICS["novelty"].fn(ctx))


def test_cost_metrics_report_what_was_measured(fake_problem: Any) -> None:
    """Sampling time was already recorded; registering it makes it rankable."""
    rng = np.random.default_rng(2)
    shape = fake_problem.design_space.shape
    ctx = _context(
        fake_problem,
        rng.random((4, *shape)),
        rng.random((4, *shape)),
        sample_seconds=2.5,
        model_params=1234,
    )
    assert METRICS["gen_seconds"].fn(ctx) == pytest.approx(2.5)
    assert METRICS["params"].fn(ctx) == pytest.approx(1234)


def test_cost_metrics_are_nan_when_unmeasured(fake_problem: Any) -> None:
    """A missing measurement is reported as missing, not as zero cost."""
    rng = np.random.default_rng(3)
    shape = fake_problem.design_space.shape
    ctx = _context(fake_problem, rng.random((4, *shape)), rng.random((4, *shape)))
    assert np.isnan(METRICS["gen_seconds"].fn(ctx))
    assert np.isnan(METRICS["params"].fn(ctx))


# ----------------------------------------------------------------------
# Declared requirements
# ----------------------------------------------------------------------


def _evaluator_with(spec: EvalSpec) -> Any:
    """An Evaluator stub carrying only what metric selection reads."""
    from types import SimpleNamespace

    from engiopt.evaluation.evaluator import Evaluator

    evaluator = Evaluator.__new__(Evaluator)
    evaluator.registry = METRICS
    evaluator.resolved = SimpleNamespace(spec=spec)
    return evaluator


def test_a_metric_needing_an_instrument_is_refused_before_anything_is_scored() -> None:
    """Failing partway through a leaderboard discards everything already computed."""
    evaluator = _evaluator_with(EvalSpec(problem_id="beams2d", metrics=("mmd", "lv_mmd")))
    with pytest.raises(ValueError, match="latent_instrument"):
        evaluator._selected(None, include_expensive=False)


def test_the_dual_gap_requirement_is_reported_separately() -> None:
    """A pinned instrument is not enough; the gap also needs its companion."""
    from engiopt.evaluation.spec import LatentInstrument

    spec = EvalSpec(
        problem_id="beams2d",
        metrics=("lv_mmd", "lv_dual_gap"),
        latent_instrument=LatentInstrument(algo="constrained_plvae_2d", seed=1, config_fingerprint="abc"),
    )
    with pytest.raises(ValueError, match="recon_only_config_fingerprint"):
        _evaluator_with(spec)._selected(None, include_expensive=False)


def test_metrics_without_requirements_need_no_setup() -> None:
    """The default path for a new contributor stays free of latent machinery."""
    evaluator = _evaluator_with(EvalSpec(problem_id="beams2d", metrics=("mmd", "dpp", "viol", "novelty", "cond_err")))
    selected = evaluator._selected(None, include_expensive=False)
    assert {spec.name for spec in selected} == {"mmd", "dpp", "viol", "novelty", "cond_err"}


def test_the_committed_specs_need_no_instrument() -> None:
    """v1 is what a contributor evaluates against by default; it must work unconfigured."""
    for path in Path("engiopt/specs").glob("*/v1.json"):
        spec = EvalSpec.load(f"{path.parent.name}/v1")
        required = {req for name in spec.metrics for req in METRICS[name].requires}
        assert not required, f"{path.parent.name}/v1 selects metrics requiring {required}"
