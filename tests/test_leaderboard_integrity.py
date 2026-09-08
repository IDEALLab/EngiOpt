"""Tests for the things a *public* leaderboard needs that an internal one does not.

An internal board can assume its rows were produced in good faith by someone who
had no reason to fabricate them. A public board cannot assume either, and the
two failure modes are independent:

- Someone reports numbers no model produced. Defended by re-execution, not by
  schema -- see `engiopt.evaluation.verify`.
- Someone reports honest numbers from a model that games the metric rather than
  solving the problem. Defended by measuring the gaming, since the scoring
  protocol is public and cannot be un-published.

The headline case for the second is a retrieval system. The spec names which
conditions are scored and the dataset supplies the optimal design for each, so
returning those designs tops `mmd`, `iog`, and `fog` by construction. The tests
below assert that such a model is *detected*, not that it scores badly: it
should score wonderfully and be visibly a copier.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from engiopt.evaluation.context import EvaluationContext
from engiopt.evaluation.leaderboard import eligible_rows
from engiopt.evaluation.leaderboard import rank
from engiopt.evaluation.registry import METRICS
from engiopt.evaluation.spec import EvalSpec
from engiopt.evaluation.submission import admission_problems
from engiopt.evaluation.submission import FLAG_IGNORES_CONDITIONS
from engiopt.evaluation.submission import FLAG_MEMORIZED
from engiopt.evaluation.submission import FLAG_UNVERIFIED
from engiopt.evaluation.submission import integrity_flags
from engiopt.evaluation.submission import prepare_submission
from engiopt.evaluation.submission import SubmissionRejectedError

# Importing the metrics package registers the built-ins.
import engiopt.evaluation.metrics  # noqa: F401  # isort: skip


def _context(problem: Any, gen: np.ndarray, ref: np.ndarray, **kwargs: Any) -> EvaluationContext:
    return EvaluationContext(problem=problem, problem_id="fake", gen_designs=gen, ref_designs=ref, **kwargs)


# ----------------------------------------------------------------------
# Memorization: a retrieval system must be visible as one
# ----------------------------------------------------------------------


def test_a_generator_returning_the_reference_designs_is_caught(fake_problem: Any) -> None:
    """The attack this metric exists for.

    A lookup table keyed on the condition vector returns the dataset-optimal
    design for each scored condition. That is the *definition* of a perfect
    `mmd` and a near-zero `iog`, so no amount of care in those metrics can
    distinguish it from a model that learned the problem. This one can.
    """
    rng = np.random.default_rng(0)
    ref = rng.random((10, *fake_problem.design_space.shape))
    ctx = _context(fake_problem, ref.copy(), ref)

    scores = METRICS["novelty"].fn(ctx)

    assert scores["copy_rate"] == 1.0
    assert scores["novelty"] == pytest.approx(0.0, abs=1e-12)
    # And it does indeed post a perfect distribution score, which is the point.
    assert METRICS["mmd"].fn(ctx) == pytest.approx(0.0, abs=1e-9)


def test_a_generator_producing_its_own_designs_is_not_flagged(fake_problem: Any) -> None:
    """The check must not fire on a model that merely resembles the data."""
    rng = np.random.default_rng(1)
    ref = rng.random((10, *fake_problem.design_space.shape))
    ctx = _context(fake_problem, rng.random((10, *fake_problem.design_space.shape)), ref)

    scores = METRICS["novelty"].fn(ctx)

    assert scores["copy_rate"] == 0.0
    assert scores["novelty"] > 0


def test_copying_the_training_split_is_caught_too(fake_problem: Any) -> None:
    """Not only the scored references are copyable -- the whole public dataset is.

    A model that memorized the training split and emits rows of it under
    whatever conditions it is given never touches a reference design, so a check
    that looked only at those would report it as perfectly novel.
    """
    rng = np.random.default_rng(2)
    ref = rng.random((6, *fake_problem.design_space.shape))
    train = rng.random((20, *fake_problem.design_space.shape))
    ctx = _context(fake_problem, train[:6].copy(), ref, copy_corpus_fn=lambda: train)

    assert METRICS["novelty"].fn(ctx)["copy_rate"] == 1.0


def test_near_copies_count_as_copies(fake_problem: Any) -> None:
    """Adding imperceptible noise to a retrieved design must not launder it.

    Otherwise the defense is defeated by one line, and the tolerance is what
    decides how much perturbation counts as having generated something.
    """
    rng = np.random.default_rng(3)
    ref = rng.random((8, *fake_problem.design_space.shape))
    barely_perturbed = ref + rng.normal(scale=1e-4, size=ref.shape)

    scores = METRICS["novelty"].fn(_context(fake_problem, barely_perturbed, ref, copy_tol=0.01))

    assert scores["copy_rate"] == 1.0


def test_novelty_is_diagnostic_and_cannot_be_ranked_on() -> None:
    """Ranking on novelty would put pure noise in first place.

    The metric has no good direction -- zero means retrieval, large means only
    "unlike the data", which is what a broken model also achieves. Registering a
    direction would have created a new thing to game while closing an old one.
    """
    assert METRICS["novelty"].higher_is_better is None
    with pytest.raises(ValueError, match="diagnostic"):
        rank(pd.DataFrame([{"problem_id": "p", "algo_id": "a", "novelty": 0.5}]), "novelty")


# ----------------------------------------------------------------------
# Conditioning: a model must be shown to read its brief
# ----------------------------------------------------------------------


def test_a_model_ignoring_its_conditions_scores_zero_sensitivity(fake_problem: Any) -> None:
    """Same output whatever the brief: the signature of conditions never reaching the network."""
    rng = np.random.default_rng(4)
    designs = rng.random((8, *fake_problem.design_space.shape))
    ctx = _context(fake_problem, designs, designs.copy(), resample_permuted=lambda _order: designs)

    assert METRICS["cond_sens"].fn(ctx) == pytest.approx(0.0, abs=1e-12)


def test_a_model_reading_its_conditions_scores_above_zero(fake_problem: Any) -> None:
    """A generator whose output follows the condition order must register a response."""
    rng = np.random.default_rng(5)
    designs = rng.random((8, *fake_problem.design_space.shape))
    ctx = _context(fake_problem, designs, designs.copy(), resample_permuted=lambda order: designs[order])

    assert METRICS["cond_sens"].fn(ctx) > 0


def test_sensitivity_is_unavailable_rather_than_zero_without_a_comparison(fake_problem: Any) -> None:
    """An unconditional *problem* must not be reported as a model ignoring conditions.

    Zero would be a claim about the model. NaN is the honest answer: there was
    nothing to shuffle, so nothing was measured.
    """
    rng = np.random.default_rng(6)
    designs = rng.random((8, *fake_problem.design_space.shape))

    assert np.isnan(METRICS["cond_sens"].fn(_context(fake_problem, designs, designs.copy())))


def test_an_honestly_unconditional_model_is_not_flagged() -> None:
    """Being unconditional is allowed by the contract; claiming otherwise is not.

    The flag fires on the *inconsistency* between what a model declares and what
    it does, so a model registered as unconditional keeps a clean row.
    """
    row = {"algo_id": "gan_cnn_2d", "cond_sens": 0.0, "verified": True}

    from engiopt.utils.all_generators import BUILTIN_GENERATORS

    assert not BUILTIN_GENERATORS["gan_cnn_2d"].conditional
    assert FLAG_IGNORES_CONDITIONS not in integrity_flags(row)


def test_a_conditional_model_that_ignores_conditions_is_flagged() -> None:
    """The inconsistent case: declares conditional, output does not move."""
    row = {"algo_id": "cgan_cnn_2d", "cond_sens": 0.0, "verified": True}

    from engiopt.utils.all_generators import BUILTIN_GENERATORS

    assert BUILTIN_GENERATORS["cgan_cnn_2d"].conditional
    assert FLAG_IGNORES_CONDITIONS in integrity_flags(row)


# ----------------------------------------------------------------------
# Admission: what a public board refuses outright
# ----------------------------------------------------------------------


def _publishable(**overrides: Any) -> dict[str, Any]:
    row = {
        "problem_id": "beams2d",
        "algo_id": "cgan_cnn_2d",
        "config_fingerprint": "023dd1fb",
        "seed": 1,
        "spec_version": "v1",
        "checkpoint_repo": "someone/engiopt-cgan-cnn-2d",
        "checkpoint_path": "beams2d/seed_1",
        "checkpoint_revision": "cafe1234",
        "checkpoint_hash": "0011223344556677",
        "mmd": 0.04,
    }
    row.update(overrides)
    return row


def test_a_row_with_no_checkpoint_address_is_refused() -> None:
    """Nobody can re-run it, so it is not a result -- it is a claim with no referent."""
    problems = admission_problems(_publishable(checkpoint_repo=None, checkpoint_path=None))

    assert any("no fetchable checkpoint" in problem for problem in problems)


def test_a_row_evaluated_from_a_local_directory_is_refused() -> None:
    """`--model-source local` leaves no repo or revision, which is the common accident.

    The message has to name the fix, because the submitter's evaluation
    succeeded and they have no reason to suspect the checkpoint was the problem.
    """
    problems = admission_problems(_publishable(checkpoint_repo=None, checkpoint_path=None, checkpoint_revision=None))

    assert any("--checkpoint-backend hf" in problem for problem in problems)


def test_a_submitter_cannot_stamp_their_own_row_as_verified() -> None:
    """Verification means somebody else re-ran it. Self-service defeats the entire mechanism."""
    problems = admission_problems(_publishable(verified=True))

    assert any("cannot be self-asserted" in problem for problem in problems)


def test_preparing_a_submission_clears_any_verification_stamp() -> None:
    """Even absent malice, a stale True from a re-published row must not survive."""
    prepared = prepare_submission(pd.DataFrame([_publishable()]))

    assert not prepared["verified"].iloc[0]
    assert FLAG_UNVERIFIED in prepared["flags"].iloc[0]


def test_every_rejected_row_is_reported_at_once() -> None:
    """Fixing a batch one error per round-trip is the difference between usable and not."""
    frame = pd.DataFrame([_publishable(checkpoint_repo=None), _publishable(seed=2, algo_id=None)])

    with pytest.raises(SubmissionRejectedError) as excinfo:
        prepare_submission(frame)

    assert len(excinfo.value.problems_by_row) == 2


def test_a_memorizing_row_is_published_but_flagged() -> None:
    """Published, because hiding it throws away what the board just learned.

    Flagged, because calling it first place would be the board endorsing the
    thing it exists to detect.
    """
    prepared = prepare_submission(pd.DataFrame([_publishable(copy_rate=0.9)]), EvalSpec(problem_id="beams2d"))

    assert len(prepared) == 1
    assert FLAG_MEMORIZED in prepared["flags"].iloc[0]


# ----------------------------------------------------------------------
# Eligibility: what gets ranked, as distinct from what gets published
# ----------------------------------------------------------------------


def _scored(**overrides: Any) -> dict[str, Any]:
    row = _publishable(verified=True, flags="")
    row.update(overrides)
    return row


def test_unverified_rows_are_published_but_never_ranked() -> None:
    """The whole trust model in one assertion."""
    frame = pd.DataFrame([_scored(verified=False, flags=FLAG_UNVERIFIED)])

    assert eligible_rows(frame).empty


def test_flagged_rows_stay_on_the_board_and_out_of_the_ranking() -> None:
    """A retrieval system belongs in the table and not in the ordering."""
    frame = pd.DataFrame([_scored(algo_id="honest"), _scored(algo_id="copier", flags=FLAG_MEMORIZED)])

    eligible = eligible_rows(frame)

    assert list(eligible["algo_id"]) == ["honest"]
    assert len(frame) == 2


def test_an_entry_missing_a_required_seed_is_not_ranked() -> None:
    """The cherry-picking defense.

    Twenty seeds run, the best three published, and a median over a maximum
    looks exactly like an honest median in every individual row. Requiring
    *named* seeds removes the choice; requiring a count would not.
    """
    spec = EvalSpec(problem_id="beams2d", required_seeds=(1, 2, 3))
    frame = pd.DataFrame([_scored(seed=1), _scored(seed=2)])

    assert eligible_rows(frame, spec).empty


def test_an_entry_covering_every_required_seed_is_ranked() -> None:
    """And the rule must not lock out the people who followed it."""
    spec = EvalSpec(problem_id="beams2d", required_seeds=(1, 2, 3))
    frame = pd.DataFrame([_scored(seed=1), _scored(seed=2), _scored(seed=3)])

    ranked = rank(frame, "mmd", eval_spec=spec)

    assert len(ranked) == 1
    assert ranked["n_seeds"].iloc[0] == 3


def test_seed_coverage_is_judged_per_entry_not_across_the_board() -> None:
    """One complete entry must not lend its seeds to an incomplete one."""
    spec = EvalSpec(problem_id="beams2d", required_seeds=(1, 2, 3))
    frame = pd.DataFrame(
        [
            _scored(algo_id="complete", seed=1),
            _scored(algo_id="complete", seed=2),
            _scored(algo_id="complete", seed=3),
            _scored(algo_id="partial", seed=1),
        ]
    )

    assert set(eligible_rows(frame, spec)["algo_id"]) == {"complete"}


def test_ranking_can_be_asked_for_the_raw_table() -> None:
    """A local run has nothing verified yet, and still wants to see its own ordering."""
    frame = pd.DataFrame([_scored(algo_id="a", verified=False, mmd=0.1), _scored(algo_id="b", verified=False, mmd=0.2)])

    ranked = rank(frame, "mmd", eligible_only=False)

    assert list(ranked["algo_id"]) == ["a", "b"]


def test_ranking_an_empty_eligible_set_does_not_explode() -> None:
    """A board where nothing is verified yet is the normal state on day one."""
    frame = pd.DataFrame([_scored(verified=False)])

    assert rank(frame, "mmd").empty


# ----------------------------------------------------------------------
# End to end: the evaluator has to actually feed the integrity metrics
# ----------------------------------------------------------------------


def _evaluator(problem: Any, spec: EvalSpec, ref: np.ndarray, conditions: Any) -> Any:
    """An `Evaluator` wired to a fake problem, bypassing spec resolution.

    `Evaluator.for_problem` would load a real EngiBench problem and draw a real
    dataset. What is under test here is the wiring between the evaluator and the
    integrity metrics -- the permuted re-sample and the copy corpus -- which is
    the part that no direct-context test can reach.
    """
    import torch as th

    from engiopt.evaluation.evaluator import Evaluator
    from engiopt.evaluation.spec import ResolvedSpec

    resolved = ResolvedSpec(
        spec=spec,
        conditions_tensor=th.tensor([[float(i), 0.5] for i in range(len(ref))]),
        conditions=conditions,
        ref_designs=ref,
        indices=np.arange(len(ref)),
        condition_keys=("volfrac", "rmin"),
    )
    return Evaluator(problem=problem, problem_id="fake", resolved=resolved, device=th.device("cpu"))


def _generator(problem: Any, produce: Any, *, algo_id: str = "demo", conditional: bool = True) -> Any:
    """A minimal real `Generator` whose `_sample` is supplied by the caller."""
    from engiopt.core import Generator

    class _Demo(Generator):
        design_kinds = ("2d",)

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)

        @classmethod
        def build(cls, resolved: Any, problem: Any, device: Any, **base: Any) -> Any:
            raise NotImplementedError

        def _sample(self, conditions: Any, n: int) -> Any:
            return produce(conditions, n)

    _Demo.algo_id = algo_id
    _Demo.conditional = conditional
    import torch as th

    return _Demo(problem=problem, problem_id="fake", seed=1, device=th.device("cpu"))


def test_the_evaluator_detects_a_lookup_table_end_to_end(fake_problem: Any) -> None:
    """The full path: sample, build the corpus, score, and report `copy_rate`.

    This is the claim the whole memorization defense rests on, so it is asserted
    through the evaluator rather than against a hand-built context -- a metric
    that works but is never fed would pass every other test in this file.
    """
    spec = EvalSpec(problem_id="fake", n_samples=4, metrics=("mmd", "novelty"))
    ref = np.stack([np.full(fake_problem.design_space.shape, v) for v in (0.4, 0.5, 0.6, 0.7)])
    conditions = fake_problem.dataset["train"].select([0, 1, 2, 0])

    evaluator = _evaluator(fake_problem, spec, ref, conditions)
    lookup_table = _generator(fake_problem, lambda _conditions, _n: ref)
    row = evaluator.score(lookup_table)

    assert row["copy_rate"] == 1.0
    assert row["mmd"] == pytest.approx(0.0, abs=1e-9)
    assert FLAG_MEMORIZED in integrity_flags(row, spec)


def test_the_evaluator_leaves_an_honest_model_unflagged(fake_problem: Any) -> None:
    """The same path, for a model that generates rather than retrieves."""
    spec = EvalSpec(problem_id="fake", n_samples=4, metrics=("mmd", "novelty", "cond_sens"))
    ref = np.stack([np.full(fake_problem.design_space.shape, v) for v in (0.4, 0.5, 0.6, 0.7)])
    conditions = fake_problem.dataset["train"].select([0, 1, 2, 0])
    rng = np.random.default_rng(7)

    evaluator = _evaluator(fake_problem, spec, ref, conditions)
    # Output tracks the first condition column, so it is genuinely conditional.
    honest = _generator(
        fake_problem,
        lambda conditions, n: (
            np.stack([np.full(fake_problem.design_space.shape, 0.05 * float(c[0])) for c in conditions.tensor])
            + rng.normal(scale=0.05, size=(n, *fake_problem.design_space.shape))
        ),
    )
    row = evaluator.score(honest)

    assert row["copy_rate"] == 0.0
    assert row["cond_sens"] > 0
    assert not [flag for flag in integrity_flags(row, spec) if flag != FLAG_UNVERIFIED]


def test_the_evaluator_catches_a_conditional_model_that_ignores_conditions(fake_problem: Any) -> None:
    """Declares itself conditional, returns the identical batch for any brief."""
    spec = EvalSpec(problem_id="fake", n_samples=4, metrics=("cond_sens",))
    ref = np.stack([np.full(fake_problem.design_space.shape, v) for v in (0.4, 0.5, 0.6, 0.7)])
    conditions = fake_problem.dataset["train"].select([0, 1, 2, 0])
    fixed = np.full((4, *fake_problem.design_space.shape), 0.25)

    evaluator = _evaluator(fake_problem, spec, ref, conditions)
    row = evaluator.score(_generator(fake_problem, lambda _conditions, _n: fixed, conditional=True))

    assert row["cond_sens"] == pytest.approx(0.0, abs=1e-12)


def test_the_permuted_re_sample_holds_the_latent_draw_fixed(fake_problem: Any) -> None:
    """The comparison is only meaningful if the *only* thing that changed is the brief.

    Both draws are seeded identically, so a model whose randomness is the same
    and whose conditions differ isolates the conditional response. If the seed
    were not replayed, every model would look condition-sensitive because every
    model would have re-rolled its noise.
    """
    import torch as th

    spec = EvalSpec(problem_id="fake", n_samples=4, metrics=("cond_sens",))
    ref = np.stack([np.full(fake_problem.design_space.shape, v) for v in (0.4, 0.5, 0.6, 0.7)])
    conditions = fake_problem.dataset["train"].select([0, 1, 2, 0])

    def _pure_noise(_conditions: Any, n: int) -> Any:
        return th.rand(n, *fake_problem.design_space.shape)

    evaluator = _evaluator(fake_problem, spec, ref, conditions)
    row = evaluator.score(_generator(fake_problem, _pure_noise))

    # Noise that ignores its conditions must read as zero response, not as a
    # large one produced by re-rolling the random draw.
    assert row["cond_sens"] == pytest.approx(0.0, abs=1e-12)


def test_the_copy_corpus_is_drawn_once_for_a_whole_sweep(fake_problem: Any) -> None:
    """Every model in a sweep is checked against the same corpus, fetched once.

    Both halves matter: a per-model corpus would be slow, and a corpus that
    varied between models would make their `copy_rate` values incomparable.
    """
    spec = EvalSpec(problem_id="fake", n_samples=4, metrics=("novelty",))
    ref = np.stack([np.full(fake_problem.design_space.shape, v) for v in (0.4, 0.5, 0.6, 0.7)])
    conditions = fake_problem.dataset["train"].select([0, 1, 2, 0])
    evaluator = _evaluator(fake_problem, spec, ref, conditions)

    draws = 0
    original = evaluator._draw_copy_corpus

    def _counting_draw() -> Any:
        nonlocal draws
        draws += 1
        return original()

    evaluator._draw_copy_corpus = _counting_draw  # type: ignore[method-assign]
    for _ in range(3):
        evaluator.score(_generator(fake_problem, lambda _c, n: np.zeros((n, *fake_problem.design_space.shape))))

    assert draws == 1


# ----------------------------------------------------------------------
# Two people, one model name
# ----------------------------------------------------------------------


def test_two_contributors_of_the_same_model_do_not_share_a_row() -> None:
    """On a public board the model name plus config plus seed is not unique.

    Two people who both train `cgan_cnn_2d` with this repository's default
    hyperparameters on seed 1 produce identical identity columns and genuinely
    different weights. Merging them would let either one's push replace the
    other's verified row, silently and with no malice required.
    """
    from engiopt.evaluation.leaderboard import merge_rows

    mine = pd.DataFrame([_scored(checkpoint_repo="me/engiopt-cgan-cnn-2d", mmd=0.10)])
    theirs = pd.DataFrame([_scored(checkpoint_repo="you/engiopt-cgan-cnn-2d", mmd=0.20)])

    merged = merge_rows(mine, theirs)

    assert len(merged) == 2
    assert set(merged["mmd"]) == {0.10, 0.20}


def test_re_evaluating_your_own_checkpoint_still_supersedes_your_row() -> None:
    """The de-duplication that `ROW_KEY` exists for must keep working."""
    from engiopt.evaluation.leaderboard import merge_rows

    first = pd.DataFrame([_scored(mmd=0.10)])
    second = pd.DataFrame([_scored(mmd=0.20)])

    merged = merge_rows(first, second)

    assert len(merged) == 1
    assert merged["mmd"].iloc[0] == 0.20


def test_two_contributors_are_ranked_as_separate_entries() -> None:
    """And their seeds must not be pooled into one median over a mixture."""
    spec = EvalSpec(problem_id="beams2d", required_seeds=(1,))
    frame = pd.DataFrame(
        [
            _scored(checkpoint_repo="me/engiopt-cgan-cnn-2d", seed=1, mmd=0.10),
            _scored(checkpoint_repo="you/engiopt-cgan-cnn-2d", seed=1, mmd=0.20),
        ]
    )

    ranked = rank(frame, "mmd", eval_spec=spec)

    assert len(ranked) == 2
    assert list(ranked["rank"]) == [1, 2]
    assert ranked["checkpoint_repo"].iloc[0] == "me/engiopt-cgan-cnn-2d"


# ----------------------------------------------------------------------
# A model must never fall off a run without saying so
# ----------------------------------------------------------------------


def test_a_model_no_fingerprint_is_scoped_to_falls_back_to_canonical() -> None:
    """Otherwise it is dropped before the load is attempted -- silently.

    `--generators gan_cnn_2d vqgan --config-fingerprints gan_cnn_2d:6293adb3`
    scopes the only entry to `gan_cnn_2d`, leaving `vqgan` with an empty
    fingerprint list. An empty list means the loop body never runs, so vqgan
    produces no row, no error, and no message. A leaderboard quietly missing an
    entrant is worse than one reporting a load failure.
    """
    from engiopt.evaluate import _fingerprints_for

    assert _fingerprints_for(("gan_cnn_2d:6293adb3",), "vqgan") == (None,)
    # The cases that already worked must keep working.
    assert _fingerprints_for(("gan_cnn_2d:6293adb3",), "gan_cnn_2d") == ("6293adb3",)
    assert _fingerprints_for(("abc123",), "vqgan") == ("abc123",)
    assert _fingerprints_for((), "vqgan") == (None,)
