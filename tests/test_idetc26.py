"""Verification for the IDETC'26 challenge harness.

The tests that matter here are the ones whose failure mode is "in front of a
room": the cheap board silently invoking a simulator and stalling the live
segment, a suspect unreachable by the name it is displayed under, or the config
resolving differently depending on which directory a notebook was launched from.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from engiopt.workshops.idetc26 import Case
from engiopt.workshops.idetc26 import WorkshopConfig
from engiopt.workshops.idetc26.seal import seal
from engiopt.workshops.idetc26.seal import SealError
from engiopt.workshops.idetc26.seal import unseal

PROBLEM_ID = "beams2d"


@pytest.fixture(scope="module")
def case(tmp_path_factory: pytest.TempPathFactory) -> Case:
    """One opened case, shared: opening it reads the dataset."""
    return Case.open(PROBLEM_ID, artifact_dir=tmp_path_factory.mktemp("idetc"))


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


def test_config_round_trips() -> None:
    """Every declared metric must be one the registry actually knows."""
    from engiopt.evaluation.registry import METRICS

    config = WorkshopConfig.load(PROBLEM_ID)
    assert config.problem_id == PROBLEM_ID
    for name in config.metrics:
        assert name in METRICS, f"{name} is configured but not registered"


def test_cost_is_read_from_the_registry_and_never_declared() -> None:
    """A column a team believed was cheap and which starts a simulator would stall the session.

    The config used to declare its own cheap/expensive split, which could drift
    away from the registry's. It no longer can: both properties are derived, so
    the two halves partition the declared metrics exactly.
    """
    from engiopt.evaluation.registry import METRICS

    config = WorkshopConfig.load(PROBLEM_ID)
    assert all(METRICS[name].cost == "cheap" for name in config.cheap_metrics)
    assert all(METRICS[name].cost == "expensive" for name in config.expensive_metrics)
    assert set(config.cheap_metrics) | set(config.expensive_metrics) == set(config.metrics)


def test_config_resolves_the_same_from_any_working_directory(tmp_path: Path) -> None:
    """The DCC'26 bug, tested away: config paths anchor on `__file__`, never on the cwd."""
    original = Path.cwd()
    try:
        from_repo = WorkshopConfig.load(PROBLEM_ID)
        os.chdir(tmp_path)
        from_elsewhere = WorkshopConfig.load(PROBLEM_ID)
    finally:
        os.chdir(original)
    assert from_repo == from_elsewhere


def test_unavailable_metrics_are_derived_rather_than_declared() -> None:
    """beams2d has a volume budget, so nothing is unavailable -- and that answer comes from the spec."""
    from engiopt.evaluation.spec import EvalSpec

    config = WorkshopConfig.load(PROBLEM_ID)
    assert EvalSpec.load(config.spec).volume_condition is not None
    assert config.unavailable_metrics() == {}


# ----------------------------------------------------------------------
# The bank
# ----------------------------------------------------------------------


def test_models_are_named_for_what_they_are(case: Case) -> None:
    """Every suspect is reachable by the name it is displayed under, and the names are unique."""
    labels = case.bank.labels
    declared = {str(entry.get("name") or entry["algo"]) for entry in case.config.bank}
    assert len(set(labels)) == len(labels)
    for label in labels:
        assert case.bank.resolve(label).label == label
        assert label.split("#")[0].split(".")[0] in declared


def test_no_two_suspects_differ_only_by_training_seed() -> None:
    """A line-up is a set of *different models*, and a seed is not a model.

    Two entries of one algorithm at one hyperparameter configuration differ by
    nothing a participant is being asked to judge, and having them in the
    line-up invites a ranking that separates them -- which is a ranking of the
    random seed.
    """
    for problem in ("beams2d", "heatconduction2d", "photonics2d"):
        config = WorkshopConfig.load(problem)
        identities = [(entry["algo"], entry.get("config_fingerprint")) for entry in config.bank]
        assert len(identities) == len(set(identities)), f"{problem} has two suspects differing only by seed"


def test_a_suspect_that_is_not_a_seed_variant_is_not_named_like_one() -> None:
    """`cgan_cnn_2d#1` beside `cgan_cnn_2d#42` claims a seed difference that is not there.

    The derived suffix is the seed, so two configurations of one algorithm come
    out named as if the seed were what separated them, and an entry that
    declares a `name` overrides that. The override has to win *before* the
    collision check: otherwise the other member keeps a `#seed` suffix that no
    longer separates it from anything, and the fragment `"cgan_cnn_2d"` matches
    two members and refuses to resolve.

    Asserted against the naming rule rather than against whichever line-up is
    curated today, so retiring a suspect cannot quietly retire the check.
    """
    from engiopt.workshops.idetc26.bank import _handles

    keys = ["cgan_cnn_2d#1", "cgan_cnn_2d#42", "vqgan#1"]
    assert _handles(keys, declared=["", "cgan_cnn_2d_tuned", ""]) == ["cgan_cnn_2d", "cgan_cnn_2d_tuned", "vqgan"]
    assert _handles(keys) == ["cgan_cnn_2d#1", "cgan_cnn_2d#42", "vqgan"]


def test_every_suspect_can_be_explained_at_length(case: Case) -> None:
    """A one-line summary cannot support an argument about whether a column is wrong.

    Every member needs a real description, not just the ones somebody
    remembered to write: a table row that says "a model somebody trained" is
    the state this command exists to replace.
    """
    for member in case.bank:
        assert len(member.description.strip()) > 80, f"{member.label} has no usable description"


def test_explaining_a_suspect_does_not_reveal_the_construction(case: Case) -> None:
    """`explain` runs before any board does, so it must not spoil the reveal.

    The planted models' descriptions are written to be literally true and
    non-spoiling -- a careful reader can infer, which is the intended reward --
    but `built_to` names the column each was engineered to top, and that only
    becomes readable when the physics board is unsealed.
    """
    planted = [member for member in case.bank if member.kind == "planted"]
    assert planted, "this test proves nothing without a planted member"

    for member in planted:
        assert member.built_to, f"{member.label} carries no disclosure to withhold"
        assert member.built_to not in member.description
        assert member.built_to not in member.summary
        # The give-away words, rather than the exact string: a paraphrase of
        # the disclosure would spoil it just as thoroughly.
        assert not {"planted", "constructed for", "built to"} & set(member.description.lower().split())


def test_a_fragment_reaches_one_model_and_says_so_when_it_does_not(case: Case) -> None:
    """Nobody should have to type `constrained_plvae_2d` to look at a picture."""
    assert case.bank.resolve("knn").label.startswith("knn")

    with pytest.raises(KeyError, match="No model matching"):
        case.bank.resolve("bogus")


def test_the_bank_reports_what_it_could_not_load(case: Case) -> None:
    """Skipped members are surfaced, not swallowed -- and never silently.

    Which entries skip depends on what the Hub currently holds, so this asserts
    the accounting rather than a specific list: every configured entry either
    loaded or was reported.
    """
    declared = {str(entry["algo"]) for entry in case.config.bank}
    assert set(case.bank.skipped) <= declared
    assert len(case.bank) + len(case.bank.skipped) == len(case.config.bank)


def test_unknown_labels_fail_with_the_valid_ones(case: Case) -> None:
    """A typo in a notebook must say what the options were."""
    with pytest.raises(KeyError, match="In the bank"):
        case.bank["not_a_model"]


# ----------------------------------------------------------------------
# The board
# ----------------------------------------------------------------------


def test_the_cheap_board_never_touches_the_simulator(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    """The load-bearing guarantee of the live segment.

    The session assumes every cheap column is instant. If any of them reaches
    for `simulate` or `optimize`, the room waits.
    """

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("the cheap board invoked the simulator")

    monkeypatch.setattr(case.evaluator.problem, "simulate", forbidden)
    monkeypatch.setattr(case.evaluator.problem, "optimize", forbidden)

    board = case.evaluate(["mmd", "dpp_geometric", "viol"], show_cli=False)
    assert list(board.columns) == ["mmd", "dpp_geometric", "viol"]
    assert board.notna().to_numpy().all()


def test_the_cheap_tier_never_touches_the_simulator(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every cheap question, asked together, must stay off the solver.

    The expensive tier is reachable only by naming it.
    """

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("the cheap board invoked the simulator")

    monkeypatch.setattr(case.evaluator.problem, "simulate", forbidden)
    monkeypatch.setattr(case.evaluator.problem, "optimize", forbidden)

    board = case.evaluate(list(case.config.cheap_metrics), show_cli=False)
    assert set(board.columns) <= set(case.config.cheap_metrics) | {
        column for name in case.config.cheap_metrics for column in _columns_of(name)
    }
    assert not set(board.columns) & set(case.config.expensive_metrics)


def test_the_expensive_tier_replays_what_is_already_published(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    """A gap published beside the weights must not be paid for a second time.

    `case.physics()` read the Hub and `case.evaluate("cog")` did not, so asking
    for the same number through the other door started hours of optimizer for an
    answer already sitting in the package.
    """

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a published gap was recomputed instead of read")

    monkeypatch.setattr(case.evaluator.problem, "simulate", forbidden)
    monkeypatch.setattr(case.evaluator.problem, "optimize", forbidden)

    board = case.evaluate(["mmd", "cog"], models="knn_retrieval", show_cli=False)
    assert board.loc["knn_retrieval", "cog"] == case._published_row("knn_retrieval")["cog"]
    assert np.isfinite(board.loc["knn_retrieval", "mmd"])


def test_a_truncated_request_is_not_answered_from_the_published_board(case: Case) -> None:
    """`n_samples=2` asks about two designs; the published gap is over fifty of them."""
    assert case._replayable(seed=1, n_samples=None, indices=None, sigma=None, fresh=False)
    assert not case._replayable(seed=1, n_samples=2, indices=None, sigma=None, fresh=False)
    assert not case._replayable(seed=2, n_samples=None, indices=None, sigma=None, fresh=False)
    assert not case._replayable(seed=1, n_samples=None, indices=[0, 1], sigma=None, fresh=False)
    assert not case._replayable(seed=1, n_samples=None, indices=None, sigma=None, fresh=True)


def test_the_expensive_tier_prices_itself_before_it_starts(
    case: Case, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Interrupting the cell is the only brake, so the estimate has to arrive before the wait does.

    There is deliberately no confirmation flag: it protected nothing a Ctrl-C
    does not, and it made the one family that matters look locked. What the
    contract still owes a participant is the price, printed before the first
    solver call rather than after it.
    """

    class SolverStartedError(RuntimeError):
        """Raised in place of the first solver call, to stand for the wait."""

    def stop(*_args: object, **_kwargs: object) -> None:
        raise SolverStartedError

    monkeypatch.setattr(case.evaluator.problem, "simulate", stop)
    monkeypatch.setattr(case.evaluator.problem, "optimize", stop)

    with pytest.raises(SolverStartedError):
        case.evaluate("performance", models="knn_retrieval", n_samples=1, show_cli=False)

    printed = capsys.readouterr().out
    assert "optimizer runs" in printed
    assert "Interrupt the cell" in printed


def test_the_board_disagrees_with_itself(case: Case) -> None:
    """The premise of the whole session: no single suspect tops every column."""
    board = case.evaluate(list(case.config.cheap_metrics), show_cli=False)
    ranked = case.rank(board)
    winners = {column: set(ranked.index[ranked[column] == 1]) for column in ranked.columns}
    distinct = {frozenset(names) for names in winners.values()}
    assert len(distinct) >= 3, f"the line-up does not make the point: {winners}"


def test_constant_columns_are_not_ranked(case: Case) -> None:
    """Ranking a column where every model ties would invent an ordering."""
    import pandas as pd

    frame = pd.DataFrame({"mmd": [0.5, 0.5, 0.5], "novelty": [1.0, 2.0, 3.0]}, index=["A", "B", "C"])
    ranked = case.rank(frame)
    assert "mmd" not in ranked.columns
    assert ranked["novelty"].tolist() == [3, 2, 1]


# ----------------------------------------------------------------------
# Asking a question
# ----------------------------------------------------------------------


def test_a_line_of_questioning_expands_to_its_metrics(case: Case) -> None:
    """The one thing a participant has to learn: name a family, get the family.

    Nobody arrives knowing what `pca_coverage` is. They can hold seven questions
    in their head, so `evaluate("diversity")` has to be as real a call as
    `evaluate("mmd")`.
    """
    from engiopt.workshops.idetc26.families import family_of

    board = case.evaluate("diversity", show_cli=False)
    assert len(board.columns) > 1
    assert all(family_of(column) == "diversity" for column in board.columns)


def test_metrics_and_families_mix_in_one_call(case: Case) -> None:
    """`evaluate(["cost", "mmd"])` must not need the caller to know which is which."""
    board = case.evaluate(["cost", "mmd"], show_cli=False)
    assert "mmd" in board.columns
    assert "params" in board.columns


def test_a_typo_names_both_vocabularies(case: Case) -> None:
    """The most likely error in the room, and the message has to cover both kinds of name."""
    with pytest.raises(KeyError, match="Lines of questioning"):
        case.evaluate("mdd", show_cli=False)


def test_there_is_no_blanket_board(case: Case) -> None:
    """`case.evaluate()` must not answer everything at once.

    A board of every column against every suspect is read rather than argued
    with, and choosing the three columns you would report is the exercise. The
    message has to say which call to make instead, because a bare TypeError in
    front of ninety people is a room full of raised hands.
    """
    with pytest.raises(TypeError, match="needs a question"):
        case.evaluate(show_cli=False)


def test_there_is_no_blanket_contact_sheet(case: Case) -> None:
    """`case.show()` must not draw every suspect at once, and nor must `how=`.

    Ten grids on one screen get skimmed. Every drawing call names who it is
    about, so a participant can say what they looked at.
    """
    with pytest.raises(TypeError, match="needs something to look at"):
        case.show()
    with pytest.raises(TypeError, match="at least one suspect"):
        case.show(how="compare")
    with pytest.raises(TypeError, match="needs a source"):
        case.show(how="space_map")


def test_controls_are_appended_and_never_ranked(case: Case) -> None:
    """A diversity number means nothing until you know what it reads at a known input.

    The controls are marked in the index rather than mixed in, because a scale
    bar that can be mistaken for a measurement is worse than no scale bar.
    """
    plain = case.evaluate("diversity", show_cli=False)
    with_controls = case.evaluate("diversity", controls=True, show_cli=False)

    added = [name for name in with_controls.index if name not in plain.index]
    assert added, "no controls were appended"
    assert all(name.startswith("[control] ") for name in added)


def test_evaluate_offers_no_seed_argument() -> None:
    """The spec freezes the conditions, so a seed here could only redraw noise.

    "Does the ranking survive a different noise draw on the same weights" is a
    much weaker question than "does it survive retraining", and offering the
    weak one under the name `seeds` invited it to be read as the strong one.
    Removed until the line-up pulls checkpoints at several *training* seeds.
    """
    import inspect

    assert "seeds" not in inspect.signature(Case.evaluate).parameters
    # The plumbing stays: the design cache is keyed by sampling seed.
    assert "seed" in inspect.signature(Case.designs).parameters


# ----------------------------------------------------------------------
# Sealing
# ----------------------------------------------------------------------


def test_seal_round_trips(tmp_path: Path) -> None:
    """Identity, and the published digest matches the plaintext."""
    payload = "key,iog\nregurgitator#1,1.7\n"
    destination = tmp_path / "board.csv.enc"

    digest = seal(payload, "open sesame", destination)

    assert unseal(destination, "open sesame") == payload
    assert destination.with_suffix(destination.suffix + ".sha256").read_text().strip() == digest


def test_the_wrong_passphrase_is_refused(tmp_path: Path) -> None:
    """And says so in words a facilitator can act on."""
    destination = tmp_path / "board.csv.enc"
    seal("key,iog\na,1\n", "correct", destination)

    with pytest.raises(SealError, match="does not open"):
        unseal(destination, "incorrect")


def test_a_tampered_board_is_detected(tmp_path: Path) -> None:
    """The digest is the point: it catches the answer being changed after publication."""
    destination = tmp_path / "board.csv.enc"
    seal("key,iog\na,1\n", "phrase", destination)
    seal("key,iog\na,999\n", "phrase", destination)  # rewrite the ciphertext...
    destination.with_suffix(destination.suffix + ".sha256").write_text("0" * 64 + "\n")  # ...and a stale digest

    with pytest.raises(SealError, match="moved the goalposts"):
        unseal(destination, "phrase")


def test_a_suspect_the_board_predates_comes_back_blank_not_refused(tmp_path: Path) -> None:
    """A line-up gains a member before the simulator has been run over it.

    That is a normal state -- the cheap board moves in minutes and the physics
    board takes a cluster job -- so the eight rows that are ready must not be
    taken down by the three that are not. Only a board covering nobody is an
    error, because that means the wrong file entirely.
    """
    import pandas as pd

    from engiopt.workshops.idetc26.seal import seal

    board = "key,iog,cog\nknn_retrieval#1,1.0,2.0\n"
    path = tmp_path / "board.csv.enc"
    seal(board, "pass", path)

    from engiopt.workshops.idetc26.bank import BankMember
    from engiopt.workshops.idetc26.bank import ModelBank

    def member(key: str, label: str) -> BankMember:
        return BankMember(label=label, key=key, kind="pretrained", identity=label, summary="", load=lambda: None)

    case = _bare(tmp_path)
    case.bank = ModelBank(members=[member("knn_retrieval#1", "knn_retrieval"), member("planted_2d#1", "planted_2d")])

    rows = case.physics("pass", path=path)

    assert list(rows.index) == ["knn_retrieval", "planted_2d"]
    assert rows.loc["knn_retrieval", "iog"] == 1.0
    assert pd.isna(rows.loc["planted_2d", "iog"]), "an uncovered suspect must be blank, not fabricated"


def test_a_missing_board_says_where_it_looked(tmp_path: Path) -> None:
    """A missing seal on the day must not surface as a stack trace."""
    with pytest.raises(SealError, match="No sealed board"):
        unseal(tmp_path / "absent.csv.enc", "phrase")


# ----------------------------------------------------------------------
# The design cache
# ----------------------------------------------------------------------


def test_cached_designs_round_trip(tmp_path: Path) -> None:
    """What comes back must be exactly what was scored, or a board is not reproducible."""
    import numpy as np

    from engiopt.workshops.idetc26.designs import DesignStore

    store = DesignStore(PROBLEM_ID, spec_version="v2", condition_digest="abc", roots=(tmp_path,))
    designs = np.random.default_rng(0).random((8, 50, 100)).astype(np.float32)
    store.store("cgan_cnn_2d#3", 1, designs, sample_seconds=12.5, model_params=17)

    entry = store.load("cgan_cnn_2d#3", 1)
    assert entry is not None
    assert np.array_equal(entry.designs, designs)
    assert entry.sample_seconds == 12.5
    assert "measured on" in entry.replayed_cost_note


def test_designs_drawn_against_other_conditions_are_refused(tmp_path: Path) -> None:
    """A cache built under a different spec draw would put two question sets in one column."""
    import numpy as np

    from engiopt.workshops.idetc26.designs import DesignStore

    DesignStore(PROBLEM_ID, spec_version="v2", condition_digest="abc", roots=(tmp_path,)).store(
        "knn_retrieval#1", 1, np.zeros((4, 50, 100), dtype=np.float32)
    )
    other = DesignStore(PROBLEM_ID, spec_version="v2", condition_digest="def", roots=(tmp_path,))

    assert other.load("knn_retrieval#1", 1) is None


def test_an_unmeasured_sampling_cost_stays_unmeasured(tmp_path: Path) -> None:
    """`gen_seconds` must read NaN rather than zero when nobody timed it.

    "Free" and "not measured" are different claims, and a cost column that
    silently reports the first for the second is exactly the undeclared
    provenance the session is about.
    """
    import numpy as np

    from engiopt.workshops.idetc26.designs import DesignStore

    store = DesignStore(PROBLEM_ID, spec_version="v2", roots=(tmp_path,))
    store.store("knn_retrieval#1", 1, np.zeros((4, 50, 100), dtype=np.float32))

    entry = store.load("knn_retrieval#1", 1)
    assert entry is not None
    assert entry.sample_seconds is None
    assert entry.replayed_cost_note == "not measured"


# ----------------------------------------------------------------------
# The metric catalogue and its lock
# ----------------------------------------------------------------------


def _columns_of(metric: str) -> tuple[str, ...]:
    """Every board column one metric emits. `lv_residual` emits two."""
    from engiopt.evaluation.registry import METRICS

    return METRICS[metric].columns


def _bare(tmp_path: Path) -> Case:
    """A case with a real config and nothing loaded: enough to ask what it offers."""
    config = WorkshopConfig.load(PROBLEM_ID)
    return Case(
        config=config,
        evaluator=None,  # type: ignore[arg-type]
        bank=None,  # type: ignore[arg-type]
        controls=None,  # type: ignore[arg-type]
        artifact_dir=tmp_path,
        store=None,  # type: ignore[arg-type]
    )


def test_the_catalogue_groups_metrics_by_the_question_they_ask(tmp_path: Path) -> None:
    """`lv_mmd` asks what `mmd` asks. Filing them apart is what stops anyone noticing they disagree."""
    catalogue = _bare(tmp_path).metrics()

    groups = catalogue["line of questioning"].to_dict()
    assert groups["mmd"] == groups["pca_mmd"] == groups["lv_mmd"] == "similarity"
    assert groups["pixel_vendi"] == groups["lv_vendi"] == "diversity"
    assert groups["cond_err"] == groups["viol"] == "obedience"
    assert groups["lv_paired_distance"] == groups["mmd"] == "similarity"
    assert groups["params"] == groups["train_minutes"] == groups["gen_seconds"] == "cost"


def test_memorization_is_split_out_of_similarity(tmp_path: Path) -> None:
    """The registry files `novelty_ratio` under `distribution`, beside the columns it refutes.

    `mmd` is *minimized* by handing back the training set, and the memorization
    columns are what catch you doing it. Filing them together is precisely what
    lets a board look coherent while containing its own contradiction, so the
    catalogue splits them whatever the registry says.
    """
    from engiopt.evaluation.registry import METRICS

    groups = _bare(tmp_path).metrics()["line of questioning"].to_dict()
    assert METRICS["novelty_ratio"].family == METRICS["mmd"].family == "distribution"
    assert groups["novelty_ratio"] == groups["lv_novelty"] == "memorization"
    assert groups["mmd"] == "similarity"


def test_the_catalogue_says_which_space_each_metric_measures_in(tmp_path: Path) -> None:
    """The space is a modelling choice, and it is the one no results table declares."""
    catalogue = _bare(tmp_path).metrics()

    spaces = catalogue["space"].to_dict()
    assert spaces["mmd"] == "pixels"
    assert spaces["pca_mmd"] == "PCA subspace"
    assert spaces["lv_mmd"] == "learned latent"
    # A parameter count is not measured in a space, and saying "pixels" would be noise.
    assert spaces["params"] == "--"


def test_the_catalogue_reports_what_cannot_be_computed(tmp_path: Path) -> None:
    """A column that needs the simulator must say so before somebody waits on it."""
    statuses = _bare(tmp_path).metrics()["status"].to_dict()

    assert statuses["mmd"] == "available"
    assert "simulator" in statuses["iog"]


def test_a_single_line_of_questioning_can_be_asked_for(tmp_path: Path) -> None:
    """Twenty-odd columns at once is a wall; one question's worth is readable."""
    bare = _bare(tmp_path)

    diversity = bare.metrics("diversity")
    assert set(diversity["line of questioning"]) == {"diversity"}
    # The bounded n-th-root form, not the raw determinant: that one underflows
    # to indistinguishable zeros and is deliberately not offered here.
    assert "dpp_geometric" in diversity.index
    assert "dpp" not in diversity.index

    with pytest.raises(KeyError, match="No such line of questioning"):
        bare.metrics("not-a-group")


def test_every_configured_metric_lands_in_a_real_family() -> None:
    """A column filed nowhere would vanish from the catalogue without anybody noticing."""
    from engiopt.workshops.idetc26.families import FAMILIES
    from engiopt.workshops.idetc26.families import family_of

    for problem in ("beams2d", "heatconduction2d", "photonics2d"):
        for name in WorkshopConfig.load(problem).metrics:
            assert family_of(name) in FAMILIES, f"{name} has no line of questioning"


def test_model_names_are_short_where_they_can_be_and_unique_where_they_cannot() -> None:
    """A bank with one VQGAN calls it `vqgan`; a bank with two must not call them both that."""
    from engiopt.workshops.idetc26.bank import _handles

    assert _handles(["knn_retrieval#1", "vqgan#1", "vqgan#2"]) == ["knn_retrieval", "vqgan#1", "vqgan#2"]
    # Same algorithm and seed, two hyperparameter configurations: the key cannot
    # tell them apart, so the name has to.
    assert _handles(["cgan#1", "cgan#1"]) == ["cgan#1", "cgan#1.2"]


def test_a_declared_name_resolves_the_collision_for_both_members() -> None:
    """Renaming one of a pair must un-suffix the other, or the fragment breaks.

    Deriving suffixes before applying declared names leaves `cgan#1` sitting
    beside `cgan_tuned` -- a `#1` that separates it from nothing. Members are
    reachable by fragment, so `"cgan"` would then match both and refuse to
    resolve, which is exactly the call the notebook makes.
    """
    from engiopt.workshops.idetc26.bank import _handles

    assert _handles(["cgan#1", "cgan#42"], declared=["", "cgan_tuned"]) == ["cgan", "cgan_tuned"]
    # Two genuine seed variants still get told apart.
    assert _handles(["cgan#1", "cgan#2"], declared=["", ""]) == ["cgan#1", "cgan#2"]
    # When a declared name collides with one another member would have derived,
    # the declared one is honoured exactly and the derived one yields. A name
    # somebody wrote down is a stronger claim than one this function invented.
    assert _handles(["cgan#1", "gan#1"], declared=["", "cgan"]) == ["cgan#1", "cgan"]


# ----------------------------------------------------------------------
# The retrieval baseline's k = 1 path
# ----------------------------------------------------------------------


def test_pure_retrieval_returns_whole_designs(monkeypatch: pytest.MonkeyPatch) -> None:
    """k=1 must return the retrieved designs, not their row means.

    `DesignBank.nearest` documents that it drops the neighbour axis at k=1, so
    averaging over `axis=1` regardless collapses each design along its own rows
    and yields `(n, width)` -- which is not a design, and only fails later at a
    reshape far from the cause.
    """
    import numpy as np

    from engiopt.baselines.honest import KNNRetrieval

    designs = np.arange(4 * 6 * 5, dtype=np.float64).reshape(4, 6, 5)
    requested = np.array([[0.3], [0.4], [0.5]])

    class _Split:
        pass

    split = _Split()
    split.designs = designs  # type: ignore[attr-defined]

    model = KNNRetrieval.__new__(KNNRetrieval)
    model.k = 1
    model.bank = _Split()  # type: ignore[assignment]
    model.bank.nearest = lambda *_args, **_kwargs: np.array([2, 0, 3])  # type: ignore[attr-defined]
    model.bank.split = lambda _name: split  # type: ignore[attr-defined]
    model.bank.column = lambda _name: None  # type: ignore[attr-defined]
    monkeypatch.setattr(KNNRetrieval, "requested", lambda *_a, **_k: requested)

    out = model._sample(None, len(requested))  # type: ignore[arg-type]

    assert out.shape == (3, 6, 5), f"k=1 returned {out.shape}, not whole designs"
    # Pure retrieval: each row is exactly the training design it selected.
    assert np.array_equal(out, designs[[2, 0, 3]])


def test_a_problem_without_a_volume_budget_loses_only_the_volume_columns() -> None:
    """photonics2d has no volume budget -- but a paired distance never needed one.

    The exclusion used to be by metric *family*, which took
    `pixel_paired_distance` and `lv_paired_distance` down with it: both compare a
    design to the optimum for its own condition and are perfectly computable
    without a budget to hit.
    """
    config = WorkshopConfig.load("photonics2d")

    unavailable = config.unavailable_metrics()
    assert set(unavailable) == {"viol", "cond_err"}
    assert "pixel_paired_distance" in config.available(config.cheap_metrics)


def test_every_problem_config_declares_registered_metrics_and_a_full_bank() -> None:
    """Each problem the workshop offers has to be loadable and internally consistent."""
    from engiopt.evaluation.registry import METRICS

    for problem in ("beams2d", "heatconduction2d", "photonics2d"):
        config = WorkshopConfig.load(problem)
        assert all(name in METRICS for name in config.metrics), problem
        assert len(config.metrics) == len(set(config.metrics)), f"{problem} declares a column twice"
        assert len(config.bank) >= 8, problem
        assert config.controls, f"{problem} has no known-answer models to calibrate against"


def test_the_pinned_autoencoder_is_never_also_a_bank_member() -> None:
    """Measuring a model inside its own latent space is not a measurement."""
    from engiopt.evaluation.spec import EvalSpec

    for problem in ("beams2d", "photonics2d"):
        config = WorkshopConfig.load(problem)
        pinned = EvalSpec.load(config.spec).latent_instrument
        assert pinned is not None
        clashes = [
            entry
            for entry in config.bank
            if entry["algo"] == pinned.algo and entry.get("config_fingerprint") == pinned.config_fingerprint
        ]
        assert not clashes, f"{problem} ranks its own latent instrument: {clashes}"
