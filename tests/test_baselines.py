"""The dataset-fitted models must behave as their two roles require.

Two separate contracts are under test here, and conflating them is the mistake
this module exists to prevent:

- **Baselines** (`knn_retrieval`, `linear_regression`) are competitive methods
  that may sit in a bank next to trained checkpoints. Habibi et al. found kNN
  beats deconvolutional networks for topology-optimization warm-starting at
  limited data, so these have to actually work, not merely run.
- **Reference instruments** (`collapsed`, `volume_only`, `noise_doped`,
  `checkerboard`) are calibration standards. They must be *barred* from banks,
  and each must exhibit the specific metric reading it exists to demonstrate --
  otherwise the scale bar is lying about the magnification.
- **Planted models** (`ensemble_2d`, `portfolio_2d`, `coarse_to_fine_2d`,
  `annealed_2d`) are ranked in a line-up under names that imply methods, so the
  two things that must hold are properties of *construction*: each exhibits the
  mechanism it claims, and none of them reads anything but the training split.
  A planted model that reached into `val` or `test` would win by leak, and its
  row would teach the opposite of the intended lesson.

The planted checks are split in two on purpose. The mechanism tests build a
synthetic twelve-design bank and touch no dataset, so they run anywhere and fail
for exactly one reason; whether a construction actually *tops* its target column
on real data is a measurement against the real board, not something a unit test
can assert into being.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from engiopt.baselines import ALL_DATASET_MODELS
from engiopt.baselines import BANK_ELIGIBLE
from engiopt.baselines import PLANTED_MODELS
from engiopt.baselines import REFERENCE_INSTRUMENTS
from engiopt.baselines.base import match_volume_fraction
from engiopt.baselines.base import NoCheckpointError
from engiopt.evaluation import Evaluator

PROBLEM_ID = "beams2d"
CHEAP = ["mmd", "pixel_vendi", "viol", "cond_err", "novelty"]


@pytest.fixture(scope="module")
def evaluator() -> Evaluator:
    """One evaluator shared across the module; building it reads the dataset."""
    return Evaluator.for_problem(PROBLEM_ID, spec=f"{PROBLEM_ID}/v1")


@pytest.fixture(scope="module")
def board(evaluator: Evaluator) -> dict[str, dict[str, float]]:
    """Every dataset-fitted model scored once, baselines and instruments alike."""
    rows = {}
    for name, cls in ALL_DATASET_MODELS.items():
        generator = cls.from_problem(evaluator.problem, problem_id=PROBLEM_ID, seed=1)
        rows[name] = evaluator.score(generator, only=CHEAP)
    return rows


# ----------------------------------------------------------------------
# The shared contract
# ----------------------------------------------------------------------


def test_every_model_matches_the_generator_interface(evaluator: Evaluator) -> None:
    """A dataset-fitted model must be indistinguishable from a checkpoint at the interface."""
    expected = (evaluator.resolved.n_samples, *evaluator.problem.design_space.shape)
    for name, cls in ALL_DATASET_MODELS.items():
        generator = cls.from_problem(evaluator.problem, problem_id=PROBLEM_ID, seed=1)
        designs = generator.sample(evaluator.resolved.conditions_tensor, n=evaluator.resolved.n_samples)
        assert designs.shape == expected, f"{name} produced {designs.shape}"
        assert np.isfinite(designs).all(), f"{name} produced non-finite values"


def test_loading_from_a_checkpoint_fails_loudly(evaluator: Evaluator) -> None:
    """`build` must refuse rather than silently invent weights."""
    for name, cls in ALL_DATASET_MODELS.items():
        with pytest.raises(NoCheckpointError, match=name):
            cls.build(None, evaluator.problem, None)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Baselines: they have to be genuinely competitive
# ----------------------------------------------------------------------


def test_the_catalogues_do_not_overlap() -> None:
    """A model is a bank candidate, a construction, or a calibration standard -- never two."""
    catalogues = (set(BANK_ELIGIBLE), set(PLANTED_MODELS), set(REFERENCE_INSTRUMENTS))
    for left, right in ((0, 1), (0, 2), (1, 2)):
        assert catalogues[left] & catalogues[right] == set()

    assert all(cls.bank_eligible for cls in BANK_ELIGIBLE.values())
    assert not any(cls.bank_eligible for cls in REFERENCE_INSTRUMENTS.values())
    # A planted model is ranked, but never by inheriting a permissive default:
    # `bank_eligible` stays False and the `planted` kind is the only way in.
    assert not any(cls.bank_eligible for cls in PLANTED_MODELS.values())
    assert all(cls.planted for cls in PLANTED_MODELS.values())
    assert not any(cls.planted for cls in {**BANK_ELIGIBLE, **REFERENCE_INSTRUMENTS}.values())


def test_every_planted_model_declares_what_it_was_built_to_break() -> None:
    """The disclosure is the deal: a construction with no `built_to` cannot be revealed."""
    for name, cls in PLANTED_MODELS.items():
        assert cls.built_to.strip(), f"{name} would be disclosed as a blank line"
        assert cls.summary.strip(), f"{name} has no summary, so the line-up would describe it as nothing"
        # Both halves, because a construction with only a `wins` list makes an
        # unfalsifiable claim: something has to be predicted to go wrong.
        assert cls.wins, f"{name} names no column it should top"
        assert cls.loses, f"{name} names no column it should fail, so nothing about it is falsifiable"


def test_knn_satisfies_the_volume_budget_it_was_given(board: dict[str, dict[str, float]]) -> None:
    """The retrieval baseline's headline claim: it actually respects its brief.

    Across the published beams2d pool no trained checkpoint managed a violation
    rate below 0.18. If this stops holding, the workshop's Habibi framing goes
    with it.
    """
    assert board["knn_retrieval"]["viol"] == 0.0
    assert board["knn_retrieval"]["cond_err"] < 0.005


def test_knn_beats_the_regression_on_distribution(board: dict[str, dict[str, float]]) -> None:
    """Retrieval should look more like the data than a linear map does."""
    assert board["knn_retrieval"]["mmd"] < board["linear_regression"]["mmd"]


def test_the_regression_is_the_least_diverse_thing_in_the_set(board: dict[str, dict[str, float]]) -> None:
    """No latent variable means one design per condition, and the metric should say so."""
    baselines = {n: board[n]["pixel_vendi"] for n in BANK_ELIGIBLE}
    assert min(baselines, key=baselines.get) == "linear_regression"


# ----------------------------------------------------------------------
# Reference instruments: each must read what it exists to demonstrate
# ----------------------------------------------------------------------


def test_collapsed_defines_the_floor_of_the_diversity_scale(board: dict[str, dict[str, float]]) -> None:
    """One design repeated must score exactly 1 effective sample."""
    assert board["collapsed"]["pixel_vendi"] == pytest.approx(1.0, abs=1e-6)


def test_volume_only_is_perfectly_feasible_and_nothing_else(board: dict[str, dict[str, float]]) -> None:
    """Feasibility is a floor: load-bearing nonsense can sit exactly on the budget."""
    assert board["volume_only"]["viol"] == 0.0
    assert board["volume_only"]["cond_err"] < 0.005
    # ...while looking nothing like a real design.
    assert board["volume_only"]["mmd"] > board["knn_retrieval"]["mmd"]


def test_noise_doped_shows_pixel_diversity_rewarding_damage(board: dict[str, dict[str, float]]) -> None:
    """The instrument's whole purpose: added noise must *raise* the diversity reading.

    If this ever fails on this dataset, the claim that pixel-space diversity
    rewards corruption is not supported here and the segment resting on it has
    to be rebuilt around whichever distortion family does break it.
    """
    assert board["noise_doped"]["pixel_vendi"] > board["knn_retrieval"]["pixel_vendi"]
    assert board["noise_doped"]["novelty"] > board["knn_retrieval"]["novelty"]
    # ...while every metric that knows what a design is for gets worse.
    assert board["noise_doped"]["viol"] > board["knn_retrieval"]["viol"]
    assert board["noise_doped"]["mmd"] > board["knn_retrieval"]["mmd"]


def test_checkerboard_carries_high_spatial_frequency_energy(evaluator: Evaluator) -> None:
    """A computable proxy for "looks like noise": energy at the Nyquist frequency."""
    clean = BANK_ELIGIBLE["knn_retrieval"].from_problem(evaluator.problem, problem_id=PROBLEM_ID, seed=1)
    dirty = REFERENCE_INSTRUMENTS["checkerboard"].from_problem(evaluator.problem, problem_id=PROBLEM_ID, seed=1)

    conditions = evaluator.resolved.conditions_tensor
    n = evaluator.resolved.n_samples

    def nyquist_energy(designs: np.ndarray) -> float:
        spectrum = np.abs(np.fft.fft2(np.asarray(designs)))
        return float(spectrum[:, spectrum.shape[1] // 2, spectrum.shape[2] // 2].mean())

    assert nyquist_energy(dirty.sample(conditions, n=n)) > 10 * nyquist_energy(clean.sample(conditions, n=n))


def test_a_reference_instrument_cannot_be_put_in_a_bank(evaluator: Evaluator) -> None:
    """Ranking a calibration standard against real models would be a trick, not a measurement."""
    from engiopt.workshops.idetc26.bank import _member_from_entry

    with pytest.raises(ValueError, match="reference instrument"):
        _member_from_entry({"kind": "baseline", "algo": "collapsed"}, evaluator.problem, PROBLEM_ID)


# ----------------------------------------------------------------------
# Planted models: the mechanism, on a synthetic bank
# ----------------------------------------------------------------------


def _synthetic_bank(shape: tuple[int, int] = (16, 16), n: int = 12) -> Any:
    """A twelve-design training-only bank, with no dataset behind it.

    Built by hand rather than from a fixture so these tests state their own
    inputs: the designs are binary, the volume fraction of each is its own
    condition, and there is deliberately **no `val` or `test` split** -- a
    construction that reaches for one gets a KeyError here rather than a quiet
    pass on real data, where `DesignBank.split` falls back to train.
    """
    from engiopt.baselines.base import DesignBank
    from engiopt.baselines.base import Split

    rng = np.random.default_rng(0)
    designs = np.zeros((n, *shape), dtype=np.float32)
    for i in range(n):
        # A solid block whose size sets the volume fraction, plus a scattered
        # pair of cells so designs differ by more than their density. Blocks and
        # scatter are both **even-sized and even-placed**: an odd block would
        # make the source designs themselves alternate between row parities, and
        # the checkerboard test would then pass on the fixture rather than on
        # the construction.
        designs[i, : 2 + 2 * (i % 6), :] = 1.0
        designs[i, 2 * rng.integers(0, shape[0] // 2, 2), 2 * rng.integers(0, shape[1] // 2, 2)] = 1.0

    conditions = designs.reshape(n, -1).mean(axis=1, keepdims=True).astype(np.float64)

    bank = DesignBank.__new__(DesignBank)
    bank.problem_id = "synthetic"
    bank.keys = ("volfrac",)
    bank.__dict__["_splits"] = {"train": Split(designs=designs, conditions=conditions, keys=("volfrac",))}
    bank.__dict__["_scale"] = np.array([1.0])
    return bank


def _replace_keys(split: Any, keys: tuple[str, ...]) -> Any:
    """The same split under a different condition name, for the per-problem sweep."""
    from engiopt.baselines.base import Split

    return Split(designs=split.designs, conditions=split.conditions, keys=keys)


class _Batch:
    """The two attributes `DatasetGenerator.requested` reads off a condition batch."""

    def __init__(self, values: np.ndarray) -> None:
        import torch as th

        self.tensor = th.as_tensor(np.asarray(values, dtype=np.float64))


def _planted(algo: str, bank: Any) -> Any:
    """A planted model wired to a synthetic bank, without going near a problem."""
    model = PLANTED_MODELS[algo].__new__(PLANTED_MODELS[algo])
    model.bank = bank
    model.seed = 1
    return model


def _sample_planted(algo: str, bank: Any, requested: np.ndarray) -> np.ndarray:
    """Draw from a planted model, bypassing `Generator.sample`'s device plumbing."""
    return np.asarray(_planted(algo, bank)._sample(_Batch(requested), len(requested)))


@pytest.mark.parametrize("algo", sorted(PLANTED_MODELS))
def test_a_planted_model_reads_only_the_training_split(algo: str) -> None:
    """The rule that separates a construction from a leak.

    Every checkpoint in the line-up was fitted on `train`. A construction that
    retrieved from `val` or `test` would be answering with the marking scheme,
    and its win would say nothing about the metric it was built to break.
    """
    bank = _synthetic_bank()
    asked: list[str] = []
    plain_split = bank.split

    def record(name: str) -> Any:
        asked.append(name)
        if name != "train":
            raise AssertionError(f"{algo} read the {name!r} split")
        return plain_split(name)

    bank.split = record
    _sample_planted(algo, bank, np.array([[0.2], [0.35], [0.5]]))

    assert asked, f"{algo} never read the dataset at all"
    assert set(asked) == {"train"}


def test_the_ensemble_hedges_into_densities_that_are_not_a_structure() -> None:
    """Its win is a theorem, and so is its defect: the mean of designs is not a design."""
    bank = _synthetic_bank()
    requested = np.array([[0.2], [0.35], [0.5]])

    designs = _sample_planted("ensemble_2d", bank, requested)

    intermediate = np.mean((designs > 0.05) & (designs < 0.95))
    assert intermediate > 0.3, f"only {intermediate:.0%} of the field is intermediate; this is not hedging"
    # ...and the hedging costs it nothing on the budget column, which is the trap.
    assert np.allclose(designs.reshape(len(requested), -1).mean(axis=1), requested[:, 0], atol=1e-3)


def _parity_ratio(designs: np.ndarray) -> float:
    """How much the four (row, column) parity classes disagree in mean density.

    The direct proxy for a period-two artifact, and the right one here: a single
    Nyquist FFT bin mixes the artifact with the design's *own* high-frequency
    content, so a coarsened structure can lose more real detail than it gains
    artifact and read as clean while looking obviously checkerboarded.
    """
    classes = [designs[:, row::2, col::2].mean() for row in (0, 1) for col in (0, 1)]
    return float(max(classes) / max(min(classes), 1e-12))


def test_the_upsampler_carries_a_checkerboard_the_numbers_may_not_see() -> None:
    """Uneven kernel overlap, period two in both axes, exactly as Odena et al. describe."""
    bank = _synthetic_bank()
    requested = np.array([[0.2], [0.35], [0.5]])

    retrieved = bank.split("train").designs[bank.nearest(requested, split="train")].astype(np.float64)
    upsampled = _sample_planted("coarse_to_fine_2d", bank, requested)

    assert _parity_ratio(retrieved) < 1.2, "the source designs already alternate; this test proves nothing"
    assert _parity_ratio(upsampled) > 1.5, "no visible checkerboard survived the volume match"
    # It still hits the budget: the artifact is free on every column but the eye.
    assert np.allclose(upsampled.reshape(len(requested), -1).mean(axis=1), requested[:, 0], atol=1e-3)


def test_the_upsampling_artifact_is_controlled_by_its_own_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    """At zero it must vanish, which is what makes the default a tuning choice rather than luck.

    Per-pixel normalization is Odena et al.'s fix, so `artifact = 0` has a
    known answer to check against. Not quite 1.0: dividing by the overlap
    removes the *magnitude* artifact, while an even output pixel is still an
    average of two taps and an odd one of a single tap, so the two are blurred
    differently and a little parity structure survives. What must collapse is
    the 2x-and-up modulation, which is the part a person sees.
    """
    bank = _synthetic_bank()
    requested = np.array([[0.2], [0.35], [0.5]])

    monkeypatch.setattr(PLANTED_MODELS["coarse_to_fine_2d"], "artifact", 0.0)
    assert _parity_ratio(_sample_planted("coarse_to_fine_2d", bank, requested)) < 1.25

    monkeypatch.setattr(PLANTED_MODELS["coarse_to_fine_2d"], "artifact", 1.0)
    assert _parity_ratio(_sample_planted("coarse_to_fine_2d", bank, requested)) > 2.0


def test_the_upsampler_works_on_a_grid_its_factor_does_not_divide() -> None:
    """heatconduction2d is 101x101, and an even-only construction would skip it.

    The first version silently fell back to a factor that *did* divide, which
    is how a declared `coarsen = 4` became 2 on beams2d's 50x100 grid and two
    rounds of tuning came back reporting no change. Padding and cropping keeps
    the declared factor and works on any shape.
    """
    bank = _synthetic_bank(shape=(11, 11))
    requested = np.array([[0.2], [0.35]])

    designs = _sample_planted("coarse_to_fine_2d", bank, requested)

    assert designs.shape == (2, 11, 11), f"the padding was not cropped back: {designs.shape}"
    assert _parity_ratio(designs) > 1.4, "an odd grid lost the artifact"
    assert np.allclose(designs.reshape(2, -1).mean(axis=1), requested[:, 0], atol=1e-3)


def test_the_portfolio_returns_real_designs_and_ignores_the_brief() -> None:
    """Set-level realism and per-brief correctness are different claims, and it satisfies one."""
    bank = _synthetic_bank()
    low = np.array([[0.2], [0.25], [0.3], [0.35]])
    high = np.array([[0.5], [0.55], [0.6], [0.65]])

    met = 0
    for requested in (low, high):
        achieved = _sample_planted("portfolio_2d", bank, requested).reshape(len(requested), -1).mean(axis=1)
        # Never *over* budget, which is what the feasibility column measures. It
        # may fall short when no portfolio member carries enough material to be
        # scaled up to the request -- legal, and it shows up honestly as
        # condition error, which this construction is meant to lose anyway.
        assert (achieved <= requested[:, 0] + 1e-3).all(), f"over budget: {achieved} for {requested[:, 0]}"
        met += int(np.isclose(achieved, requested[:, 0], atol=1e-3).sum())

    assert met >= len(low), "the budget is being missed everywhere, so nothing is being volume-matched at all"

    # The same structures come back whatever is asked for -- only the density
    # scaling moves, which is what makes the budget column look healthy while
    # the paired columns collapse.
    def pattern(designs: np.ndarray) -> np.ndarray:
        return np.argsort(designs.reshape(len(designs), -1), axis=1)

    assert np.array_equal(
        pattern(_sample_planted("portfolio_2d", bank, low)),
        pattern(_sample_planted("portfolio_2d", bank, high)),
    )


def test_the_annealed_sampler_keeps_the_structure_and_adds_only_entropy() -> None:
    """A design plus noise is still that design, which is why a diversity column is fooled."""
    bank = _synthetic_bank()
    requested = np.array([[0.2], [0.35], [0.5]])

    retrieved = bank.split("train").designs[bank.nearest(requested, split="train")].astype(np.float64)
    noisy = _sample_planted("annealed_2d", bank, requested)

    # Clipping into [0, 1] eats part of the noise on a near-binary design, so
    # the surviving amplitude is bounded above by the temperature rather than
    # equal to it. What must not happen is it being *scaled* away.
    temperature = PLANTED_MODELS["annealed_2d"].temperature
    assert 0.5 * temperature < (noisy - retrieved).std() <= 1.05 * temperature
    for clean, dirty in zip(retrieved, noisy):
        assert np.corrcoef(clean.ravel(), dirty.ravel())[0, 1] > 0.8


def test_the_annealed_sampler_pays_nothing_for_its_noise_on_the_budget() -> None:
    """The perturbation is offset to the budget, never rescaled to it.

    Rescaling is the obvious way to put a noisy design back on budget and it
    quietly undoes the construction: measured on beams2d it dropped
    `pixel_vendi` from 12.9 to 9.8, below plain retrieval. An additive offset
    hits the same budget while leaving the amplitude alone.
    """
    bank = _synthetic_bank()
    requested = np.array([[0.2], [0.35], [0.5]])

    designs = _sample_planted("annealed_2d", bank, requested)

    assert np.allclose(designs.reshape(len(requested), -1).mean(axis=1), requested[:, 0], atol=1e-3)
    assert designs.min() >= 0.0
    assert designs.max() <= 1.0


@pytest.mark.parametrize(
    ("shape", "condition"),
    [((50, 100), "volfrac"), ((101, 101), "volume"), ((120, 120), "lambda1")],
    ids=["beams2d", "heatconduction2d", "photonics2d"],
)
@pytest.mark.parametrize("algo", sorted(PLANTED_MODELS))
def test_every_construction_runs_on_every_problem_geometry(algo: str, shape: tuple[int, int], condition: str) -> None:
    """The three problems disagree about shape and about what a budget is called.

    heatconduction2d is 101x101 and calls its budget `volume`; photonics2d is
    120x120 and has no budget at all. Each of those broke a construction that
    was written against beams2d's 50x100 `volfrac`, and every one of the breaks
    was silent -- a skipped volume match, a factor quietly halved -- rather than
    an error anybody would have seen before the day.
    """
    bank = _synthetic_bank(shape=shape, n=14)
    bank._splits["train"] = _replace_keys(bank.split("train"), (condition,))
    bank.keys = (condition,)

    requested = np.array([[0.2], [0.3], [0.45]])
    designs = _sample_planted(algo, bank, requested)

    assert designs.shape == (3, *shape), f"{algo} produced {designs.shape} on {shape}"
    assert np.isfinite(designs).all(), f"{algo} produced non-finite values on {shape}"

    budget_column = condition in ("volfrac", "volume")
    achieved = designs.reshape(3, -1).mean(axis=1)
    if budget_column:
        assert (achieved <= requested[:, 0] + 1e-3).all(), f"{algo} went over budget: {achieved}"
    else:
        # No budget to hit, and nothing may crash reaching for one.
        assert (achieved > 0).all()


def test_a_severity_declared_in_config_reaches_the_model_and_keys_its_own_cache() -> None:
    """One class, one entry per rung: that is what lets a ladder be pure config.

    The workshop draws two rungs of a distortion family and a distortion study
    sweeps all of them. Both should be entries in a file rather than edits to a
    class, and each rung has to cache separately or the second one served the
    first one's designs -- which is exactly how two rounds of tuning here came
    back reporting "no change".
    """
    import torch as th

    from engiopt.workshops.idetc26.bank import _member_from_entry

    bank = _synthetic_bank()
    built = {
        "bank": bank,
        "problem": None,
        "problem_id": "synthetic",
        "seed": 1,
        "device": th.device("cpu"),
        "condition_keys": ("volfrac",),
    }
    mild = PLANTED_MODELS["annealed_2d"](**built, temperature=0.05)
    default = PLANTED_MODELS["annealed_2d"](**built)

    assert mild.temperature == 0.05
    assert default.temperature == PLANTED_MODELS["annealed_2d"].temperature
    assert mild.temperature != default.temperature, "the declared severity did not shadow the class default"

    keys = {
        _member_from_entry({"kind": "planted", "algo": "annealed_2d", "temperature": severity}, None, PROBLEM_ID).key
        for severity in (0.05, 0.10, 0.15)
    }
    assert len(keys) == 3, f"rungs share a cache entry: {keys}"


def test_two_rungs_of_one_ladder_do_not_share_a_published_package() -> None:
    """The digest addresses metrics on the Hub, so it must cover the knobs too.

    Every rung of a severity ladder shares one source file. A fingerprint over
    the mechanism alone would file `temperature` 0.05 and 0.15 under the same
    `cfg_<digest>/seed_1/metrics.json`, and the second physics run -- hours of
    optimizer time -- would overwrite the first with nothing reporting a
    problem. This is the check that keeps that from being possible.
    """
    from engiopt.workshops.idetc26.bank import _member_from_entry
    from engiopt.workshops.idetc26.case import _package_of

    cls = PLANTED_MODELS["annealed_2d"]
    assert cls.package_fingerprint({"temperature": 0.05}) != cls.package_fingerprint({"temperature": 0.15})

    packages = {
        _package_of(_member_from_entry({"kind": "planted", "algo": "annealed_2d", "temperature": t}, None, PROBLEM_ID).key)
        for t in (0.05, 0.10, 0.15)
    }
    assert len(packages) == 3, f"rungs collide on the Hub: {packages}"

    # ...and the mechanism still counts: same knobs, edited source, new package.
    assert cls.package_fingerprint({"temperature": 0.15}) != PLANTED_MODELS["portfolio_2d"].package_fingerprint(
        {"temperature": 0.15}
    )


def test_a_planted_model_cannot_be_smuggled_in_as_a_baseline() -> None:
    """Filing a construction as a baseline would rank it and never disclose it."""
    from engiopt.workshops.idetc26.bank import _member_from_entry

    with pytest.raises(ValueError, match="planted construction"):
        _member_from_entry({"kind": "baseline", "algo": "ensemble_2d"}, None, PROBLEM_ID)


def test_a_planted_member_carries_its_disclosure_into_the_bank() -> None:
    """`built_to` has to survive assembly, since the reveal reads it off the member."""
    from engiopt.workshops.idetc26.bank import _member_from_entry

    # `load` is a lambda, so nothing here needs a problem or a dataset.
    member = _member_from_entry({"kind": "planted", "algo": "portfolio_2d"}, None, PROBLEM_ID)

    assert member.kind == "planted"
    assert member.summary == PLANTED_MODELS["portfolio_2d"].summary
    # The disclosure has to name a column, not gesture at a moral -- checked
    # against the member's own declared wins rather than a literal, so retuning
    # a construction and rewriting its disclosure to match cannot break it.
    assert any(column in member.built_to for column in member.wins), member.built_to


# ----------------------------------------------------------------------
# The shared volume-matching helper
# ----------------------------------------------------------------------


def test_match_volume_fraction_hits_the_target_and_stays_in_range() -> None:
    """Both baselines and the diffusion clean-up experiment depend on this being exact."""
    rng = np.random.default_rng(0)
    designs = rng.random((8, 20, 20))
    targets = rng.uniform(0.15, 0.6, size=8)

    matched = match_volume_fraction(designs, targets)

    assert np.allclose(matched.reshape(8, -1).mean(axis=1), targets, atol=1e-4)
    assert matched.min() >= 0.0
    assert matched.max() <= 1.0


def test_match_volume_fraction_leaves_structure_alone() -> None:
    """It is a calibration step, not a redesign: the material ordering is preserved."""
    rng = np.random.default_rng(1)
    designs = rng.random((4, 10, 10))

    matched = match_volume_fraction(designs, np.full(4, 0.3))

    for original, adjusted in zip(designs, matched):
        unsaturated = adjusted < 1.0
        assert (np.argsort(original[unsaturated]) == np.argsort(adjusted[unsaturated])).all()
