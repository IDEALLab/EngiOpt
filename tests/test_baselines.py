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
"""

from __future__ import annotations

import numpy as np
import pytest

from engiopt.baselines import ALL_DATASET_MODELS
from engiopt.baselines import BANK_ELIGIBLE
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
    """A model is a bank candidate or a calibration standard, never both."""
    assert set(BANK_ELIGIBLE) & set(REFERENCE_INSTRUMENTS) == set()
    assert all(cls.bank_eligible for cls in BANK_ELIGIBLE.values())
    assert not any(cls.bank_eligible for cls in REFERENCE_INSTRUMENTS.values())


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
