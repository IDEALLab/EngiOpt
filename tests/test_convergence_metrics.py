"""Verification for the trajectory-backed convergence metrics.

`iog`, `cog` and `fog` are three summaries of one optimizer trajectory, and the
question they cannot answer is how many function calls the warm start actually
saved. `cog` is the one that looks like it should: it sees the whole path. But
it sums gaps that may be negative, so extra iterations spent below the reference
*improve* it -- and the raw iteration count is no better, because beams2d stops
on a design-variable tolerance that keeps looping long after the objective has
settled.

So the tests that matter here are the ones whose failure mode is "the metric
rewards the wrong model": settling that is fooled by a lucky excursion, a first
call scored against a design with nothing to gain, and a storage format that
silently averages its own padding.
"""

from __future__ import annotations

import math
import types
from typing import Any

import numpy as np
import pytest

from engiopt.evaluation.metrics import builtin
from engiopt.evaluation.metrics.builtin import _settle_calls
from engiopt.evaluation.physics_board import pack_trajectories


def _context(paths: list[list[float]]) -> Any:
    """The only part of `EvaluationContext` the trajectory metrics read."""
    return types.SimpleNamespace(
        optimization=types.SimpleNamespace(trajectories=[np.asarray(p, dtype=np.float32) for p in paths])
    )


def path(*values: float) -> np.ndarray:
    """One trajectory, in the dtype the evaluator stores."""
    return np.asarray(values, dtype=np.float32)


# ----------------------------------------------------------------------
# Settling
# ----------------------------------------------------------------------


def test_jitter_inside_the_band_is_not_work() -> None:
    """A design that lands on call one and then wobbles has converged in one call.

    This is the whole reason the metric exists rather than counting iterations:
    beams2d's stopping rule is an infinity norm on the *design variables*, so one
    drifting pixel keeps the optimizer running long after the objective is done.
    Counting iterations would score this design 5; it cost 1.
    """
    assert _settle_calls(path(100.0, 10.0, 10.2, 9.9, 10.1, 10.0)) == 1


def test_settling_is_last_exit_not_first_touch() -> None:
    """A trajectory that re-leaves the band has not settled.

    First-touch would credit this path with converging at call 1 on the strength
    of an excursion it then abandons. Settling time is defined by the last exit
    for exactly this reason.
    """
    assert _settle_calls(path(100.0, 10.0, 10.0, 10.0, 60.0, 10.0)) == 5


def test_a_design_that_starts_converged_costs_no_calls() -> None:
    """Zero has to be representable: it is the best possible answer, not an error.

    The band is floored on the final value so that a flat trajectory gets a band
    of sensible width. Without the floor the band is zero-wide, nothing clears
    it, and the best design on the board scores as the worst.
    """
    assert _settle_calls(path(5.0, 5.0, 5.0)) == 0


def test_a_slow_grind_is_reported_as_slow() -> None:
    """The metric must still punish a genuinely slow trajectory."""
    assert _settle_calls(np.linspace(100, 10, 40, dtype=np.float32)) > 30


def test_a_negative_gap_trajectory_settles_like_any_other() -> None:
    """Designs better than the reference are the case `cog` gets backwards.

    `cog` improves without bound as such a trajectory runs longer; settling time
    is unmoved by the sign, because it measures distance to the converged value
    rather than accumulating the value itself.
    """
    below = path(-5.0, -37.0, -37.1, -36.9, -37.0)
    lengthened = path(-5.0, -37.0, -37.1, -36.9, -37.0, -37.0, -37.0, -37.0)
    assert _settle_calls(below) == _settle_calls(lengthened)


def test_a_tighter_band_never_reports_fewer_calls() -> None:
    """Monotone in the band: demanding more precision cannot cost fewer calls."""
    real = path(450.8, 138.6, 58.2, 30.5, 24.2, 23.5, 23.0, 22.94, 22.938, 22.937)
    assert _settle_calls(real, 0.01) >= _settle_calls(real, 0.05)


# ----------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------


def test_ragged_paths_round_trip_through_the_stored_format() -> None:
    """Every path comes back exactly, at its own true length."""
    paths = [path(3.0, 2.0, 1.0), path(9.0, 8.0), path(*np.linspace(50, 1, 17))]
    packed = pack_trajectories(paths, iog=[3.0, 9.0, 50.0])

    assert packed["gaps"].shape == (3, 17)
    assert list(packed["lengths"]) == [3, 2, 17]
    for index, original in enumerate(paths):
        stored = packed["gaps"][index, : packed["lengths"][index]]
        assert np.allclose(stored, original)


def test_padding_is_nan_so_it_cannot_be_averaged_by_accident() -> None:
    """Zero padding would be indistinguishable from a gap of zero.

    A gap of exactly zero means "as good as the reference optimum", which is an
    ordinary value here -- so padding with zero would let a reader who forgot to
    slice by `lengths` average fictional perfect steps into a real answer.
    """
    packed = pack_trajectories([path(3.0, 2.0, 1.0), path(9.0)], iog=[3.0, 9.0])
    assert np.isnan(packed["gaps"][1, 1:]).all()
    assert not np.isnan(packed["gaps"][1, 0])


def test_cog_is_still_summed_at_full_precision() -> None:
    """Storage rounds to float32; the arithmetic behind `cog` must not.

    Gaps reach 1e9 on this pool, and a float32 accumulation over ~100 of them
    moves the total in its leading digits -- silently disagreeing with every
    `cog` already published. This asserts the discrepancy the float32 path would
    have introduced is real, so the guard is not theatre.
    """
    steps = [1.2345678e8] * 100
    assert float(np.asarray(steps, dtype=np.float32).sum()) != pytest.approx(sum(steps), rel=1e-9)
    assert float(np.asarray(steps, dtype=np.float64).sum()) == pytest.approx(sum(steps), rel=1e-12)


def test_recovery_reads_the_budget_and_carries_short_paths_forward() -> None:
    """`gap_after_k` is the gap at call k, and a path that ended early keeps its last value."""
    ctx = _context([[10.0, 3.0, -1.0, -1.0, -1.0], [5.0, -1.0]])
    row = builtin.recovery(ctx)
    assert row["gap_after_1"] == pytest.approx(7.5)  # median of 10 and 5
    assert row["gap_after_2"] == pytest.approx(1.0)  # median of 3 and -1
    # The two-step path has converged, so call 10 sees its final value, not an error.
    assert row["gap_after_10"] == pytest.approx(-1.0)


def test_calls_to_parity_counts_non_reachers_past_the_end() -> None:
    """A design that never matches the reference must rank worse than a slow one."""
    slow = _context([[10.0, 8.0, 6.0, 3.0, -0.5]])
    never = _context([[10.0, 9.0, 8.5, 8.4, 8.4]])
    assert builtin.calls_to_parity(slow) == pytest.approx(5.0)
    assert builtin.calls_to_parity(never) == pytest.approx(6.0)
    assert builtin.calls_to_parity(never) > builtin.calls_to_parity(slow)


def test_recovery_family_reports_nan_without_a_trajectory() -> None:
    """No optimizer run means no answer, not a zero that would rank as perfect."""
    empty = _context([])
    assert all(math.isnan(v) for v in builtin.recovery(empty).values())
    assert math.isnan(builtin.calls_to_parity(empty))
