"""Reference instruments: models built to make a metric's reading interpretable.

A metric value on its own says nothing. Is `pixel_vendi = 30` good? You cannot
answer that without knowing what the scale does at its endpoints -- and the only
way to know is to feed it inputs whose character you already know.

These models are those inputs. Each is a *calibration standard*, not a
contestant:

    collapsed        one design repeated. Whatever a diversity metric reports
                     here is that metric's floor.
    volume_only      hits the volume budget exactly with load-bearing nonsense.
                     Whatever feasibility reports here is what feasibility is
                     worth on its own.
    noise_doped      real optimal designs plus Gaussian noise at known severity.
                     A diversity metric that *rises* here rewards damage.
    checkerboard     real designs plus the classic alternating artifact. Reads
                     as broken to a person; an unfiltered solver disagrees.

`bank_eligible` is False on every one of them. They belong on a reference row
underneath the leaderboard, the way a scale bar belongs on a micrograph -- never
in the ranking, always in the figure. Putting them in the bank would be a trick;
putting them beside it is the measurement.
"""

from __future__ import annotations

from typing import Any, ClassVar, TYPE_CHECKING

import numpy as np

from engiopt.baselines.base import DatasetGenerator

if TYPE_CHECKING:
    import numpy.typing as npt

    from engiopt.core import ConditionBatch


def _smooth(field: npt.NDArray[Any], passes: int = 2) -> npt.NDArray[Any]:
    """Cheap separable box blur, so a random field reads as blobs rather than static."""
    out = field.astype(np.float64)
    for _ in range(passes):
        out = (out + np.roll(out, 1, axis=-1) + np.roll(out, -1, axis=-1)) / 3.0
        out = (out + np.roll(out, 1, axis=-2) + np.roll(out, -1, axis=-2)) / 3.0
    return out


class Collapsed(DatasetGenerator):
    """One design, repeated. The floor of every diversity metric."""

    algo_id = "collapsed"
    conditional = False
    bank_eligible = False
    summary = "A single design returned for every request. Defines the zero point of the diversity scale."
    loses = ("pixel_vendi", "cond_err", "viol")

    def _sample(self, conditions: ConditionBatch, n: int) -> npt.NDArray[Any]:  # noqa: ARG002
        """Return the median-condition validation design, `n` times.

        `conditions` is accepted and ignored, which is the entire specification.
        """
        split = self.bank.split("val")
        centre = np.median(split.conditions, axis=0)
        pick = int(np.linalg.norm(split.conditions - centre, axis=1).argmin())
        return np.repeat(split.designs[pick][None, ...], n, axis=0)


class VolumeOnly(DatasetGenerator):
    """Hits the volume budget exactly with a structure that carries no load."""

    algo_id = "volume_only"
    conditional = True
    bank_eligible = False
    summary = "Blobs placed to hit the volume budget exactly. Shows what perfect feasibility is worth alone."
    wins = ("viol", "cond_err")
    loses = ("iog", "cog", "fog")

    volume_condition: ClassVar[str] = "volfrac"

    def _sample(self, conditions: ConditionBatch, n: int) -> npt.NDArray[Any]:
        """Threshold a smooth random field at the quantile realizing the requested fraction.

        Raises:
            ValueError: If the problem has no volume-fraction condition.
        """
        column = self.bank.column(self.volume_condition)
        if column is None:
            raise ValueError(f"{self.algo_id} needs a {self.volume_condition!r} condition; {self.problem_id} has none.")

        requested = self.requested(conditions, n)[:, column]
        shape = self.bank.split("train").designs.shape[1:]
        rng = self.rng()

        designs = np.empty((n, *shape), dtype=np.float32)
        for i, target in enumerate(requested):
            field = _smooth(rng.normal(size=shape), passes=3)
            keep = max(round(float(target) * field.size), 1)
            cutoff = np.partition(field.ravel(), -keep)[-keep]
            designs[i] = (field >= cutoff).astype(np.float32)
        return designs


class NoiseDoped(DatasetGenerator):
    """Real optimal designs plus Gaussian noise at a known severity.

    The rung of a distortion ladder that matters most: a diversity metric which
    *increases* here is rewarding damage, and that is a property of the metric
    which no amount of arguing about models can settle.
    """

    algo_id = "noise_doped"
    conditional = True
    bank_eligible = False
    tuning = ("severity",)
    summary = "Real optimal designs with Gaussian noise added. Tests whether a diversity metric rewards corruption."
    wins = ("dpp", "pixel_vendi", "novelty")
    loses = ("mmd", "viol", "iog")

    severity: ClassVar[float] = 0.25

    def _sample(self, conditions: ConditionBatch, n: int) -> npt.NDArray[Any]:
        """Return nearest validation designs, corrupted with additive noise."""
        idx = self.bank.nearest(self.requested(conditions, n), split="val")
        clean = self.bank.split("val").designs[idx].astype(np.float64)
        return clean + self.rng().normal(scale=self.severity, size=clean.shape)


class Checkerboard(DatasetGenerator):
    """Real designs plus the classic topology-optimization checkerboard artifact."""

    algo_id = "checkerboard"
    conditional = True
    bank_eligible = False
    tuning = ("amplitude",)
    summary = "Real designs with an alternating solid/void artifact. Reads as broken; scores well."
    wins = ("mmd", "cond_err")
    loses = ("dpp",)

    amplitude: ClassVar[float] = 0.35

    def _sample(self, conditions: ConditionBatch, n: int) -> npt.NDArray[Any]:
        """Return nearest validation designs with a checkerboard perturbation applied."""
        idx = self.bank.nearest(self.requested(conditions, n), split="val")
        clean = self.bank.split("val").designs[idx].astype(np.float64)

        grids = np.indices(clean.shape[1:]).sum(axis=0)
        pattern = np.where(grids % 2 == 0, 1.0, -1.0)
        # Only perturb material that is already there: an artifact in the void
        # would be a different failure, and the eye should see structure, not snow.
        return clean + self.amplitude * pattern * clean


REFERENCE_INSTRUMENTS: dict[str, type[DatasetGenerator]] = {
    cls.algo_id: cls for cls in (Collapsed, VolumeOnly, NoiseDoped, Checkerboard)
}
"""Calibration standards. Shown beside a leaderboard, never ranked inside one."""
