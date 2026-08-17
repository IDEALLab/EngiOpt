"""Baselines a reviewer would accept: no neural network, no checkpoint, real results.

These are not straw men. Habibi et al., *When Is it Actually Worth Learning
Inverse Design?* (J. Mech. Des. 148(6):061704, 2026; first presented at
IDETC-CIE 2023) compared inverse-design model families on a topology-optimization
warm-start task and found that **k-nearest neighbours and random forests
outperform deconvolutional networks when training data is limited**, once the
cost of generating that data is counted. On this problem, with this dataset, kNN
is a competitive method rather than a cautionary tale.

That is why they belong in the bank and not in an appendix. A leaderboard whose
top entry is a nearest-neighbour lookup is not a broken leaderboard -- it is a
result, and it is the result that paper reports. The interesting question is not
"how do we exclude these" but "which column, if any, tells you when the network
was worth training".

Both fit at load time from the dataset split the evaluation spec already pins,
so they are exactly as reproducible as a checkpoint and rather more portable.
"""

from __future__ import annotations

from typing import Any, ClassVar, TYPE_CHECKING

import numpy as np

from engiopt.baselines.base import DatasetGenerator
from engiopt.baselines.base import match_volume_fraction

if TYPE_CHECKING:
    import numpy.typing as npt

    from engiopt.core import ConditionBatch


class KNNRetrieval(DatasetGenerator):
    """Nearest-neighbour retrieval over the training set, rescaled to the budget.

    The method Habibi et al. found hard to beat at small data sizes. For each
    requested condition it takes the nearest training design and rescales it to
    the requested volume fraction -- the same post-hoc feasibility step any
    practitioner would apply.

    **`k = 1` is the default, and the only value worth defaulting to.** At k = 1
    this is pure retrieval: it cannot produce a design that is not already in the
    dataset, which is a real limitation and one that only the memorization
    columns measure. Any k > 1 averages k designs and returns a blurred blend
    that exists nowhere in the data -- so it is neither retrieval nor a
    generative model, and a board containing it is measuring an artifact.
    Larger k remains available via `neighbours=`, but nothing defaults to it.
    """

    algo_id = "knn_retrieval"
    conditional = True
    bank_eligible = True
    summary = "k-nearest-neighbour retrieval from the training set, rescaled to the requested volume fraction."
    reference = "Habibi et al., J. Mech. Des. 148(6):061704 (2026)"

    neighbours: ClassVar[int] = 1
    volume_condition: ClassVar[str] = "volfrac"

    def __init__(self, neighbours: int | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.k = neighbours or self.neighbours

    def parameter_count(self) -> int:
        """Every number the method must keep: the training designs themselves.

        A retrieval model has no weights, and reporting nothing would let it sit
        on a cost board for free. Its training set *is* its parameters -- it
        cannot answer a query without them -- so that is what it declares, and
        a kNN's data cost lands on the same axis as a network's weight cost.
        """
        split = self.bank.split("train")
        return int(np.prod(split.designs.shape))

    def _sample(self, conditions: ConditionBatch, n: int) -> npt.NDArray[Any]:
        """Average the `k` nearest training designs, then match the volume budget."""
        requested = self.requested(conditions, n)
        indices = self.bank.nearest(requested, split="train", k=self.k)
        picked = self.bank.split("train").designs[indices]
        # `nearest` drops the neighbour axis at k = 1, so there is nothing to
        # average over: the retrieved design *is* the answer. Averaging anyway
        # collapses each design along its own rows, which is not a design.
        designs = picked if self.k == 1 else picked.mean(axis=1)

        column = self.bank.column(self.volume_condition)
        if column is None:
            return designs
        return match_volume_fraction(designs, requested[:, column])


class LinearRegression(DatasetGenerator):
    """Ridge regression from conditions to pixels: the cheapest thing that could work.

    A quadratic feature expansion of the conditions, fitted in closed form on the
    training split. It has no latent variable, so it produces exactly one design
    per condition and cannot express a design portfolio at all -- which is the
    honest trade the diversity columns are there to price.

    Reported not because anyone would ship it, but because a generative model
    that cannot beat it has not earned its training budget. It is the baseline
    that makes the rest of the leaderboard interpretable.
    """

    algo_id = "linear_regression"
    conditional = True
    bank_eligible = True
    summary = "Ridge regression from conditions to pixels, fitted in closed form. No latent variable."

    fit_samples: ClassVar[int] = 2000
    ridge: ClassVar[float] = 1e-2

    def parameter_count(self) -> int:
        """Size of the fitted weight matrix: one row per feature, one column per pixel.

        Declared rather than measured because the weights are solved for inside
        `_sample` and never stored, so there is nothing for a torch walk to find.
        The shape is fixed by the problem, so the count is exact.
        """
        split = self.bank.split("train")
        n_features = 1 + 2 * split.conditions.shape[1]
        return int(n_features * np.prod(split.designs.shape[1:]))

    def _features(self, conditions: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Bias, the conditions, and their squares."""
        return np.hstack([np.ones((len(conditions), 1)), conditions, conditions**2])

    def _sample(self, conditions: ConditionBatch, n: int) -> npt.NDArray[Any]:
        """Fit on a training subsample, then predict the conditional mean design."""
        split = self.bank.split("train")
        rng = self.rng()
        take = rng.choice(len(split), size=min(self.fit_samples, len(split)), replace=False)

        features = self._features(split.conditions[take])
        targets = split.designs[take].reshape(len(take), -1).astype(np.float64)

        gram = features.T @ features + self.ridge * np.eye(features.shape[1])
        weights = np.linalg.solve(gram, features.T @ targets)

        predicted = self._features(self.requested(conditions, n)) @ weights
        return predicted.reshape(n, *split.designs.shape[1:]).astype(np.float32)


HONEST_BASELINES: dict[str, type[DatasetGenerator]] = {cls.algo_id: cls for cls in (KNNRetrieval, LinearRegression)}
"""Baselines eligible to sit in the workshop bank alongside trained checkpoints."""
