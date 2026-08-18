"""Models fitted from the dataset rather than trained: baselines and reference instruments.

Two groups, and the distinction is the whole point of the module.

**Bank-eligible baselines** (`honest.py`) are legitimate methods a reviewer would
accept, and on this task they are competitive. Habibi et al., *When Is it
Actually Worth Learning Inverse Design?* (J. Mech. Des. 148(6):061704, 2026;
IDETC-CIE 2023) found k-nearest neighbours beats deconvolutional networks for
topology-optimization warm-starting at limited data sizes once data-generation
cost is counted. They belong in a leaderboard, not beneath one.

These two also exist as full generator packages under
`engiopt/generators/{knn_retrieval,linear_regression}/`, with training scripts
that publish checkpoints to HuggingFace like any other model. The versions here
fit in-process, which is what a notebook wants; the packaged versions are what
the pool and the leaderboard use. Same method, two entry points.

**Reference instruments** (`references.py`) are calibration standards: a
collapsed model, a feasible-but-useless one, a noise-doped one, a checkerboard.
They exist to tell you what a metric reads at a known input, the way a scale bar
tells you what a micrograph's magnification is. Every one of them carries
`bank_eligible = False`, and putting one in a bank would be a trick rather than a
measurement.

**Planted models** (`planted.py`) are the third thing, and the distinction from
the second is the one worth holding: a reference instrument is *labelled* and
never ranked, while a planted model is ranked, carries a name that implies a
method, and is disclosed at the reveal. They exist because a line-up assembled
from real checkpoints alone may have a consistent winner, and an afternoon spent
arguing about metrics that all agree teaches nothing. Each is built from the
training split only -- never `val` or `test`, which would make its win a leak
rather than a measurement -- and declares its real cost.
"""

from engiopt.baselines.base import DatasetGenerator
from engiopt.baselines.base import DesignBank
from engiopt.baselines.base import match_volume_fraction
from engiopt.baselines.base import NoCheckpointError
from engiopt.baselines.honest import HONEST_BASELINES
from engiopt.baselines.honest import KNNRetrieval
from engiopt.baselines.honest import LinearRegression
from engiopt.baselines.planted import Annealed2D
from engiopt.baselines.planted import CoarseToFine2D
from engiopt.baselines.planted import Ensemble2D
from engiopt.baselines.planted import PLANTED_MODELS
from engiopt.baselines.planted import PlantedModel
from engiopt.baselines.planted import Portfolio2D
from engiopt.baselines.references import Checkerboard
from engiopt.baselines.references import Collapsed
from engiopt.baselines.references import NoiseDoped
from engiopt.baselines.references import REFERENCE_INSTRUMENTS
from engiopt.baselines.references import VolumeOnly

BANK_ELIGIBLE = dict(HONEST_BASELINES)
"""Dataset-fitted models allowed to compete in a workshop bank."""

ALL_DATASET_MODELS = {**HONEST_BASELINES, **PLANTED_MODELS, **REFERENCE_INSTRUMENTS}
"""Every model in this package, eligible or not. For tests and diagnostics."""

__all__ = [
    "ALL_DATASET_MODELS",
    "BANK_ELIGIBLE",
    "HONEST_BASELINES",
    "PLANTED_MODELS",
    "REFERENCE_INSTRUMENTS",
    "Annealed2D",
    "Checkerboard",
    "CoarseToFine2D",
    "Collapsed",
    "DatasetGenerator",
    "DesignBank",
    "Ensemble2D",
    "KNNRetrieval",
    "LinearRegression",
    "NoCheckpointError",
    "NoiseDoped",
    "PlantedModel",
    "Portfolio2D",
    "VolumeOnly",
    "match_volume_fraction",
]
