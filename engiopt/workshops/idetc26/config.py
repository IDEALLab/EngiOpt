"""Per-problem configuration for the case.

The whole session is parameterized by one JSON file per problem, so a second
team can run the same arc on photonics2d without a line of code changing. What
belongs here is anything that differs between problems: which suspects are in
the line-up, which columns can be asked of them, which are unavailable and why.

`unavailable_metrics` is derived from the frozen eval spec rather than written
down, so it cannot drift away from what the evaluator will actually do -- and
when a team discovers mid-session that the feasibility column their neighbours
are arguing about does not exist for their problem, that is content, not a bug.

There is **one** metric list, not four. The old config split its columns into
opening / withheld / manifold / expensive to drive a staged reveal; the session
no longer has stages, every column is askable from the first minute, and cost is
a property the registry already knows. A split that exists only to hide things
from participants is a split that has to be maintained for no measurement
reason.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from engiopt.evaluation.registry import METRICS
from engiopt.evaluation.spec import EvalSpec

PACKAGE_DIR = Path(__file__).resolve().parent
"""This package, which is all a Colab runtime has.

Colab installs EngiOpt with `pip install git+...`, and that ships the `engiopt`
package and nothing else -- no `workshops/` tree, no notebooks, no sealed board.
So anything the case must be able to read on the day lives *inside the package*:
the problem configs, the sealed board, and the design cache.
"""

CONFIG_DIRS = (
    PACKAGE_DIR / "problems",
    Path(__file__).resolve().parents[3] / "workshops" / "idetc26" / "problems",
)
"""Where per-problem JSON files are looked for, package copy first.

Resolved from `__file__` rather than the cwd. The DCC'26 notebooks resolved
their artifact directory relative to the working directory, which put a second
copy under `simple/workshops/dcc26/` the moment a notebook was run from
anywhere else. The second entry keeps an existing checkout working.
"""

SEALED_DIRS = (
    PACKAGE_DIR / "sealed",
    Path(__file__).resolve().parents[3] / "workshops" / "idetc26" / "sealed",
)
"""Where sealed boards are looked for, on the same rule as the configs."""


@dataclass(frozen=True)
class WorkshopConfig:
    """Everything the case needs to know about one problem.

    Attributes:
        problem_id: EngiBench registry key.
        spec: Frozen evaluation spec, as `"<problem_id>/<version>"`.
        display_name: Human-readable name shown in the notebook.
        metrics: Every column this problem offers, cheap and expensive
            together. Cost is read from the registry rather than declared here,
            so a column cannot be filed as cheap and then start a simulator.
        bank: The suspects, as `{"kind": "constructed"|"pretrained", ...}` specs.
        controls: Models whose answer is already known, scored beside the board
            as a scale bar and never ranked against it.
    """

    problem_id: str
    spec: str
    display_name: str
    metrics: tuple[str, ...]
    bank: tuple[dict[str, Any], ...] = ()
    controls: tuple[dict[str, Any], ...] = ()

    @classmethod
    def load(cls, problem_id: str, *, config_dir: Path | None = None) -> WorkshopConfig:
        """Read a problem's config from the first directory that has one.

        Raises:
            FileNotFoundError: If no config exists for that problem.
        """
        directories = (config_dir,) if config_dir else CONFIG_DIRS
        path = next(
            (directory / f"{problem_id}.json" for directory in directories if (directory / f"{problem_id}.json").exists()),
            None,
        )
        if path is None:
            available = sorted({p.stem for directory in directories for p in directory.glob("*.json")})
            raise FileNotFoundError(f"No workshop config for {problem_id!r}. Available: {available or 'none'}.")
        payload = json.loads(path.read_text())
        return cls(
            problem_id=payload["problem_id"],
            spec=payload["spec"],
            display_name=payload.get("display_name", payload["problem_id"]),
            metrics=tuple(dict.fromkeys(payload["metrics"])),
            bank=tuple(payload.get("bank", ())),
            controls=tuple(payload.get("controls", ())),
        )

    @property
    def cheap_metrics(self) -> tuple[str, ...]:
        """Every column that answers without a simulator."""
        return tuple(name for name in self.metrics if METRICS[name].cost == "cheap")

    @property
    def expensive_metrics(self) -> tuple[str, ...]:
        """Every column that needs the simulator."""
        return tuple(name for name in self.metrics if METRICS[name].cost == "expensive")

    def sealed_board_path(self) -> Path:
        """Where this problem's sealed physics board is, package copy first.

        Returns the package location when nothing exists yet, so a missing board
        reports the path it should have been written to.
        """
        name = f"{self.problem_id}_physics.csv.enc"
        return next((directory / name for directory in SEALED_DIRS if (directory / name).exists()), SEALED_DIRS[0] / name)

    def has_latent_instrument(self) -> bool:
        """Whether this problem's spec pins the autoencoder the latent columns need.

        Asked rather than assumed: the latent columns are the only ones in the
        suite that depend on a fitted instrument, and a spec that pins none has
        to say so plainly rather than fail inside a metric.
        """
        return EvalSpec.load(self.spec).latent_instrument is not None

    def unavailable_metrics(self) -> dict[str, str]:
        """Configured metrics that cannot produce a number on this problem, and why.

        Derived from the frozen spec, not declared, so the answer stays true
        when the spec changes. Two things can make a column impossible: a
        volume-fraction metric on a problem with no volume budget (photonics2d,
        whose spec sets `volume_condition` to null), and a latent column on a
        spec that pins no autoencoder.
        """
        spec = EvalSpec.load(self.spec)
        reasons: dict[str, str] = {}

        if spec.volume_condition is None:
            budget = f"{self.problem_id} has no volume-fraction budget, so there is nothing to be feasible against."
            # Asked of each metric, not of its family. `pixel_paired_distance` and
            # `lv_paired_distance` compare a design to the optimum for its own
            # condition; they need no volume budget, and excluding them by family
            # silently removed two working columns from every problem lacking one.
            reasons.update({name: budget for name in self.metrics if "volume_condition" in METRICS[name].requires})

        if spec.latent_instrument is None:
            pinned = f"{self.spec} pins no autoencoder, so there is no latent space to measure in."
            reasons.update({name: pinned for name in self.metrics if name.startswith(("lv_", "lvoff_"))})

        return reasons

    def available(self, metrics: tuple[str, ...] | list[str]) -> list[str]:
        """Those of `metrics` that this problem can actually compute."""
        unavailable = self.unavailable_metrics()
        return [name for name in metrics if name not in unavailable]

    def direction(self, metric: str) -> bool | None:
        """Whether higher is better for a column, straight from the registry."""
        return METRICS[metric].higher_is_better
