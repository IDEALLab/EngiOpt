"""Per-problem configuration for the challenge.

The whole session is parameterized by one JSON file per problem, so a second
team can run the same arc on photonics2d without a line of code changing. What
belongs here is anything that differs between problems: which metrics are shown
first, which are held back for the reveal, which are unavailable and why.

`unavailable_metrics` is derived from the frozen eval spec rather than written
down, so it cannot drift away from what the evaluator will actually do -- and
when a team discovers mid-session that the feasibility column their neighbours
are arguing about does not exist for their problem, that is content, not a bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
from pathlib import Path
from typing import Any

from engiopt.evaluation.registry import METRICS
from engiopt.evaluation.spec import EvalSpec

CONFIG_DIR = Path(__file__).resolve().parents[3] / "workshops" / "idetc26" / "problems"
"""Where the per-problem JSON files live, resolved from this file rather than the cwd.

The DCC'26 notebooks resolved their artifact directory relative to the working
directory, which put a second copy under `simple/workshops/dcc26/` the moment a
notebook was run from anywhere else. Anchoring on `__file__` is the fix.
"""


@dataclass(frozen=True)
class WorkshopConfig:
    """Everything the challenge needs to know about one problem.

    Attributes:
        problem_id: EngiBench registry key.
        spec: Frozen evaluation spec, as `"<problem_id>/<version>"`.
        display_name: Human-readable name shown in the notebook.
        opening_metrics: Columns teams get before they commit to a winner.
        withheld_metrics: Cheap columns held back for the reveal. These were
            affordable all along, which is the point they make.
        expensive_metrics: Simulator-backed columns, revealed last.
        manifold_metrics: Cheap columns that need a fitted latent instrument.
            Held separately from `withheld_metrics` because they are the one
            group with a prerequisite a team can fail to have: the spec must
            pin a `latent_instrument`. A problem whose spec pins none simply
            leaves this empty and the notebook says so instead of raising.
        bank: Bank members, as `{"kind": "constructed"|"pretrained", ...}` specs.
        reveal_seeds: Seeds the lottery re-runs the opening metrics at.
        narrative: Free-text strings the notebook prints, kept out of the code.
    """

    problem_id: str
    spec: str
    display_name: str
    opening_metrics: tuple[str, ...]
    withheld_metrics: tuple[str, ...] = ()
    expensive_metrics: tuple[str, ...] = ()
    manifold_metrics: tuple[str, ...] = ()
    bank: tuple[dict[str, Any], ...] = ()
    reference_instruments: tuple[dict[str, Any], ...] = ()
    reveal_seeds: tuple[int, ...] = (1, 2, 3)
    narrative: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, problem_id: str, *, config_dir: Path | None = None) -> WorkshopConfig:
        """Read a problem's config from `workshops/idetc26/problems/<problem_id>.json`.

        Raises:
            FileNotFoundError: If no config exists for that problem.
        """
        path = (config_dir or CONFIG_DIR) / f"{problem_id}.json"
        if not path.exists():
            available = sorted(p.stem for p in (config_dir or CONFIG_DIR).glob("*.json"))
            raise FileNotFoundError(f"No workshop config for {problem_id!r}. Available: {available or 'none'}.")
        payload = json.loads(path.read_text())
        return cls(
            problem_id=payload["problem_id"],
            spec=payload["spec"],
            display_name=payload.get("display_name", payload["problem_id"]),
            opening_metrics=tuple(payload["opening_metrics"]),
            withheld_metrics=tuple(payload.get("withheld_metrics", ())),
            expensive_metrics=tuple(payload.get("expensive_metrics", ())),
            manifold_metrics=tuple(payload.get("manifold_metrics", ())),
            bank=tuple(payload.get("bank", ())),
            reference_instruments=tuple(payload.get("reference_instruments", ())),
            reveal_seeds=tuple(payload.get("reveal_seeds", (1, 2, 3))),
            narrative=payload.get("narrative", {}),
        )

    @property
    def cheap_metrics(self) -> tuple[str, ...]:
        """Every simulation-free column, opening and withheld together."""
        return (*self.opening_metrics, *self.withheld_metrics)

    def has_latent_instrument(self) -> bool:
        """Whether this problem's spec pins the autoencoder the latent columns need.

        Asked rather than assumed: the manifold columns are the only ones in the
        suite that depend on a fitted instrument, and a spec that pins none has
        to say so plainly rather than fail inside a metric.
        """
        return EvalSpec.load(self.spec).latent_instrument is not None

    def unavailable_metrics(self) -> dict[str, str]:
        """Configured metrics that cannot produce a number on this problem, and why.

        Derived from the frozen spec, not declared, so the answer stays true
        when the spec changes. The only case today is a volume-fraction metric
        on a problem with no volume budget -- photonics2d, whose spec sets
        `volume_condition` to null.
        """
        spec = EvalSpec.load(self.spec)
        if spec.volume_condition is not None:
            return {}
        reason = f"{self.problem_id} has no volume-fraction budget, so there is nothing to be feasible against."
        return {name: reason for name in self.cheap_metrics if METRICS[name].family in {"feasibility", "conditions"}}

    def available(self, metrics: tuple[str, ...]) -> list[str]:
        """Those of `metrics` that this problem can actually compute."""
        unavailable = self.unavailable_metrics()
        return [name for name in metrics if name not in unavailable]

    def direction(self, metric: str) -> bool | None:
        """Whether higher is better for a column, straight from the registry."""
        return METRICS[metric].higher_is_better
