"""The anonymized model bank.

Teams see `Model A`, `Model B`, ... and nothing else. The anonymization is not
theatre: knowing that one entry is a diffusion model and another is a lookup
table is exactly the prior that stops people from reading the numbers, and the
session is about reading the numbers.

Two properties matter for the room:

1. **Order is permuted per team**, deterministically from the team name, so
   neighbouring groups cannot compare notes by letter and so no letter acquires
   a reputation across sessions.
2. **Members load lazily and one at a time.** A bank is an iterator, never a
   dict of loaded models -- a Colab free-tier runtime will not hold eight
   generators and their sampled designs at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import hashlib
from typing import Any, Callable, TYPE_CHECKING

from engiopt.baselines import BANK_ELIGIBLE
from engiopt.baselines import REFERENCE_INSTRUMENTS

if TYPE_CHECKING:
    from collections.abc import Iterator

    from engibench.core import Problem

    from engiopt.core import Generator
    from engiopt.workshops.idetc26.config import WorkshopConfig

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class BankMember:
    """One anonymized entry in the bank.

    Attributes:
        label: What the team sees, e.g. `"Model C"`.
        key: Stable internal identifier, used for joins and for the reveal.
        kind: How this model came to exist -- `pretrained` (a trained
            checkpoint), `baseline` (fitted from the dataset), or `reference`
            (a calibration instrument, never ranked).
        identity: The real name, shown only at the reveal.
        summary: One line saying what this model actually does.
        wins: Columns the construction was built to top.
        loses: Columns it was built to fail.
        load: Zero-argument callable returning a loaded `Generator`.
    """

    label: str
    key: str
    kind: str
    identity: str
    summary: str
    load: Callable[[], Generator]
    wins: tuple[str, ...] = ()
    loses: tuple[str, ...] = ()


class ModelLoadError(RuntimeError):
    """Raised when a bank member cannot be loaded and was not marked optional."""


@dataclass
class ModelBank:
    """A permuted, lazily-loaded collection of anonymized generators.

    Attributes:
        members: Bank entries in the team's own presentation order.
        skipped: Entries that could not be loaded, as `{spec: reason}`. Printed
            rather than raised, because a bank that is one checkpoint short
            still runs a workshop and a stack trace on the conference wifi does
            not.
    """

    members: list[BankMember]
    skipped: dict[str, str] = field(default_factory=dict)

    @classmethod
    def assemble(cls, config: WorkshopConfig, problem: Problem, *, team: str = "") -> ModelBank:
        """Build the bank described by a workshop config.

        Args:
            config: The problem's workshop configuration.
            problem: The loaded EngiBench problem, shared by every member.
            team: Team name; seeds the presentation-order permutation. An empty
                name gives the canonical order, which is what solutions and
                tests want.

        Returns:
            The assembled bank, with unloadable optional members recorded in
            `skipped` rather than raised.
        """
        entries: list[BankMember] = []
        skipped: dict[str, str] = {}

        for entry in config.bank:
            try:
                entries.append(_member_from_entry(entry, problem, config.problem_id))
            except Exception as exc:  # noqa: PERF203 - a handful of entries, and one bad checkpoint must not sink the rest
                name = str(entry.get("algo", entry))
                if not entry.get("optional", False):
                    raise ModelLoadError(f"Required bank member {name!r} failed to load: {exc}") from exc
                skipped[name] = f"{type(exc).__name__}: {exc}"

        order = _permutation(len(entries), team)
        members = [
            BankMember(
                label=f"Model {LETTERS[position]}",
                key=entries[source].key,
                kind=entries[source].kind,
                identity=entries[source].identity,
                summary=entries[source].summary,
                wins=entries[source].wins,
                loses=entries[source].loses,
                load=entries[source].load,
            )
            for position, source in enumerate(order)
        ]
        return cls(members=members, skipped=skipped)

    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self) -> Iterator[BankMember]:
        return iter(self.members)

    def __getitem__(self, label: str) -> BankMember:
        """Look a member up by its anonymous label.

        Raises:
            KeyError: If no member carries that label.
        """
        for member in self.members:
            if member.label == label:
                return member
        raise KeyError(f"No {label!r} in this bank. Members: {[m.label for m in self.members]}")

    @property
    def labels(self) -> list[str]:
        """Every member's anonymous label, in presentation order."""
        return [member.label for member in self.members]

    def identities(self) -> dict[str, str]:
        """The mapping from label to real model name. For the reveal only."""
        return {member.label: member.identity for member in self.members}


def _member_from_entry(entry: dict[str, Any], problem: Problem, problem_id: str) -> BankMember:
    """Turn one config bank entry into a loadable member.

    Raises:
        ValueError: If the entry names an unknown kind or an unknown model.
    """
    kind = entry.get("kind", "pretrained")
    algo = entry["algo"]
    seed = int(entry.get("seed", 1))

    if kind in {"baseline", "reference"}:
        catalogue = BANK_ELIGIBLE if kind == "baseline" else REFERENCE_INSTRUMENTS
        if algo not in catalogue:
            raise ValueError(f"Unknown {kind} model {algo!r}. Known: {sorted(catalogue)}.")
        cls = catalogue[algo]
        if kind == "baseline" and not cls.bank_eligible:
            raise ValueError(
                f"{algo!r} is a reference instrument, not a baseline: it exists to calibrate what a metric "
                "reads at a known input, and ranking it against real models would be a trick rather than a "
                'measurement. Declare it with kind="reference" to show it beside the board.'
            )
        return BankMember(
            label="",
            key=f"{algo}#{seed}",
            kind=kind,
            identity=algo,
            summary=cls.summary,
            wins=cls.wins,
            loses=cls.loses,
            load=lambda: cls.from_problem(problem, problem_id=problem_id, seed=seed),
        )

    if kind == "pretrained":
        from engiopt.utils.all_generators import BUILTIN_GENERATORS

        if algo not in BUILTIN_GENERATORS:
            raise ValueError(f"Unknown generator {algo!r}. Known: {sorted(BUILTIN_GENERATORS)}.")
        cls_pretrained = BUILTIN_GENERATORS[algo]
        kwargs = {
            "problem_id": problem_id,
            "seed": seed,
            "model_source": entry.get("model_source", "auto"),
            "local_model_dir": _resolve_local_dir(entry.get("local_model_dir")),
            "config_fingerprint": entry.get("config_fingerprint"),
        }
        # Fail here, at assembly, rather than at first sample: a missing
        # checkpoint should be reported while the bank is being built and the
        # entry can still be skipped.
        loaded = cls_pretrained.from_pretrained(problem, **kwargs)
        return BankMember(
            label="",
            key=f"{algo}#{seed}",
            kind="pretrained",
            identity=f"{algo} (seed {seed})",
            summary=entry.get("summary", "A model somebody trained."),
            load=lambda: loaded,
        )

    raise ValueError(f"Unknown bank entry kind {kind!r}; expected 'pretrained', 'baseline', or 'reference'.")


def _resolve_local_dir(path: str | None) -> str | None:
    """Resolve a config-declared local checkpoint directory against the repo root.

    Paths in `problems/<id>.json` are written relative to the repository so the
    same config works from a notebook, a test, and the CLI. Resolving against
    the cwd instead is the DCC'26 bug that put a second artifact tree on disk.
    """
    if not path:
        return None
    from pathlib import Path

    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(Path(__file__).resolve().parents[3] / candidate)


def _permutation(n: int, team: str) -> list[int]:
    """A deterministic per-team ordering of `n` members.

    Seeded from a hash of the team name rather than from the clock, so a team
    that reruns the notebook -- or a facilitator reproducing a dispute an hour
    later -- sees the same letters attached to the same models.
    """
    if not team:
        return list(range(n))
    digest = hashlib.sha256(team.strip().lower().encode()).digest()
    import numpy as np

    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return rng.permutation(n).tolist()
