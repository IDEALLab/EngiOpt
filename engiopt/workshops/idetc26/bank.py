"""The model bank: what is available to measure, named for what it is.

Every member is called what it actually is -- `diffusion_2d_cond`,
`knn_retrieval`, `vqgan#2` -- because the interesting comparisons are between
*kinds* of model, and a reader who does not know that one entry is a lookup
table cannot ask why it is beating the networks. The names are the ones
`engiopt` uses elsewhere, so what a participant learns here is the real
vocabulary rather than a workshop's private one.

Members are referred to by any unambiguous fragment: `"diffusion"` reaches
`diffusion_2d_cond`, and `"cgan"` reports that it matches three and asks which.
Nobody should have to type an exact identifier to look at a picture.

**Members load lazily and one at a time.** A bank is an iterator, never a dict
of loaded models -- a Colab free-tier runtime will not hold ten generators and
their sampled designs at once -- and a member whose designs are already cached
never loads its checkpoint at all.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import Any, Callable, TYPE_CHECKING

from engiopt.baselines import BANK_ELIGIBLE
from engiopt.baselines import REFERENCE_INSTRUMENTS

if TYPE_CHECKING:
    from collections.abc import Iterator

    from engibench.core import Problem

    from engiopt.core import Generator
    from engiopt.workshops.idetc26.config import WorkshopConfig


@dataclass(frozen=True)
class BankMember:
    """One entry in the bank.

    Attributes:
        label: How this model is referred to everywhere -- its algorithm name,
            with `#seed` appended only when the bank holds more than one of
            them, since an unnecessary suffix is just noise to type.
        key: Stable internal identifier (`algo#seed`), used for joins with the
            sealed board and the design cache.
        kind: How this model came to exist -- `pretrained` (a trained
            checkpoint), `baseline` (fitted from the dataset), or `reference`
            (a calibration instrument, never ranked beside real models).
        identity: The algorithm and seed, spelled out.
        summary: One line saying what this model actually does.
        train_minutes: What this model cost to train, when that is known.
            Declared rather than measured: nothing in a checkpoint package
            records it today, so the figure comes from the family's finished
            W&B runs and is carried here with that provenance stated.
        wins: Columns this construction is expected to top.
        loses: Columns it is expected to fail.
        load: Zero-argument callable returning a loaded `Generator`.
    """

    label: str
    key: str
    kind: str
    identity: str
    summary: str
    load: Callable[[], Generator]
    train_minutes: float | None = None
    wins: tuple[str, ...] = ()
    loses: tuple[str, ...] = ()


class ModelLoadError(RuntimeError):
    """Raised when a bank member cannot be loaded and was not marked optional."""


@dataclass
class ModelBank:
    """A lazily-loaded collection of the generators available to measure.

    Attributes:
        members: Bank entries, in the order the config declares them.
        skipped: Entries that could not be loaded, as `{spec: reason}`. Printed
            rather than raised, because a bank that is one checkpoint short
            still runs a workshop and a stack trace on the conference wifi does
            not.
    """

    members: list[BankMember]
    skipped: dict[str, str] = field(default_factory=dict)

    @classmethod
    def assemble(
        cls,
        config: WorkshopConfig,
        problem: Problem,
        *,
        device: Any = None,
        is_cached: Callable[[str, int], bool] | None = None,
        entries: tuple[dict[str, Any], ...] | None = None,
    ) -> ModelBank:
        """Build the bank described by a workshop config.

        Args:
            config: The problem's workshop configuration.
            problem: The loaded EngiBench problem, shared by every member.
            device: Where checkpoints are loaded and sampled. Defaults to the
                repo-wide choice, which takes a GPU when there is one. Passed
                explicitly so that a bank cannot end up on a different device
                from the evaluator that scores it.
            is_cached: Asked whether a member's designs are already on disk, as
                `(key, seed) -> bool`. A member whose designs are cached does
                not have its checkpoint downloaded at assembly -- which is the
                difference between a workshop that opens in seconds and one that
                pulls two gigabytes of weights before showing anybody anything.
            entries: Bank specs to assemble instead of `config.bank`. Used for
                the controls, which are a second small bank built the same way
                -- one assembly path, so a control loads, caches and samples
                exactly as a suspect does and cannot quietly diverge from one.

        Returns:
            The assembled bank, with unloadable optional members recorded in
            `skipped` rather than raised.
        """
        members: list[BankMember] = []
        skipped: dict[str, str] = {}

        for entry in config.bank if entries is None else entries:
            try:
                members.append(_member_from_entry(entry, problem, config.problem_id, is_cached, device))
            except Exception as exc:  # noqa: PERF203 - a handful of entries, and one bad checkpoint must not sink the rest
                name = str(entry.get("algo", entry))
                if not entry.get("optional", False):
                    raise ModelLoadError(f"Required bank member {name!r} failed to load: {exc}") from exc
                skipped[name] = f"{type(exc).__name__}: {exc}"

        # Members carry any name their entry declared in `label` already; the
        # rest get one derived from their key. Zipped over `members` rather than
        # over the entries, because an entry that failed to load is not here.
        labels = _handles([member.key for member in members], declared=[member.label for member in members])
        return cls(members=[replace(member, label=label) for member, label in zip(members, labels)], skipped=skipped)

    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self) -> Iterator[BankMember]:
        return iter(self.members)

    def __getitem__(self, name: str) -> BankMember:
        """Look a member up by name, exactly or by any unambiguous fragment."""
        return self.resolve(name)

    def resolve(self, name: str) -> BankMember:
        """Find the member `name` refers to.

        Exact matches win; otherwise any fragment that picks out exactly one
        member is accepted, so `"diffusion"` and `"knn"` work and nobody has to
        type `constrained_plvae_2d` to look at a picture. An ambiguous fragment
        reports what it matched rather than guessing, since guessing would
        quietly show somebody the wrong model.

        Raises:
            KeyError: If nothing matches, or more than one thing does.
        """
        needle = name.strip().lower()
        for member in self.members:
            if needle in {member.label.lower(), member.key.lower()}:
                return member
        matches = [member for member in self.members if needle in member.label.lower()]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"No model matching {name!r}. In the bank: {self.labels}")
        raise KeyError(f"{name!r} matches {[m.label for m in matches]} -- which one did you mean?")

    @property
    def labels(self) -> list[str]:
        """Every member's name, in declaration order."""
        return [member.label for member in self.members]


class _LazyCheckpoint:
    """A checkpoint that loads on first use and is remembered afterwards.

    Used only for members whose designs are already cached. The trade is
    deliberate: a missing checkpoint is then reported when somebody asks for a
    fresh sample rather than when the bank is built, and in exchange nobody
    downloads a 1.8 GB VQGAN to look at designs that are sitting on disk.
    """

    def __init__(self, cls: Any, problem: Problem, kwargs: dict[str, Any]) -> None:
        self._cls = cls
        self._problem = problem
        self._kwargs = kwargs
        self._loaded: Generator | None = None

    def __call__(self) -> Generator:
        if self._loaded is None:
            self._loaded = self._cls.from_pretrained(self._problem, **self._kwargs)
        return self._loaded


def _member_from_entry(
    entry: dict[str, Any],
    problem: Problem,
    problem_id: str,
    is_cached: Callable[[str, int], bool] | None = None,
    device: Any = None,
) -> BankMember:
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
        # A dataset-fitted baseline is configurable -- kNN's `neighbours` is the
        # difference between pure retrieval and a blend of five designs -- and
        # that setting has to be declarable in the bank JSON. Without this the
        # bank silently takes the class default, which is how a bank meant to
        # hold a lookup table ended up holding a five-way average.
        options = _baseline_options(entry)
        return BankMember(
            label=entry.get("name", ""),
            key=_key_for(algo, seed, options),
            kind=kind,
            identity=algo,
            summary=entry.get("summary") or cls.summary,
            train_minutes=entry.get("train_minutes"),
            wins=cls.wins,
            loses=cls.loses,
            load=lambda: cls.from_problem(problem, problem_id=problem_id, seed=seed, **options),
        )

    if kind == "pretrained":
        from engiopt.utils.all_generators import BUILTIN_GENERATORS

        if algo not in BUILTIN_GENERATORS:
            raise ValueError(f"Unknown generator {algo!r}. Known: {sorted(BUILTIN_GENERATORS)}.")
        cls_pretrained = BUILTIN_GENERATORS[algo]
        fingerprint = entry.get("config_fingerprint")
        key = _key_for(algo, seed, {"cfg": fingerprint} if fingerprint else {})
        kwargs = {
            "problem_id": problem_id,
            "seed": seed,
            "device": device,
            "model_source": entry.get("model_source", "auto"),
            "local_model_dir": _resolve_local_dir(entry.get("local_model_dir")),
            "config_fingerprint": entry.get("config_fingerprint"),
        }
        lazy = _LazyCheckpoint(cls_pretrained, problem, kwargs)
        if is_cached is None or not is_cached(key, 1):
            # Fail here, at assembly, rather than at first sample: a missing
            # checkpoint should be reported while the bank is being built and
            # the entry can still be skipped. With cached designs there is
            # nothing to fail at yet, so the load is deferred instead.
            lazy()
        return BankMember(
            label=entry.get("name", ""),
            key=key,
            kind="pretrained",
            identity=f"{algo} (seed {seed})",
            summary=entry.get("summary", "A model somebody trained."),
            train_minutes=entry.get("train_minutes"),
            load=lazy,
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


DECLARED_ENTRY_KEYS = frozenset(
    {
        "kind",
        "algo",
        "name",
        "seed",
        "optional",
        "summary",
        "model_source",
        "local_model_dir",
        "config_fingerprint",
        "train_minutes",
    }
)
"""Bank-entry keys the loader consumes itself. Anything else configures the model."""


def _baseline_options(entry: dict[str, Any]) -> dict[str, Any]:
    """Constructor options for a dataset-fitted baseline, taken from its bank entry."""
    return {key: value for key, value in entry.items() if key not in DECLARED_ENTRY_KEYS}


def _key_for(algo: str, seed: int, options: dict[str, Any]) -> str:
    """The stable identifier a cache entry and a sealed board row are filed under.

    It has to separate two entries that differ only in configuration. `algo#seed`
    alone does not: a k=1 kNN and a k=5 kNN are the same algorithm at the same
    seed, and filing both under `knn_retrieval#1` would serve one model's cached
    designs for the other -- silently, since the conditions and the spec match.
    """
    base = f"{algo}#{seed}"
    if not options:
        return base
    return base + "[" + ",".join(f"{key}={value}" for key, value in sorted(options.items())) + "]"


def _handles(keys: list[str], declared: list[str] | None = None) -> list[str]:
    """Turn internal `algo#seed` keys into the shortest names that stay unique.

    A bank with one VQGAN calls it `vqgan`; a bank with two calls them `vqgan#1`
    and `vqgan#2`. The suffix appears exactly when it carries information, so
    the common case is a name somebody can type from memory and the ambiguous
    case is never silently collapsed.

    A bank entry may declare its own `name`, and that wins outright. It has to:
    the derived suffix is the *seed*, so two entries of one algorithm that
    differ by hyperparameter configuration come out as `cgan_cnn_2d#1` and
    `cgan_cnn_2d#42` -- names that say "these differ by seed" about two models
    that do not. A line-up with no seed-pairs in it must not be described by
    names that imply otherwise.

    **The collision check runs after the declared names are applied, not
    before.** Deriving first and overriding afterwards leaves the *other* member
    of a resolved pair carrying a suffix that no longer separates it from
    anything -- `cgan_cnn_2d#1` beside `cgan_cnn_2d_tuned` -- and that is worse
    than cosmetic: members are reachable by fragment, so `"cgan_cnn_2d"` would
    then match both and refuse to resolve.

    Args:
        keys: Internal `algo#seed[options]` identifiers.
        declared: Names taken from the bank entries, aligned with `keys`. An
            empty string means "derive one".

    Returns:
        One name per key, unique across the bank.
    """
    # The configuration suffix a key may carry (`[neighbours=1]`) keeps cache
    # entries apart; it has no business in a name somebody has to type.
    plain = [key.split("[")[0] for key in keys]
    algos = [key.split("#")[0] for key in plain]
    chosen = list(declared) if declared is not None else [""] * len(keys)

    # What each member would be called if nothing collided. A declared name is
    # already final; the rest would like to be the bare algorithm name.
    candidates = [name or algo for name, algo in zip(chosen, algos)]
    repeated = {candidate for candidate, count in Counter(candidates).items() if count > 1}
    handles = [
        name or (bare if candidate in repeated else algo)
        for name, bare, algo, candidate in zip(chosen, plain, algos, candidates)
    ]

    # Two entries of the same algorithm and seed differ only by hyperparameter
    # configuration, which the key does not carry; number them rather than
    # hand back a duplicate name.
    seen: Counter[str] = Counter()
    unique = []
    for handle in handles:
        seen[handle] += 1
        unique.append(handle if seen[handle] == 1 else f"{handle}.{seen[handle]}")
    return unique
