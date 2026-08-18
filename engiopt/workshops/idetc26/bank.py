"""The model bank: what is available to measure, named for the method it implies.

Real members are called what they actually are -- `diffusion_2d_cond`,
`knn_retrieval`, `vqgan#2` -- because the interesting comparisons are between
*kinds* of model, and a reader who does not know that one entry is a lookup
table cannot ask why it is beating the networks. The names are the ones
`engiopt` uses elsewhere, so what a participant learns here is the real
vocabulary rather than a workshop's private one.

**A `planted` member is the exception, deliberately.** It carries a name that
implies a method it does not implement, because its job is to be ranked first by
a column it does not deserve, and a name that announced the construction would
be ranked last by everybody. Its `summary` is still literally true, its cost
columns are still real, and it is disclosed with `built_to` when the physics
board is unsealed -- see `engiopt/baselines/planted.py` for why those three
constraints are the whole difference between a teaching device and a trick.

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
from engiopt.baselines import PLANTED_MODELS
from engiopt.baselines import REFERENCE_INSTRUMENTS

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from engibench.core import Problem

    from engiopt.baselines.base import DatasetGenerator
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
            checkpoint), `baseline` (fitted from the dataset), `planted` (built
            to top a column it does not deserve, ranked with the rest and
            disclosed at the reveal), or `reference` (a calibration instrument,
            never ranked beside real models).
        built_to: For a planted member, what it was built to break. Empty for
            everything else, which is what the reveal keys off.
        identity: The algorithm and seed, spelled out.
        summary: One line saying what this model actually does.
        description: The same thing at length, in plain words, for
            `case.explain`. A dataset-fitted model carries its own on the class
            beside the code it describes; a checkpoint's is declared in the
            problem config, since the adapter's docstring is written for
            somebody reading the source rather than somebody in the room.
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
    description: str = ""
    wins: tuple[str, ...] = ()
    loses: tuple[str, ...] = ()
    built_to: str = ""


_CATALOGUES: dict[str, Mapping[str, type[DatasetGenerator]]] = {
    "baseline": BANK_ELIGIBLE,
    "planted": PLANTED_MODELS,
    "reference": REFERENCE_INSTRUMENTS,
}
"""Which catalogue each dataset-fitted `kind` draws from.

One table rather than three lookups, because the three kinds are three different
promises about how a member will be treated -- competed with, ranked and then
disclosed, or never ranked -- and a kind that resolves in one place but not
another is how a construction ends up in a line-up without its disclosure.
"""


TRAINED_DESCRIPTIONS = {
    "knn_retrieval": (
        "Looks up the one training design whose brief is closest to yours and hands it back, rescaled to "
        "your volume budget. No network, no noise input: the same brief always gives the same design, and "
        "it can never produce anything that is not already in the dataset."
    ),
    "deconv_regression": (
        "Supervised. The brief goes in, one design comes out, and it was trained by penalising the "
        "pixel-by-pixel difference from the right answer. It shares its upsampling stack with the "
        "conditional GANs, so what differs between them is the training objective rather than the "
        "architecture. Habibi et al.'s deconvolutional network."
    ),
    "vqgan": (
        "Compresses designs into a grid of discrete codes drawn from a learned codebook, then trains a "
        "transformer to write out those codes one at a time, conditioned on the brief. Sampling means "
        "generating a code sequence and decoding it back to pixels."
    ),
    "diffusion_2d_cond": (
        "Starts from pure noise and removes a little of it at a time, hundreds of times over, each step "
        "guided by the brief. The most expensive model here to sample from, and the one whose training "
        "objective is closest to 'match the whole distribution'."
    ),
    "cgan_cnn_2d": (
        "A generator network turns a random vector plus your brief into a design, trained against a "
        "discriminator that learns to tell generated designs from real ones. The randomness means it can "
        "offer a different answer to the same brief each time you ask."
    ),
    "gan_cnn_2d": (
        "The same adversarial setup with one thing removed: it never sees the brief. It learns what "
        "designs look like in general and samples from that, so nothing connects what you asked for to "
        "what comes back."
    ),
    "constrained_plvae_2d": (
        "An autoencoder trained to squeeze designs into as few latent dimensions as it can while still "
        "reconstructing them, with a further constraint tying that latent space to performance. Sampling "
        "means drawing a point in the latent space and decoding it. It is also the family the lv_ columns "
        "measure in -- though never the exact checkpoint being scored."
    ),
}
"""Plain-language descriptions of the trained families, for `case.explain`.

In code rather than in `problems/<id>.json` for two reasons. The same seven
families appear in all three problems, so a config copy is three copies to keep
in step; and the configs are edited constantly as line-ups change, which makes
them the worst place to park prose that almost never changes. A bank entry may
still override with its own `description` when a particular checkpoint needs
saying something different about it.

Not taken from the adapter docstrings, which are written for whoever maintains
the class: "Conditional denoising diffusion model over 2D designs." teaches the
words rather than the mechanism, which is the gap `explain` exists to close.
"""


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

    if kind in _CATALOGUES:
        catalogue = _CATALOGUES[kind]
        if algo not in catalogue:
            # A model filed under the wrong kind is a different mistake from a
            # typo, and "unknown baseline" would send the reader looking for a
            # missing class that is sitting right there under another heading.
            _refuse_wrong_catalogue(algo, kind)
            raise ValueError(f"Unknown {kind} model {algo!r}. Known: {sorted(catalogue)}.")
        cls = catalogue[algo]
        # A dataset-fitted baseline is configurable -- kNN's `neighbours` is the
        # difference between pure retrieval and a blend of five designs -- and
        # that setting has to be declarable in the bank JSON. Without this the
        # bank silently takes the class default, which is how a bank meant to
        # hold a lookup table ended up holding a five-way average.
        options = _baseline_options(entry)
        # The key carries the *effective* settings and a digest of the model's
        # own source, not just what the entry declared. A checkpoint's weights
        # are what change its fingerprint; a dataset-fitted model has knobs on
        # the class and a mechanism in its code, and neither is in the entry.
        # Without both, retuning a construction and re-scoring it replays the
        # designs the previous version produced.
        # `code` folds the mechanism *and* the knobs into one digest, because it
        # is also what addresses this model's published metrics on the Hub. A
        # digest over the source alone would give every rung of a severity
        # ladder the same package path, and the second rung's physics would
        # overwrite the first's with nothing downstream reporting a problem.
        settings = {**cls.settings(), **{k: v for k, v in options.items() if k in cls.tuning}}
        digest = cls.package_fingerprint(settings)
        fingerprint = {"code": digest} if digest else {}
        return BankMember(
            label=entry.get("name", ""),
            key=_key_for(algo, seed, {**cls.settings(), **fingerprint, **options}),
            kind=kind,
            identity=algo,
            summary=entry.get("summary") or cls.summary,
            description=entry.get("description") or cls.description,
            train_minutes=entry.get("train_minutes"),
            wins=cls.wins,
            loses=cls.loses,
            built_to=getattr(cls, "built_to", ""),
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
            # An adapter docstring is the last resort rather than the default:
            # it is written for whoever maintains the class, and a participant
            # reading "Conditional denoising diffusion model over 2D designs."
            # learns the words rather than the mechanism.
            description=(
                entry.get("description") or TRAINED_DESCRIPTIONS.get(algo) or _first_paragraph(cls_pretrained.__doc__)
            ),
            train_minutes=entry.get("train_minutes"),
            load=lazy,
        )

    raise ValueError(f"Unknown bank entry kind {kind!r}; expected 'pretrained', 'baseline', 'planted', or 'reference'.")


def _first_paragraph(text: str | None) -> str:
    """The opening paragraph of a docstring, unwrapped, or empty."""
    if not text:
        return ""
    paragraph = text.strip().split("\n\n")[0]
    return " ".join(line.strip() for line in paragraph.splitlines())


def _refuse_wrong_catalogue(algo: str, kind: str) -> None:
    """Explain a model that exists but was declared under the wrong heading.

    The three catalogues are three different promises to a participant -- a
    baseline competes honestly, a planted model is ranked and disclosed at the
    reveal, a reference instrument is never ranked at all -- so filing one under
    another kind is not a naming slip, it changes what the session claims.

    Args:
        algo: The model named in the bank entry.
        kind: The kind it was declared as.

    Raises:
        ValueError: If `algo` is a known model of some other kind.
    """
    actual = next((name for name, catalogue in _CATALOGUES.items() if algo in catalogue), None)
    if actual is None:
        return

    named, why = {
        "baseline": (
            "baseline",
            "a published method that competes honestly, and demoting it to a control would hide a result "
            "rather than calibrate one",
        ),
        "planted": (
            "planted construction",
            'built to top a column it does not deserve. Ranked, yes -- but only under kind="planted", '
            "which is what carries its disclosure into the reveal",
        ),
        "reference": (
            "reference instrument",
            "there to calibrate what a metric reads at a known input, and ranking it against real models "
            "would be a trick rather than a measurement",
        ),
    }[actual]
    raise ValueError(f'{algo!r} was declared kind="{kind}", but it is a {named}: {why}.')


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
