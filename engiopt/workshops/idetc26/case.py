"""The case file: the suspects, and everything you are allowed to ask them.

A line-up of generative models for one engineering design problem. One of them
is the best. Your job is to work out which, and -- harder -- to say what you had
to measure before you were entitled to claim it.

You get four things and no more:

    case = Case.open("beams2d")

    case.models()                     who is in the line-up
    case.metrics()                    what you are allowed to ask, by line of questioning
    case.evaluate("diversity")        put a question to the suspects
    case.show("diffusion")            look at what a suspect actually produced

**`case.evaluate` is the only way to measure anything.** It takes questions and
suspects -- one of each, lists of each, or neither, which means all of them --
and it does not care whether a question is one metric, a whole line of
questioning, cheap, or backed by the simulator:

    case.evaluate()                                  every cheap question, every suspect
    case.evaluate("mmd")                             one question, every suspect
    case.evaluate("diversity")                       one line of questioning
    case.evaluate(["cost", "memorization"])          two of them
    case.evaluate("mmd", models=["knn", "diffusion"])       two suspects
    case.evaluate("performance", n_samples=2)         the expensive one

and the knobs are there when you want them -- `sigma=` to change the kernel
bandwidth, `n_samples=` and `random_conditions=` to change which designs get
compared, `controls=True` for a scale bar under the board.

Two things are deliberate. **Nothing here reimplements evaluation**: every
number comes from `engiopt.evaluation.Evaluator` running the frozen spec, and
every scoring call prints the `python -m engiopt.evaluate` line that would
produce the same board, so what is learned here is the tool rather than the
workshop. And **sampling is cached to disk** (`designs.DesignStore`), because
waiting for a diffusion model to draw fifty designs teaches nothing; the costs
worth feeling -- a fresh sample you asked to time, the simulator -- are still
paid in full.

### What this object is for

Three jobs, and it is worth knowing which:

1. **It carries the setup.** The loaded EngiBench problem, the frozen `EvalSpec`
   that fixes the conditions *every* suspect answers, an `Evaluator`, the device,
   and the assembled checkpoints. Without it each measurement is twenty lines of
   construction, and any one of them getting the spec wrong silently destroys
   comparability between two rows of the same board.
2. **It is the cache.** Drawn designs and computed columns are memoized on
   `(model, seed)`, which is why asking the same thing twice is free.
3. **It is the menu.** `case.<tab>` is the entire toolbox, and there are four
   entries in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import io
from pathlib import Path
import textwrap
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

from engiopt.evaluation import Evaluator
from engiopt.evaluation.registry import METRICS
from engiopt.workshops.idetc26.bank import ModelBank
from engiopt.workshops.idetc26.config import WorkshopConfig
from engiopt.workshops.idetc26.designs import DesignStore
from engiopt.workshops.idetc26.families import FAMILIES
from engiopt.workshops.idetc26.families import family_of
from engiopt.workshops.idetc26.families import space_of
from engiopt.workshops.idetc26.seal import SealError
from engiopt.workshops.idetc26.seal import unseal

if TYPE_CHECKING:
    from engiopt.workshops.idetc26.views import Views

HELP = """
THE CASE FILE -- four commands, and everything is optional inside them.

WHO IS IN THE LINE-UP
  case.models()                          the suspects, and what each one is
  case.explain("diffusion")              one suspect, in full: what it does, step by step
  case.metrics()                         what you may ask, grouped by line of questioning
  case.metrics("diversity")              one line of questioning, in detail

ASK A QUESTION            case.evaluate(what, models=who)
  case.evaluate()                        every cheap question, every suspect
  case.evaluate("mmd")                   one question
  case.evaluate("diversity")             a whole line of questioning
  case.evaluate(["cost", "similarity"])   several of either, mixed freely
  case.evaluate("mmd", models="knn")     one suspect
  case.evaluate("mmd", models=["knn", "diffusion"])
  case.evaluate(ranks=True)              answers as ranks, 1 = best
  case.evaluate(controls=True)           score models whose answer is already known
  case.evaluate("performance", n_samples=2)   the simulator, on two briefs

  ...and the knobs, all optional, all off by default:
  ranks=True           print each column as a placing (1 = best) instead of its value
  controls=True        add the known-answer rows under the board, as a scale bar
  n_samples=10         score on 10 of the spec's briefs instead of all of them
  random_conditions=True   ...and take those 10 at random rather than the first 10
  sigma=0.5            force the kernel width used by mmd / dpp / vendi, instead of
                       the median-distance default nobody reports
  fresh=True           ignore the cached designs: sample from the checkpoints now.
                       Slower, and the only way gen_seconds times *this* machine

THE CONTROLS -- models whose answer is known before you measure
  case.evaluate("pixel_vendi", controls=True)    put the scale bar under the board
  case.explain("noise_doped")                    what a control is built to do
  case.show("noise_doped")                       ...and what it looks like

  collapsed     one design, repeated for every brief. A diversity column's floor.
  noise_doped   real optimal designs plus Gaussian noise. Noise cannot improve an
                optimal design, so a column that *rises* here is rewarding damage.
  volume_only   blobs thresholded to hit the volume budget exactly, carrying no load.
                Whatever feasibility reads here is what feasibility is worth alone.

  They are never ranked and never suspects. Read them as endpoints: a column whose
  suspects all sit between `collapsed` and `noise_doped` has not separated anything.

LOOK AT SOMETHING         case.show(what)
  case.show()                            a few designs from every suspect
  case.show("diffusion")                 one suspect's designs
  case.show("diffusion", n=20)           more of them
  case.show("knn", "diffusion")          two suspects, same condition, side by side
  case.show("cgan", "test")              a suspect against the real optimum
  case.show("train")                     the designs every suspect was fitted on
  case.show("test")                      the held-out designs they are scored against
  case.show(answers)                     a table of answers, drawn as ranks
  case.show(cheap, physics)              two tables joined and drawn as one
  case.show(answers, "mmd", "novelty_ratio")   two columns against each other

  ...and `how=`, which picks one of five pictures, each named for what it draws:
  how="designs"           a grid of one suspect's designs -- the default, said out loud
  how="compare"           one row per suspect, one brief per column, real optimum on top
  how="conditions"        each design captioned "asked 0.30 / got 0.41": did it obey?
  how="nearest_training"  each design with its closest training design beneath it, and
                          the distance between them printed
  how="space_map"         a scatter: your sources in the top two dimensions of a fitted
                          space, the training set faded behind them

  ...and the rest of the knobs:
  n=12               how many designs to draw (where the view draws designs)
  seed=2             which sampling draw -- redraws the noise, never the briefs
  space="pca"        which space how="space_map" draws in; "lv" is the default
  color="volfrac"    what how="space_map" grades the training backdrop by:
                     "performance" (the default), any condition, or "none"
  fresh=True         resample from the checkpoint instead of reading the design cache

THE REST OF THE FILE
  case.designs("knn")                    the raw array, if you want to compute your own
  case.designs("test")                   ...and the real designs it is scored against
  case.problem.render(design)            EngiBench's own renderer for this problem
  case.instrument()                      which autoencoder the latent columns measure in
  case.physics(passphrase)               the precomputed simulator board
"""


@dataclass
class Case:
    """One problem's line-up of suspects, and everything askable about them.

    Attributes:
        config: The problem's workshop configuration.
        evaluator: The scorer, running the frozen spec.
        bank: The suspects.
        controls: Models whose answer is known, used as a scale bar.
        artifact_dir: Where freshly sampled designs are written.
        store: The design cache this case reads and writes.
    """

    config: WorkshopConfig
    evaluator: Evaluator
    bank: ModelBank
    controls: ModelBank
    artifact_dir: Path
    store: DesignStore
    _rows: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict, repr=False)
    _designs: dict[tuple[str, int], np.ndarray] = field(default_factory=dict, repr=False)
    _sample_meta: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict, repr=False)
    _published: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _views: Views | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Opening the file
    # ------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        problem_id: str,
        *,
        artifact_dir: str | Path | None = None,
        device: Any = None,
        cache_dir: str | Path | None = None,
    ) -> Case:
        """Load the problem and assemble the line-up.

        Args:
            problem_id: Which problem to work on.
            artifact_dir: Where to write fresh samples; defaults to
                `./idetc26_work`.
            device: Torch device for the trained checkpoints. Defaults to the
                repo-wide choice, which picks up a Colab GPU when there is one.
            cache_dir: Extra directory to search for cached designs, ahead of
                the copy that ships with the package.

        Returns:
            The opened case.
        """
        config = WorkshopConfig.load(problem_id)
        evaluator = Evaluator.for_problem(problem_id, spec=config.spec, device=device)

        directory = Path(artifact_dir) if artifact_dir else Path.cwd() / "idetc26_work"
        directory.mkdir(parents=True, exist_ok=True)
        store = _store_for(config, evaluator, artifact_dir=directory, cache_dir=cache_dir)
        bank = ModelBank.assemble(config, evaluator.problem, device=evaluator.device, is_cached=store.holds)
        controls = ModelBank.assemble(
            config, evaluator.problem, device=evaluator.device, is_cached=store.holds, entries=config.controls
        )

        print(
            f"{config.display_name}: {len(bank)} suspects, every one asked the same "
            f"{evaluator.resolved.n_samples} design problems from spec {config.spec}."
        )
        # Say the device out loud. Sampling a diffusion model is a thousand
        # sequential network calls, so "cpu" here is the difference between
        # seconds and ten minutes -- and in Colab it means somebody forgot to
        # ask for a GPU runtime, which is fixable in ten seconds if they know.
        print(
            f"Sampling on: {evaluator.device}"
            + ("  (Colab: Runtime -> Change runtime type -> GPU)" if str(evaluator.device) == "cpu" else "")
        )
        for name, reason in bank.skipped.items():
            print(f"  [missing suspect] {name}: {reason}")
        for metric, reason in config.unavailable_metrics().items():
            print(f"  [unavailable] {metric}: {reason}")
        print("\n  case.models()    who is in the line-up")
        print("  case.metrics()   what you are allowed to ask")
        print("  case.help()      the whole toolbox, on one card")

        return cls(config=config, evaluator=evaluator, bank=bank, controls=controls, artifact_dir=directory, store=store)

    @property
    def problem(self) -> Any:
        """The EngiBench problem itself, for anything this module does not wrap.

        An accessor, not a facade. `problem.render(design)` draws a design the
        way *that problem* thinks it should be drawn -- for photonics2d, the
        field magnitudes at both wavelengths beside the permittivity -- and
        there is no reason for the workshop to own a second version of that.
        """
        return self.evaluator.problem

    @staticmethod
    def help() -> None:
        """Print every command there is, with the line that runs it."""
        print(HELP)

    # ------------------------------------------------------------------
    # Who is here, and what may be asked
    # ------------------------------------------------------------------

    def models(self) -> pd.DataFrame:
        """The line-up: what each suspect is, and what it cost to put here.

        Named rather than lettered. Which entry is a diffusion model and which
        is a lookup table is not a spoiler -- it is the first thing you need in
        order to ask why the lookup table is winning.

        Returns:
            A frame indexed by the name every other command accepts.
        """
        rows = [
            {
                "suspect": member.label,
                "what it is": member.summary,
                "train min": member.train_minutes,
                "ready": "yes" if self.store.holds(member.key, 1) else "no",
            }
            for member in self.bank
        ]
        print("Name any suspect by any unambiguous part of its name: 'diffusion', 'knn', 'plvae'.")
        return pd.DataFrame(rows).set_index("suspect")

    def explain(self, model: str | None = None) -> None:
        """Everything known about one suspect, in plain words.

        `models()` is a table, and a table has room for a sentence. That is
        enough to tell a lookup table from a diffusion model and not enough to
        argue about either -- so the question "what *is* this thing actually
        doing" had no answer in the notebook, and a participant who cannot
        answer it cannot say why a column is wrong about it.

        What it does **not** print is anything the board has not shown you yet.
        A model's construction is described; whether that construction is a
        good idea is left to the evidence.

        Args:
            model: Any unambiguous part of a suspect's name, or of a control's.
                Omitted, it lists everyone who can be asked about.
        """
        if model is None:
            self._explain_index()
            return

        member = self.resolve(model)
        is_control = member.label in set(self.controls.labels)
        rule = "-" * min(len(member.label) + 4, 78)
        print(f"\n{member.label}\n{rule}")
        if is_control:
            print("A CONTROL, not a suspect: a model whose answer is already known, built to say what a")
            print("column reads at an input you can already judge. Never ranked against the line-up.\n")
        print(f"{member.summary}\n")

        if member.description:
            for line in textwrap.wrap(" ".join(member.description.split()), width=78):
                print(line)
            print()

        cost = []
        if member.train_minutes is not None:
            cost.append(f"about {member.train_minutes:g} minutes to train")
        if self.store.holds(member.key, 1):
            cost.append("its designs are already cached, so looking at it costs nothing")
        if cost:
            print(f"Cost: {'; '.join(cost)}.")
        print(f"Filed as: {member.identity}   (cache key {member.key})")
        print(
            f"\nLook at it:   case.show({member.label!r})"
            f"\nAgainst real: case.show({member.label!r}, 'test')"
            f"\nIs it copying? case.show({member.label!r}, how='nearest_training')"
        )

    def _explain_index(self) -> None:
        """List everyone `explain` can be asked about, suspects first."""
        print("Ask about any of these:\n")
        for member in self.bank:
            print(f"  case.explain({member.label!r})")
        if len(self.controls):
            print("\n...and the controls, which are calibration standards rather than suspects:\n")
            for member in self.controls:
                print(f"  case.explain({member.label!r})")

    def metrics(self, family: str | None = None) -> pd.DataFrame:
        """What you are allowed to ask, grouped by the line of questioning it belongs to.

        The grouping is the content. Every column in the benchmark answers one of
        a handful of plain questions, and several of those questions are asked in
        more than one **space** -- `mmd`, `pca_mmd` and `lv_mmd` are one question
        in raw pixels, in a PCA subspace, and in a learned latent space. Those
        columns disagree with each other. That disagreement is a fact about the
        spaces rather than about the suspects, and a list sorted by name hides it
        completely.

        Args:
            family: Show one line of questioning in detail, e.g. `"diversity"`.

        Returns:
            A frame indexed by metric name.

        Raises:
            KeyError: If `family` is not a line of questioning.
        """
        if family is not None and family not in FAMILIES:
            raise KeyError(f"No such line of questioning: {family!r}. Try one of {list(FAMILIES)}.")

        unavailable = self.config.unavailable_metrics()
        rows = [
            {
                "metric": name,
                "line of questioning": family_of(name),
                "asks": METRICS[name].description,
                "space": space_of(name),
                "better": {True: "higher", False: "lower", None: "no direction"}[METRICS[name].higher_is_better],
                "cost": "the simulator" if METRICS[name].cost == "expensive" else "seconds",
                "status": self._status_of(name, unavailable),
            }
            for name in self.config.metrics
            if family is None or family_of(name) == family
        ]
        frame = pd.DataFrame(rows).set_index("metric")
        order = {key: position for position, key in enumerate(FAMILIES)}
        frame = frame.sort_values("line of questioning", key=lambda column: column.map(order), kind="stable")

        for key in dict.fromkeys(frame["line of questioning"]):
            print(f"{key:<14} {FAMILIES[key].question}")
            if family is not None:
                print(f"{'':<14} {FAMILIES[key].detail}")
        print(f"\nAsk a whole line of questioning at once:  case.evaluate({next(iter(frame['line of questioning']))!r})")
        print(
            f"{len(frame)} columns configured for this problem; the benchmark registers {len(METRICS)}. "
            "`python -m engiopt.evaluate --list-metrics` shows every one."
        )
        return frame

    # ------------------------------------------------------------------
    # Asking
    # ------------------------------------------------------------------

    def evaluate(
        self,
        metrics: str | list[str] | tuple[str, ...] | None = None,
        models: str | list[str] | tuple[str, ...] | None = None,
        *,
        ranks: bool = False,
        controls: bool = False,
        n_samples: int | None = None,
        random_conditions: bool = False,
        sigma: float | None = None,
        fresh: bool = False,
        show_cli: bool = True,
    ) -> pd.DataFrame:
        """Put questions to suspects. The only way anything gets measured.

        This is `python -m engiopt.evaluate` in a notebook, and it prints the
        command line that would produce the same answers so the skill transfers
        out of it.

        There is deliberately **no `seeds=` argument**. The spec freezes the
        conditions every suspect answers (`EvalSpec.condition_seed`), so the only
        thing a seed could vary here is the *sampling noise* on a fixed
        checkpoint -- and "does the ranking survive a different noise draw" is a
        far weaker question than "does it survive retraining". The interesting
        version needs checkpoints at several **training** seeds, which the Hub
        has for the replicated configurations but which the line-up does not
        currently pull. See `_score(seed=...)`, which still threads a sampling
        seed for the design cache.

        Args:
            metrics: What to ask. A metric name (`"mmd"`), a line of questioning
                (`"diversity"`, which asks every metric in it), or a list mixing
                both. `None` asks every cheap question this problem has.
            models: Which suspects, by name or by any unambiguous fragment.
                `None` asks all of them.
            ranks: Return per-column ranks, 1 = best, instead of raw values.
            controls: Also score the models whose answer is already known, as a
                scale bar beneath the board. They are never ranked.
            n_samples: Ask about only `n_samples` of the spec's conditions.
                Worth using with the simulator, where it is the difference
                between two minutes and two hours.
            random_conditions: Draw those `n_samples` at random rather than
                taking the first. The draw is made once and shared by every
                suspect, so the board still compares like with like; it is
                seeded on `n_samples`, so the same call gives the same subset.
            sigma: Kernel bandwidth for `mmd`, `dpp` and the vendi family,
                replacing the median heuristic. Applies in whichever space each
                metric measures in. A board computed under an override is not
                comparable to one computed without it, and says so.
            fresh: Resample from the checkpoints instead of using cached
                designs. Slower, and the only way to time the models here.
            show_cli: Print the equivalent `engiopt.evaluate` command.

        Returns:
            A board indexed by suspect.
        """
        chosen = self._resolve_metrics(metrics)
        expensive = [name for name in chosen if METRICS[name].cost == "expensive"]
        labels = self._resolve_models(models)
        # Drawn before pricing, because whether a subset was asked for decides
        # whether the published physics can answer at all.
        indices_preview = self._condition_subset(n_samples, random_conditions=random_conditions)

        if show_cli:
            self._print_cli(chosen, expensive=bool(expensive))
        if expensive:
            replayable = self._replayable(seed=1, n_samples=n_samples, indices=indices_preview, sigma=sigma, fresh=fresh)
            self._price_expensive(labels, chosen, n_samples or self.evaluator.resolved.n_samples, replayable=replayable)

        indices = indices_preview
        if sigma is not None:
            print(
                f"  [note] kernel bandwidth forced to sigma={sigma:g}, replacing the median heuristic "
                "calibrated on the validation split. These numbers are not comparable to a default board."
            )

        board = self._board(
            labels,
            chosen,
            # A random subset is already expressed as explicit indices, so
            # passing `n_samples` as well would truncate the truncation.
            n_samples=None if indices is not None else n_samples,
            indices=indices,
            sigma=sigma,
            fresh=fresh,
            include_expensive=bool(expensive),
        )
        board.index.name = "suspect"

        self._note_replayed_costs(board, labels)
        board = self.rank(board) if ranks else board
        if controls:
            board = self._with_controls(board, chosen)
        return board

    # ------------------------------------------------------------------
    # Looking
    # ------------------------------------------------------------------

    def show(
        self,
        *what: Any,
        how: str | None = None,
        n: int = 4,
        seed: int = 1,
        space: str = "lv",
        color: str = "performance",
        fresh: bool = False,
    ) -> Any:
        """Look at something. The only way anything gets drawn.

        What you hand it decides what you get, because in every case there is
        exactly one sensible picture:

            case.show()                     a few designs from every suspect
            case.show("diffusion")          one suspect's designs
            case.show("knn", "diffusion")   two suspects, same condition, side by side
            case.show("cgan", "test")       a suspect against the real optimum
            case.show("train")              the designs every model was fitted on
            case.show(answers)              a table of answers, drawn as ranks
            case.show(answers, "mmd", "viol")   two columns of it against each other

        Five views are specific enough to need naming, via `how=`. Each is named
        for the picture it draws rather than for the point it makes, so that a
        participant reading somebody else's cell can tell what came out of it:

            how="designs"           a grid of one suspect's designs -- the default
            how="compare"           every suspect on one brief, real optimum on top
            how="conditions"        what was asked of it, against what came back
            how="nearest_training"  each design beside its closest training design
            how="space_map"         where its designs sit against the real ones

        Args:
            *what: A suspect name, two names, one or more boards, or a board and
                two of its column names. Nothing at all means every suspect.
            how: One of the named views above.
            n: How many designs to draw, where that applies.
            seed: Sampling seed.
            space: For `how="space_map"`, which space to draw in -- `"lv"` or `"pca"`.
            color: For `how="space_map"`, what to grade the training backdrop by:
                `"performance"` (the problem's objective, the default), any
                condition by name, or `"none"` for a flat backdrop.
            fresh: Resample rather than using cached designs.

        Returns:
            The matplotlib figure, so it can be saved or resized, or the
            interactive widget for the slider views.

        Raises:
            KeyError: If `how` is not one of the named views.
            TypeError: If the arguments are not a shape `show` can draw.
        """
        views = self._views_object()

        if how is not None:
            if how not in _VIEWS:
                raise KeyError(f"No such view: {how!r}. Try one of {sorted(_VIEWS)}.")
            return _VIEWS[how](views, [str(item) for item in what], n=n, seed=seed, space=space, color=color)

        if what and isinstance(what[0], pd.DataFrame):
            return _show_board(views, what)

        names = [str(item) for item in what]
        if len(names) == _PAIR:
            return views.side_by_side(names[0], names[1], seed=seed)
        if len(names) == 1 and names[0].lower() in _SPLITS:
            return views.dataset(names[0].lower())
        return views.designs(*names, n=n, seed=seed, fresh=fresh)

    def designs(self, model: str, seed: int = 1, *, fresh: bool = False) -> np.ndarray:
        """One suspect's designs for every condition in the spec.

        The raw array, for when you want to compute something the benchmark does
        not offer. Read from the design cache when it holds them, which is what
        makes looking at a model free. `fresh=True` samples again from the
        checkpoint and writes the result back -- slower, and the only way to
        watch what a model costs to run.

        Args:
            model: Any unambiguous part of a suspect's name, or `"test"` for the
                reference designs the spec scores everything against.
            seed: Sampling seed.
            fresh: Ignore every cache and sample from the model itself.

        Returns:
            An array of designs, one per condition in the spec.
        """
        if model.lower() in _REFERENCE_NAMES:
            # The same vocabulary `show` accepts, so "test" does not mean the
            # reference optimum in one call and nothing in the next.
            return np.asarray(self.evaluator.resolved.ref_designs)

        member = self.resolve(model)
        label = member.label
        if not fresh and (held := self._designs.get((label, seed))) is not None:
            return held

        if not fresh:
            entry = self.store.load(member.key, seed)
            if entry is not None:
                self._designs[(label, seed)] = entry.designs
                self._sample_meta[(label, seed)] = {
                    "sample_seconds": entry.sample_seconds,
                    "model_params": entry.model_params,
                    "machine": entry.machine,
                    "replayed": True,
                }
                return entry.designs

        print(f"  sampling {label} ({self.evaluator.resolved.n_samples} designs) ... ", end="", flush=True)
        generator = member.load()
        generator.seed = seed
        context = self.evaluator.context_for(generator)
        seconds = context.sample_seconds
        print(f"{seconds:.1f}s" if seconds is not None else "done")

        self._designs[(label, seed)] = context.gen_designs
        self._sample_meta[(label, seed)] = {
            "sample_seconds": seconds,
            "model_params": context.model_params,
            "machine": "this machine",
            "replayed": False,
        }
        self.store.store(member.key, seed, context.gen_designs, sample_seconds=seconds, model_params=context.model_params)
        return context.gen_designs

    # ------------------------------------------------------------------
    # The rest of the file
    # ------------------------------------------------------------------

    def instrument(self) -> pd.Series:
        """Which autoencoder the latent columns measure in.

        Printed rather than assumed. A latent metric is only comparable between
        two rows encoded by the same instrument, and the only way a reader can
        check that is if the row says which one it was.

        The active width is read off the loaded encoder's own pruning mask, not
        off the spec. `expected_n_active` is a declaration that nothing
        enforces, and the published metadata disagrees with it on several
        packages -- so the number reported here is the one the columns were
        actually computed in.

        Returns:
            The instrument's identity, or an empty series when the spec pins none.
        """
        pinned = self.evaluator.spec.latent_instrument
        if pinned is None:
            print(f"{self.config.spec} pins no autoencoder, so this problem has no latent columns.")
            return pd.Series(dtype=object, name="latent instrument")
        active = _measured_active_dims(self.evaluator.latent_lvae)
        fields: dict[str, object] = {
            "algo": pinned.algo,
            "config_fingerprint": pinned.config_fingerprint,
            "seed": pinned.seed,
            "measured_n_active": active,
            # The linear control is fitted to the instrument's width on purpose:
            # a PCA subspace of some other size would answer a different
            # question, and "the latent space beat PCA" would be a statement
            # about dimensionality rather than about the manifold.
            "pca_components": active,
            "declared_n_active": pinned.expected_n_active,
            "recon_only_config_fingerprint": pinned.recon_only_config_fingerprint,
            "recon_only_n_active": _measured_active_dims(self.evaluator.latent_recon_lvae),
            "revision": pinned.revision,
        }
        return pd.Series(fields, name="latent instrument")

    def physics(self, passphrase: str | None = None, path: str | Path | None = None) -> pd.DataFrame:
        """The precomputed simulator board, read from the Hub.

        `iog`/`cog`/`fog` cost hours per model, so nobody computes them during a
        session. They are computed once and published into each checkpoint's own
        `metrics.json`, beside the weights they describe -- so this reads the
        same Hub the models themselves came from, and there is no local file to
        go stale or to be forgotten when the package is installed elsewhere.

        A suspect with no published physics comes back as a row of NaN rather
        than an error. That is the honest rendering: the board is a statement
        about which models have been measured, and a model nobody has run the
        simulator on has not been measured. Constructed models are the usual
        case -- they have no checkpoint package to attach metrics to.

        Args:
            passphrase: Only for the legacy sealed board. Given one, this reads
                the encrypted CSV instead of the Hub.
            path: Sealed file to use with `passphrase`.

        Returns:
            The expensive-metric board, indexed by suspect.
        """
        if passphrase is not None:
            return self._sealed_physics(passphrase, path)

        from engiopt.evaluation.physics_board import published_physics

        rows, absent = {}, []
        for member in self.bank:
            algo, fingerprint, seed = _package_of(member.key)
            found = published_physics(self.config.problem_id, algo, fingerprint, seed)
            if found is None:
                absent.append(member.label)
            rows[member.label] = found or {}

        frame = pd.DataFrame(rows).T
        frame.index.name = "suspect"
        columns = [c for c in self.config.expensive_metrics if c in frame.columns]
        frame = frame[columns + [c for c in frame.columns if c not in columns]]

        print(f"Read from the Hub: {len(self.bank) - len(absent)}/{len(self.bank)} suspects have published physics.")
        if absent:
            print(
                f"  no simulator run published for: {', '.join(absent)}. "
                "They are shown as blank rather than dropped -- 'not measured' is a fact about the board, "
                "not a reason to hide a model from it."
            )
        self._note_partial_physics(frame, [member.label for member in self.bank if member.label not in absent])
        return frame

    def _note_partial_physics(self, frame: pd.DataFrame, published: list[str]) -> None:
        """Name any column a *published* row is missing, and say so as a gap in the board.

        The rows are the union of what each package carries, so a package
        published before a column existed reads as NaN in that column and as a
        number everywhere else. That is a different thing from a model nobody
        ran the simulator on, and a bare NaN cannot tell the two apart -- which
        is how "this model failed the physics" gets read off a board where the
        model was simply scored by an earlier version of the scorer.
        """
        gaps: dict[tuple[str, ...], list[str]] = {}
        for column in frame.columns:
            incomplete = tuple(label for label in published if pd.isna(frame.loc[label, column]))
            if incomplete:
                gaps.setdefault(incomplete, []).append(column)
        for labels, columns in gaps.items():
            print(
                f"  [note] {', '.join(columns)} was never published for {', '.join(labels)}, so those cells are "
                "blank. Every other column in those rows was measured on the same run as everybody else's."
            )

    def _sealed_physics(self, passphrase: str, path: str | Path | None) -> pd.DataFrame:
        """The legacy encrypted board, for a session that still uses one.

        A suspect the board does not cover comes back **blank rather than
        refused**: a line-up gains a member before the simulator has been run
        over it, and that is a normal state rather than a corrupt file. Only a
        board covering nobody is an error, since that means the wrong file.

        Raises:
            SealError: If the board is missing, unreadable, or covers none of
                this line-up.
        """
        sealed = Path(path) if path else self.config.sealed_board_path()
        frame = pd.read_csv(io.StringIO(unseal(sealed, passphrase)))

        if "key" not in frame.columns:
            raise SealError(f"{sealed} has no `key` column; it was not written by `build_sealed_board`.")
        by_key = frame.set_index("key")
        missing = [member for member in self.bank if member.key not in by_key.index]
        if len(missing) == len(self.bank):
            raise SealError(
                f"The sealed board covers none of this line-up ({[m.key for m in self.bank]}). "
                "It was sealed against a different problem, or a different spec."
            )
        if missing:
            # Blank rather than refuse. A suspect added after the board was
            # sealed has no physics *yet*, and that is a normal state between a
            # line-up change and the next simulator run -- refusing the whole
            # board would take the eight rows that are ready down with the
            # three that are not. Said out loud, because a blank cell a reader
            # mistakes for a zero is worse than an error.
            print(
                f"  [sealed board] no physics for {[m.label for m in missing]} -- "
                f"added to the line-up after this board was sealed, so their rows are blank. "
                f"Rebuild with `build_sealed_board.py --problem-id {self.config.problem_id}`."
            )

        rows = by_key.reindex([member.key for member in self.bank])
        rows.index = pd.Index(self.bank.labels, name="suspect")
        # `spec` and `source` are provenance for the sealed file, not results.
        return rows.drop(columns=[c for c in ("algo", "problem_id", "spec", "source") if c in rows.columns])

    def rank(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Convert a board of values into per-column ranks, 1 being best.

        Columns with no intrinsic direction, and columns that are constant
        across the line-up, are dropped: ranking them would manufacture an
        ordering the numbers do not support.

        Args:
            frame: A board as `evaluate` returns it.

        Returns:
            The same board as integer ranks.
        """
        ranked = {}
        for column in frame.columns:
            higher_is_better = column_direction(column)
            if higher_is_better is None or _is_constant(frame[column]):
                continue
            ranked[column] = frame[column].rank(ascending=not higher_is_better, method="min")
        out = pd.DataFrame(ranked, index=frame.index).astype("Int64")
        out.index.name = frame.index.name
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _views_object(self) -> Views:
        """The prebuilt figures, built on first use."""
        if self._views is None:
            from engiopt.workshops.idetc26.views import Views

            self._views = Views(self)
        return self._views

    def resolve(self, model: str) -> Any:
        """Resolve a name against the suspects, then against the controls.

        Controls are resolvable everywhere a suspect is -- `case.show`,
        `case.designs`, `case.explain` -- because a calibration standard you are
        told to read the board against but cannot look at is not a standard,
        it is a footnote. What controls are *not* is rankable: `evaluate` still
        resolves only suspects, and puts controls on their own rows.
        """
        try:
            return self.bank.resolve(model)
        except KeyError:
            if len(self.controls):
                return self.controls.resolve(model)
            raise

    def _resolve_models(self, models: str | list[str] | tuple[str, ...] | None) -> list[str]:
        """Which suspects a call meant, in line-up order."""
        if models is None:
            return list(self.bank.labels)
        requested = [models] if isinstance(models, str) else list(models)
        return [self.bank.resolve(name).label for name in requested]

    def _resolve_metrics(self, metrics: str | list[str] | tuple[str, ...] | None) -> list[str]:
        """Expand whatever was asked for into a list of metric names.

        A name is a metric, or a whole line of questioning, and a list may mix
        the two freely. Everything is filtered to what this problem can actually
        compute, with a printed reason for anything dropped.

        Raises:
            KeyError: If a name is neither a metric nor a line of questioning.
        """
        if metrics is None:
            requested = list(self.config.cheap_metrics)
        else:
            names = [metrics] if isinstance(metrics, str) else list(metrics)
            requested = []
            unknown = []
            for name in names:
                if name in METRICS:
                    requested.append(name)
                elif name in FAMILIES:
                    requested.extend(m for m in self.config.metrics if family_of(m) == name)
                else:
                    unknown.append(name)
            if unknown:
                raise KeyError(
                    f"Not a metric or a line of questioning: {unknown}. "
                    f"Lines of questioning: {list(FAMILIES)}. `case.metrics()` lists the columns."
                )
            requested = list(dict.fromkeys(requested))

        available = self.config.available(requested)
        unavailable = self.config.unavailable_metrics()
        for name in requested:
            if name not in available:
                print(f"  [skipped] {name}: {unavailable[name]}")
        if not available:
            print("  Nothing left to ask -- every column requested is unavailable on this problem.")
        return available

    def _board(
        self,
        labels: list[str],
        names: list[str],
        *,
        seed: int = 1,
        n_samples: int | None,
        indices: Any,
        sigma: float | None,
        fresh: bool,
        include_expensive: bool,
    ) -> pd.DataFrame:
        """Score every suspect on every column at one seed."""
        rows = {
            label: self._score(
                label,
                names,
                seed=seed,
                n_samples=n_samples,
                indices=indices,
                sigma=sigma,
                fresh=fresh,
                include_expensive=include_expensive,
            )
            for label in labels
        }
        columns = [column for name in names for column in METRICS[name].columns]
        frame = pd.DataFrame(rows).T
        return frame[[column for column in columns if column in frame.columns]]

    def _score(
        self,
        model: str,
        metrics: list[str],
        *,
        seed: int = 1,
        n_samples: int | None = None,
        indices: Any = None,
        sigma: float | None = None,
        fresh: bool = False,
        include_expensive: bool = False,
    ) -> dict[str, Any]:
        """Score one suspect, reusing anything already computed at this seed.

        A metric is not always one column: `lv_residual` fills `lv_residual_mean`
        and `lv_residual_p90`. The cache is keyed by the columns a metric
        actually emits, so a multi-output metric is recognised as already
        computed instead of being recomputed on every board.

        Expensive columns and truncated boards are never cached, since their
        value depends on how many samples were asked for.
        """
        member = self.resolve(model)
        label = member.label
        reusable = n_samples is None and indices is None and sigma is None and not include_expensive and not fresh
        row = self._rows.setdefault((label, seed), {}) if reusable else {}
        missing = [name for name in metrics if any(column not in row for column in METRICS[name].columns)]
        if not missing:
            return row

        # Hours of optimizer already paid for and published beside the weights.
        # Recomputing them because they were asked for through `evaluate` rather
        # than through `physics` would be the same run, twice, for the same
        # answer.
        if include_expensive and self._replayable(
            seed=seed, n_samples=n_samples, indices=indices, sigma=sigma, fresh=fresh
        ):
            published = self._published_row(label)
            row.update(
                {column: published[column] for name in missing for column in METRICS[name].columns if column in published}
            )
            missing = [name for name in missing if any(column not in row for column in METRICS[name].columns)]
            if not missing:
                return row

        designs = self.designs(label, seed=seed, fresh=fresh)
        meta = self._sample_meta.get((label, seed), {})
        context = self.evaluator.context_from_designs(
            designs,
            sample_seconds=meta.get("sample_seconds"),
            model_params=meta.get("model_params"),
            train_minutes=member.train_minutes,
            n_samples=n_samples,
            indices=indices,
            sigma=sigma,
        )
        row.update(self.evaluator.score_context(context, only=missing, include_expensive=include_expensive))
        return row

    def _published_row(self, label: str) -> dict[str, Any]:
        """This suspect's simulator columns as published on the Hub, or `{}`.

        Read once per suspect and kept, because the answer does not change
        during a session and the alternative is a Hub round trip per board.
        """
        if label not in self._published:
            from engiopt.evaluation.physics_board import published_physics

            algo, fingerprint, seed = _package_of(self.resolve(label).key)
            found = published_physics(self.config.problem_id, algo, fingerprint, seed, spec=self.config.spec)
            self._published[label] = dict(found or {})
        return self._published[label]

    def _replayable(self, *, seed: int, n_samples: int | None, indices: Any, sigma: float | None, fresh: bool) -> bool:
        """Whether a published number answers exactly the question being asked.

        A gap is defined against the designs it was measured on. The published
        one was computed over the whole spec at sampling seed 1, so any call
        that changes which designs are compared -- fewer conditions, a random
        subset, a fresh draw, another seed -- is asking a different question and
        has to pay for it. `sigma` cannot touch a gap, but a board mixing a
        forced bandwidth with replayed physics would still be describing two
        different runs in one table.
        """
        return seed == _PUBLISHED_SAMPLING_SEED and n_samples is None and indices is None and sigma is None and not fresh

    def _with_controls(self, board: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
        """Append rows for models whose answer is already known.

        `collapsed` is one design repeated; `noise_doped` is real optima with
        noise added, which cannot be an improvement; `volume_only` hits the
        budget with material that carries no load. None of them is a suspect and
        none is ever ranked. They say what a column *reads* at a known input,
        which is the difference between a diversity number and a diversity
        number you can interpret.
        """
        if not len(self.controls):
            print("  [note] this problem declares no controls, so there is nothing to calibrate against.")
            return board
        cheap = [name for name in metrics if METRICS[name].cost == "cheap"]
        rows = {member.label: self._score(member.label, cheap) for member in self.controls}
        frame = pd.DataFrame(rows).T
        frame = frame[[column for column in board.columns if column in frame.columns]]
        frame.index = pd.Index([f"[control] {label}" for label in frame.index], name=board.index.name)
        print("  [note] rows marked [control] are known-answer models, not suspects. They are never ranked.")
        return pd.concat([board, frame])

    def _condition_subset(self, n_samples: int | None, *, random_conditions: bool) -> Any:
        """Which of the spec's conditions to score, drawn once for the whole board.

        Returns None when every condition is in play, which keeps the common
        case on the plain path. A random subset is seeded on its own size rather
        than on the clock, so the same call twice is the same subset -- a metric
        that moves between two runs should be telling you about the metric, not
        about which thirty conditions each run happened to draw.
        """
        if not random_conditions:
            return None
        drawn = self.evaluator.condition_subset(n_samples, random=True)
        if drawn is not None:
            total = self.evaluator.resolved.n_samples
            print(f"  [note] scoring {len(drawn)} of {total} conditions, drawn at random and shared by every suspect.")
        return drawn

    def _status_of(self, name: str, unavailable: dict[str, str]) -> str:
        """Whether a column can actually be produced here, and if not why not."""
        if name in unavailable:
            return f"unavailable -- {unavailable[name]}"
        if METRICS[name].cost == "expensive":
            return "needs the simulator -- minutes per design"
        return "available"

    def _print_cli(self, metrics: list[str], *, expensive: bool) -> None:
        """Print the `engiopt.evaluate` command that produces the same answers."""
        parts = [
            "python -m engiopt.evaluate",
            f"--problem-id {self.config.problem_id}",
            f"--spec {self.config.spec}",
            "--generators all",
            f"--metrics {' '.join(metrics)}",
        ]
        if expensive:
            parts.append("--include-expensive")
        print("Same numbers, outside this notebook:\n  " + " \\\n    ".join(parts) + "\n")

    def _price_expensive(self, labels: list[str], metrics: list[str], n_samples: int, *, replayable: bool) -> None:
        """Say what has to be run, and what is being read instead of run.

        Every sample runs one optimization and two simulations, so the cost is
        linear in the suspects that actually need running. Most of them do not:
        the line-up's physics is published beside its weights, and a default
        board replays it. Pricing the whole line-up when nine tenths of it is a
        Hub read taught the wrong lesson about what the simulator costs.
        """
        columns = [column for name in metrics if METRICS[name].cost == "expensive" for column in METRICS[name].columns]
        replayed = (
            [label for label in labels if all(column in self._published_row(label) for column in columns)]
            if replayable
            else []
        )
        if replayed:
            print(
                f"  [note] {len(replayed)} of {len(labels)} suspects have this published beside their weights "
                "and are read from the Hub, not recomputed: " + ", ".join(replayed)
            )

        running = [label for label in labels if label not in replayed]
        if not running:
            print("  Nothing to run: every suspect asked for has already been through the simulator.\n")
            return

        per_sample = _SECONDS_PER_PHYSICS_SAMPLE.get(self.config.problem_id, 3.4)
        total = n_samples * len(running)
        print(
            f"{len(running)} suspects x {n_samples} designs = {total} optimizer runs, "
            f"about {total * per_sample / 60:.0f} min on this machine "
            f"({per_sample:.0f}s per design, measured on this problem). Interrupt the cell to stop it.\n"
        )

    def _note_replayed_costs(self, board: pd.DataFrame, labels: list[str]) -> None:
        """Say so when a cost column was replayed from a cache built elsewhere.

        `gen_seconds` is wall-clock, so it only compares models measured on the
        same machine. Reporting a cached timing without saying whose machine it
        came from would be the kind of undeclared provenance these columns
        exist to catch.
        """
        if "gen_seconds" not in board.columns:
            return
        machines = {
            self._sample_meta[(label, 1)]["machine"]
            for label in labels
            if self._sample_meta.get((label, 1), {}).get("replayed")
        }
        if machines:
            print(
                f"  [note] gen_seconds was measured when the design cache was built ({', '.join(sorted(machines))}), "
                "not on this machine. Pass fresh=True to time the suspects here instead."
            )


_VIEWS = {
    "designs": lambda views, names, **kw: views.designs(*names, n=kw["n"], seed=kw["seed"]),
    "compare": lambda views, names, **kw: views.compare(*names, n=kw["n"], seed=kw["seed"]),
    "conditions": lambda views, names, **kw: views.conditions(
        _one(names, "conditions"), n=max(kw["n"], 6), seed=kw["seed"]
    ),
    "nearest_training": lambda views, names, **kw: views.nearest_training(
        _one(names, "nearest_training"), n=kw["n"], seed=kw["seed"]
    ),
    "space_map": lambda views, names, **kw: views.space_map(*names, space=kw["space"], seed=kw["seed"], color=kw["color"]),
}
"""The named views `show(how=...)` dispatches to.

A table rather than a chain of `if`s, so the error message for an unknown view
can list what there is -- which is the only documentation somebody typing into a
notebook actually reads.
"""


_PAIR = 2
"""Two names means a comparison: two suspects, or a suspect against the data."""

_SCATTER = 3
"""A board plus two of its column names means a scatter of one against the other."""

_REFERENCE_NAMES = frozenset({"test", "reference", "ref"})
"""Names for the real held-out designs the spec scores every suspect against."""

_SPLITS = frozenset({"train", "val", "test"})
"""Dataset splits, which `show` browses rather than treating as suspects.

Named for the split rather than for anything vaguer. "The real data" is two
different sets of designs -- the ones every model was fitted on and the ones it
is being scored against -- and which of them you are looking at is the whole
point of the memorization question later on. A word that blurs them would be
hiding exactly the distinction the session is about.
"""


def _show_board(views: Views, what: tuple[Any, ...]) -> Any:
    """Draw boards: as ranks, or as a scatter of two named columns of one.

    Several boards are joined column-wise and drawn as one, so putting the cheap
    answers beside the simulator's needs no pandas in the notebook -- which is
    the only thing that put an `import pandas` in front of a participant.

    Raises:
        TypeError: If given a board and some other number of arguments.
    """
    if all(isinstance(item, pd.DataFrame) for item in what):
        return views.board(what[0] if len(what) == 1 else pd.concat(what, axis=1))
    if len(what) == _SCATTER:
        return views.scatter(what[0], str(what[1]), str(what[2]))
    raise TypeError(
        "case.show(board) draws ranks, case.show(board, other_board) joins them, and "
        'case.show(board, "mmd", "novelty_ratio") scatters two of its columns.'
    )


def _one(names: list[str], how: str) -> str:
    """The single suspect a one-suspect view needs.

    Raises:
        TypeError: If none was named. `conditions` and `copying` are about one
            model's relationship to its own inputs and to the training set;
            there is no sensible default for "whose".
    """
    if not names:
        raise TypeError(f'how={how!r} is about one suspect -- name it: case.show("diffusion", how={how!r}).')
    return names[0]


_PUBLISHED_SAMPLING_SEED = 1
"""The sampling seed the published physics was computed at.

`physics_board` defaults to it, so a board asked for at any other seed describes
designs nobody has run the simulator on."""

_SECONDS_PER_PHYSICS_SAMPLE = {"beams2d": 3.4, "heatconduction2d": 3.0, "photonics2d": 36.0}
"""Measured per-sample cost of the expensive tier, by problem.

Published figures rather than guesses: beams2d runs one optimization and two
simulations per sample at about 3.4 s on a laptop CPU. Problems without a
measurement fall back to that, which is the right order of magnitude for the 2D
topology problems and stated so it can be corrected.
"""


def column_direction(column: str) -> bool | None:
    """Whether higher is better for a leaderboard *column*, not a metric name.

    The two differ whenever a metric emits several columns, and ranking is done
    per column. A column nothing in the registry claims has no direction, which
    is the honest answer rather than an error: it simply will not be ranked.

    Args:
        column: A board column name.

    Returns:
        True, False, or None when the column has no direction.
    """
    if column in METRICS:
        return METRICS[column].higher_is_better
    for spec in METRICS.select():
        if column in spec.columns:
            return spec.higher_is_better
    return None


def _package_of(key: str) -> tuple[str, str | None, int]:
    """Split a bank key back into the checkpoint package it names.

    Bank keys are `algo#seed[opt=value,...]`, and the Hub addresses a package by
    `(algo, config_fingerprint, seed)`. A key carrying neither `cfg` nor `code`
    names the canonical package -- or no package at all, which is why the caller
    must tolerate a miss.
    """
    head, _, options = key.partition("[")
    algo, _, seed = head.partition("#")
    fingerprint = None
    for option in options.rstrip("]").split(","):
        name, _, value = option.partition("=")
        # `cfg` is a trained checkpoint's hyperparameter fingerprint. `code` is
        # the constructed models' equivalent -- a digest of the mechanism that
        # built them -- and it addresses a package the same way, so both map to
        # the same `cfg_<fp>/seed_<n>/` path on the Hub.
        if name in {"cfg", "code"}:
            fingerprint = value
    return algo, fingerprint, int(seed or 1)


def _measured_active_dims(lvae: Any) -> int | None:
    """Active latent width read off a loaded encoder, or None if unreadable."""
    if lvae is None:
        return None
    from engiopt.lvae.encode import get_active_mask

    try:
        return int(get_active_mask(lvae.encoder).sum())
    except Exception:  # noqa: BLE001 - a width we cannot read must not break the print
        return None


def _is_constant(series: pd.Series) -> bool:
    """Whether a column carries no information to rank on."""
    values = series.dropna()
    return values.empty or not (values != values.iloc[0]).any()


def _store_for(
    config: WorkshopConfig,
    evaluator: Evaluator,
    *,
    artifact_dir: Path,
    cache_dir: str | Path | None,
) -> DesignStore:
    """Build the design cache search path for one case.

    The working directory comes first so a fresh sample somebody paid for is the
    one they get back, then any directory they pointed at, then the caches that
    ship with the checkout and the package.
    """
    from engiopt.workshops.idetc26.designs import PACKAGE_CACHE
    from engiopt.workshops.idetc26.designs import REPO_CACHE

    roots = [artifact_dir / "designs"]
    if cache_dir:
        roots.append(Path(cache_dir))
    roots += [REPO_CACHE, PACKAGE_CACHE]
    return DesignStore(
        config.problem_id,
        spec_version=evaluator.spec.version,
        condition_digest=evaluator.spec.condition_digest,
        roots=tuple(roots),
        write_root=roots[0],
    )
