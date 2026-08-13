"""The challenge itself: look, measure, commit, and then watch it come apart.

The arc is four moves, and the order is the argument:

1. `gallery()` -- rank by eye, before any number exists.
2. `board()` -- the metrics a team can actually afford, and a public commitment.
3. `seed_lottery()` -- the *same* metrics at other seeds. The ranking moves
   before anything has been unsealed, which is why the point cannot be
   dismissed as a trick played with hidden data.
4. `reveal()` -- the withheld cheap columns, then the identities.

`unseal()` adds the physics board when one has been prepared.

Nothing here reimplements evaluation. Every number comes from
`engiopt.evaluation.Evaluator` running the frozen spec, because a workshop that
scored models on a private code path would be teaching about the workshop rather
than about the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import hashlib
import io
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

from engiopt.evaluation import Evaluator
from engiopt.evaluation.context import EvaluationContext
from engiopt.evaluation.registry import METRICS
from engiopt.workshops.idetc26.bank import ModelBank
from engiopt.workshops.idetc26.config import WorkshopConfig
from engiopt.workshops.idetc26.seal import SealError
from engiopt.workshops.idetc26.seal import unseal

if TYPE_CHECKING:
    from matplotlib.figure import Figure

GALLERY_COLUMNS = 4
"""Designs shown per model in the eyeball round.

Small on purpose. Twenty designs would let a team spot mode collapse reliably,
and the fact that four do not is part of what the session says about visual
inspection: it is the metric everyone uses and nobody reports, and its sample
size is always whatever fits on a slide.
"""


class NotCommittedError(RuntimeError):
    """Raised when a reveal is requested before the team has committed a verdict.

    The reveal only teaches anything against a prediction, so it is gated on one
    existing -- not to be coy, but because a team that reads the answer first
    learns that the answer is surprising rather than that their reasoning was.
    """


@dataclass(frozen=True)
class Verdict:
    """A team's public commitment, hashed so it cannot be quietly revised.

    Attributes:
        team: Team name.
        winner: The label the team picked, e.g. `"Model C"`.
        why: One sentence of justification. "I would not ship any of these
            because ___" is an accepted -- and winning -- answer.
        ranking: Optional full ordering, best first.
        eyeball_ranking: The ordering the team committed to before seeing any
            metric, when they recorded one.
        digest: SHA256 over the commitment.
    """

    team: str
    winner: str
    why: str
    ranking: tuple[str, ...] = ()
    eyeball_ranking: tuple[str, ...] = ()
    digest: str = ""

    @classmethod
    def create(cls, **kwargs: Any) -> Verdict:
        """Build a verdict and stamp it with the digest of its own contents."""
        payload = {k: list(v) if isinstance(v, tuple) else v for k, v in kwargs.items()}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return cls(**kwargs, digest=digest)

    def to_json(self) -> str:
        """Serialize the commitment, digest included."""
        return json.dumps(
            {
                "team": self.team,
                "winner": self.winner,
                "why": self.why,
                "ranking": list(self.ranking),
                "eyeball_ranking": list(self.eyeball_ranking),
                "digest": self.digest,
            },
            indent=2,
        )


@dataclass
class Challenge:
    """One team's run through the bank on one problem.

    Attributes:
        config: The problem's workshop configuration.
        evaluator: The scorer, running the frozen spec.
        bank: The anonymized models, in this team's order.
        team: Team name, which fixes the letter permutation.
        artifact_dir: Where verdicts and cached designs are written.
        verdict: The team's commitment, once made.
    """

    config: WorkshopConfig
    evaluator: Evaluator
    bank: ModelBank
    team: str
    artifact_dir: Path
    verdict: Verdict | None = None
    _rows: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict, repr=False)
    _designs: dict[tuple[str, int], np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def open(
        cls,
        problem_id: str,
        *,
        team: str = "",
        artifact_dir: str | Path | None = None,
        device: Any = None,
    ) -> Challenge:
        """Load the problem, assemble the bank, and report what is in it.

        Args:
            problem_id: Which problem this team is working on.
            team: Team name; fixes the per-team letter permutation.
            artifact_dir: Where to write verdicts; defaults to `./idetc26_work`.
            device: Torch device for the trained checkpoints. Defaults to the
                repo-wide choice, which picks up a Colab GPU when there is one.

        Returns:
            The opened challenge, ready for `gallery()`.
        """
        config = WorkshopConfig.load(problem_id)
        evaluator = Evaluator.for_problem(problem_id, spec=config.spec, device=device)
        bank = ModelBank.assemble(config, evaluator.problem, team=team)

        directory = Path(artifact_dir) if artifact_dir else Path.cwd() / "idetc26_work"
        directory.mkdir(parents=True, exist_ok=True)

        print(f"{config.display_name}: {len(bank)} models in the bank -- {', '.join(bank.labels)}")
        print(f"Each is scored on the same {evaluator.resolved.n_samples} conditions from spec {config.spec}.")
        for name, reason in bank.skipped.items():
            print(f"  [skipped] {name}: {reason}")
        for metric, reason in config.unavailable_metrics().items():
            print(f"  [unavailable] {metric}: {reason}")

        return cls(config=config, evaluator=evaluator, bank=bank, team=team, artifact_dir=directory)

    # ------------------------------------------------------------------
    # Sampling and scoring
    # ------------------------------------------------------------------

    def designs(self, label: str, seed: int = 1) -> np.ndarray:
        """Sample one model's designs for the spec's conditions, cached per seed."""
        cached = self._designs.get((label, seed))
        if cached is not None:
            return cached

        member = self.bank[label]
        generator = member.load()
        generator.seed = seed
        designs = self.evaluator.context_for(generator).gen_designs
        self._designs[(label, seed)] = designs
        return designs

    def score(self, label: str, metrics: tuple[str, ...], seed: int = 1) -> dict[str, Any]:
        """Score one model on the given metrics, reusing anything already computed.

        A metric is not always one column: `lv_residual` fills `lv_residual_mean`
        and `lv_residual_p90`. The cache is keyed by the columns a metric
        actually emits, so a multi-output metric is recognised as already
        computed instead of being recomputed on every board.
        """
        row = self._rows.setdefault((label, seed), {})
        missing = [
            name
            for name in self.config.available(metrics)
            if any(column not in row for column in METRICS[name].columns)
        ]
        if not missing:
            return row

        member = self.bank[label]
        generator = member.load()
        generator.seed = seed
        context = self.evaluator.context_for(generator)
        self._designs[(label, seed)] = context.gen_designs
        row.update(self.evaluator.score_context(context, only=missing, include_expensive=False))
        return row

    def board(self, metrics: tuple[str, ...] | None = None, seed: int = 1, *, ranks: bool = False) -> pd.DataFrame:
        """Compute a metric board over the whole bank.

        Args:
            metrics: Metric names to compute; defaults to the config's opening
                set. A name may expand to more than one column.
            seed: Sampling seed every model is drawn at.
            ranks: Return competition ranks (1 = best) instead of raw values.

        Returns:
            A DataFrame indexed by anonymous label.
        """
        names = self.config.available(metrics or self.config.opening_metrics)
        rows = {label: self.score(label, tuple(names), seed=seed) for label in self.bank.labels}
        columns = [column for name in names for column in METRICS[name].columns]
        frame = pd.DataFrame(rows).T[columns]
        frame.index.name = "model"
        return self.rank(frame) if ranks else frame

    def rank(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Convert a board of values into per-column ranks, 1 being best.

        Columns with no intrinsic direction, and columns that are constant
        across the bank, are dropped: ranking them would manufacture an ordering
        the numbers do not support.
        """
        ranked = {}
        for column in frame.columns:
            higher_is_better = _direction_of(column)
            if higher_is_better is None or _is_constant(frame[column]):
                continue
            ranked[column] = frame[column].rank(ascending=not higher_is_better, method="min")
        out = pd.DataFrame(ranked, index=frame.index).astype("Int64")
        out.index.name = "model"
        return out

    def winners(self, frame: pd.DataFrame) -> pd.Series:
        """The rank-1 model in each column. The shortest form of the argument."""
        ranked = self.rank(frame)
        return pd.Series(
            {column: ", ".join(ranked.index[ranked[column] == 1]) for column in ranked.columns},
            name="rank-1 model",
        )

    # ------------------------------------------------------------------
    # Commitment
    # ------------------------------------------------------------------

    def submit(
        self,
        winner: str,
        why: str,
        ranking: list[str] | None = None,
        eyeball_ranking: list[str] | None = None,
    ) -> Verdict:
        """Record the team's pick, write it to disk, and print its digest.

        Args:
            winner: The label the team is backing.
            why: One sentence of justification.
            ranking: Optional full ordering, best first.
            eyeball_ranking: The pre-metrics ordering, if the team recorded one.

        Returns:
            The stamped verdict.

        Raises:
            KeyError: If `winner` is not a label in this bank.
        """
        self.bank[winner]
        self.verdict = Verdict.create(
            team=self.team or "anonymous",
            winner=winner,
            why=why,
            ranking=tuple(ranking or ()),
            eyeball_ranking=tuple(eyeball_ranking or ()),
        )
        path = self.artifact_dir / "verdict.json"
        path.write_text(self.verdict.to_json())

        print(f"Committed: {winner} -- {why}")
        print(f"SHA256 {self.verdict.digest}")
        print(f"Written to {path}. Read the digest out when your team presents.")
        return self.verdict

    def _require_verdict(self) -> Verdict:
        """Return the verdict, or explain why the reveal is gated.

        Raises:
            NotCommittedError: If nothing has been committed yet.
        """
        if self.verdict is None:
            raise NotCommittedError(
                "Commit first: ch.submit(winner=..., why=...). The reveal is only worth "
                "anything against a prediction, and 'I would not ship any of these because ___' counts."
            )
        return self.verdict

    # ------------------------------------------------------------------
    # The reveal ladder
    # ------------------------------------------------------------------

    def seed_lottery(self, seeds: tuple[int, ...] | None = None) -> pd.DataFrame:
        """Re-rank the bank on the *same* metrics at several sampling seeds.

        Nothing is unsealed and no new column appears: this is the board the
        team already has, drawn again. If the ranking moves here, every ranking
        reported from a single seed anywhere is suspect, and that includes most
        published ones.

        Returns:
            Ranks indexed by label, columns `("<metric>", "seed <n>")`.
        """
        self._require_verdict()
        chosen = seeds or self.config.reveal_seeds
        frames = {seed: self.rank(self.board(seed=seed)) for seed in chosen}
        wide = pd.concat(frames, axis=1)
        wide.columns = pd.MultiIndex.from_tuples([(metric, f"seed {seed}") for seed, metric in wide.columns])
        return wide.sort_index(axis=1)

    def withheld(self, seed: int = 1) -> pd.DataFrame:
        """The cheap columns the team was not given. Still no simulator involved.

        That is the point they make: these were affordable the whole time. A
        benchmark reports what it chose to report, and the choice is an argument
        nobody wrote down.
        """
        self._require_verdict()
        if not self.config.withheld_metrics:
            return pd.DataFrame()
        return self.board(metrics=self.config.withheld_metrics, seed=seed)

    def manifold(self, seed: int = 1) -> pd.DataFrame:
        """The same questions, asked in a learned latent space instead of in pixels.

        Costs what the pixel columns cost -- an encode and a decode -- so this is
        not the expensive tier arriving early. What changes is the space the
        distance is measured in, and the argument of the segment is that the
        space is a modelling choice nobody in a results table declares.

        Returns:
            A board of the configured manifold columns, or an empty frame when
            this problem's spec pins no latent instrument.
        """
        self._require_verdict()
        if not self.config.manifold_metrics:
            return pd.DataFrame()
        if not self.config.has_latent_instrument():
            print(
                f"{self.config.spec} pins no latent instrument, so the manifold columns cannot be computed "
                "here. That is a property of the spec, not of the models: someone has to train and pin an "
                "autoencoder for a problem before anyone can report a latent metric on it."
            )
            return pd.DataFrame()
        return self.board(metrics=self.config.manifold_metrics, seed=seed)

    def instrument(self) -> pd.Series:
        """Which autoencoder the manifold columns were measured in.

        Printed rather than assumed. A latent metric is only comparable between
        two rows that were encoded by the same instrument, and the only way a
        reader can check that is if the row says which one it was.
        """
        pinned = self.evaluator.spec.latent_instrument
        if pinned is None:
            return pd.Series(dtype=object, name="latent instrument")
        fields = {
            "algo": pinned.algo,
            "config_fingerprint": pinned.config_fingerprint,
            "seed": pinned.seed,
            "expected_n_active": pinned.expected_n_active,
            "recon_only_config_fingerprint": pinned.recon_only_config_fingerprint,
            "revision": pinned.revision,
        }
        return pd.Series(fields, name="latent instrument")

    def reference_row(self, metrics: tuple[str, ...] | None = None, seed: int = 1) -> pd.DataFrame:
        """Score the calibration instruments -- what a metric reads at a known input.

        These are never ranked and never in the bank. A collapsed model tells you
        what a diversity column reads on one design repeated fifty times; a
        noise-doped one tells you what it reads on real optima plus noise. That
        is a scale bar under the board, and without one a diversity number is
        just a number.

        Args:
            metrics: Columns to compute; defaults to the opening set.
            seed: Sampling seed.

        Returns:
            A frame indexed by instrument name, or empty if none are configured.
        """
        from engiopt.baselines import REFERENCE_INSTRUMENTS

        names = tuple(entry["algo"] for entry in self.config.reference_instruments)
        if not names:
            return pd.DataFrame()

        columns = self.config.available(metrics or self.config.opening_metrics)
        rows = {}
        for name in names:
            factory = REFERENCE_INSTRUMENTS.get(name)
            if factory is None:
                continue
            generator = factory.from_problem(
                self.evaluator.problem,
                problem_id=self.config.problem_id,
                seed=seed,
            )
            context = self.evaluator.context_for(generator)
            rows[name] = self.evaluator.score_context(context, only=columns, include_expensive=False)
        frame = pd.DataFrame(rows).T
        frame.index.name = "reference instrument"
        return frame

    def run_physics(
        self,
        labels: list[str] | None = None,
        n_samples: int = 3,
        seed: int = 1,
        *,
        confirm: bool = False,
    ) -> pd.DataFrame:
        """Compute the expensive columns yourself, on as many samples as you can afford.

        The sealed board exists because nobody can run the simulator during a
        session. This is the other half of that lesson: run it on a handful of
        designs and watch what it costs, so "expensive" stops being a word in a
        table caption and becomes a number you waited for.

        Every sample runs one optimization and two simulations, so cost is linear
        in `n_samples * len(labels)` and the estimate below is honest rather than
        reassuring. The sealed board is computed at the spec's full sample count;
        a board you compute here on 3 samples is *not* comparable to it, and
        seeing how far a 3-sample estimate lands from the sealed 50-sample one is
        the most useful thing this method does.

        Args:
            labels: Models to score; defaults to the whole bank.
            n_samples: Conditions per model. Kept small on purpose.
            seed: Sampling seed the designs are drawn at.
            confirm: Pass True to actually run. Without it the method prints a
                cost estimate and returns an empty frame, because a cell that
                silently starts a twenty-minute job in a workshop is a trap.

        Returns:
            A frame of the expensive columns, indexed by label. Empty when
            `confirm` is False.
        """
        import time

        chosen = labels or list(self.bank.labels)
        columns = self.config.available(self.config.expensive_metrics)
        total = n_samples * len(chosen)
        seconds = total * self._seconds_per_physics_sample()
        print(
            f"{len(chosen)} models x {n_samples} samples = {total} optimizer runs, "
            f"about {seconds / 60:.0f} min on this machine "
            f"({self._seconds_per_physics_sample():.0f}s per sample, measured on this problem).\n"
            "Sampling is on top of that for any model you have not already scored -- "
            "the diffusion model alone takes minutes to draw its designs."
        )
        if not confirm:
            print("Nothing has run. Re-run with confirm=True when you are ready to wait.")
            return pd.DataFrame()

        rows = {}
        measured: list[float] = []
        resolved = self.evaluator.resolved
        for label in chosen:
            started = time.perf_counter()
            # `designs()` is cached per (label, seed), so a team that already
            # computed a board pays nothing to sample again here. Going through
            # `context_for` would re-sample all 50 designs before truncating,
            # which on the diffusion model costs minutes to then optimize two.
            designs = self.designs(label, seed=seed)
            # Truncate every per-sample array together, so sample i still means
            # the same condition in all of them.
            trimmed = EvaluationContext(
                problem=self.evaluator.problem,
                problem_id=self.config.problem_id,
                gen_designs=designs[:n_samples],
                ref_designs=resolved.ref_designs[:n_samples],
                conditions=resolved.conditions.select(range(n_samples)) if resolved.conditions is not None else None,
                sigma=self.evaluator.spec.sigma,
                volfrac_tol=self.evaluator.spec.volfrac_tol,
                volume_condition=self.evaluator.spec.volume_condition,
                objective_weights=self.evaluator.spec.objective_weights,
                objective_weight_condition=self.evaluator.spec.objective_weight_condition,
            )
            rows[label] = self.evaluator.score_context(trimmed, only=columns, include_expensive=True)
            elapsed = time.perf_counter() - started
            measured.append(elapsed)
            print(f"  {label}: {elapsed:5.1f}s  ({elapsed / n_samples:.1f}s per sample)")

        actual = sum(measured)
        per_sample = actual / max(total, 1)
        print(
            f"\nActually took {actual / 60:.1f} min ({per_sample:.1f}s per sample) against an estimate of "
            f"{seconds / 60:.1f} min. The published per-sample figure is a starting guess; "
            "what you just measured is the number for this machine."
        )

        frame = pd.DataFrame(rows).T[columns]
        frame.index.name = "model"
        return frame

    def _seconds_per_physics_sample(self) -> float:
        """Measured per-sample cost of the expensive tier on this problem.

        A published figure rather than a guess: beams2d runs one optimization and
        two simulations per sample at about 3.4 s on a laptop CPU. Problems
        without a measurement fall back to that, which is the right order of
        magnitude for the 2D topology problems and stated so it can be corrected.
        """
        return {"beams2d": 3.4, "heatconduction2d": 3.0, "photonics2d": 36.0}.get(self.config.problem_id, 3.4)

    def unseal_physics(self, passphrase: str, path: str | Path | None = None) -> pd.DataFrame:
        """Open the sealed physics board.

        Args:
            passphrase: The phrase announced in the room.
            path: Sealed file; defaults to the problem's board under
                `workshops/idetc26/sealed/`.

        Returns:
            The expensive-metric board, indexed by anonymous label.

        Raises:
            SealError: If the board is missing, or was sealed for another bank.
        """
        self._require_verdict()
        sealed = Path(path) if path else _sealed_path(self.config.problem_id)
        frame = pd.read_csv(io.StringIO(unseal(sealed, passphrase)))

        if "key" not in frame.columns:
            raise SealError(f"{sealed} has no `key` column; it was not written by `build_sealed_board`.")
        by_key = frame.set_index("key")
        missing = [m.key for m in self.bank if m.key not in by_key.index]
        if missing:
            raise SealError(f"The sealed board does not cover {missing}. It was sealed against a different bank.")

        rows = by_key.loc[[m.key for m in self.bank]]
        rows.index = pd.Index(self.bank.labels, name="model")
        return rows.drop(columns=[c for c in ("algo", "problem_id") if c in rows.columns])

    def identities(self) -> pd.DataFrame:
        """Who each model actually was, and what it was built to exploit."""
        verdict = self._require_verdict()
        frame = pd.DataFrame(
            [
                {
                    "model": member.label,
                    "identity": member.identity,
                    "kind": member.kind,
                    "built to win": ", ".join(member.wins) or "--",
                    "built to lose": ", ".join(member.loses) or "--",
                    "what it does": member.summary,
                }
                for member in self.bank
            ]
        ).set_index("model")

        picked = self.bank[verdict.winner]
        print(f"You picked {verdict.winner}: {picked.identity} -- {picked.summary}")
        if picked.kind == "baseline":
            print(
                "That is a dataset-fitted baseline, not a trained network. Habibi et al. (J. Mech. Des. 2026) "
                "found exactly this on a topology-optimization warm-start task: at limited data sizes the "
                "simple method wins. It is a result, not a mistake in the bank."
            )
        return frame

    # ------------------------------------------------------------------
    # Looking at designs
    # ------------------------------------------------------------------

    def gallery(self, n_per_model: int = GALLERY_COLUMNS, seed: int = 1) -> Figure:
        """Show a few designs from every model, labelled only by letter.

        Args:
            n_per_model: Designs shown per model.
            seed: Sampling seed.

        Returns:
            The figure, so a notebook can save or resize it.
        """
        import matplotlib.pyplot as plt

        labels = self.bank.labels
        figure, axes = plt.subplots(len(labels), n_per_model, figsize=(2.0 * n_per_model, 1.6 * len(labels)), squeeze=False)
        for row, label in enumerate(labels):
            designs = self.designs(label, seed=seed)
            for column in range(n_per_model):
                axis = axes[row][column]
                axis.imshow(np.asarray(designs[column]).squeeze(), cmap="gray_r", vmin=0, vmax=1)
                axis.set_xticks([])
                axis.set_yticks([])
                if column == 0:
                    axis.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=11)
        figure.suptitle(f"{self.config.display_name}: the bank, unlabelled", fontsize=13)
        figure.tight_layout()
        return figure

    def compare(self, labels: list[str], n: int = 4, seed: int = 1) -> Figure:
        """Show two or three models side by side against the reference designs."""
        import matplotlib.pyplot as plt

        rows = [
            ("reference", np.asarray(self.evaluator.resolved.ref_designs)),
            *[(l, self.designs(l, seed)) for l in labels],
        ]
        figure, axes = plt.subplots(len(rows), n, figsize=(2.0 * n, 1.6 * len(rows)), squeeze=False)
        for row, (name, designs) in enumerate(rows):
            for column in range(n):
                axis = axes[row][column]
                axis.imshow(np.asarray(designs[column]).squeeze(), cmap="gray_r", vmin=0, vmax=1)
                axis.set_xticks([])
                axis.set_yticks([])
                if column == 0:
                    axis.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=11)
        figure.suptitle("Same conditions, every row", fontsize=13)
        figure.tight_layout()
        return figure


def _direction_of(column: str) -> bool | None:
    """Whether higher is better for a leaderboard *column*, not a metric name.

    The two differ whenever a metric emits several columns, and ranking is done
    per column. A column nothing in the registry claims has no direction, which
    is the honest answer rather than an error: it simply will not be ranked.
    """
    if column in METRICS:
        return METRICS[column].higher_is_better
    for spec in METRICS.select():
        if column in spec.columns:
            return spec.higher_is_better
    return None


def _is_constant(series: pd.Series) -> bool:
    """Whether a column carries no information to rank on."""
    values = series.dropna()
    return values.empty or not (values != values.iloc[0]).any()


def _sealed_path(problem_id: str) -> Path:
    """Default location of a problem's sealed board, anchored on this file."""
    return Path(__file__).resolve().parents[3] / "workshops" / "idetc26" / "sealed" / f"{problem_id}_physics.csv.enc"
