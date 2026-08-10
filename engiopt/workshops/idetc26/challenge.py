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
        """Score one model on the given columns, reusing anything already computed."""
        row = self._rows.setdefault((label, seed), {})
        missing = [name for name in self.config.available(metrics) if name not in row]
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
            metrics: Columns to compute; defaults to the config's opening set.
            seed: Sampling seed every model is drawn at.
            ranks: Return competition ranks (1 = best) instead of raw values.

        Returns:
            A DataFrame indexed by anonymous label.
        """
        columns = self.config.available(metrics or self.config.opening_metrics)
        rows = {label: self.score(label, tuple(columns), seed=seed) for label in self.bank.labels}
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
            higher_is_better = METRICS[column].higher_is_better
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


def _is_constant(series: pd.Series) -> bool:
    """Whether a column carries no information to rank on."""
    values = series.dropna()
    return values.empty or not (values != values.iloc[0]).any()


def _sealed_path(problem_id: str) -> Path:
    """Default location of a problem's sealed board, anchored on this file."""
    return Path(__file__).resolve().parents[3] / "workshops" / "idetc26" / "sealed" / f"{problem_id}_physics.csv.enc"
