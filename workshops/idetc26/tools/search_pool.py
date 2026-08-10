"""Search the published checkpoint pool for the bank with the most metric disagreement.

The workshop only works if the answer is genuinely unknown, and that is a
property of the *bank*, not of the notebook. So rather than pick models and hope
they disagree, this scores everything affordable in the pool and then searches
for the subset where the cheap metrics most stubbornly refuse to agree.

Two stages, because they have very different costs:

    python workshops/idetc26/tools/search_pool.py score   # hours, restartable
    python workshops/idetc26/tools/search_pool.py select  # seconds, pure pandas

`score` appends one row per package to `pool_cheap.csv` and skips anything
already there, so it can be killed and resumed. `select` reads that file and
reports the highest-disagreement banks it can find.

Sampling cost varies by four orders of magnitude across families -- a cGAN
samples 50 designs in 0.09 s, a conditional diffusion model takes 450 s -- so
the slow families are capped by `--max-per-algo` rather than swept.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import time

import pandas as pd

from engiopt.evaluation import Evaluator
from engiopt.evaluation.registry import METRICS

BUILD_DIR = Path(__file__).resolve().parent
POOL_CSV = BUILD_DIR / "pool_cheap.csv"

CHEAP_METRICS = ["mmd", "pca_mmd", "dpp", "pixel_vendi", "viol", "cond_err", "novelty", "gen_seconds", "params"]
"""Every simulation-free column. The search runs on these because they are what
a participant can actually compute in the session."""

REPOS = {
    "cgan_cnn_2d": "engiopt-cgan-cnn-2d",
    "gan_cnn_2d": "engiopt-gan-cnn-2d",
    "vqgan": "engiopt-vqgan",
    "diffusion_2d_cond": "engiopt-diffusion-2d-cond",
    "constrained_plvae_2d": "engiopt-constrained-plvae-2d",
}

SLOW_ALGOS = {"vqgan", "diffusion_2d_cond"}
"""Families whose sampling cost makes an exhaustive sweep unaffordable."""

HYPERPARAMETER_COLUMNS = ("n_epochs", "latent_dim", "lr_gen", "lr", "batch_size")
"""Run-config keys recorded alongside the scores, for reading the board by hand.

The union across families, not the intersection: `diffusion_2d_cond` has `lr`
where the GANs have `lr_gen`, and a row missing a column must write an empty
cell rather than shift the ones after it."""

CSV_COLUMNS = ("key", "algo", "config_fingerprint", "seed", *HYPERPARAMETER_COLUMNS, *CHEAP_METRICS)
"""The fixed output schema. Every appended row is reindexed onto it."""

HF_ENTITY = "IDEALLab"


@dataclass(frozen=True)
class PoolEntry:
    """One published checkpoint package.

    Attributes:
        algo: Model family, which selects the adapter and the repo.
        config_fingerprint: Hyperparameter fingerprint, or None for a package
            stored at the canonical default-config path.
        seed: Training seed.
        run_config: The hyperparameters the package records.
    """

    algo: str
    config_fingerprint: str | None
    seed: int
    run_config: dict

    @property
    def key(self) -> str:
        """Stable identifier, also the leaderboard row key."""
        return f"{self.algo}/{self.config_fingerprint or 'default'}/seed_{self.seed}"


def discover(problem_id: str, *, max_per_algo: dict[str, int] | None = None) -> list[PoolEntry]:
    """List every published package for a problem, reading each one's run config.

    Args:
        problem_id: Which problem's packages to enumerate.
        max_per_algo: Per-family caps, for families too slow to sweep.

    Returns:
        Pool entries, ordered so that a truncated run still covers every family.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub import HfApi

    api = HfApi()
    caps = max_per_algo or {}
    by_algo: dict[str, list[PoolEntry]] = {}

    for algo, repo in REPOS.items():
        repo_id = f"{HF_ENTITY}/{repo}"
        try:
            files = [f for f in api.list_repo_files(repo_id) if f.startswith(f"{problem_id}/")]
        except Exception as exc:  # noqa: BLE001 - a missing repo is a gap in the pool, not a crash
            print(f"  [skip] {repo_id}: {exc}")
            continue

        entries = []
        for package in sorted({"/".join(f.split("/")[:-1]) for f in files}):
            parts = package.split("/")
            fingerprint = next((p.removeprefix("cfg_") for p in parts if p.startswith("cfg_")), None)
            seed_part = next((p for p in parts if p.startswith("seed_")), None)
            if seed_part is None:
                continue
            try:
                config = json.loads(Path(hf_hub_download(repo_id, f"{package}/run_config.json")).read_text())
            except Exception as exc:  # noqa: BLE001 - a package without a config cannot be rebuilt
                print(f"  [skip] {package}: no run_config ({type(exc).__name__})")
                continue
            entries.append(PoolEntry(algo, fingerprint, int(seed_part.removeprefix("seed_")), config))

        cap = caps.get(algo)
        by_algo[algo] = entries[:cap] if cap else entries
        print(f"  {algo:24s} {len(by_algo[algo]):3d} packages" + (f" (capped from {len(entries)})" if cap else ""))

    # Round-robin, so killing a long run still leaves every family represented.
    return [e for e in itertools.chain.from_iterable(itertools.zip_longest(*by_algo.values())) if e is not None]


def score(problem_id: str, spec: str, entries: list[PoolEntry], *, output: Path) -> None:
    """Score each entry on the cheap metrics, appending to a restartable CSV."""
    from engiopt.utils.all_generators import BUILTIN_GENERATORS

    evaluator = Evaluator.for_problem(problem_id, spec=spec)
    done = set(pd.read_csv(output)["key"]) if output.exists() else set()
    print(f"{len(entries)} packages, {len(done)} already scored.")

    for index, entry in enumerate(entries, start=1):
        if entry.key in done:
            continue
        started = time.perf_counter()
        try:
            generator = BUILTIN_GENERATORS[entry.algo].from_pretrained(
                evaluator.problem,
                problem_id=problem_id,
                seed=entry.seed,
                model_source="hf",
                config_fingerprint=entry.config_fingerprint,
            )
            row = evaluator.score(generator, only=CHEAP_METRICS)
        except Exception as exc:  # noqa: BLE001 - one unloadable package must not end a multi-hour sweep
            print(f"  [{index}/{len(entries)}] {entry.key}: FAILED {type(exc).__name__}: {str(exc)[:120]}")
            continue

        record = {
            "key": entry.key,
            "algo": entry.algo,
            "config_fingerprint": entry.config_fingerprint or "",
            "seed": entry.seed,
            **{k: entry.run_config.get(k) for k in HYPERPARAMETER_COLUMNS},
            **{m: row.get(m) for m in CHEAP_METRICS},
        }
        # Reindex to the fixed schema before appending. Families do not share a
        # hyperparameter vocabulary -- cgan_cnn_2d has `lr_gen`, diffusion has
        # `lr` -- so a per-row frame has per-row columns, and `mode="a"` writes
        # values positionally against whatever header landed first. That silently
        # shifts the metric columns for every family whose config keys differ
        # from the first row's, which is a corrupted board that still parses.
        frame = pd.DataFrame([record]).reindex(columns=CSV_COLUMNS)
        frame.to_csv(output, mode="a", header=not output.exists(), index=False)
        print(f"  [{index}/{len(entries)}] {entry.key}: {time.perf_counter() - started:5.1f}s  mmd={row.get('mmd'):.4f}")


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------


def rank_frame(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Per-metric competition ranks, 1 = best, using each metric's declared direction."""
    return pd.DataFrame(
        {m: frame[m].rank(ascending=not METRICS[m].higher_is_better, method="min") for m in metrics},
        index=frame.index,
    )


def disagreement(frame: pd.DataFrame, metrics: list[str]) -> dict[str, float]:
    """How badly a candidate bank's metrics disagree about the winner.

    Three numbers, because they say different things:

    - `distinct_winners`: how many different models come first somewhere. This
        is the one participants experience directly.
    - `mean_tau`: average Kendall rank correlation between metric pairs. Low
        means the columns are ordering the bank differently all the way down,
        not just swapping the top slot.
    - `spread`: mean rank range per model. High means no model is consistently
        anywhere, which is what makes a defensible pick impossible.
    """
    from scipy.stats import kendalltau

    ranks = rank_frame(frame, metrics)
    winners = {ranks[m].idxmin() for m in metrics}

    taus = [
        kendalltau(ranks[a], ranks[b]).statistic
        for a, b in itertools.combinations(metrics, 2)
        if ranks[a].notna().all() and ranks[b].notna().all()
    ]
    finite = [t for t in taus if pd.notna(t)]

    return {
        "distinct_winners": float(len(winners)),
        "mean_tau": float(sum(finite) / len(finite)) if finite else float("nan"),
        "spread": float((ranks.max(axis=1) - ranks.min(axis=1)).mean()),
    }


def select(pool: pd.DataFrame, *, size: int, metrics: list[str], trials: int, stratified: bool = True) -> pd.DataFrame:
    """Search candidate banks and return them ranked by disagreement.

    Args:
        pool: The scored pool, indexed by key.
        size: How many models the bank holds.
        metrics: Columns the disagreement is measured over.
        trials: Random candidate banks to try.
        stratified: Guarantee every family is represented before filling the
            remaining slots. A bank of one architecture's hyperparameter variants
            would disagree for reasons nobody wants to teach, and a bank that
            silently drops a family loses the comparison that motivates the
            session.

    Returns:
        Candidate banks, best first, with their disagreement scores.

    Raises:
        ValueError: If the pool cannot supply a bank of the requested size.
    """
    import numpy as np

    usable = pool.dropna(subset=metrics)
    if len(usable) < size:
        raise ValueError(f"Pool has {len(usable)} scorable packages, fewer than the requested bank size {size}.")

    rng = np.random.default_rng(0)
    families = list(usable.groupby("algo"))
    seen: set[tuple[str, ...]] = set()
    results = []

    for _ in range(trials):
        state = lambda: int(rng.integers(1 << 30))  # noqa: E731 - a name for `rng.integers`, not a function worth defining
        if stratified:
            # One from each family first, then fill the rest from the whole pool.
            # With more families than slots, take a random subset of families --
            # which is why this cannot just concatenate one row per group.
            chosen = families if size >= len(families) else [families[i] for i in rng.permutation(len(families))[:size]]
            picked = pd.concat([group.sample(1, random_state=state()) for _, group in chosen])
            remaining = size - len(picked)
            if remaining > 0:
                rest = usable.drop(index=picked.index)
                picked = pd.concat([picked, rest.sample(min(remaining, len(rest)), random_state=state())])
            candidate = picked
        else:
            candidate = usable.sample(size, random_state=state())

        names = tuple(sorted(candidate.index))
        if names in seen:
            continue
        seen.add(names)
        results.append({"bank": " | ".join(names), "families": candidate["algo"].nunique(), **disagreement(candidate, metrics)})

    if not results:
        raise ValueError(f"No candidate banks found in {trials} trials; the pool may be too small.")

    frame = pd.DataFrame(results)
    # Distinct winners is what the room actually experiences; mean_tau breaks
    # ties by how deep the disagreement runs below the top slot.
    return frame.sort_values(["distinct_winners", "mean_tau"], ascending=[False, True]).reset_index(drop=True)


def main() -> None:
    """Run the requested stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["score", "select"])
    parser.add_argument("--problem-id", default="beams2d")
    parser.add_argument("--spec", default="beams2d/v1")
    parser.add_argument("--output", type=Path, default=POOL_CSV)
    parser.add_argument("--max-slow", type=int, default=4, help="Cap on packages per slow-sampling family.")
    parser.add_argument("--size", type=int, default=8, help="Bank size, for `select`.")
    parser.add_argument("--trials", type=int, default=4000)
    parser.add_argument("--metrics", nargs="*", default=None)
    parser.add_argument("--any-algo", action="store_true", help="Allow several models from the same family.")
    args = parser.parse_args()

    if args.stage == "score":
        entries = discover(args.problem_id, max_per_algo=dict.fromkeys(SLOW_ALGOS, args.max_slow))
        score(args.problem_id, args.spec, entries, output=args.output)
        return

    pool = pd.read_csv(args.output).set_index("key")
    metrics = args.metrics or [m for m in CHEAP_METRICS if m in pool.columns and pool[m].notna().any()]
    print(f"Pool: {len(pool)} packages across {pool['algo'].nunique()} families. Metrics: {metrics}\n")

    banks = select(pool, size=args.size, metrics=metrics, trials=args.trials, stratified=not args.any_algo)
    pd.set_option("display.max_colwidth", 200)
    print(banks.head(10).to_string())


if __name__ == "__main__":
    main()
