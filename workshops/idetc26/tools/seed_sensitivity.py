"""How much of the line-up's ranking is a property of the training seed?

The notebook's closing section says this is the question it cannot answer, and
the reason given is that every column came from one checkpoint each. That reason
is now out of date: the four trained generative families in the bank each have
**ten training seeds** published at the banked configuration, and every one of
them carries a full `metrics.json` evaluated against `beams2d/v1` -- the same
spec the session scores against.

So this costs a Hub read and nothing else. No sampling, no simulator, no
dataset: the numbers were paid for once by the pool sweep and are read back
beside the weights they describe.

    python workshops/idetc26/tools/seed_sensitivity.py

Two figures, because there are two different questions:

- `seed_spread.png` -- every seed of every family on every column. The thing to
  look for is a family whose own seeds span more than the gap between families,
  because that is a ranking of the random seed.
- `seed_ranks.png` -- the same data as ranks. Ten independent line-ups, one per
  seed, each ranked as if it were the only board you had.

**Seeds are paired by index, not by anything physical.** Seed 3 of the cGAN and
seed 3 of the VQGAN are unrelated runs; pairing them makes one valid joint draw
of a line-up, and ten of those is ten line-ups somebody could have published.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

TOOLS_DIR = Path(__file__).resolve().parent
CACHE_CSV = TOOLS_DIR / "seed_sensitivity.csv"

SEEDS = range(1, 11)

MIN_SEEDS = 2
"""Below this a family has no seed evidence at all, and saying so is the finding."""

INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"
SEED_MARK = "#8a8985"
"""Undifferentiated ink for the nine seeds nobody chose. Deliberately neutral:
identity here is carried by the y position, so a hue would only add noise."""

BANKED = "#2a78d6"
"""The one seed the workshop actually banks. The only accent on the figure."""

CHEAP = ("mmd", "viol", "cond_err")
PHYSICS = ("iog", "cog", "fog")

LOG_COLUMNS = {"mmd", "cond_err"}
"""Columns spanning orders of magnitude across seeds. `viol` is a rate on [0, 1]
and the gaps go negative, so neither takes a log."""

QUESTION = {
    "mmd": "does it look like real data?",
    "viol": "does it obey the budget?",
    "cond_err": "did it answer the brief?",
    "iog": "how good does it start?",
    "cog": "what does the whole path cost?",
    "fog": "how good does it finish?",
}


def published(problem_id: str, bank: list[dict]) -> pd.DataFrame:
    """Every published seed of every bank member, as one row per checkpoint.

    A member with no package at a seed is absent rather than NaN: the question
    is what somebody could have drawn, and they could not have drawn a
    checkpoint that was never trained.
    """
    from huggingface_hub import hf_hub_download

    rows: list[dict] = []
    for entry in bank:
        algo, fingerprint = entry["algo"], entry.get("config_fingerprint")
        repo = f"IDEALLab/engiopt-{algo.replace('_', '-')}"
        path = f"{problem_id}/" + (f"cfg_{fingerprint}/" if fingerprint else "")
        for seed in SEEDS:
            try:
                local = hf_hub_download(repo, f"{path}seed_{seed}/metrics.json")
            except Exception:  # noqa: BLE001, S112 - an unpublished seed is the normal case, not an error
                continue
            with Path(local).open() as handle:
                payload = json.load(handle)
            metrics = payload["metrics"] if isinstance(payload.get("metrics"), dict) else payload
            rows.append({**metrics, "algo": algo, "cfg": fingerprint, "seed": seed, "banked": seed == entry["seed"]})
    return pd.DataFrame(rows)


def _panel(ax: plt.Axes, frame: pd.DataFrame, column: str, families: list[str]) -> None:
    """One column, every family, one dot per seed."""
    for row, algo in enumerate(families):
        seen = frame[frame.algo == algo].dropna(subset=[column])
        if seen.empty:
            ax.text(
                0.5,
                row,
                "not published",
                transform=ax.get_yaxis_transform(),
                ha="center",
                va="center",
                fontsize=8,
                color=MUTED,
                style="italic",
            )
            continue
        values = seen[column].to_numpy()
        # The range bar first, so the dots sit on top of it.
        ax.plot([values.min(), values.max()], [row, row], color=SEED_MARK, alpha=0.25, lw=6, solid_capstyle="round")
        others, banked = seen[~seen.banked], seen[seen.banked]
        # Seeds cluster hard on the well-behaved columns; without this the ten
        # dots read as three and the spread looks smaller than it is.
        jitter = np.linspace(-0.17, 0.17, len(others))
        ax.scatter(others[column], row + jitter, s=40, color=SEED_MARK, zorder=3, edgecolor=SURFACE, lw=1.2)
        ax.scatter(banked[column], [row] * len(banked), s=86, color=BANKED, zorder=4, edgecolor=SURFACE, lw=1.6)

    if column in LOG_COLUMNS:
        ax.set_xscale("log")
    elif column in PHYSICS:
        # The gaps go negative and span ten decades; symlog is the only scale
        # that shows both a -80 and a 9.6e9 without hiding one of them.
        ax.set_xscale("symlog", linthresh=1.0)
        ax.xaxis.set_major_locator(mpl.ticker.SymmetricalLogLocator(base=100.0, linthresh=1.0))
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([""] * len(families))
    ax.set_ylim(-0.6, len(families) - 0.4)
    ax.invert_yaxis()
    ax.set_title(f"{column}\n{QUESTION[column]}", fontsize=9.5, color=INK, loc="left", pad=8)
    ax.grid(axis="x", color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d3")
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)


def spread_figure(frame: pd.DataFrame, families: list[str], out: Path) -> None:
    """Every seed of every family, on the six columns that carry the argument."""
    columns = [c for c in (*CHEAP, *PHYSICS) if c in frame]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 6.6), facecolor=SURFACE)
    for ax, column in zip(axes.ravel(), columns):
        _panel(ax, frame, column, families)
    for row in range(2):
        axes[row, 0].set_yticklabels(families, fontsize=9, color=INK)
    for ax in axes.ravel()[len(columns) :]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=SEED_MARK, ms=7, label="a training seed"),
        plt.Line2D([], [], marker="o", ls="", color=BANKED, ms=9, label="the seed in the line-up"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=MUTED,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Ten training seeds per family, on the columns the session ranks by  ·  beams2d/v1",
        fontsize=11.5,
        color=INK,
        x=0.012,
        ha="left",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"Wrote {out}")


def rank_frame(frame: pd.DataFrame, families: list[str], column: str) -> pd.DataFrame:
    """Rank the families against each other once per seed, `1` best.

    Only seeds where every family published are ranked; a partial line-up would
    hand a family a better rank for the absence of a rival.
    """
    ranks = {}
    for seed in SEEDS:
        row = frame[(frame.seed == seed) & frame.algo.isin(families)].dropna(subset=[column])
        if len(row) < len(families):
            continue
        ranks[seed] = row.set_index("algo")[column].rank(ascending=True).reindex(families)
    return pd.DataFrame(ranks)


def ranks_figure(frame: pd.DataFrame, families: list[str], out: Path) -> None:
    """The same numbers as ranks: ten line-ups somebody could have published."""
    columns = [c for c in (*CHEAP, *PHYSICS) if c in frame]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 6.0), facecolor=SURFACE)
    for ax, column in zip(axes.ravel(), columns):
        ranked = rank_frame(frame, families, column)
        for row, algo in enumerate(families):
            if algo not in ranked.index:
                continue
            values = ranked.loc[algo].dropna().to_numpy()
            ax.plot([values.min(), values.max()], [row, row], color=SEED_MARK, alpha=0.25, lw=6, solid_capstyle="round")
            jitter = np.linspace(-0.16, 0.16, len(values))
            ax.scatter(values, row + jitter, s=40, color=SEED_MARK, zorder=3, edgecolor=SURFACE, lw=1.2)
        ax.set_xticks(range(1, len(families) + 1))
        ax.set_xlim(0.4, len(families) + 0.6)
        ax.set_yticks(range(len(families)))
        ax.set_yticklabels([""] * len(families))
        ax.set_ylim(-0.6, len(families) - 0.4)
        ax.invert_yaxis()
        ax.set_title(f"{column}\n{QUESTION[column]}", fontsize=9.5, color=INK, loc="left", pad=8)
        ax.grid(axis="x", color="#e6e5e1", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#d8d7d3")
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for row in range(2):
        axes[row, 0].set_yticklabels(families, fontsize=9, color=INK)
    for ax in axes.ravel()[len(columns) :]:
        ax.set_visible(False)
    fig.suptitle(
        "The rank each family takes when the line-up is drawn at one seed  ·  1 = best, ten independent line-ups",
        fontsize=11.5,
        color=INK,
        x=0.012,
        ha="left",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.96))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"Wrote {out}")


def main() -> None:
    """Read the published per-seed metrics and draw both figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-id", default="beams2d")
    parser.add_argument("--out-dir", type=Path, default=TOOLS_DIR)
    parser.add_argument("--refresh", action="store_true", help="Re-read the Hub instead of the cached CSV.")
    args = parser.parse_args()

    from engiopt.workshops.idetc26.config import WorkshopConfig

    config = WorkshopConfig.load(args.problem_id)
    if args.refresh or not CACHE_CSV.exists():
        frame = published(args.problem_id, [dict(entry) for entry in config.bank])
        frame.to_csv(CACHE_CSV, index=False)
    else:
        frame = pd.read_csv(CACHE_CSV)

    # Only families with more than one published seed can answer the question at
    # all; the rest are named so their absence is a stated fact, not a gap.
    counts = frame.groupby("algo")["seed"].nunique()
    families = sorted(counts[counts >= MIN_SEEDS].index, key=lambda a: -counts[a])
    single = sorted({entry["algo"] for entry in config.bank} - set(families))
    print(f"{len(families)} families with >=2 seeds: {dict(counts[families])}")
    print(f"one checkpoint only, so no seed evidence exists: {single}")

    spread_figure(frame, families, args.out_dir / "seed_spread.png")
    ranks_figure(frame, [f for f in families if counts[f] >= len(SEEDS)], args.out_dir / "seed_ranks.png")


if __name__ == "__main__":
    main()
