r"""Draw the LV-metrics paper's figures from measured boards, not from notes.

Three figures, each backed by a file this repository can regenerate:

1. **The constraint ablation** -- partial Spearman against *residual* performance
   for every candidate instrument, split by whether the performance constraint
   was on. This is the paper's central claim about why the constraint is not
   optional, and it is the one figure whose ground truth needs no simulator.
   Source: `engiopt.lvae.select_instrument --out <csv>`.

2. **Real-model selection** -- |Spearman| of each cheap column against the
   simulator's IOG/COG/FOG over a physics board of real generators. The
   distortion battery could not separate the latent metrics from a matched PCA
   projection; this is the experiment that can.
   Source: a physics board CSV, e.g. `workshops/idetc26/tools/ground_truth_16.csv`.

3. **The pruning frontier** -- active dimensions against achieved held-out
   reconstruction, one point per training arm. The knee is the dimension
   estimate, and it is read off the frontier rather than chosen: no threshold
   appears anywhere in the figure.
   Source: a W&B summary export with `active_dims` and `val_nmse_rec`.

Each figure is skipped with a printed reason when its input is absent, so a
partial run still produces what it can rather than failing on the first missing
file.

Example:
    python -m engiopt.lvae.paper_figures --out-dir paper/figures/lv \
        --instrument-csv instr_photonics.csv \
        --board-csv workshops/idetc26/tools/ground_truth_16.csv \
        --frontier-csv plvae_runs.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PERF_OFF_THRESHOLD = 100.0
"""`nmse_threshold_perf` at or above this means the constraint was disabled."""

LATENT_PREFIX = "lv_"
PIXEL_BASELINES = ("mmd", "dpp", "pixel_vendi", "novelty", "cond_err", "viol")
PCA_BASELINES = ("pca_mmd", "pca_vendi", "pca_coverage")

PHYSICS_COLUMNS = ("iog", "cog", "fog")

MIN_PAIRS_FOR_RHO = 3
"""Fewest usable rows a Spearman correlation is computed over."""


RECON_ONLY_PREFIX = "lvoff_"


def _space_of(metric: str) -> str:
    """Which representation a column is measured in.

    `lvoff_` is checked before `lv_` and would otherwise fall through to pixel:
    "lvoff_mmd" does not start with "lv_", so the recon-only rung was silently
    coloured as a pixel baseline the moment it was added.
    """
    if metric.startswith(RECON_ONLY_PREFIX):
        return "LV recon-only"
    if metric.startswith(LATENT_PREFIX):
        return "LV latent"
    if metric.startswith("pca_"):
        return "PCA"
    return "pixel"


SPACE_COLORS = {
    "LV latent": "#1f4e79",
    "LV recon-only": "#5b8db8",
    "PCA": "#c0743a",
    "pixel": "#8a8a8a",
}
"""Ordered as the ladder climbs: grey pixels, orange linear control, then the
two learned spaces in light and dark, so the constraint's step reads as a shade
change within one family rather than as a fourth unrelated thing."""


def _save(figure: plt.Figure, out_dir: Path, stem: str) -> None:
    """Write one figure as both PDF and PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = out_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"  wrote {out_dir / stem}.{{pdf,png}}")


def constraint_ablation(csv: Path, out_dir: Path) -> None:
    """Residual-performance correlation for every candidate, split by constraint.

    The figure the claim rests on: if the performance constraint is what makes a
    latent space performance-relevant, then arms trained with it must separate
    from arms trained without it on a criterion that never saw the constraint.
    """
    frame = pd.read_csv(csv)
    if "rho_residual" not in frame.columns:
        print(f"  [skip] {csv} has no rho_residual column")
        return

    frame = frame.copy()
    frame["constraint"] = np.where(frame["candidate"].str.startswith("ON"), "performance ON", "reconstruction only")
    frame = frame.sort_values("rho_residual", ascending=True)

    # An arm is only useful if it does both jobs. Counting that here rather than
    # asserting it in the caption is what makes the figure state a measurement:
    # on photonics 9 of 14 arms manage it, on heat 1 of 19.
    both = int(((frame["rho_residual"] > 0) & frame["gate1_compressed"]).sum())
    problem = str(frame["problem"].iloc[0]) if "problem" in frame.columns else csv.stem.replace("instr_", "")

    figure, axis = plt.subplots(figsize=(7.4, 0.34 * len(frame) + 1.7))
    colors = ["#1f4e79" if c == "performance ON" else "#c0743a" for c in frame["constraint"]]
    positions = np.arange(len(frame))
    # Arms that never pruned are drawn hollow: they are not instruments, whatever
    # their correlation says, and on heat they are the only ones with any signal.
    hatches = ["" if c else "///" for c in frame["gate1_compressed"]]
    bars = axis.barh(positions, frame["rho_residual"], color=colors, height=0.68, edgecolor="white", linewidth=0.4)
    for bar, hatch in zip(bars, hatches, strict=True):
        bar.set_hatch(hatch)
    axis.set_yticks(positions)
    axis.set_yticklabels([f"{c}  ({n:.0f}d)" for c, n in zip(frame["candidate"], frame["n_active"], strict=True)])
    axis.axvline(0.0, color="black", linewidth=0.9)
    axis.set_xlabel(r"partial Spearman $\rho$ vs residual performance (conditions partialled out)")
    axis.set_title(
        f"{problem}: {both} of {len(frame)} arms both compressed and carry performance signal",
        fontsize=11,
    )

    for position, (rho, eligible) in enumerate(zip(frame["rho_residual"], frame["eligible"], strict=True)):
        if eligible:
            axis.text(
                rho + (0.008 if rho >= 0 else -0.008),
                position,
                "*",
                va="center",
                ha="left" if rho >= 0 else "right",
                fontsize=13,
            )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#1f4e79"),
        plt.Rectangle((0, 0), 1, 1, color="#c0743a"),
    ]
    axis.legend(handles, ["performance constraint ON", "reconstruction only"], loc="lower right", frameon=False)
    axis.margins(y=0.01)
    figure.text(
        0.01,
        -0.03,
        "* cleared all four instrument gates.   Hatched = never compressed (dims = latent_dim), so not an "
        "instrument whatever its correlation.\nBar length is the task criterion; the gates decide eligibility.",
        fontsize=8,
        color="#555555",
    )
    _save(figure, out_dir, "fig_constraint_ablation")


def real_model_selection(csv: Path, out_dir: Path) -> None:
    """|Spearman| of each cheap column against the simulator, over real generators.

    Grouped by the space the column is measured in, because the paper's question
    is not which metric wins but whether the *space* is what buys the accuracy.
    """
    board = pd.read_csv(csv)
    physics = [c for c in PHYSICS_COLUMNS if c in board.columns]
    if not physics:
        print(f"  [skip] {csv} carries none of {PHYSICS_COLUMNS}")
        return

    candidates = [
        c
        for c in board.columns
        if (c.startswith(LATENT_PREFIX) or c in PIXEL_BASELINES or c in PCA_BASELINES)
        and pd.api.types.is_numeric_dtype(board[c])
    ]

    rows = []
    for metric in candidates:
        entry = {"metric": metric, "space": _space_of(metric)}
        for target in physics:
            pair = board[[metric, target]].replace([np.inf, -np.inf], np.nan).dropna()
            usable = len(pair) >= MIN_PAIRS_FOR_RHO
            entry[target] = abs(stats.spearmanr(pair[metric], pair[target]).statistic) if usable else np.nan
        rows.append(entry)

    table = pd.DataFrame(rows).dropna(subset=physics, how="all")
    table = table.sort_values(physics[0], ascending=True)

    figure, axis = plt.subplots(figsize=(7.6, 0.42 * len(table) + 1.4))
    positions = np.arange(len(table))
    width = 0.8 / len(physics)
    hatches = ("", "//", "..")
    for index, target in enumerate(physics):
        offset = (index - (len(physics) - 1) / 2) * width
        axis.barh(
            positions + offset,
            table[target],
            height=width,
            color=[SPACE_COLORS[s] for s in table["space"]],
            hatch=hatches[index % len(hatches)],
            edgecolor="white",
            linewidth=0.4,
            label=target.upper(),
        )
    axis.set_yticks(positions)
    axis.set_yticklabels(table["metric"])
    axis.set_xlabel(rf"$|\rho|$ against the simulator (Spearman, n = {len(board)} generators)")
    axis.set_title("Same question, different space: which cheap column recovers the simulator's ranking")
    axis.legend(loc="lower right", frameon=False, title="physics target")
    for label, space in zip(axis.get_yticklabels(), table["space"], strict=True):
        label.set_color(SPACE_COLORS[space])
    axis.margins(y=0.01)
    _save(figure, out_dir, "fig_real_model_selection")

    table.to_csv(out_dir / "fig_real_model_selection.csv", index=False)


def pruning_frontier(csv: Path, out_dir: Path) -> None:
    """Active dimensions against achieved held-out reconstruction, per problem.

    Every point is one trained arm. No threshold appears in the figure: the
    thresholds only generated the candidates, and what is read off the plot is
    the frontier they trace.
    """
    runs = pd.read_csv(csv)
    needed = {"problem", "active_dims", "val_nmse_rec", "thr_perf"}
    if not needed.issubset(runs.columns):
        print(f"  [skip] {csv} is missing {sorted(needed - set(runs.columns))}")
        return

    runs = runs.dropna(subset=["active_dims", "val_nmse_rec"]).copy()
    runs["constraint"] = np.where(runs["thr_perf"] >= PERF_OFF_THRESHOLD, "reconstruction only", "performance ON")
    problems = sorted(runs["problem"].unique())

    figure, axes = plt.subplots(1, len(problems), figsize=(4.2 * len(problems), 3.6), squeeze=False)
    for axis, problem in zip(axes[0], problems, strict=True):
        subset = runs[runs["problem"] == problem]
        for constraint, color in (("performance ON", "#1f4e79"), ("reconstruction only", "#c0743a")):
            arm = subset[subset["constraint"] == constraint]
            axis.scatter(
                arm["val_nmse_rec"],
                arm["active_dims"],
                s=34,
                c=color,
                alpha=0.75,
                edgecolor="white",
                linewidth=0.5,
                label=constraint,
            )
        latent_dim = subset["latent_dim"].max() if "latent_dim" in subset else 100
        axis.axhline(latent_dim, color="#999999", linestyle=":", linewidth=1.0)
        axis.text(
            axis.get_xlim()[1],
            latent_dim,
            " no compression",
            va="bottom",
            ha="right",
            fontsize=8,
            color="#777777",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(problem)
        axis.set_xlabel("achieved val NMSE (reconstruction)")
        axis.set_ylabel("active latent dimensions")
    axes[0][0].legend(loc="lower left", frameon=False, fontsize=8)
    figure.suptitle("The frontier the thresholds generate -- an arm that never compressed is not an instrument")
    figure.tight_layout()
    _save(figure, out_dir, "fig_pruning_frontier")


def two_lens(board_csv: Path, distortion_csv: Path, out_dir: Path) -> None:
    """Physics alignment against failure detection, for every metric at once.

    A metric that ranks generators the way the simulator does, but cannot tell a
    corrupted design set from a clean one, is not a good metric -- it is a good
    *predictor on this board*, and the two come apart. Ranking on one axis is how
    a column that rewards noise keeps its place near the top of a table.

    So both axes are plotted together and neither is allowed to stand alone:

    - **x** -- |Spearman| against the simulator's IOG over real generators.
    - **y** -- fraction of distortion families the metric responds correctly to,
      where "correct" is the sign declared in `distortion_capture.EXPECTED`
      before any of it was measured.

    The top-right quadrant is the only one that earns a recommendation. A metric
    far right and low has learned this board; a metric high and left is honest
    and uninformative.
    """
    from engiopt.lvae.distortion_report import grade

    board = pd.read_csv(board_csv)
    if "iog" not in board.columns:
        print(f"  [skip] {board_csv} has no iog column")
        return

    graded = grade(pd.read_csv(distortion_csv))
    # A metric passes a family when it is neither wrong, rewarded, nor blind.
    graded["passed"] = graded["verdict"].isin(("ok", "weak"))
    detection = graded.groupby("metric")["passed"].mean()
    rewarded = graded[graded["verdict"] == "REWARDED"].groupby("metric").size()

    rows = []
    for metric in sorted(detection.index):
        column = metric if metric in board.columns else f"{metric}_mean"
        if column not in board.columns:
            continue
        pair = board[[column, "iog"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < MIN_PAIRS_FOR_RHO:
            continue
        rows.append(
            {
                "metric": metric,
                "space": _space_of(metric),
                "physics_rho": abs(stats.spearmanr(pair[column], pair["iog"]).statistic),
                "detection": float(detection[metric]),
                "rewarded": int(rewarded.get(metric, 0)),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        print("  [skip] no metric appears in both the board and the battery")
        return

    figure, axis = plt.subplots(figsize=(7.8, 6.0))
    for space, group in table.groupby("space"):
        axis.scatter(
            group["physics_rho"],
            group["detection"],
            s=[150 if r == 0 else 90 for r in group["rewarded"]],
            c=SPACE_COLORS[space],
            marker="o",
            edgecolor=["white" if r == 0 else "#c1121f" for r in group["rewarded"]],
            linewidth=[0.8 if r == 0 else 2.2 for r in group["rewarded"]],
            label=space,
            zorder=3,
        )
    for _, row in table.iterrows():
        axis.annotate(
            row["metric"],
            (row["physics_rho"], row["detection"]),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=8.5,
            color=SPACE_COLORS[row["space"]],
        )

    axis.axhline(0.8, color="#bbbbbb", linestyle=":", linewidth=1.0, zorder=1)
    axis.axvline(0.6, color="#bbbbbb", linestyle=":", linewidth=1.0, zorder=1)
    axis.set_xlabel(r"lens 1: $|\rho|$ against simulator IOG, real generators")
    axis.set_ylabel("lens 2: fraction of injected failures detected correctly")
    axis.set_title("Neither lens alone is a recommendation")
    axis.set_ylim(-0.04, 1.08)
    axis.legend(loc="lower left", frameon=False, title="measured in")
    figure.text(
        0.01,
        -0.02,
        "Red outline: the metric was REWARDED by at least one corruption -- it scored a damaged set "
        "better than the clean one.\nMarker size shrinks for those. Dotted lines are reading aids, not thresholds.",
        fontsize=8,
        color="#555555",
    )
    _save(figure, out_dir, "fig_two_lens")
    table.sort_values("physics_rho", ascending=False).to_csv(out_dir / "fig_two_lens.csv", index=False)


LADDER_RUNGS = ("pixel", "pca", "lvoff", "lv")
LADDER_LABELS = ("pixels", "PCA\n(matched dims)", "least volume\n(recon only)", "least volume\n(+ performance)")
LADDER_QUESTIONS = {
    "does it match the data?": ("mmd", "pca_mmd", "lvoff_mmd", "lv_mmd"),
    "did it cover the modes?": (None, "pca_coverage", "lvoff_coverage", "lv_coverage"),
    "how many distinct designs?": ("pixel_vendi", "pca_vendi", "lvoff_vendi", "lv_vendi"),
    "did it answer the condition?": (
        "pixel_paired_distance",
        "pca_paired_distance",
        "lvoff_paired_distance",
        "lv_paired_distance",
    ),
}


def ablation_ladder(boards: dict[str, Path], out_dir: Path, target: str = "iog_median") -> None:
    """The four-rung ladder, one panel per problem.

    The figure the whole method rests on. Pixels to PCA is dimensionality
    reduction; PCA to recon-only is nonlinearity; recon-only to the instrument
    is the performance constraint and nothing else, because the two learned arms
    are trained identically apart from it.

    Args:
        boards: Rescored board CSV per problem id. Each needs the cheap columns
            and `target`.
        out_dir: Where the figure is written.
        target: Physics column to correlate against. The median by default --
            the mean optimality gap is set by its single worst design.
    """
    usable = {p: pd.read_csv(path) for p, path in boards.items() if path.exists()}
    if not usable:
        print("  [skip] no board CSVs exist")
        return

    figure, axes = plt.subplots(1, len(usable), figsize=(4.6 * len(usable), 4.1), sharey=True, squeeze=False)
    for axis, (problem, board) in zip(axes[0], usable.items(), strict=True):
        if target not in board:
            axis.set_title(f"{problem}\n(no {target} column)")
            continue
        for question, rungs in LADDER_QUESTIONS.items():
            xs, ys = [], []
            for index, metric in enumerate(rungs):
                if metric is None or metric not in board:
                    continue
                pair = board[[metric, target]].dropna()
                if len(pair) < MIN_PAIRS_FOR_RHO:
                    continue
                xs.append(index)
                ys.append(abs(stats.spearmanr(pair[metric], pair[target]).statistic))
            if xs:
                axis.plot(xs, ys, marker="o", linewidth=1.8, markersize=5, label=question)
        axis.set_xticks(range(len(LADDER_RUNGS)))
        axis.set_xticklabels(LADDER_LABELS, fontsize=7.5)
        axis.set_title(f"{problem}  (n={len(board)})", fontsize=10)
        axis.grid(axis="y", alpha=0.3)
        axis.axvspan(2.5, 3.5, color="#1f4e79", alpha=0.06)
    axes[0][0].set_ylabel(f"|Spearman| vs simulator {target}")
    axes[0][-1].legend(fontsize=7.5, loc="best")
    figure.suptitle(
        "The same question, climbing four spaces. The shaded step is the performance\n"
        "constraint alone -- both of its endpoints are learned, compressed and nonlinear.",
        fontsize=9,
        y=1.04,
    )
    _save(figure, out_dir, "fig_ablation_ladder")


def garbage_map(embedding_csv: Path, distances_csv: Path, out_dir: Path) -> None:
    """Where deliberately-bad designs fall, per space.

    Top row: the same points projected to 2D in pixels, in the recon-only
    latent, and in the instrument's latent. Bottom row: the standardized
    distance to the nearest real optimum in all four spaces, which is where the
    PCA control earns its place -- truncation moves noisy designs inward without
    any constraint having acted.

    Args:
        embedding_csv: From `engiopt.lvae.figure_data`.
        distances_csv: Likewise.
        out_dir: Where the figure is written.
    """
    embedding = pd.read_csv(embedding_csv)
    distance = pd.read_csv(distances_csv)
    panels = [s for s in ("pixel", "lv_off", "lv_on") if s in set(embedding["space"])]
    spaces = [s for s in ("pixel", "pca", "lv_off", "lv_on") if s in set(distance["space"])]
    titles = {"pixel": "pixels", "pca": "PCA (matched)", "lv_off": "LV recon-only", "lv_on": "LV + performance"}
    known_bad = {"collapsed", "noise_doped", "checkerboard", "volume_only"}

    figure, axes = plt.subplots(2, max(len(panels), len(spaces)), figsize=(4.0 * max(len(panels), len(spaces)), 7.4))
    for axis in axes.ravel():
        axis.set_visible(False)

    for axis, space in zip(axes[0], panels, strict=False):
        axis.set_visible(True)
        frame = embedding[embedding["space"] == space]
        backdrop = frame[frame["source"] == "reference"]
        axis.scatter(backdrop["x"], backdrop["y"], s=9, color="#cfcfcf", label="real optima", zorder=1)
        for source, group in frame[frame["source"] != "reference"].groupby("source"):
            bad = source in known_bad
            axis.scatter(
                group["x"],
                group["y"],
                s=14 if bad else 9,
                marker="x" if bad else "o",
                alpha=0.85 if bad else 0.5,
                label=source,
                zorder=3 if bad else 2,
            )
        explained = frame["explained"].iloc[0] if len(frame) else float("nan")
        axis.set_title(f"{titles.get(space, space)}\n2 axes carry {explained:.0%} of real variance", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    axes[0][0].legend(fontsize=6.5, loc="best", framealpha=0.9)

    for axis, space in zip(axes[1], spaces, strict=False):
        axis.set_visible(True)
        frame = distance[distance["space"] == space]
        order = frame.groupby("source")["distance"].median().sort_values()
        axis.boxplot(
            [frame[frame["source"] == s]["distance"].to_numpy() for s in order.index],
            labels=list(order.index),
            vert=False,
            showfliers=False,
            widths=0.6,
        )
        axis.axvline(1.0, color="#8a8a8a", linestyle=":", linewidth=1)
        axis.set_title(titles.get(space, space), fontsize=9)
        axis.set_xscale("log")
        axis.tick_params(axis="y", labelsize=6.5)
        axis.tick_params(axis="x", labelsize=7)
    axes[1][0].set_xlabel("distance to nearest real optimum\n(reference NN spacing = 1, dotted)", fontsize=7.5)
    _save(figure, out_dir, "fig_garbage_map")


def latent_spectra(csvs: dict[str, Path], out_dir: Path) -> None:
    """Sorted latent standard deviations for both arms of each canonical pair.

    A collapsed arm is obvious here and invisible in `n_active`: it shows as one
    dimension orders of magnitude above the rest. The participation ratio in the
    legend is the effective width the columns really have.

    Args:
        csvs: Spectra CSV per problem id, from `engiopt.lvae.figure_data`.
        out_dir: Where the figure is written.
    """
    usable = {p: pd.read_csv(path) for p, path in csvs.items() if path.exists()}
    if not usable:
        print("  [skip] no spectra CSVs exist")
        return
    colors = {"perf_on": SPACE_COLORS["LV latent"], "recon_only": SPACE_COLORS["LV recon-only"]}
    figure, axes = plt.subplots(1, len(usable), figsize=(4.3 * len(usable), 3.8), squeeze=False)
    for axis, (problem, frame) in zip(axes[0], usable.items(), strict=True):
        for arm, arm_rows in frame.groupby("arm"):
            group = arm_rows.sort_values("rank")
            ratio = group["participation_ratio"].iloc[0]
            axis.plot(
                group["rank"],
                group["sigma"],
                marker="o",
                markersize=3,
                color=colors.get(str(arm), "#888888"),
                label=f"{arm}: {int(group['n_active'].iloc[0])} active, PR {ratio:.1f}",
            )
        axis.set_yscale("log")
        axis.set_title(problem, fontsize=10)
        axis.set_xlabel("latent dimension, sorted")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    axes[0][0].set_ylabel("standard deviation")
    figure.suptitle(
        "A single dimension far above the rest is the trivial solution the Lipschitz bound exists to prevent.\n"
        "Participation ratio (PR) is the effective width; the active count is not.",
        fontsize=8.5,
        y=1.06,
    )
    _save(figure, out_dir, "fig_latent_spectra")


def rank_disagreement(board_csv: Path, out_dir: Path, target: str = "iog_median", top: int = 20) -> None:
    """Models ranked by each cheap column, against the simulator's ranking.

    One line per model, from its rank under a cheap metric to its rank under the
    simulator. Lines that cross are the workshop's whole argument, and a metric
    whose lines run flat is one that would have picked the right model.

    Args:
        board_csv: A rescored board carrying cheap columns and `target`.
        out_dir: Where the figure is written.
        target: The physics column that defines the true ranking.
        top: Show only this many models, chosen as the simulator's best.
    """
    board = pd.read_csv(board_csv)
    if target not in board:
        print(f"  [skip] {board_csv} has no {target} column")
        return
    board = board.dropna(subset=[target]).nsmallest(top, target).reset_index(drop=True)
    columns = [c for c in ("mmd", "pca_mmd", "lvoff_mmd", "lv_mmd", "lv_paired_distance") if c in board]
    if not columns:
        print("  [skip] no cheap columns on this board")
        return

    truth = board[target].rank()
    figure, axes = plt.subplots(1, len(columns), figsize=(2.6 * len(columns), 5.6), sharey=True, squeeze=False)
    for axis, column in zip(axes[0], columns, strict=True):
        cheap = board[column].rank()
        for i in range(len(board)):
            moved = abs(cheap[i] - truth[i])
            axis.plot(
                [0, 1],
                [cheap[i], truth[i]],
                color=SPACE_COLORS[_space_of(column)],
                alpha=min(0.9, 0.25 + moved / len(board)),
                linewidth=0.8 + 1.6 * moved / len(board),
            )
        tau = stats.kendalltau(cheap, truth).statistic
        axis.set_title(f"{column}\n" + rf"$\tau$ = {tau:+.2f}", fontsize=9)
        axis.set_xticks([0, 1])
        axis.set_xticklabels([column, "simulator"], fontsize=7, rotation=20)
        axis.invert_yaxis()
    axes[0][0].set_ylabel(f"rank (1 = best), top {len(board)} by {target}")
    figure.suptitle(
        "Each line is one model. Where the lines cross, the cheap metric picked a different winner.", fontsize=9, y=1.02
    )
    _save(figure, out_dir, "fig_rank_disagreement")


def main() -> None:
    """Draw whichever figures their inputs are present for."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--instrument-csv", type=Path, default=None)
    parser.add_argument("--board-csv", type=Path, default=None)
    parser.add_argument("--frontier-csv", type=Path, default=None)
    parser.add_argument("--distortion-csv", type=Path, default=None)
    parser.add_argument(
        "--figdata-dir",
        type=Path,
        default=None,
        help="Directory of engiopt.lvae.figure_data output; draws the garbage map and the spectra.",
    )
    parser.add_argument(
        "--rescored",
        nargs="*",
        default=(),
        metavar="PROBLEM=CSV",
        help="Rescored boards for the ladder, e.g. beams2d=board_rescored_beams2d.csv",
    )
    args = parser.parse_args()

    jobs = (
        ("constraint ablation", args.instrument_csv, constraint_ablation),
        ("real-model selection", args.board_csv, real_model_selection),
        ("pruning frontier", args.frontier_csv, pruning_frontier),
    )
    for name, path, draw in jobs:
        print(f"{name}:")
        if path is None:
            print("  [skip] no input given")
        elif not path.exists():
            print(f"  [skip] {path} does not exist")
        else:
            draw(path, args.out_dir)

    print("two-lens comparison:")
    if args.board_csv is None or args.distortion_csv is None:
        print("  [skip] needs both --board-csv and --distortion-csv")
    elif not args.board_csv.exists() or not args.distortion_csv.exists():
        print("  [skip] one of the two inputs does not exist")
    else:
        two_lens(args.board_csv, args.distortion_csv, args.out_dir)

    _draw_canonical(args)


def _draw_canonical(args: argparse.Namespace) -> None:
    """The figures built on the canonical instrument pairs.

    Split out of `main` because each of these has its own two-way skip and the
    dispatcher was doing more branching than any one reader should have to hold.
    """
    boards = {p: Path(c) for p, c in (pair.split("=", 1) for pair in args.rescored)}

    print("ablation ladder:")
    if boards:
        ablation_ladder(boards, args.out_dir)
    else:
        print("  [skip] no --rescored boards given")

    print("rank disagreement:")
    first = next((path for path in boards.values() if path.exists()), None)
    if first is not None:
        rank_disagreement(first, args.out_dir)
    else:
        print("  [skip] needs at least one --rescored board")

    print("garbage map and latent spectra:")
    if args.figdata_dir is None:
        print("  [skip] no --figdata-dir given")
        return
    spectra = {p.stem.removeprefix("spectra_"): p for p in sorted(args.figdata_dir.glob("spectra_*.csv"))}
    latent_spectra(spectra, args.out_dir)
    for embedding_csv in sorted(args.figdata_dir.glob("embedding_*.csv")):
        problem = embedding_csv.stem.removeprefix("embedding_")
        distances_csv = args.figdata_dir / f"distances_{problem}.csv"
        if distances_csv.exists():
            garbage_map(embedding_csv, distances_csv, args.out_dir / problem)
        else:
            print(f"  [skip] {problem}: no distances CSV beside the embedding")


if __name__ == "__main__":
    main()
