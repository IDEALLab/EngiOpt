r"""Simulate augmented EngiBench-2D designs to get physics ground truth for the labels.

`build_augmented_datasets` stamps every row with two labels -- `is_valid_design` and
`is_condition_matched` -- but those are asserted by construction: a blurred design is
labelled off-manifold because we blurred it, not because anything checked. This script
runs the actual simulator and turns both labels into measured quantities, so the paper
can say a corruption hurt performance instead of assuming it.

Two independent questions are answered, and they are recorded separately:

  performance_loss   Did the corruption damage the design?
                     simulate(corrupted design, the ORIGINAL design's conditions)
                       vs simulate(original design, the same conditions).
                     Same conditions on both sides, so this isolates the damage done
                     to the design itself. Expected: ~0 for `clean` and for
                     `condition_jumble` (whose design is untouched), and growing with
                     severity for `gaussian_noise` / `blur` / `intensity_shift`.

  wrong_condition_   Did the design fail the conditions it was ASKED for?
  loss               simulate(design, the REQUESTED conditions)
                       vs simulate(the real optimum for those requested conditions,
                                   the same conditions).
                     Expected: ~0 wherever requested == original, and large for
                     `condition_jumble` -- the cost of "right design, wrong problem".

Both are direction-corrected per objective (a maximized objective like photonics2d's
`total_overlap` is negated), so **positive always means worse than the real optimum**.

Alongside them, `pixel_change_vs_original` measures how different the corrupted design
*looks* (relative L2). Crossing the two is the point: rows where the design looks very
different but performs the same are cosmetic corruption, and that is the ground truth a
per-design cosmetic-vs-functional metric has to reproduce to be worth anything. On
beams2d, holding `pixel_change_vs_original` inside [0.42, 0.47] still leaves
`performance_loss` spanning -60% to +692%, so pixel distance on its own says almost
nothing about functional damage.

Two caveats before trusting these numbers:

1. Not every condition reaches the simulator. beams2d's `volfrac` is an optimizer
   *constraint*, so compliance does not depend on it -- a volume-fraction mismatch is
   invisible to simulation and has to be caught by measuring mean density instead
   (`mean_density` is recorded for exactly that). The conditions that do move the
   physics are beams2d `forcedist`, heatconduction2d `length`, photonics2d
   `lambda1`/`lambda2`, and thermoelastic2d `weight`. `intensity_shift` is the family
   this bites: it violates only the volume fraction.

2. Simulated performance is not a universally valid validity oracle. On photonics2d,
   blurring a design *raises* `total_overlap` (0.22 -> 1.07 at sigma=5), so
   `performance_loss` comes out strongly negative for exactly the rows the validity
   label calls off-manifold. The mass-penalty term does not explain it -- it accounts
   for 0.016 of that 0.85 swing -- it is the FDFD model rewarding intermediate
   permittivity, the grayscale artifact that density-based photonic topology
   optimization has to binarize away. Unmanufacturable designs scoring better is the
   physics-side analogue of pixel-DPP rewarding noise. Read a negative
   `performance_loss` on photonics2d as a finding about the objective, not a bug here.

Cost (measured, per simulate call, one design): beams2d ~0.08 s, photonics2d ~0.18 s,
thermoelastic2d ~0.18 s, heatconduction2d ~9 s (it round-trips through a container).
A full 2112-row dataset is therefore minutes for the first three and hours for
heatconduction2d -- use `--dry-run` to price a sweep, then `--n-sources` / `--repeats` /
`--max-sims` to cut it down. Reference simulations are cached per (design, conditions),
and a row whose requested conditions equal its original conditions is simulated once
rather than twice.

Results stream to CSV row by row and re-running skips rows already present, so a killed
job resumes. `--n-shards` / `--shard-idx` splits the work for a SLURM array.

Example:
    # price it first
    python -m engiopt.simulate_augmented_datasets --problem-id beams2d \\
        --dataset-dir aug_out --dry-run

    python -m engiopt.simulate_augmented_datasets --problem-id beams2d \\
        --dataset-dir aug_out --output-dir sim_out

    # the expensive one: subsample hard
    python -m engiopt.simulate_augmented_datasets --problem-id heatconduction2d \\
        --dataset-dir aug_out --output-dir sim_out --n-sources 16 --repeats 1
"""

from __future__ import annotations

import contextlib
import csv
from dataclasses import dataclass
import io
import json
import os
import time
from typing import Any

from datasets import load_from_disk
from engibench.core import ObjectiveDirection
from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import numpy.typing as npt
import tyro

from engiopt.transforms import get_scalar_condition_keys

# Two condition values this close are treated as the same condition, both when deciding
# whether a row was relabelled and when matching a relabelled row back to its source.
CONDITION_MATCH_TOLERANCE = 1e-6

# Measured seconds per simulate call, used only to price a sweep in --dry-run.
SECONDS_PER_SIMULATION = {
    "beams2d": 0.08,
    "photonics2d": 0.18,
    "thermoelastic2d": 0.18,
    "heatconduction2d": 9.0,
}
DEFAULT_SECONDS_PER_SIMULATION = 0.2


@dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "beams2d"
    """EngiBench 2D problem identifier (must match the augmented dataset)."""
    dataset_dir: str = "aug_out"
    """Directory holding `{problem_id}_{split}_augmented`."""
    output_dir: str = "sim_out"
    """Directory for the results CSV and summary JSON."""
    split: str = "test"
    """Split suffix used when the dataset was built."""
    families: tuple[str, ...] = ()
    """Restrict to these corruption types (empty = all present in the dataset)."""
    n_sources: int = 0
    """Keep only the first N original designs (0 = all). The main cost lever."""
    repeats: int = 1
    """Keep repeat indices `< repeats` (graded corruptions have several random draws)."""
    severities: tuple[float, ...] = ()
    """Restrict to these severities (empty = all). Uncorrupted rows are always kept."""
    max_sims: int = 0
    """Stop after this many simulator calls (0 = no cap). Counts reference runs too."""
    n_shards: int = 1
    """Split the selected rows across this many shards (SLURM array support)."""
    shard_idx: int = 0
    """Which shard to run, in `[0, n_shards)`."""
    dry_run: bool = False
    """Report the row/simulation count and estimated wall time, then exit."""
    resume: bool = True
    """Skip rows already present in the output CSV."""
    quiet_simulator: bool = True
    """Swallow simulator stdout (photonics2d prints its conditions on every call)."""


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


def all_conditions_of(dataset_row: dict[str, Any], condition_names: list[str]) -> dict[str, Any]:
    """Every condition the simulator accepts for one dataset row, arrays included.

    The augmented dataset stores scalar conditions only, so array-valued ones
    (thermoelastic2d's 65x65 boundary maps) are recovered here from the original split
    via the row's source index. Without them the simulator would silently fall back to
    its default boundary conditions and every thermoelastic2d number would be wrong.
    """
    return {name: dataset_row[name] for name in condition_names if name in dataset_row}


def conditions_identity(conditions: dict[str, Any]) -> tuple:
    """Hashable identity of a condition set, so identical simulations run only once."""
    identity = []
    for name in sorted(conditions):
        value = np.asarray(conditions[name])
        identity.append((name, value.tobytes() if value.ndim else float(value)))
    return tuple(identity)


def objective_worse_is_positive_signs(problem: Any) -> npt.NDArray[np.float64]:
    """`+1` for objectives that are minimized, `-1` for objectives that are maximized.

    Multiplying a raw `objective - reference` difference by this makes "positive means
    worse" true on every problem, including photonics2d's maximized `total_overlap`.
    """
    return np.array(
        [-1.0 if direction == ObjectiveDirection.MAXIMIZE else 1.0 for _name, direction in problem.objectives],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# How different two designs look
# ---------------------------------------------------------------------------


def pixel_rmse(design: npt.NDArray, reference: npt.NDArray) -> float:
    """Per-pixel RMSE between two designs -- the plainest "how different do these look"."""
    return float(np.sqrt(np.mean((design - reference) ** 2)))


def pixel_relative_l2(design: npt.NDArray, reference: npt.NDArray) -> float:
    """`||design - reference|| / ||reference||`, i.e. pixel difference as a fraction.

    Scaled this way it sits on the same footing as a relative performance loss, so the
    two can be crossed directly: a design that looks 45% different but performs the same
    is cosmetic corruption.
    """
    reference_norm = float(np.linalg.norm(reference))
    return float(np.linalg.norm(design - reference) / reference_norm) if reference_norm else float("nan")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class CachingSimulator:
    """Runs `problem.simulate`, reusing identical (design, conditions) calls."""

    def __init__(self, problem: Any, *, quiet: bool) -> None:
        self.problem = problem
        self.quiet = quiet
        self.results_by_call: dict[tuple, npt.NDArray[np.float64]] = {}
        self.n_simulations = 0
        self.total_seconds = 0.0

    def simulate(
        self, design: npt.NDArray, conditions: dict[str, Any], *, design_identity: tuple
    ) -> npt.NDArray[np.float64]:
        """Simulate `design` under `conditions`, reusing an earlier identical call.

        Non-finite objectives are returned as-is rather than raised on. A corrupted
        design that makes the solve diverge is a result -- and on thermoelastic2d it is
        the dominant one, where `gaussian_noise` NaNs `structural_compliance` while
        leaving `thermal_compliance` perfectly finite. Discarding the row would throw
        away both the surviving objectives and the divergence itself.
        """
        call_identity = (design_identity, conditions_identity(conditions))
        if call_identity in self.results_by_call:
            return self.results_by_call[call_identity]

        started = time.perf_counter()
        # beams2d caches its FE setup across calls and only rebuilds it after a reset,
        # so skipping this would silently simulate under the previous row's conditions.
        self.problem.reset()
        if self.quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                raw_objectives = self.problem.simulate(design, config=conditions)
        else:
            raw_objectives = self.problem.simulate(design, config=conditions)
        self.total_seconds += time.perf_counter() - started
        self.n_simulations += 1

        objectives = np.atleast_1d(np.asarray(raw_objectives, dtype=np.float64))
        self.results_by_call[call_identity] = objectives
        return objectives


# ---------------------------------------------------------------------------
# Row selection
# ---------------------------------------------------------------------------


def select_rows(augmented_rows: list[dict[str, Any]], args: Args) -> list[int]:
    """Indices of the augmented rows to simulate, after every subsampling filter."""
    kept_source_indices = None
    if args.n_sources:
        every_source = sorted({row["source_idx"] for row in augmented_rows})
        kept_source_indices = set(every_source[: args.n_sources])

    selected_indices = []
    for index, row in enumerate(augmented_rows):
        if args.families and row["family"] not in args.families:
            continue
        if kept_source_indices is not None and row["source_idx"] not in kept_source_indices:
            continue
        if row["repeat"] >= args.repeats:
            continue
        # Uncorrupted rows are the baseline for every comparison, so a severity filter
        # must never drop them.
        wanted_severity = not args.severities or any(
            abs(row["severity"] - severity) < CONDITION_MATCH_TOLERANCE for severity in args.severities
        )
        if row["family"] != "clean" and not wanted_severity:
            continue
        selected_indices.append(index)

    if args.n_shards > 1:
        selected_indices = [i for i in selected_indices if i % args.n_shards == args.shard_idx]
    return selected_indices


def row_was_relabelled(row: dict[str, Any], condition_names: list[str]) -> bool:
    """True when the row's requested conditions differ from the design's own conditions."""
    return any(abs(row[f"stated_{name}"] - row[f"true_{name}"]) > CONDITION_MATCH_TOLERANCE for name in condition_names)


def source_index_of_requested_conditions(
    row: dict[str, Any], uncorrupted_rows: list[dict[str, Any]], condition_names: list[str]
) -> int | None:
    """Index of the original design that is genuinely optimal for this row's requested conditions.

    `condition_jumble` relabels a design with another sampled row's conditions but does
    not record which one, so it is recovered by matching the requested condition values
    back against the uncorrupted rows. The match identifies the design that *is* optimal
    for the requested conditions -- the only honest reference for the wrong-condition
    comparison.
    """
    for candidate in uncorrupted_rows:
        if all(
            abs(row[f"stated_{name}"] - candidate[f"true_{name}"]) < CONDITION_MATCH_TOLERANCE for name in condition_names
        ):
            return int(candidate["source_idx"])
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: Args) -> None:  # noqa: C901, PLR0915
    """Simulate the selected augmented rows and stream one result row each to CSV."""
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=0)
    design_shape = tuple(problem.design_space.shape)

    augmented_path = os.path.join(args.dataset_dir, f"{args.problem_id}_{args.split}_augmented")
    augmented_dataset = load_from_disk(augmented_path)
    augmented_rows = list(augmented_dataset)
    condition_names = [c[len("stated_") :] for c in augmented_dataset.column_names if c.startswith("stated_")]

    original_split = problem.dataset[args.split]
    simulator_condition_names = [k for k in problem.conditions_keys if k in original_split.column_names]
    scalar_condition_names = get_scalar_condition_keys(problem, original_split)
    objective_names = [name for name, _direction in problem.objectives]
    worse_is_positive = objective_worse_is_positive_signs(problem)

    uncorrupted_rows = [row for row in augmented_rows if row["family"] == "clean"]
    selected_indices = select_rows(augmented_rows, args)

    # Each row costs one simulation, plus a second when its requested conditions differ
    # from its own. Reference runs are shared and amortise to about one per original.
    n_relabelled = sum(1 for i in selected_indices if row_was_relabelled(augmented_rows[i], condition_names))
    n_originals_used = len({augmented_rows[i]["source_idx"] for i in selected_indices})
    estimated_simulations = len(selected_indices) + n_relabelled + 2 * n_originals_used
    seconds_each = SECONDS_PER_SIMULATION.get(args.problem_id, DEFAULT_SECONDS_PER_SIMULATION)
    estimated_minutes = estimated_simulations * seconds_each / 60

    print(f"{args.problem_id}: {len(augmented_rows)} augmented rows -> {len(selected_indices)} selected", end="")
    if args.n_shards > 1:
        print(f" (shard {args.shard_idx}/{args.n_shards})", end="")
    print(f"; ~{estimated_simulations} simulations, ~{estimated_minutes:.1f} min at {seconds_each:.2f} s/sim")
    if args.dry_run:
        return

    os.makedirs(args.output_dir, exist_ok=True)
    shard_suffix = f"_shard{args.shard_idx}" if args.n_shards > 1 else ""
    csv_path = os.path.join(args.output_dir, f"{args.problem_id}_{args.split}_simulated{shard_suffix}.csv")

    column_names = [
        "problem_id",
        "split",
        "augmented_row_index",
        "corruption_type",
        "severity",
        "repeat",
        "original_row_index",
        "requested_conditions_row_index",
        "labelled_valid_design",
        "labelled_condition_matched",
        "mean_density",
        "pixel_rmse_vs_original",
        "pixel_change_vs_original",
        "pixel_rmse_vs_requested_optimum",
        "pixel_change_vs_requested_optimum",
        "status",
        "error",
        "seconds",
        # Under the design's own conditions: the design, the original it came from, and
        # how much worse the design is (positive = worse).
        *[f"objective_of_design_{name}" for name in objective_names],
        *[f"objective_of_original_{name}" for name in objective_names],
        *[f"performance_loss_{name}" for name in objective_names],
        # Under the conditions the design was asked for: the design, the design that is
        # genuinely optimal there, and how much worse the design is.
        *[f"objective_under_requested_{name}" for name in objective_names],
        *[f"objective_of_requested_optimum_{name}" for name in objective_names],
        *[f"wrong_condition_loss_{name}" for name in objective_names],
    ]

    already_done: set[int] = set()
    if args.resume and os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            already_done = {int(r["augmented_row_index"]) for r in csv.DictReader(f)}
        print(f"resuming: {len(already_done)} rows already in {csv_path}")

    simulator = CachingSimulator(problem, quiet=args.quiet_simulator)
    remaining_indices = [i for i in selected_indices if i not in already_done]

    csv_file = open(csv_path, "a", newline="")  # noqa: SIM115
    writer = csv.DictWriter(csv_file, fieldnames=column_names)
    if not already_done:
        writer.writeheader()

    n_failed = 0
    try:
        for position, augmented_row_index in enumerate(remaining_indices):
            row = augmented_rows[augmented_row_index]
            corrupted_design = np.asarray(row["design"], dtype=np.float64).reshape(design_shape)

            original_row_index = int(row["source_idx"])
            original_row = original_split[original_row_index]
            original_design = np.asarray(original_row["optimal_design"], dtype=np.float64).reshape(design_shape)

            problem_conditions = all_conditions_of(original_row, simulator_condition_names)
            own_conditions = {
                **problem_conditions,
                **{k: float(row[f"true_{k}"]) for k in condition_names if k in scalar_condition_names},
            }
            requested_conditions = {
                **problem_conditions,
                **{k: float(row[f"stated_{k}"]) for k in condition_names if k in scalar_condition_names},
            }
            conditions_are_identical = conditions_identity(own_conditions) == conditions_identity(requested_conditions)

            result: dict[str, Any] = {
                "problem_id": args.problem_id,
                "split": args.split,
                "augmented_row_index": augmented_row_index,
                "corruption_type": row["family"],
                "severity": row["severity"],
                "repeat": row["repeat"],
                "original_row_index": original_row_index,
                "requested_conditions_row_index": "",
                "labelled_valid_design": row["is_valid_design"],
                "labelled_condition_matched": row["is_condition_matched"],
                "mean_density": float(corrupted_design.mean()),
                "pixel_rmse_vs_original": pixel_rmse(corrupted_design, original_design),
                "pixel_change_vs_original": pixel_relative_l2(corrupted_design, original_design),
                "pixel_rmse_vs_requested_optimum": "",
                "pixel_change_vs_requested_optimum": "",
                "status": "ok",
                "error": "",
            }

            started = time.perf_counter()
            try:
                # --- Did the corruption damage the design? ------------------------
                # Same conditions on both sides, so only the design differs.
                objective_of_design = simulator.simulate(
                    corrupted_design, own_conditions, design_identity=("corrupted", augmented_row_index)
                )
                objective_of_original = simulator.simulate(
                    original_design, own_conditions, design_identity=("original", original_row_index)
                )
                performance_loss = worse_is_positive[: objective_of_design.size] * (
                    objective_of_design - objective_of_original
                )

                # --- Did it fail the conditions it was asked for? -----------------
                objective_under_requested = objective_of_design
                objective_of_requested_optimum = None
                wrong_condition_loss = None
                requested_row_index = (
                    original_row_index
                    if conditions_are_identical
                    else source_index_of_requested_conditions(row, uncorrupted_rows, condition_names)
                )
                if requested_row_index is not None:
                    requested_row = original_split[requested_row_index]
                    requested_optimum = np.asarray(requested_row["optimal_design"], dtype=np.float64).reshape(design_shape)
                    # The requested conditions are that row's whole problem, arrays included.
                    conditions_for_reference = {
                        **all_conditions_of(requested_row, simulator_condition_names),
                        **{k: float(row[f"stated_{k}"]) for k in condition_names if k in scalar_condition_names},
                    }
                    objective_of_requested_optimum = simulator.simulate(
                        requested_optimum, conditions_for_reference, design_identity=("original", requested_row_index)
                    )
                    objective_under_requested = simulator.simulate(
                        corrupted_design, conditions_for_reference, design_identity=("corrupted", augmented_row_index)
                    )
                    wrong_condition_loss = worse_is_positive[: objective_under_requested.size] * (
                        objective_under_requested - objective_of_requested_optimum
                    )
                    result["requested_conditions_row_index"] = requested_row_index
                    # How different the design looks from the one that IS optimal here.
                    # For condition_jumble both sides are real optima, so this is the
                    # on-manifold "different design, same job" pixel distance.
                    result["pixel_rmse_vs_requested_optimum"] = pixel_rmse(corrupted_design, requested_optimum)
                    result["pixel_change_vs_requested_optimum"] = pixel_relative_l2(corrupted_design, requested_optimum)

                for j, name in enumerate(objective_names):
                    result[f"objective_of_design_{name}"] = objective_of_design[j]
                    result[f"objective_of_original_{name}"] = objective_of_original[j]
                    result[f"performance_loss_{name}"] = performance_loss[j]
                    result[f"objective_under_requested_{name}"] = objective_under_requested[j]
                    result[f"objective_of_requested_optimum_{name}"] = (
                        "" if objective_of_requested_optimum is None else objective_of_requested_optimum[j]
                    )
                    result[f"wrong_condition_loss_{name}"] = "" if wrong_condition_loss is None else wrong_condition_loss[j]

                # "the solve diverged" is a distinct outcome from "the simulator raised",
                # and a distinct one again from "the design is merely worse".
                if not np.all(np.isfinite(np.concatenate([objective_of_design, objective_of_original]))):
                    result["status"] = "diverged"
                    result["error"] = f"non-finite objectives {objective_of_design.tolist()}"

            except Exception as exc:  # noqa: BLE001 -- a refused design is data, not a crash
                # A corrupted design the solver refuses outright is itself evidence of
                # being off-manifold, so it is recorded and the sweep continues.
                result["status"] = type(exc).__name__
                result["error"] = str(exc)[:200].replace("\n", " ")
                n_failed += 1

            result["seconds"] = round(time.perf_counter() - started, 3)
            writer.writerow(result)
            csv_file.flush()

            if (position + 1) % 50 == 0 or position + 1 == len(remaining_indices):
                seconds_per_simulation = simulator.total_seconds / max(simulator.n_simulations, 1)
                print(
                    f"  [{position + 1}/{len(remaining_indices)}] {simulator.n_simulations} simulations, "
                    f"{seconds_per_simulation:.2f} s each, {n_failed} failed",
                    flush=True,
                )
            if args.max_sims and simulator.n_simulations >= args.max_sims:
                print(f"  stopping: hit --max-sims={args.max_sims}")
                break
    finally:
        csv_file.close()

    summarize(csv_path, args, objective_names)


def summarize(csv_path: str, args: Args, objective_names: list[str]) -> None:
    """Print and store per-corruption statistics -- did the corruption hurt performance?

    Medians lead the table because these losses are heavy-tailed: a jumbled design that
    puts a void where the load lands drives compliance to ~1e9 and drags any mean with
    it. The mean and 90th percentile go to the JSON so the tail stays visible.
    """
    with open(csv_path, newline="") as f:
        results = list(csv.DictReader(f))
    if not results:
        print("no rows simulated")
        return

    primary_objective = objective_names[0]
    summary: dict[str, Any] = {
        "problem_id": args.problem_id,
        "primary_objective": primary_objective,
        "corruptions": {},
    }

    def finite_values(group: list[dict[str, Any]], column: str) -> npt.NDArray[np.float64]:
        """Parsed values of `column`, dropping blanks and the NaNs of diverged solves."""
        values = np.array([float(r[column]) for r in group if r.get(column, "") != ""], dtype=np.float64)
        return values[np.isfinite(values)]

    print(f"\nper-corruption medians on '{primary_objective}' (positive = worse than the real optimum):")
    print(
        f"  {'corruption':<17} {'sev':>5} {'n':>5} {'pixel_change':>13} {'perf_loss':>13} {'perf_loss_%':>12} "
        f"{'wrong_cond_loss':>16} {'diverged':>9} {'failed':>7}"
    )

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in results:
        groups.setdefault((r["corruption_type"], r["severity"]), []).append(r)

    for (corruption, severity), group in sorted(groups.items(), key=lambda kv: (kv[0][0], float(kv[0][1]))):
        n_failed = sum(1 for r in group if r["status"] not in ("ok", "diverged"))
        n_diverged = sum(1 for r in group if r["status"] == "diverged")
        scored = [r for r in group if r["status"] in ("ok", "diverged")]

        performance_loss = finite_values(scored, f"performance_loss_{primary_objective}")
        wrong_condition_loss = finite_values(scored, f"wrong_condition_loss_{primary_objective}")
        original_objective = finite_values(scored, f"objective_of_original_{primary_objective}")
        pixel_change = finite_values(scored, "pixel_change_vs_original")

        median_pixel_change = float(np.median(pixel_change)) if pixel_change.size else float("nan")
        if performance_loss.size == 0:
            print(
                f"  {corruption:<17} {float(severity):>5.1f} {0:>5} {median_pixel_change:>13.3f} "
                f"{'-':>13} {'-':>12} {'-':>16} {n_diverged:>9} {n_failed:>7}"
            )
            summary["corruptions"][f"{corruption}@{float(severity):.1f}"] = {
                "n": 0,
                "diverged": n_diverged,
                "failed": n_failed,
                "pixel_change_vs_original_median": median_pixel_change,
            }
            continue

        median_performance_loss = float(np.median(performance_loss))
        typical_original = float(np.median(np.abs(original_objective))) if original_objective.size else 0.0
        performance_loss_percent = median_performance_loss / typical_original if typical_original else float("nan")
        median_wrong_condition_loss = float(np.median(wrong_condition_loss)) if wrong_condition_loss.size else float("nan")

        print(
            f"  {corruption:<17} {float(severity):>5.1f} {performance_loss.size:>5} {median_pixel_change:>13.3f} "
            f"{median_performance_loss:>13.4g} {performance_loss_percent:>12.1%} "
            f"{median_wrong_condition_loss:>16.4g} {n_diverged:>9} {n_failed:>7}"
        )
        summary["corruptions"][f"{corruption}@{float(severity):.1f}"] = {
            "n": int(performance_loss.size),
            "diverged": n_diverged,
            "failed": n_failed,
            "pixel_change_vs_original_median": median_pixel_change,
            "performance_loss_median": median_performance_loss,
            "performance_loss_mean": float(performance_loss.mean()),
            "performance_loss_p90": float(np.percentile(performance_loss, 90)),
            "performance_loss_percent_of_optimum": performance_loss_percent,
            "wrong_condition_loss_median": median_wrong_condition_loss,
            "wrong_condition_loss_mean": float(wrong_condition_loss.mean()) if wrong_condition_loss.size else float("nan"),
            "wrong_condition_loss_p90": (
                float(np.percentile(wrong_condition_loss, 90)) if wrong_condition_loss.size else float("nan")
            ),
        }

    summary_path = csv_path.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nrows -> {csv_path}\nsummary -> {summary_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
