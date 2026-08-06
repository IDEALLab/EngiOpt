r"""Render augmentation panels (design + conditions pre/post) for augmented 2D datasets.

Companion visual check for `engiopt.build_augmented_datasets`: it reads the Arrow
datasets that script writes and renders, per problem, a 2-row x n-family grid of
designs annotated with the two ground-truth labels every downstream metric is scored
against:

  is_valid_design       green "on-man"  / red "OFF-man"    -> the validity axis
  is_condition_matched  green "cond OK" / red "cond WRONG" -> the conditional axis

Each panel's x-label shows the stated (asked) vs true (realized) scalar conditions,
printed as `vf 0.35` when they agree and `vf 0.35->0.48` when the augmentation moved
them -- so intensity_shift's analytic volfrac drift and condition_jumble's relabel are
both visible in the figure itself.

Two source designs are shown per problem, picked from opposite ends of the first
condition's range so the condition_jumble swap is actually distinguishable.

Example:
    python -m engiopt.plot_augmented_datasets \\
        --dataset-dir aug_out --output-dir paper/figures/augmentation \\
        --problem-ids beams2d heatconduction2d photonics2d thermoelastic2d
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import os

from datasets import load_from_disk
import matplotlib as mpl
import numpy as np
import tyro

mpl.use("Agg")  # headless: must be set before pyplot is imported
import matplotlib.pyplot as plt

# Requested and realized conditions closer than this are treated as unchanged.
CONDITION_MATCH_TOLERANCE = 1e-3

# Severity at which each graded corruption is rendered: high enough to be visible, low
# enough to still look like a corrupted design rather than pure noise.
SEVERITY_SHOWN = {"gaussian_noise": 0.6, "blur": 1.0, "intensity_shift": 1.0}

# Column order; corruptions absent from a given problem's dataset are skipped.
CORRUPTION_ORDER = ("clean", "gaussian_noise", "blur", "intensity_shift", "condition_jumble")

# Conditions preferred for the caption, most interpretable first (at most 3 are shown).
PREFERRED_CONDITION_NAMES = (
    "volfrac",
    "volume",
    "volume_fraction_target",
    "forcedist",
    "length",
    "weight",
    "lambda1",
    "lambda2",
)

SHORT_NAMES = {
    "volfrac": "vf",
    "volume": "vf",
    "volume_fraction_target": "vf",
    "forcedist": "fd",
    "lambda1": "λ1",
    "lambda2": "λ2",
    "weight": "w",
    "length": "len",
}


@dataclass
class Args:
    """Command-line arguments."""

    dataset_dir: str = "aug_out"
    """Directory holding the `{problem_id}_{split}_augmented` datasets."""
    output_dir: str = "aug_out"
    """Directory to write `augmentation_panel_{problem_id}.png` into."""
    split: str = "test"
    """Split suffix used when the datasets were built."""
    problem_ids: tuple[str, ...] = field(
        default_factory=lambda: ("beams2d", "heatconduction2d", "photonics2d", "thermoelastic2d")
    )
    """Problems to render (one figure each)."""
    dpi: int = 125
    """Output resolution."""


def render(problem_id: str, args: Args) -> str:
    """Render one problem's augmentation panel.

    Args:
        problem_id: EngiBench 2D problem identifier.
        args: Parsed command-line arguments.

    Returns:
        Path to the written PNG.
    """
    dataset = load_from_disk(os.path.join(args.dataset_dir, f"{problem_id}_{args.split}_augmented"))
    augmented_rows = list(dataset)
    condition_names = [c[len("stated_") :] for c in dataset.column_names if c.startswith("stated_")]
    corruption_types = [
        corruption for corruption in CORRUPTION_ORDER if any(r["family"] == corruption for r in augmented_rows)
    ]

    def row_for(corruption: str, original_index: int, severity: float | None = None, repeat: int = 0) -> dict:
        """Row for one (corruption, original design, repeat), nearest the wanted severity."""
        candidates = [
            r
            for r in augmented_rows
            if r["family"] == corruption and r["source_idx"] == original_index and r["repeat"] == repeat
        ]
        if not candidates:
            msg = f"{problem_id}: no row for {corruption}, original design {original_index}, repeat {repeat}"
            raise ValueError(msg)
        if severity is not None:
            candidates = sorted(candidates, key=lambda r: abs(r["severity"] - severity))
        return candidates[0]

    # Two original designs from opposite ends of the first condition's range, so the
    # condition_jumble swap between them is actually visible.
    uncorrupted_by_original = {r["source_idx"]: r for r in augmented_rows if r["family"] == "clean"}
    sorting_condition = condition_names[0]
    by_condition = sorted(uncorrupted_by_original, key=lambda i: uncorrupted_by_original[i][f"stated_{sorting_condition}"])
    low_condition_design, high_condition_design = by_condition[1], by_condition[-2]

    captioned_conditions = [name for name in PREFERRED_CONDITION_NAMES if name in condition_names]
    captioned_conditions = (captioned_conditions or condition_names)[:3]

    def condition_caption(row: dict) -> str:
        """Requested vs realized conditions, arrowed only where the corruption moved them."""
        parts = []
        for name in captioned_conditions:
            requested, realized = row[f"stated_{name}"], row[f"true_{name}"]
            label = SHORT_NAMES.get(name, name[:4])
            unchanged = abs(requested - realized) < CONDITION_MATCH_TOLERANCE
            parts.append(f"{label} {realized:.2f}" if unchanged else f"{label} {requested:.2f}→{realized:.2f}")
        return "  ".join(parts)

    fig, axes = plt.subplots(2, len(corruption_types), figsize=(3.0 * len(corruption_types), 6.6), squeeze=False)
    for panel_row, original_index in enumerate([low_condition_design, high_condition_design]):
        for panel_column, corruption in enumerate(corruption_types):
            row = row_for(corruption, original_index, SEVERITY_SHOWN.get(corruption))
            ax = axes[panel_row, panel_column]
            ax.imshow(np.asarray(row["design"]), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if panel_row == 0:
                severity_note = f"\n(sev={row['severity']:.1f})" if corruption in SEVERITY_SHOWN else ""
                ax.set_title(corruption + severity_note, fontsize=10, fontweight="bold")
            ax.set_xlabel(condition_caption(row), fontsize=8.5)
            validity_colour = "tab:green" if row["is_valid_design"] else "tab:red"
            condition_colour = "tab:green" if row["is_condition_matched"] else "tab:red"
            ax.text(
                0.02,
                0.97,
                "on-man" if row["is_valid_design"] else "OFF-man",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="white",
                bbox={"boxstyle": "round,pad=0.15", "fc": validity_colour, "ec": "none"},
            )
            ax.text(
                0.98,
                0.97,
                "cond OK" if row["is_condition_matched"] else "cond WRONG",
                transform=ax.transAxes,
                va="top",
                ha="right",
                fontsize=8,
                color="white",
                bbox={"boxstyle": "round,pad=0.15", "fc": condition_colour, "ec": "none"},
            )

    fig.suptitle(
        f"{problem_id} — augmentation + conditions (pre → post).  green=should pass, red=should flag",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"augmentation_panel_{problem_id}.png")
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    low_value = uncorrupted_by_original[low_condition_design][f"stated_{sorting_condition}"]
    high_value = uncorrupted_by_original[high_condition_design][f"stated_{sorting_condition}"]
    print(f"{problem_id}: {sorting_condition}={low_value:.2f} and {high_value:.2f}  ->  {output_path}")
    return output_path


def main(args: Args) -> None:
    """Render one panel per requested problem."""
    for problem_id in args.problem_ids:
        render(problem_id, args)


if __name__ == "__main__":
    main(tyro.cli(Args))
