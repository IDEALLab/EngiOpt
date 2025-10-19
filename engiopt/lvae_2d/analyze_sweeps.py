"""Analyze LVAE sweep results to understand pruning method and spectral norm effects.

Usage:
    python engiopt/lvae_2d/analyze_sweeps.py <sweep-id>
    python engiopt/lvae_2d/analyze_sweeps.py <sweep-id> --problem beams2d
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb

sns.set_theme(style="whitegrid")


def fetch_sweep_data(sweep_id: str, entity: str | None = None, project: str = "lvae") -> pd.DataFrame:
    """Fetch all runs from a sweep into a DataFrame."""
    api = wandb.Api()
    if entity:
        sweep_path = f"{entity}/{project}/{sweep_id}"
    else:
        sweep_path = f"{project}/{sweep_id}"

    print(f"Fetching sweep: {sweep_path}")
    sweep = api.sweep(sweep_path)

    data = []
    for run in sweep.runs:
        row = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            # Config
            "problem_id": run.config.get("problem_id"),
            "pruning_strategy": run.config.get("pruning_strategy"),
            "use_spectral_norm": run.config.get("use_spectral_norm"),
            "plummet_threshold": run.config.get("plummet_threshold"),
            "alpha": run.config.get("alpha"),
            "percentile": run.config.get("percentile"),
            "latent_dim": run.config.get("latent_dim"),
            "w_v": run.config.get("w_v"),
            "eta": run.config.get("eta"),
            "pruning_epoch": run.config.get("pruning_epoch"),
            "beta": run.config.get("beta"),
            "lr": run.config.get("lr"),
            "seed": run.config.get("seed"),
            # Summary metrics
            "val_rec": run.summary.get("val_rec"),
            "val_vol_loss": run.summary.get("val_vol_loss"),
            "val_total_loss": run.summary.get("val_total_loss"),
            "active_dims": run.summary.get("active_dims"),
            "rec_loss": run.summary.get("rec_loss"),
            "vol_loss": run.summary.get("vol_loss"),
        }
        data.append(row)

    df = pd.DataFrame(data)
    print(f"Fetched {len(df)} runs ({(df['state'] == 'finished').sum()} finished)")
    return df


def plot_pruning_method_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare plummet vs lognorm performance."""
    finished = df[df["state"] == "finished"].copy()

    if finished.empty:
        print("No finished runs to analyze")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Validation reconstruction loss
    sns.boxplot(
        data=finished,
        x="pruning_strategy",
        y="val_rec",
        hue="use_spectral_norm",
        ax=axes[0],
    )
    axes[0].set_title("Validation Reconstruction Loss by Pruning Method")
    axes[0].set_ylabel("Val Reconstruction Loss")
    axes[0].set_xlabel("Pruning Strategy")

    # Active dimensions
    sns.boxplot(
        data=finished,
        x="pruning_strategy",
        y="active_dims",
        hue="use_spectral_norm",
        ax=axes[1],
    )
    axes[1].set_title("Active Dimensions by Pruning Method")
    axes[1].set_ylabel("Active Dimensions")
    axes[1].set_xlabel("Pruning Strategy")

    plt.tight_layout()
    fig.savefig(output_dir / "pruning_method_comparison.png", dpi=150)
    print(f"Saved: {output_dir / 'pruning_method_comparison.png'}")


def plot_plummet_sensitivity(df: pd.DataFrame, output_dir: Path):
    """Analyze sensitivity to plummet threshold."""
    plummet_runs = df[(df["pruning_strategy"] == "plummet") & (df["state"] == "finished")].copy()

    if plummet_runs.empty:
        print("No finished plummet runs")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reconstruction vs threshold
    for spec_norm in [False, True]:
        subset = plummet_runs[plummet_runs["use_spectral_norm"] == spec_norm]
        if not subset.empty:
            axes[0].scatter(
                subset["plummet_threshold"],
                subset["val_rec"],
                label=f"Spectral Norm: {spec_norm}",
                alpha=0.6,
                s=100,
            )
    axes[0].set_xlabel("Plummet Threshold")
    axes[0].set_ylabel("Val Reconstruction Loss")
    axes[0].set_xscale("log")
    axes[0].set_title("Plummet Threshold Sensitivity")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Active dims vs threshold
    for spec_norm in [False, True]:
        subset = plummet_runs[plummet_runs["use_spectral_norm"] == spec_norm]
        if not subset.empty:
            axes[1].scatter(
                subset["plummet_threshold"],
                subset["active_dims"],
                label=f"Spectral Norm: {spec_norm}",
                alpha=0.6,
                s=100,
            )
    axes[1].set_xlabel("Plummet Threshold")
    axes[1].set_ylabel("Active Dimensions")
    axes[1].set_xscale("log")
    axes[1].set_title("Pruning Aggressiveness vs Threshold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "plummet_sensitivity.png", dpi=150)
    print(f"Saved: {output_dir / 'plummet_sensitivity.png'}")


def plot_lognorm_sensitivity(df: pd.DataFrame, output_dir: Path):
    """Analyze sensitivity to lognorm alpha and percentile."""
    lognorm_runs = df[(df["pruning_strategy"] == "lognorm") & (df["state"] == "finished")].copy()

    if lognorm_runs.empty:
        print("No finished lognorm runs")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Create pivot tables for heatmaps
    for i, spec_norm in enumerate([False, True]):
        subset = lognorm_runs[lognorm_runs["use_spectral_norm"] == spec_norm]
        if subset.empty:
            continue

        # Reconstruction loss heatmap
        pivot_rec = subset.pivot_table(
            values="val_rec",
            index="alpha",
            columns="percentile",
            aggfunc="mean",
        )
        sns.heatmap(pivot_rec, annot=True, fmt=".4f", cmap="YlOrRd", ax=axes[0, i])
        axes[0, i].set_title(f"Val Rec Loss (Spectral Norm: {spec_norm})")
        axes[0, i].set_xlabel("Percentile")
        axes[0, i].set_ylabel("Alpha")

        # Active dims heatmap
        pivot_dims = subset.pivot_table(
            values="active_dims",
            index="alpha",
            columns="percentile",
            aggfunc="mean",
        )
        sns.heatmap(pivot_dims, annot=True, fmt=".0f", cmap="YlGnBu", ax=axes[1, i])
        axes[1, i].set_title(f"Active Dims (Spectral Norm: {spec_norm})")
        axes[1, i].set_xlabel("Percentile")
        axes[1, i].set_ylabel("Alpha")

    plt.tight_layout()
    fig.savefig(output_dir / "lognorm_sensitivity.png", dpi=150)
    print(f"Saved: {output_dir / 'lognorm_sensitivity.png'}")


def plot_spectral_norm_effect(df: pd.DataFrame, output_dir: Path):
    """Isolate the effect of spectral normalization."""
    finished = df[df["state"] == "finished"].copy()

    if finished.empty:
        print("No finished runs")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot comparison
    sns.violinplot(
        data=finished,
        x="use_spectral_norm",
        y="val_rec",
        hue="pruning_strategy",
        ax=axes[0],
        split=True,
    )
    axes[0].set_title("Spectral Normalization Effect on Reconstruction")
    axes[0].set_xlabel("Use Spectral Norm")
    axes[0].set_ylabel("Val Reconstruction Loss")

    # Paired comparison (if we have matched runs)
    spec_on = finished[finished["use_spectral_norm"] == True].groupby(["pruning_strategy", "seed"])["val_rec"].mean()  # noqa: E712
    spec_off = finished[finished["use_spectral_norm"] == False].groupby(["pruning_strategy", "seed"])["val_rec"].mean()  # noqa: E712

    improvement = ((spec_off - spec_on) / spec_off * 100).reset_index()
    improvement.columns = ["pruning_strategy", "seed", "improvement_pct"]

    sns.barplot(
        data=improvement,
        x="pruning_strategy",
        y="improvement_pct",
        ax=axes[1],
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Spectral Norm Improvement (%)")
    axes[1].set_ylabel("% Improvement in Val Rec (positive = better)")
    axes[1].set_xlabel("Pruning Strategy")

    plt.tight_layout()
    fig.savefig(output_dir / "spectral_norm_effect.png", dpi=150)
    print(f"Saved: {output_dir / 'spectral_norm_effect.png'}")


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics."""
    finished = df[df["state"] == "finished"]

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Overall stats
    print(f"\nTotal runs: {len(df)}")
    print(f"Finished runs: {len(finished)}")
    print(f"Failed runs: {len(df[df['state'] == 'failed'])}")

    if finished.empty:
        return

    # Best run overall
    best_run = finished.loc[finished["val_rec"].idxmin()]
    print("\nBest run overall:")
    print(f"  Val Rec: {best_run['val_rec']:.6f}")
    print(f"  Pruning: {best_run['pruning_strategy']}")
    print(f"  Spectral Norm: {best_run['use_spectral_norm']}")
    print(f"  Active Dims: {best_run['active_dims']}")
    if best_run["pruning_strategy"] == "plummet":
        print(f"  Plummet Threshold: {best_run['plummet_threshold']}")
    elif best_run["pruning_strategy"] == "lognorm":
        print(f"  Alpha: {best_run['alpha']}, Percentile: {best_run['percentile']}")

    # Best by pruning method
    print("\nBest by pruning method:")
    for method in finished["pruning_strategy"].unique():
        subset = finished[finished["pruning_strategy"] == method]
        best = subset.loc[subset["val_rec"].idxmin()]
        print(f"  {method}: val_rec={best['val_rec']:.6f}, active_dims={best['active_dims']}, spec_norm={best['use_spectral_norm']}")

    # Spectral norm effect
    print("\nSpectral Norm Effect:")
    for spec_norm in [False, True]:
        subset = finished[finished["use_spectral_norm"] == spec_norm]
        if not subset.empty:
            print(f"  Spectral Norm={spec_norm}: mean val_rec={subset['val_rec'].mean():.6f} ± {subset['val_rec'].std():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze LVAE sweep results")
    parser.add_argument("sweep_id", help="Wandb sweep ID")
    parser.add_argument("--entity", help="Wandb entity (default: your default entity)")
    parser.add_argument("--project", default="lvae", help="Wandb project (default: lvae)")
    parser.add_argument("--output", default="sweep_analysis", help="Output directory for plots")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Fetch data
    df = fetch_sweep_data(args.sweep_id, args.entity, args.project)

    # Save raw data
    csv_path = output_dir / "sweep_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw data to: {csv_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_pruning_method_comparison(df, output_dir)
    plot_plummet_sensitivity(df, output_dir)
    plot_lognorm_sensitivity(df, output_dir)
    plot_spectral_norm_effect(df, output_dir)

    # Print summary
    print_summary_statistics(df)

    print(f"\n✓ Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
