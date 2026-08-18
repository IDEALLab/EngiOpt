r"""Dump the small artifacts the canonical-instrument figures are drawn from.

Figures need model loads and encodes; plotting needs neither. Splitting them
means the expensive half runs once on the cluster and the drawing half runs in a
second on a laptop, which is the difference between iterating on a figure and
re-queuing a job to move a legend.

Everything here is cheap in the sense that matters: no simulator and no
optimizer. It loads encoders, samples from a few generators, and writes
coordinates.

Two artifacts per problem:

`spectra.csv` -- sorted per-dimension latent standard deviation for the pinned
instrument and its recon-only companion, measured on real validation designs.
This is what makes a collapsed arm visible: two of the six canonical arms put
99% of their latent variance in one dimension with sigma inflated ~40x, which
`n_active` cannot show and a participation ratio summarises but does not
display.

`embedding.csv` -- real, generated and deliberately-corrupted designs projected
to 2D in pixels, in the companion's latent, and in the instrument's latent. The
same points in all three, so the panels differ only by the space. Corrupted
designs come from the reference instruments, whose answers are known by
construction, so where they land is a check rather than an illustration.

The 2D projection is a PCA fitted **on the reference designs of that space** and
applied to everything else, so the view is anchored on the real data and a
generator that leaves it is visibly outside rather than re-centred.

There is deliberately no PCA *panel*. A 2D PCA view of the matched PCA subspace
is the same picture as a 2D PCA view of raw pixels -- PCA subspaces are nested,
so the top two components do not change when you truncate to k of them. Drawing
both would show two identical panels and imply a difference that does not exist
in the view.

`distances.csv` -- where the PCA control does differ, and the quantitative half
of the same figure. For every source and all *four* spaces, the distance from
each design to its nearest reference optimum, divided by the reference set's own
median nearest-neighbour spacing in that space. One dimensionless number per
design meaning "how many typical inter-optimum steps outside the real data does
this sit", comparable across spaces that have different widths and units. This
is what separates truncation from performance-awareness: dropping components
removes high-frequency noise, so corrupted designs move inward in PCA without
the constraint having done anything.

Example:
    python -m engiopt.lvae.figure_data --problem-id beams2d --out-dir figdata
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from engiopt.evaluation import Evaluator

REFERENCES = ("collapsed", "noise_doped", "checkerboard", "volume_only")
"""Designs whose answer is known by construction. `volume_only` needs a volume
condition and is skipped on problems without one."""

PANEL_SPACES = ("pixel", "lv_off", "lv_on")
"""Spaces that get a 2D panel. PCA is excluded on purpose -- see the module
docstring: its 2D view is identical to the pixel one."""

DISTANCE_SPACES = ("pixel", "pca", "lv_off", "lv_on")
"""Spaces the standardized distance is measured in. PCA belongs here, because
truncation genuinely changes distances even though it cannot change the top two
components."""


def _flatten(designs: npt.NDArray) -> npt.NDArray:
    """Designs as `(n, -1)` regardless of grid shape."""
    array = np.asarray(designs)
    return array.reshape(len(array), -1)


def spectra(evaluator: Evaluator, designs: npt.NDArray) -> pd.DataFrame:
    """Sorted latent standard deviations for both arms of the canonical pair.

    Args:
        evaluator: Evaluator carrying the pinned instrument.
        designs: Real designs to measure the spectrum on.

    Returns:
        Long-form frame with `arm`, `rank`, `sigma`, and the cumulative
        variance fraction that rank accounts for.
    """
    from engiopt.lvae.encode import encode_active

    rows = []
    for arm, lvae in (("perf_on", evaluator.latent_lvae), ("recon_only", evaluator.latent_recon_lvae)):
        if lvae is None:
            continue
        device = next(lvae.encoder.parameters()).device
        codes = encode_active(lvae.encoder, designs, device)
        sigma = np.sort(codes.std(axis=0))[::-1]
        variance = sigma**2
        cumulative = np.cumsum(variance) / variance.sum() if variance.sum() else np.zeros_like(variance)
        # Participation ratio: the effective dimension count, which is what
        # `n_active` is repeatedly mistaken for. An arm with 32 active
        # dimensions and 90% of its variance in two has PR near 2, not 32.
        participation = float(variance.sum() ** 2 / (variance**2).sum()) if variance.sum() else float("nan")
        for rank, (s, c) in enumerate(zip(sigma, cumulative, strict=True), start=1):
            rows.append(
                {
                    "arm": arm,
                    "rank": rank,
                    "sigma": float(s),
                    "cum_var_fraction": float(c),
                    "n_active": len(sigma),
                    "participation_ratio": participation,
                }
            )
    return pd.DataFrame(rows)


def _sources(evaluator: Evaluator, problem_id: str, models: list[str], n: int) -> dict[str, npt.NDArray]:
    """Designs per source: the real data, each named generator, each reference."""
    from engiopt.baselines import REFERENCE_INSTRUMENTS
    from engiopt.baselines.base import DatasetGenerator
    from engiopt.utils.all_generators import BUILTIN_GENERATORS

    out: dict[str, npt.NDArray] = {}
    for spec in models:
        algo, _, fingerprint = spec.partition("#")
        try:
            cls = BUILTIN_GENERATORS[algo]
            if isinstance(cls, type) and issubclass(cls, DatasetGenerator):
                generator = cls.from_problem(evaluator.problem, problem_id=problem_id, seed=1)
            else:
                generator = cls.from_pretrained(
                    evaluator.problem,
                    problem_id=problem_id,
                    seed=1,
                    model_source="hf",
                    config_fingerprint=fingerprint or None,
                )
            out[algo] = np.asarray(evaluator.context_for(generator, n_samples=n).gen_designs)
            print(f"  sampled {algo}: {out[algo].shape}")
        except Exception as exc:  # noqa: BLE001 - a missing package must not cost the whole dump
            print(f"  {algo}: SKIPPED {type(exc).__name__}: {str(exc)[:110]}")

    for name in REFERENCES:
        cls = REFERENCE_INSTRUMENTS.get(name)
        if cls is None:
            continue
        try:
            generator = cls.from_problem(evaluator.problem, problem_id=problem_id, seed=1)
            out[name] = np.asarray(evaluator.context_for(generator, n_samples=n).gen_designs)
            print(f"  sampled reference {name}: {out[name].shape}")
        except Exception as exc:  # noqa: BLE001 - volume_only has no budget on photonics, by design
            print(f"  reference {name}: SKIPPED {type(exc).__name__}: {str(exc)[:110]}")
    return out


def _encoder_for(space: str, evaluator: Evaluator, reference: npt.NDArray) -> Any:
    """A function mapping designs into `space`, fitted on the reference set.

    Args:
        space: One of `pixel`, `pca`, `lv_off`, `lv_on`.
        evaluator: Evaluator carrying the pinned instrument pair.
        reference: Real designs any fitting is done on.

    Returns:
        A callable taking designs and returning `(n, d)` coordinates, or None if
        that space has no instrument on this problem.
    """
    from sklearn.decomposition import PCA

    from engiopt.lvae.encode import encode_active
    from engiopt.lvae.encode import get_active_mask

    if space == "pixel":
        return _flatten

    if space == "pca":
        # Matched to the instrument's active width, the same rule
        # `EvaluationContext.pca_codes` uses -- a linear control at matched
        # dimensionality rather than matched effort.
        width = 20
        if evaluator.latent_lvae is not None:
            width = int(get_active_mask(evaluator.latent_lvae.encoder).sum())
        flat = _flatten(reference)
        width = max(1, min(width, *flat.shape))
        fitted = PCA(n_components=width).fit(flat)
        return lambda designs: fitted.transform(_flatten(designs))

    lvae = evaluator.latent_lvae if space == "lv_on" else evaluator.latent_recon_lvae
    if lvae is None:
        return None
    device = next(lvae.encoder.parameters()).device
    return lambda designs: encode_active(lvae.encoder, np.asarray(designs), device)


def embedding(evaluator: Evaluator, reference: npt.NDArray, sources: dict[str, npt.NDArray]) -> pd.DataFrame:
    """Every source as 2D coordinates, in each space that gets a panel.

    Args:
        evaluator: Evaluator carrying the pinned instrument pair.
        reference: Real designs the projection is anchored on.
        sources: Designs per source name.

    Returns:
        Long-form frame with `space`, `source`, `x`, `y`, `explained`.
    """
    from sklearn.decomposition import PCA

    rows = []
    for space in PANEL_SPACES:
        encode = _encoder_for(space, evaluator, reference)
        if encode is None:
            print(f"  {space}: no instrument pinned, skipped")
            continue
        reference_coded = encode(reference)
        fitted = PCA(n_components=min(2, *reference_coded.shape)).fit(reference_coded)
        explained = float(fitted.explained_variance_ratio_[:2].sum())
        coords = {"reference": fitted.transform(reference_coded)}
        for name, designs in sources.items():
            coords[name] = fitted.transform(encode(designs))
        for source, points in coords.items():
            for x, y in points:
                rows.append({"space": space, "source": source, "x": float(x), "y": float(y), "explained": explained})
        print(f"  {space}: {len(coords)} sources, 2 axes carry {explained:.1%} of reference variance")
    return pd.DataFrame(rows)


def distances(evaluator: Evaluator, reference: npt.NDArray, sources: dict[str, npt.NDArray]) -> pd.DataFrame:
    """How far outside the real data each design sits, in every space.

    Standardized by the reference set's own median nearest-neighbour spacing so
    the number is dimensionless and comparable across spaces of different width
    and scale. A value near 1 means "as far from the real data as two real
    optima typically are from each other"; 10 means well outside it.

    Args:
        evaluator: Evaluator carrying the pinned instrument pair.
        reference: Real designs, the thing distance is measured to.
        sources: Designs per source name.

    Returns:
        Long-form frame with `space`, `source`, `distance` (one row per design).
    """
    from scipy.spatial.distance import cdist

    rows = []
    for space in DISTANCE_SPACES:
        encode = _encoder_for(space, evaluator, reference)
        if encode is None:
            print(f"  {space}: no instrument pinned, skipped")
            continue
        reference_coded = encode(reference)
        within = cdist(reference_coded, reference_coded)
        np.fill_diagonal(within, np.inf)
        scale = float(np.median(within.min(axis=1))) or 1.0
        for name, designs in sources.items():
            nearest = cdist(encode(designs), reference_coded).min(axis=1) / scale
            rows.extend({"space": space, "source": name, "distance": float(d)} for d in nearest)
        summary = pd.DataFrame(rows)
        summary = summary[summary["space"] == space].groupby("source")["distance"].median()
        print(f"  {space:7s} median standardized distance: " + "  ".join(f"{k}={v:.2f}" for k, v in summary.items()))
    return pd.DataFrame(rows)


def main() -> None:
    """Write the spectra and embedding artifacts for one problem."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem-id", required=True)
    ap.add_argument(
        "--spec",
        default=None,
        help="Defaults to <problem_id>/v1, the only version there is.  The evaluator resolves the same "
        "latent instrument on any problem -- passing it produces empty spectra and no latent panels.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--models",
        nargs="*",
        default=["knn_retrieval", "cgan_cnn_2d#825831f6", "gan_cnn_2d#6293adb3", "diffusion_2d_cond", "vqgan"],
        help="Generators to embed, as algo or algo#config_fingerprint.",
    )
    ap.add_argument("--n", type=int, default=50, help="Designs per source.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec = args.spec or f"{args.problem_id}/v1"
    evaluator = Evaluator.for_problem(args.problem_id, spec=spec)
    if evaluator.spec.latent_instrument is None:
        # Every artifact here is about the instrument pair, so an unpinned spec
        # yields empty files rather than a smaller figure. Failing here beats a
        # 1-byte CSV and a KeyError three lines later, which is what a spec
        # without an instrument silently produced.
        raise SystemExit(f"{spec} pins no latent_instrument; there is nothing for this tool to measure.")
    reference = np.asarray(evaluator.resolved.ref_designs)
    print(f"{args.problem_id} [{spec}]: {len(reference)} reference designs")

    print("\n-- spectra --")
    frame = spectra(evaluator, reference)
    path = args.out_dir / f"spectra_{args.problem_id}.csv"
    frame.to_csv(path, index=False)
    for arm, group in frame.groupby("arm"):
        top = group.sort_values("rank").iloc[0]
        print(f"  {arm}: {int(top['n_active'])} active, PR {top['participation_ratio']:.1f}, top sigma {top['sigma']:.3g}")
    print(f"  -> {path}")

    print("\n-- sampling sources --")
    sources = _sources(evaluator, args.problem_id, args.models, args.n)

    print("\n-- embedding --")
    path = args.out_dir / f"embedding_{args.problem_id}.csv"
    embedding(evaluator, reference, sources).to_csv(path, index=False)
    print(f"  -> {path}")

    print("\n-- distances --")
    path = args.out_dir / f"distances_{args.problem_id}.csv"
    distances(evaluator, reference, sources).to_csv(path, index=False)
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
