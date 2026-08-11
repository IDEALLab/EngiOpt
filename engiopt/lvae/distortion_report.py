r"""Grade each metric against what it *should* do under each known failure.

`distortion_capture` records responses; this decides whether they are correct.
For every (instrument, failure kind, metric) it takes the Spearman correlation
of the metric against severity and compares it to the expected sign in
`EXPECTED`:

- **ok** -- moves the right way, with enough magnitude to be usable.
- **weak** -- right direction, but the response is too small to separate a
  corrupted set from a clean one in practice.
- **WRONG** -- moves the wrong way.
- **REWARDED** -- the corruption *improved* the score. A diversity metric that
  rises under additive noise, or a distribution metric that rewards copying the
  reference set, is the failure this whole exercise exists to find.
- **blind** -- pinned across the whole severity range, so it has no resolution
  here whichever way it nominally points.

Two things are reported alongside the sign, because a correlation alone hides
both of the pathologies that actually matter. `span` is the ratio of the
metric's value at full severity to its clean value, so a metric can be
monotone and still useless. `flat` flags a metric pinned across the range
(Vendi at the sample count, DPP underflowed to zero), where it has no resolution
left regardless of direction.

Example:
    python -m engiopt.lvae.distortion_report --csv distortion_photonics2d.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from engiopt.lvae.distortion_capture import EXPECTED
from engiopt.lvae.distortion_capture import METRIC_NAMES

WEAK_SPAN = 1.15
"""Below a 15% move across the full severity range, a metric cannot separate."""
STRONG_RHO = 0.5
"""Spearman magnitude below this is not a reliable monotone response."""
FLAT_TOLERANCE = 0.01
"""Total movement below 1% of the metric's own scale counts as no resolution."""
ENDPOINT_TOLERANCE = 1.05
"""A corrupted set scoring >5% better than the clean one has been rewarded."""


def classify(rho: float, span: float, expected: str, *, flat: bool, endpoint_ratio: float) -> str:
    """Verdict for one (metric, failure kind) cell.

    `flat` means the metric never moved across the whole severity range, so it
    has no resolution here regardless of which way it points -- a Vendi score
    pinned at the sample count, or a DPP determinant underflowed to zero.

    `endpoint_ratio` is worst-severity over clean, and it is checked separately
    from the correlation because the correlation cannot see a U-shape. A
    diversity score that dips at low severity and then climbs past its clean
    value has been *rewarded* by the corruption, while its Spearman reads
    negative and looks healthy.
    """
    if not np.isfinite(rho):
        return "n/a"
    moved = abs(rho) >= STRONG_RHO and span >= WEAK_SPAN

    # Ending above the clean baseline is a reward however the path got there.
    ended_better = np.isfinite(endpoint_ratio) and endpoint_ratio > ENDPOINT_TOLERANCE

    if expected in ("0", "<=0"):
        # "<=0" is one-sided: rising is the pathology, falling or flat is fine.
        rose = ended_better or (moved and rho > 0)
        if rose or (expected == "0" and moved):
            return "REWARDED"
        return "blind" if (flat and expected == "<=0") else "ok"
    want = 1 if expected == "+" else -1
    if flat or np.sign(rho) != want:
        return "blind" if (flat or not moved) else "WRONG"
    return "ok" if moved else "weak"


def main() -> None:
    """Print the failure-response grid and the list of outright failures."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    metrics = [m for m in METRIC_NAMES if m in df.columns]

    records = []
    # Grouped per family, not per kind: the five fidelity corruptions do not
    # behave alike, and pooling them lets a pathology specific to one (noise
    # raising a diversity score) be averaged away by the other four.
    for (instrument, family), sub in df.groupby(["instrument", "family"], sort=False):
        kind = str(sub["kind"].iloc[0])
        expected = EXPECTED[kind]
        for metric in metrics:
            values = sub[metric].to_numpy(dtype=float)
            sev = sub["severity"].to_numpy(dtype=float)
            ok = np.isfinite(values)
            rho = stats.spearmanr(sev[ok], values[ok]).statistic if ok.sum() > 2 else np.nan  # noqa: PLR2004

            clean = np.nanmedian(values[sev == 0])
            worst = np.nanmedian(values[sev == sev.max()])
            endpoint_ratio = np.nan
            if not np.isfinite(clean) or clean == 0:
                span = np.inf
            else:
                ratio = abs(worst / clean)
                span = max(ratio, 1 / ratio) if ratio > 0 else np.inf
            endpoint_ratio = worst / clean if np.isfinite(clean) and clean != 0 else np.nan

            # No resolution left: the metric is pinned across the whole range.
            # Detected from the data rather than from a known bound, so it also
            # catches a determinant underflowed to a constant zero.
            spread = np.nanmax(values[ok]) - np.nanmin(values[ok]) if ok.any() else 0.0
            scale = max(abs(np.nanmedian(values[ok])), 1e-12) if ok.any() else 1.0
            flat = bool(spread / scale < FLAT_TOLERANCE)
            records.append(
                {
                    "instrument": instrument,
                    "family": family,
                    "kind": kind,
                    "metric": metric,
                    "expected": expected.get(metric, "0"),
                    "rho": rho,
                    "span": span,
                    "verdict": classify(rho, span, expected.get(metric, "0"), flat=flat, endpoint_ratio=endpoint_ratio),
                    "endpoint_ratio": endpoint_ratio,
                    "flat": flat,
                }
            )

    res = pd.DataFrame(records)
    pd.set_option("display.width", 250)

    for instrument, sub in res.groupby("instrument", sort=False):
        print(f"\n=== {instrument}: Spearman(metric, severity) — expected sign in brackets ===")
        grid = sub.pivot_table(index="metric", columns="family", values="rho", aggfunc="first").reindex(metrics)
        signs = sub.pivot_table(index="metric", columns="family", values="expected", aggfunc="first").reindex(metrics)
        shown = grid.copy().astype(object)
        for r in grid.index:
            for c in grid.columns:
                want = signs.loc[r, c]
                shown.loc[r, c] = f"{grid.loc[r, c]:+.2f} [{want}]" if np.isfinite(grid.loc[r, c]) else "n/a"
        print(shown.to_string())

        print(f"\n--- {instrument}: verdicts ---")
        print(
            sub.pivot_table(index="metric", columns="family", values="verdict", aggfunc="first")
            .reindex(metrics)
            .to_string()
        )

    bad = res[res["verdict"].isin(("WRONG", "REWARDED", "blind"))]
    if len(bad):
        print("\n=== outright failures ===")
        print(
            bad[["instrument", "kind", "metric", "rho", "span", "verdict"]].to_string(
                index=False, float_format=lambda v: f"{v:.3g}"
            )
        )

    out = args.out or args.csv.replace(".csv", "_verdicts.csv")
    res.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
