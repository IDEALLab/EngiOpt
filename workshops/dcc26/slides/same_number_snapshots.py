"""Shared engine: pick + render beams2d dataset snapshots for two 'profiles'.

Both the static slide (build_same_number_pptx.py) and the live widget notebook
use ``select_indices`` / ``render_grid`` so the snapshots stay identical.

The narrative: two papers report the SAME headline number (MSE = 0.04) but
trained/evaluated on very different slices of the SAME task. Each "profile" is a
set of toggles that map onto real beams2d dataset filters:

    budget  : "fixed"   -> volfrac clustered near 0.35
              "sampled" -> volfrac across the full 0.15-0.40 range
    filter  : "sharp"   -> small filter radius rmin ~ 1.5  (crisp members)
              "smeared" -> large filter radius rmin >= 3.25 (gray boundaries)
    split   : "mean"    -> force position near the training mean (~0.5)
              "ood"     -> force position at the OOD corners (0.0 / 1.0)

Run standalone (EngiBench312) to export the two static PNGs:

    python workshops/dcc26/slides/same_number_snapshots.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "assets"))

# Two profiles used on the slide. Each value is a toggle the widget can flip.
PAPER_A = {"budget": "fixed", "filter": "sharp", "split": "mean"}
PAPER_B = {"budget": "sampled", "filter": "smeared", "split": "ood"}


def _load():
    from engibench.utils.all_problems import BUILTIN_PROBLEMS

    p = BUILTIN_PROBLEMS["beams2d"]()
    p.reset(seed=0)
    tr = p.dataset["train"]
    return {
        "design": np.array(tr["optimal_design"]),
        "volfrac": np.array(tr["volfrac"]),
        "rmin": np.array(tr["rmin"]),
        "forcedist": np.array(tr["forcedist"]),
    }


def select_indices(data: dict, profile: dict, n: int = 6, seed: int = 0) -> list[int]:
    """Indices of dataset designs matching a profile's toggles."""
    vf, rmin, fd = data["volfrac"], data["rmin"], data["forcedist"]
    mask = np.ones(len(vf), dtype=bool)

    if profile["budget"] == "fixed":
        mask &= np.abs(vf - 0.35) <= 0.02
    # "sampled" -> no volfrac restriction (full range)

    if profile["filter"] == "sharp":
        mask &= rmin <= 1.75
    else:  # smeared
        mask &= rmin >= 3.25

    if profile["split"] == "mean":
        mask &= (fd >= 0.4) & (fd <= 0.6)
    else:  # ood corners
        mask &= (fd <= 0.05) | (fd >= 0.95)

    pool = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    if len(pool) <= n:
        return [int(i) for i in pool]

    # Spread across (volfrac, forcedist) so a "diverse" profile actually looks
    # diverse; a clustered profile will look uniform regardless.
    pts = np.stack([vf[pool], fd[pool]], axis=1)
    mn = pts.min(0); sp = np.where(pts.max(0) - mn > 0, pts.max(0) - mn, 1.0)
    norm = (pts - mn) / sp
    start = int(rng.integers(len(pool)))
    chosen = [start]
    d = np.linalg.norm(norm - norm[start], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(norm - norm[nxt], axis=1))
    return [int(pool[j]) for j in chosen]


def render_grid(data: dict, idxs: list[int], *, rows=2, cols=3, ax=None, fig=None):
    """Render the selected designs as a tight rows x cols viridis grid."""
    own = ax is None
    if own:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 0.9))
    else:
        axes = ax
    flat = np.array(axes).ravel()
    designs = data["design"]
    for k, a in enumerate(flat):
        if k < len(idxs):
            a.imshow(designs[idxs[k]], cmap="viridis", vmin=0, vmax=1, aspect="auto")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    if own:
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.06, hspace=0.06)
    return fig


def profile_caption(data: dict, idxs: list[int]) -> str:
    vf = data["volfrac"][idxs]; rmin = data["rmin"][idxs]
    return (f"volfrac {vf.min():.2f}-{vf.max():.2f}   rmin {rmin.min():.1f}-{rmin.max():.1f}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    data = _load()
    for name, profile in (("paperA", PAPER_A), ("paperB", PAPER_B)):
        idxs = select_indices(data, profile)
        fig = render_grid(data, idxs)
        path = os.path.join(OUT_DIR, f"same_number_{name}.png")
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"wrote {path}   [{profile_caption(data, idxs)}]")


if __name__ == "__main__":
    main()
