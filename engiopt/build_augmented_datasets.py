"""Build augmented EngiBench-2D design datasets for the LV-metrics diagnostics.

**Capture-only and fully decoupled.** This script imports *no* engiopt metric code
and loads *no* LVAE — it only transforms designs and writes a HuggingFace dataset.
That makes it independent of the pending metrics refactor: the augmented designs are
frozen artifacts, and every metric (residual / l_perf / condition-recovery / LV-MMD /
coverage / diversity) is computed *later*, downstream, by reading this dataset.

Each augmentation is kept only if it exposes a *specific*, verifiable failure mode
with unambiguous ground truth. Every row carries two labels so the downstream scoring
knows exactly what each metric *should* say:

  is_valid_design       design still on the good-design manifold?  (validity axis)
  is_condition_matched  design still matches its STATED condition?  (conditional axis)

Curated families -- together they cover validity + conditional failure modes:

  family           valid  cond   what it reveals
  ---------------  -----  -----  ------------------------------------------------------
  clean            T      T      the on-manifold, correctly-paired baseline
  gaussian_noise   F      T      off-manifold hi-freq: residual^, latent invariant,
                                 pixel-DPP *rewards* it (anti-gaming headline); also the
                                 building block for the mode-collapse diversity probe
  blur             F      T      off-manifold lo-freq: a distinct validity corruption so
                                 the monotonic-sensitivity claim rests on >1 failure type
  intensity_shift  F      F      shifts mean density = shifts volfrac -> EXACT analytic
                                 condition ground truth (validates condition-recovery)
  condition_jumble T      F      the 'right design, wrong problem' hard negative:
                                 real optimum relabeled with another row's condition;
                                 every UNconditional metric (pixel OR latent) scores it
                                 perfect

Coverage/diversity failures (mode_drop, mode_invent, collapse) are constructed
DOWNSTREAM by subsetting/repeating these rows by condition band, so they need no baked
family. Flip-symmetry families were evaluated and DROPPED: no 2D problem here has a
DOF-verified flip symmetry (the MBB beam is a half-model with asymmetric BCs), so a
flipped design is off-manifold, not a clean wrong-condition negative -- and
condition_jumble already carries the conditional axis without any symmetry assumption.

Example:
    python -m engiopt.build_augmented_datasets \\
        --problem-id beams2d --n-samples 128 --n-levels 6 --n-repeats 3 \\
        --output-dir aug_out --push-to-hub IDEALLab/beams_2d_50_100_v0_augmented
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import os

from datasets import Array2D
from datasets import Dataset
from datasets import Features
from datasets import Value
from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter
import tyro

# Canonical EngiOpt data-layer helpers (no metrics/LVAE code -> refactor-safe).
from engiopt.transforms import get_image_condition_keys
from engiopt.transforms import get_scalar_condition_keys

# Note on flip-symmetry augmentations (considered, then dropped): a flip is only a
# valid augmentation if it is a symmetry of the *modeled domain's* boundary conditions,
# verified at the DOF level -- NOT from a docstring. Half-models spend their geometric
# symmetry to exist: beams2d fixes {left-edge x (symmetry plane), bottom-right y
# (roller)}, which is asymmetric, so an hflip/vflip lands outside the problem family
# (off-manifold), not on a valid wrong-condition design. None of the 2D problems here
# gave a confirmed, useful flip symmetry, and condition_jumble already covers the
# conditional axis without one, so no flip family is emitted.


@dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "beams2d"
    """EngiBench 2D problem identifier."""
    split: str = "test"
    """Dataset split to augment (held out from LVAE/generator training)."""
    n_samples: int = 128
    """Number of source designs to augment (capped at split size)."""
    n_levels: int = 6
    """Severity levels per graded family, including clean (severity=0)."""
    n_repeats: int = 3
    """Independent stochastic draws per (graded family, severity>0) and per relabel."""
    seed: int = 0
    """Base RNG seed (deterministic output)."""
    output_dir: str = "aug_out"
    """Local directory to save the dataset (Arrow) + manifest JSON."""
    push_to_hub: str | None = None
    """Optional HF repo id (e.g. 'IDEALLab/beams_2d_50_100_v0_augmented'); pushes if set."""
    families: tuple[str, ...] = field(
        default_factory=lambda: (
            "gaussian_noise",
            "blur",
            "intensity_shift",
            "condition_jumble",
        )
    )
    """Augmentation families to build (clean baseline is always emitted)."""


# ---------------------------------------------------------------------------
# Per-design transforms. severity in [0,1]; severity=0 is identity for graded ones.
# (Copied here on purpose so this module imports nothing from the metrics code.)
# ---------------------------------------------------------------------------


def _gaussian_noise(x: npt.NDArray, s: float, rng: np.random.Generator) -> npt.NDArray:
    return np.clip(x + rng.normal(0.0, 0.5 * s, size=x.shape).astype(np.float32), 0.0, 1.0)


def _blur(x: npt.NDArray, s: float, _rng: np.random.Generator) -> npt.NDArray:
    if s == 0:
        return x.copy()
    return gaussian_filter(x, sigma=5.0 * s).astype(np.float32)


def _intensity_shift(x: npt.NDArray, s: float, rng: np.random.Generator) -> npt.NDArray:
    # Uniform density shift -> changes mean density == volume fraction (the condition).
    sign = 1.0 if rng.random() < 0.5 else -1.0
    return np.clip(x + sign * 0.4 * s, 0.0, 1.0).astype(np.float32)


GRADED = {
    "gaussian_noise": _gaussian_noise,
    "blur": _blur,
    "intensity_shift": _intensity_shift,
}
# Graded families that leave the *stated* condition valid (design corrupted, not moved
# to a different condition). intensity_shift is the exception: it changes volfrac.
GRADED_KEEPS_CONDITION = {"gaussian_noise": True, "blur": True, "intensity_shift": False}

# Per-family seed offsets, so each family draws its own noise stream. These are fixed
# constants and NOT hash(family): Python salts string hashes per process (PYTHONHASHSEED),
# so seeding from hash() would silently produce different corrupted designs on every run
# -- and different designs across SLURM shards that each rebuild the dataset. The whole
# point of this artifact is that --seed reproduces it exactly.
FAMILY_SEED_OFFSET = {"gaussian_noise": 101, "blur": 202, "intensity_shift": 303}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def main(args: Args) -> None:  # noqa: PLR0915, PLR0912
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    design_shape = tuple(problem.design_space.shape)
    assert len(design_shape) == 2, f"2D problems only, got shape {design_shape}"
    h, w = design_shape

    if args.split not in problem.dataset:
        raise ValueError(f"split '{args.split}' not in {list(problem.dataset.keys())}")
    ds = problem.dataset[args.split]

    # Scalar condition columns only -- the canonical EngiOpt accessor
    # (get_scalar_condition_keys uses problem.conditions_keys under the hood).
    # Array-valued conditions (thermoelastic2d's 65x65 boundary maps) are excluded
    # from the dense storage and condition-based augmentations; they still identify
    # the source row. This mirrors the paper's scalar-condition scoping.
    cond_keys = get_scalar_condition_keys(problem, ds)
    image_cond_keys = get_image_condition_keys(problem, ds)
    if not cond_keys:
        raise ValueError(f"no scalar condition columns found among {ds.column_names}")
    if image_cond_keys:
        print(
            f"[warn] {args.problem_id} has image conditions {image_cond_keys} -- excluded from "
            "condition augmentations (scalar-condition stress test)"
        )

    n = min(args.n_samples, len(ds))
    rng0 = np.random.default_rng(args.seed)
    idx = rng0.permutation(len(ds))[:n]

    designs = np.asarray(ds["optimal_design"], dtype=np.float32).reshape(len(ds), h, w)[idx]
    # per-source stated/true condition dicts (floats; bool conditions cast to 0/1)
    conds = [{k: float(ds[int(i)][k]) for k in cond_keys} for i in idx]

    # Detect the volume-fraction-like condition by meaning, not by name: it is the
    # scalar condition whose value equals mean(design). Named volfrac (beams2d),
    # volume (heatconduction2d), volume_fraction_target (thermoelastic2d), or absent
    # (photonics2d). intensity_shift changes mean density -> it violates exactly this
    # condition, and nothing else.
    means = designs.mean(axis=(1, 2))
    vf_key = next(
        (k for k in cond_keys if np.allclose([c[k] for c in conds], means, atol=0.03)),
        None,
    )
    if "intensity_shift" in args.families:
        print(f"[info] volume-fraction condition = {vf_key!r} (intensity_shift target)")

    # accumulate columns
    col_design: list[npt.NDArray] = []
    col_family: list[str] = []
    col_sev: list[float] = []
    col_rep: list[int] = []
    col_src: list[int] = []
    col_valid: list[bool] = []
    col_matched: list[bool] = []
    col_stated: dict[str, list[float]] = {f"stated_{k}": [] for k in cond_keys}
    col_true: dict[str, list[float]] = {f"true_{k}": [] for k in cond_keys}

    def emit(design, family, sev, rep, src, valid, matched, stated, true):
        col_design.append(np.asarray(design, dtype=np.float32))
        col_family.append(family)
        col_sev.append(float(sev))
        col_rep.append(int(rep))
        col_src.append(int(src))
        col_valid.append(bool(valid))
        col_matched.append(bool(matched))
        for k in cond_keys:
            col_stated[f"stated_{k}"].append(float(stated[k]))
            col_true[f"true_{k}"].append(float(true[k]))

    # --- clean baseline (on-manifold, correctly paired) ------------------------
    for j in range(n):
        emit(designs[j], "clean", 0.0, 0, int(idx[j]), True, True, conds[j], conds[j])

    severities = np.linspace(0.0, 1.0, args.n_levels)[1:]  # skip 0 (covered by clean)

    for family in args.families:
        if family in GRADED:
            keeps_cond = GRADED_KEEPS_CONDITION[family]
            for sev in severities:
                for rep in range(args.n_repeats):
                    for j in range(n):
                        rng = np.random.default_rng((args.seed, FAMILY_SEED_OFFSET[family], int(sev * 1000), rep, j))
                        aug = GRADED[family](designs[j], float(sev), rng)
                        true = dict(conds[j])
                        matched = True
                        # intensity_shift changes mean density; where a volume-fraction
                        # condition exists (mean == that condition), it is now violated.
                        if not keeps_cond and vf_key is not None:
                            true[vf_key] = float(np.asarray(aug).mean())
                            matched = False
                        emit(aug, family, sev, rep, int(idx[j]), False, matched, conds[j], true)

        elif family == "condition_jumble":
            for rep in range(args.n_repeats):
                for j in range(n):
                    rng = np.random.default_rng((args.seed, 777, rep, j))
                    other = int(rng.integers(0, n))
                    while other == j and n > 1:
                        other = int(rng.integers(0, n))
                    stated = conds[other]  # asked for a DIFFERENT condition
                    matched = all(abs(stated[k] - conds[j][k]) < 1e-6 for k in cond_keys)  # noqa: PLR2004
                    # design unchanged and valid; only the pairing is wrong
                    emit(designs[j], family, 1.0, rep, int(idx[j]), True, matched, stated, conds[j])

    # --- assemble HF dataset ---------------------------------------------------
    features = Features(
        {
            "design": Array2D(shape=(h, w), dtype="float32"),
            "family": Value("string"),
            "severity": Value("float32"),
            "repeat": Value("int32"),
            "source_idx": Value("int32"),
            "is_valid_design": Value("bool"),
            "is_condition_matched": Value("bool"),
            **{f"stated_{k}": Value("float32") for k in cond_keys},
            **{f"true_{k}": Value("float32") for k in cond_keys},
        }
    )
    data = {
        "design": col_design,
        "family": col_family,
        "severity": col_sev,
        "repeat": col_rep,
        "source_idx": col_src,
        "is_valid_design": col_valid,
        "is_condition_matched": col_matched,
        **col_stated,
        **col_true,
    }
    dataset = Dataset.from_dict(data, features=features)

    # per-family label matrix for a sanity check
    manifest = {
        "problem_id": args.problem_id,
        "split": args.split,
        "design_shape": [h, w],
        "condition_keys": cond_keys,
        "n_source": n,
        "n_rows": len(dataset),
        "seed": args.seed,
        "families": {},
    }
    fam_arr = np.asarray(col_family)
    for fam in ["clean", *args.families]:
        m = fam_arr == fam
        if not m.any():
            continue
        manifest["families"][fam] = {
            "rows": int(m.sum()),
            "is_valid_design": bool(np.asarray(col_valid)[m].all()),
            "is_condition_matched": bool(np.asarray(col_matched)[m].all()),
            "any_matched": bool(np.asarray(col_matched)[m].any()),
        }

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{args.problem_id}_{args.split}_augmented")
    dataset.save_to_disk(save_path)
    with open(os.path.join(args.output_dir, f"{args.problem_id}_{args.split}_augmented_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nbuilt {len(dataset)} rows -> {save_path}")
    print("per-family (rows | valid | cond-matched):")
    for fam, info in manifest["families"].items():
        print(
            f"  {fam:16s} {info['rows']:6d} | valid={info['is_valid_design']!s:5s} | "
            f"matched={info['is_condition_matched']!s:5s} (any={info['any_matched']!s})"
        )

    if args.push_to_hub:
        print(f"\npushing to hub: {args.push_to_hub}")
        dataset.push_to_hub(args.push_to_hub)
        print("pushed.")


if __name__ == "__main__":
    main(tyro.cli(Args))
