"""
Drop-in adapter: converts processed new-dataset pickle files into an iterable
of dicts that match the EngiBench Wings3D item format consumed by train_lvae.py
and train_ddm.py.

Expected item keys (same as Wings3D):
    case_num, slice_num, coords [9,192,2], initial, final,
    mach, reynolds, cl_target, area_case_ratio,
    alpha, area_initial (placeholder -9999), cd_val, cl_val,
    cl_con, area_con, coef_pressure [9,192],
    velocity_x [9,192], velocity_y [9,192], velocity_z [9,192],
    transforms [9],   ← span-wise eta positions
    te_shifts [9],    ← trailing-edge y-offsets (computed from coords)

Usage:
    from engiopt.data_processing.new_dataset_adapter import NewWingsDataset
    dataset = NewWingsDataset(slices_pkl, scalars_pkl, split="train")
    for item in dataset:
        ...   # same code path as Wings3D
"""

import pickle
import numpy as np
import pandas as pd
from typing import Iterator


# Fraction of cases used for train / val / test splits.
# Matches the EngiBench Wings3D convention (80 / 10 / 10).
_SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}

# Cases whose final cl constraint violation exceeds this are excluded.
_CL_CON_THRESHOLD = 0.01

_MACH_MAX = float("inf")


def _scalar_row(df_scalars: pd.DataFrame, case_num: int, final: bool) -> pd.Series:
    """Return the initial or final scalar row for a given case."""
    rows = df_scalars[df_scalars["case_num"] == case_num]
    rows = rows.sort_values("iter")
    return rows.iloc[-1] if final else rows.iloc[0]


def _build_item(
    case_num: int,
    slice_num: int,
    df_slice_group: pd.DataFrame,
    scalar_row: pd.Series,
    is_initial: bool,
    is_final: bool,
) -> dict:
    """Build a single Wings3D-compatible item dict from processed slice data."""
    sub_slice_nums = df_slice_group["sub_slice_num"].unique()
    etas = df_slice_group.groupby("sub_slice_num")["eta"].first()

    # Sort by eta (root → tip)
    sort_idx = np.argsort(etas.values)
    sub_slice_nums_sorted = sub_slice_nums[sort_idx]
    etas_sorted = etas.values[sort_idx]

    n_slices = len(sub_slice_nums_sorted)
    n_pts = len(df_slice_group[df_slice_group["sub_slice_num"] == sub_slice_nums_sorted[0]])

    coords        = np.full((n_slices, n_pts, 2), -9999.0, dtype=np.float32)
    coef_pressure = np.full((n_slices, n_pts),    -9999.0, dtype=np.float32)
    velocity_x    = np.full((n_slices, n_pts),    -9999.0, dtype=np.float32)
    velocity_y    = np.full((n_slices, n_pts),    -9999.0, dtype=np.float32)
    velocity_z    = np.full((n_slices, n_pts),    -9999.0, dtype=np.float32)

    for i, ssn in enumerate(sub_slice_nums_sorted):
        row_df = df_slice_group[df_slice_group["sub_slice_num"] == ssn]
        coords[i, :, 0] = row_df["CoordinateX"].values
        coords[i, :, 1] = row_df["CoordinateY"].values
        coef_pressure[i] = row_df["CoefPressure"].values
        velocity_x[i]    = row_df["VelocityX"].values
        velocity_y[i]    = row_df["VelocityY"].values
        velocity_z[i]    = row_df["VelocityZ"].values

    # te_shifts: trailing-edge y-offsets per span slice (first point = TE)
    te_shifts = coords[:, 0, 1].copy()  # [n_slices]

    geo_params = np.array([
        float(scalar_row["case_sweep_value"]),
        float(scalar_row["case_base_span_scaling_value"]),
        float(scalar_row["case_initial_spanwise_taper_scaling_value"]),
        float(scalar_row["case_initial_thickness_taper_scaling_value"]),
        float(scalar_row["case_chord_scaling_value0"]),
        float(scalar_row["case_chord_scaling_value1"]),
        float(scalar_row["case_chord_scaling_value2"]),
        float(scalar_row["case_chord_scaling_value3"]),
        float(scalar_row["case_chord_scaling_value4"]),
        float(scalar_row["case_chord_scaling_value5"]),
        float(scalar_row["case_chord_scaling_value6"]),
        float(scalar_row["case_chord_scaling_value7"]),
        float(scalar_row["case_twist_value0"]),
        float(scalar_row["case_twist_value1"]),
        float(scalar_row["case_twist_value2"]),
        float(scalar_row["case_twist_value3"]),
        float(scalar_row["case_twist_value4"]),
        float(scalar_row["case_twist_value5"]),
        float(scalar_row["case_twist_value6"]),
        float(scalar_row["case_thickness_taper_value0"]),
        float(scalar_row["case_thickness_taper_value1"]),
        float(scalar_row["case_thickness_taper_value2"]),
        float(scalar_row["case_thickness_taper_value3"]),
        float(scalar_row["case_thickness_taper_value4"]),
        float(scalar_row["case_thickness_taper_value5"]),
        float(scalar_row["case_thickness_taper_value6"]),
        float(scalar_row["case_thickness_taper_value7"]),
        float(scalar_row["case_dihedral_value0"]),
        float(scalar_row["case_dihedral_value1"]),
        float(scalar_row["case_dihedral_value2"]),
        float(scalar_row["case_dihedral_value3"]),
        float(scalar_row["case_dihedral_value4"]),
        float(scalar_row["case_dihedral_value5"]),
        float(scalar_row["case_dihedral_value6"]),
    ], dtype=np.float32)  # [34]

    return {
        "case_num":        int(case_num),
        "slice_num":       int(slice_num),
        "coords":          coords,                    # [9,192,2]
        "initial":         int(is_initial),
        "final":           int(is_final),
        "mach":            float(scalar_row["case_mach"]),
        "reynolds":        float(scalar_row["case_reynolds"]),
        "cl_target":       float(scalar_row["case_cl_target"]),
        "area_case_ratio": float(scalar_row["case_volume_ratio_min"]),
        "alpha":           float(scalar_row["alpha"]),
        "area_initial":    -9999.0,
        "cd_val":          float(scalar_row["cd"]),
        "cl_val":          float(scalar_row["cl"]),
        "cl_con":          float(scalar_row.get("cl_con", -9999.0)),
        "area_con":        float(scalar_row.get("vol_con", -9999.0)),
        "transforms":      etas_sorted.astype(np.float32),   # [9] span positions
        "te_shifts":       te_shifts.astype(np.float32),     # [9]
        "coef_pressure":   coef_pressure,             # [9,192]
        "velocity_x":      velocity_x,                # [9,192]
        "velocity_y":      velocity_y,                # [9,192]
        "velocity_z":      velocity_z,                # [9,192]
        "geo_params":      geo_params,                # [34] geometric scalars
    }


class NewWingsDataset:
    """Iterable dataset backed by processed pickle files.

    Emits dicts in the same format as EngiBench Wings3D so that
    train_lvae.py and train_ddm.py can be used without modification —
    just replace:
        problem = Wings3D(seed=cfg.seed)
        all_train = list(problem.dataset["train"])
    with:
        dataset = NewWingsDataset(slices_pkl, scalars_pkl, seed=cfg.seed)
        all_train = list(dataset["train"])
    """

    def __init__(
        self,
        slices_pkl: str,
        scalars_pkl: str,
        seed: int = 0,
    ):
        with open(slices_pkl, "rb") as f:
            self._df_slices: pd.DataFrame = pickle.load(f)
        with open(scalars_pkl, "rb") as f:
            self._df_scalars: pd.DataFrame = pickle.load(f)

        # Determine case split reproducibly
        all_cases = sorted(self._df_slices["case_num"].unique())
        rng = np.random.default_rng(seed)
        shuffled = np.array(all_cases, dtype=int)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(n_total * _SPLIT_FRACTIONS["train"])
        n_val   = int(n_total * _SPLIT_FRACTIONS["val"])

        self._splits = {
            "train": set(shuffled[:n_train].tolist()),
            "val":   set(shuffled[n_train : n_train + n_val].tolist()),
            "test":  set(shuffled[n_train + n_val :].tolist()),
        }

    def __getitem__(self, split: str) -> "_SplitView":
        if split not in self._splits:
            raise KeyError(f"Unknown split '{split}'. Choose from {list(self._splits)}")
        return _SplitView(self._df_slices, self._df_scalars, self._splits[split])


class _SplitView:
    """Lazy iterator over one split of the dataset."""

    def __init__(
        self,
        df_slices: pd.DataFrame,
        df_scalars: pd.DataFrame,
        case_set: set,
    ):
        self._df_slices  = df_slices[df_slices["case_num"].isin(case_set)]
        self._df_scalars = df_scalars
        self._case_set   = case_set

    def _valid_cases(self) -> set:
        """Return the subset of case_set that has converged cl_con."""
        valid = set()
        for case_num in self._case_set:
            initial_row = _scalar_row(self._df_scalars, case_num, final=False)
            if float(initial_row.get("case_mach", 0.0)) >= _MACH_MAX:
                continue
            final_row = _scalar_row(self._df_scalars, case_num, final=True)
            cl_con = float(final_row.get("cl_con", 0.0))
            if abs(cl_con) <= _CL_CON_THRESHOLD:
                valid.add(case_num)
        return valid

    def __len__(self) -> int:
        # Two items per case: initial (slice_num=0) + final (slice_num=1)
        return len(self._valid_cases()) * 2

    def __iter__(self) -> Iterator[dict]:
        converged = self._valid_cases()
        for case_num in sorted(converged):
            df_case = self._df_slices[self._df_slices["case_num"] == case_num]
            for slice_num in sorted(df_case["slice_num"].unique()):
                is_initial = (slice_num == 0)
                is_final   = (slice_num == df_case["slice_num"].max())
                group = df_case[df_case["slice_num"] == slice_num]
                scalar_row = _scalar_row(self._df_scalars, case_num, final=is_final)
                yield _build_item(
                    case_num, slice_num, group, scalar_row, is_initial, is_final
                )
