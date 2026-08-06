"""Dataset preparation shared by the LVAE training scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset


def filter_dataset_by_condition(
    dataset: Dataset,
    key: str,
    value: float | None = None,
    value_range: tuple[float, float] | None = None,
    tolerance: float = 0.01,
) -> Dataset:
    """Filter a dataset split down to rows matching one scalar condition.

    Training an LVAE on a single condition slice is how a multi-modal design
    space gets separated into its parts: `thermoelastic2d` at `weight=0.0` and
    at `weight=1.0` are effectively different manifolds, and an autoencoder
    fitted to both at once measures the union rather than either.

    Call this **before** `dataset.with_format("torch")`; HuggingFace's `filter`
    operates on the raw, non-tensor data.

    Args:
        dataset: A dataset split, e.g. `problem.dataset["train"]`.
        key: Scalar condition column to filter on, e.g. `"weight"`.
        value: Exact target value; rows within `tolerance` are kept.
        value_range: Inclusive `(lo, hi)` bounds. Takes precedence over `value`.
        tolerance: Tolerance for exact-value matching.

    Returns:
        The filtered dataset.

    Raises:
        ValueError: If `key` is not a column, or neither `value` nor
            `value_range` is given.
    """
    if key not in dataset.column_names:
        raise ValueError(f"condition key {key!r} not in dataset columns: {dataset.column_names}")
    if value is None and value_range is None:
        raise ValueError("pass either 'value' or 'value_range' to filter on")

    if value_range is not None:
        lo, hi = value_range
        filtered = dataset.filter(lambda row: lo <= float(row[key]) <= hi)
    else:
        target = float(value)  # type: ignore[arg-type]
        filtered = dataset.filter(lambda row: abs(float(row[key]) - target) < tolerance)

    print(f"Filtered dataset by {key}: {len(dataset)} -> {len(filtered)} samples")
    return filtered
