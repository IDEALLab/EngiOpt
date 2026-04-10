"""Utilities for selecting the best model checkpoint based on validation metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch as th


class BestEpochTracker:
    """Tracks validation metrics and manages best checkpoint selection."""

    def __init__(
        self,
        checkpoint_dir: Path | str,
        metric_name: str = "mmd",
        maximize: bool = False,
    ):
        """Initialize tracker.

        Args:
            checkpoint_dir: Directory where checkpoints are saved.
            metric_name: Name of the metric to track (e.g., "mmd", "dpp").
            maximize: If True, higher is better. If False, lower is better.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metric_name = metric_name
        self.maximize = maximize
        self.best_metric_value: float | None = None
        self.best_epoch: int | None = None
        self.metrics_history: dict[int, float] = {}
        self.metrics_file = self.checkpoint_dir / "validation_metrics.json"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def update(self, epoch: int, metric_value: float) -> bool:
        """Update tracker with new validation metric.

        Args:
            epoch: Current epoch number.
            metric_value: Value of the validation metric.

        Returns:
            bool: True if this is the best metric value so far.
        """
        self.metrics_history[epoch] = metric_value

        is_best = False
        if self.best_metric_value is None:
            is_best = True
        elif self.maximize and metric_value > self.best_metric_value:
            is_best = True
        elif not self.maximize and metric_value < self.best_metric_value:
            is_best = True

        if is_best:
            self.best_metric_value = metric_value
            self.best_epoch = epoch

        # Save metrics to file for reference
        self._save_metrics()

        return is_best

    def _save_metrics(self) -> None:
        """Save metrics history to JSON file."""
        history_to_save = {
            "best_epoch": self.best_epoch,
            "best_metric_value": float(self.best_metric_value)
            if self.best_metric_value is not None
            else None,
            "metric_name": self.metric_name,
            "maximize": self.maximize,
            "all_metrics": {str(k): v for k, v in self.metrics_history.items()},
        }
        with open(self.metrics_file, "w") as f:
            json.dump(history_to_save, f, indent=2)

    def get_best_checkpoint_path(
        self, checkpoint_template: str = "checkpoint_epoch_{epoch}.pth"
    ) -> Path | None:
        """Get path to best checkpoint.

        Args:
            checkpoint_template: Template for checkpoint filename.
                Should contain "{epoch}" placeholder.

        Returns:
            Path to best checkpoint, or None if no checkpoints recorded.
        """
        if self.best_epoch is None:
            return None
        return self.checkpoint_dir / checkpoint_template.format(epoch=self.best_epoch)

    @staticmethod
    def compute_mmd_batch(
        generated: th.Tensor,
        reference: th.Tensor,
        sigma: float = 1.0,
    ) -> float:
        """Compute MMD between two batches (on CPU using numpy for efficiency).

        Args:
            generated: Tensor of shape (n, ...) with generated samples.
            reference: Tensor of shape (m, ...) with reference samples.
            sigma: Bandwidth parameter.

        Returns:
            MMD value.
        """
        from scipy.spatial.distance import cdist

        # Convert to numpy and flatten
        gen_np = generated.detach().cpu().numpy().reshape(generated.shape[0], -1)
        ref_np = reference.detach().cpu().numpy().reshape(reference.shape[0], -1)

        # Compute kernel matrices
        k_xx = np.exp(-cdist(gen_np, gen_np, "sqeuclidean") / (2 * sigma**2))
        k_yy = np.exp(-cdist(ref_np, ref_np, "sqeuclidean") / (2 * sigma**2))
        k_xy = np.exp(-cdist(gen_np, ref_np, "sqeuclidean") / (2 * sigma**2))

        mmd_val = float(k_xx.mean() + k_yy.mean() - 2 * k_xy.mean())
        return mmd_val

    @staticmethod
    def compute_dpp_batch(
        generated: th.Tensor,
        sigma: float = 1.0,
    ) -> float:
        """Compute DPP diversity metric.

        Args:
            generated: Tensor of shape (n, ...) with generated samples.
            sigma: Bandwidth parameter.

        Returns:
            DPP diversity value.
        """
        from scipy.spatial.distance import cdist

        gen_np = generated.detach().cpu().numpy().reshape(generated.shape[0], -1)
        pairwise_sq_dists = cdist(gen_np, gen_np, "sqeuclidean")
        similarity_matrix = np.exp(-pairwise_sq_dists / (2 * sigma**2))
        reg_matrix = similarity_matrix + 1e-6 * np.eye(gen_np.shape[0])

        try:
            dpp_val = float(np.linalg.det(reg_matrix))
        except np.linalg.LinAlgError:
            dpp_val = 0.0

        return dpp_val
