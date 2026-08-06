"""Configuration recovered from a trained LVAE checkpoint package.

A checkpoint stores weights; rebuilding the network around them needs the
architecture arguments the run was trained with. Those live in the package's
`run_config.json`, written by `engiopt.checkpoint_store.save_checkpoint_package`.

`LVAEConfig` is the typed view of the subset that matters for reconstruction.
Reading it from `run_config` rather than from a W&B run object is what lets an
encoder load with no W&B dependency at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_PREDICTOR_HIDDEN_DIMS: tuple[int, ...] = (256, 128)
DEFAULT_RESIZE_DIMENSIONS: tuple[int, int] = (100, 100)

PERF_DIM_ALL = -1
"""Sentinel meaning "predict from every latent dimension"."""


@dataclass(frozen=True)
class LVAEConfig:
    """Architecture arguments needed to rebuild a trained LVAE.

    Args:
        latent_dim: Width of the latent space before pruning.
        perf_dim: Latent dimensions the performance predictor reads.
        resize_dimensions: Spatial size the encoder resizes inputs to.
        design_shape: The problem's native design shape.
        decoder_lipschitz_scale: Spectral-norm scale on the decoder, which sets
            how much latent distance can stretch into design space.
        predictor_lipschitz_scale: Spectral-norm scale on the predictor.
        predictor_hidden_dims: Hidden widths of the performance predictor.
        conditional_predictor: Whether the predictor also consumes conditions.
        nmse_threshold_rec: Reconstruction NMSE the run was constrained to.
        nmse_threshold_perf: Performance NMSE the run was constrained to.
        whitening: Whether PCA-rotation whitening was applied to latent codes.
        condition_filter_key: Condition column the training set was filtered on.
        condition_filter_value: Exact condition value trained on, if any.
        condition_filter_range: Inclusive condition range trained on, if any.
    """

    latent_dim: int
    perf_dim: int
    resize_dimensions: tuple[int, int]
    design_shape: tuple[int, int]
    decoder_lipschitz_scale: float = 1.0
    predictor_lipschitz_scale: float = 1.0
    predictor_hidden_dims: tuple[int, ...] = DEFAULT_PREDICTOR_HIDDEN_DIMS
    conditional_predictor: bool = False
    nmse_threshold_rec: float = 0.0
    nmse_threshold_perf: float = 0.0
    whitening: bool = False
    condition_filter_key: str | None = None
    condition_filter_value: float | None = None
    condition_filter_range: tuple[float, float] | None = None

    @classmethod
    def from_run_config(cls, run_config: dict[str, Any], design_shape: tuple[int, int]) -> LVAEConfig:
        """Build a config from a checkpoint package's `run_config.json`.

        Args:
            run_config: The training script's `Args`, exactly as saved.
            design_shape: The problem's design shape, used when the run predates
                `design_shape` being recorded.

        Returns:
            The typed configuration.

        Raises:
            KeyError: If `latent_dim` is absent, which means the package is not
                an LVAE checkpoint.
        """
        if "latent_dim" not in run_config:
            raise KeyError("run_config has no 'latent_dim'; this is not an LVAE checkpoint package")

        latent_dim = int(run_config["latent_dim"])
        perf_dim_raw = int(run_config.get("perf_dim", latent_dim))
        perf_dim = latent_dim if perf_dim_raw == PERF_DIM_ALL else perf_dim_raw

        recorded_shape = run_config.get("design_shape")
        shape = tuple(recorded_shape) if recorded_shape is not None else tuple(design_shape)
        if len(shape) != 2:  # noqa: PLR2004
            raise ValueError(f"expected a 2D design shape, got {shape!r}")

        resize = tuple(run_config.get("resize_dimensions", DEFAULT_RESIZE_DIMENSIONS))

        filter_range = run_config.get("condition_filter_range")
        return cls(
            latent_dim=latent_dim,
            perf_dim=perf_dim,
            resize_dimensions=(int(resize[0]), int(resize[1])),
            design_shape=(int(shape[0]), int(shape[1])),
            decoder_lipschitz_scale=float(run_config.get("decoder_lipschitz_scale", 1.0)),
            predictor_lipschitz_scale=float(run_config.get("predictor_lipschitz_scale", 1.0)),
            predictor_hidden_dims=tuple(run_config.get("predictor_hidden_dims", DEFAULT_PREDICTOR_HIDDEN_DIMS)),
            conditional_predictor=bool(run_config.get("conditional_predictor", False)),
            nmse_threshold_rec=float(run_config.get("nmse_threshold_rec", 0.0)),
            nmse_threshold_perf=float(run_config.get("nmse_threshold_perf", 0.0)),
            whitening=bool(run_config.get("whitening", False)),
            condition_filter_key=run_config.get("condition_filter_key"),
            condition_filter_value=run_config.get("condition_filter_value"),
            condition_filter_range=tuple(filter_range) if filter_range is not None else None,
        )
