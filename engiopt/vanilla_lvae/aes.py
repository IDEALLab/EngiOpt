"""Vanilla LVAE autoencoder implementations with dynamic pruning.

This module provides simplified autoencoder classes for EngiOpt's LVAE experiments:
- LeastVolumeAE: Volume-regularized autoencoder (no pruning)
- LeastVolumeAE_DynamicPruning: Volume AE with dimension pruning (plummet or lognorm)
- PerfLeastVolumeAE_DP: Adds performance prediction to dynamic pruning AE
- InterpretablePerfLeastVolumeAE_DP: Performance prediction using first latent dims

This vanilla version supports plummet and lognorm pruning strategies without safeguards.
"""

from __future__ import annotations

from typing import Callable, Literal, TYPE_CHECKING

from scipy.stats import norm
import torch
from torch import nn
import torch.nn.functional as f
from tqdm import tqdm

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader


class LeastVolumeAE(nn.Module):
    """Autoencoder with volume-regularization loss.

    Minimizes the volume of the latent space (geometric mean of standard deviations)
    in addition to reconstruction error, promoting a compact representation.

    Volume loss is computed as: exp(mean(log(std_i + eta)))
    where std_i is the standard deviation of each latent dimension and eta is a small
    constant for numerical stability.

    Args:
        encoder: Encoder network mapping input to latent code.
        decoder: Decoder network mapping latent code to reconstruction.
        optimizer: Optimizer instance for training.
        weights: Loss weights [reconstruction, volume]. Default: [1.0, 0.001].
        eta: Smoothing constant for volume loss computation. Default: 0.
    """

    w: torch.Tensor  # Type annotation for buffer

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        optimizer: Optimizer,
        weights: list[float] | Callable[[int], torch.Tensor] | None = None,
        eta: float = 0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optim = optimizer
        self.eta = eta

        if weights is None:
            weights = [1.0, 0.001]

        if callable(weights):
            w = weights(0)
            self._w_schedule: Callable[[int], torch.Tensor] | None = weights
        else:
            w = weights
            self._w_schedule = None

        self.register_buffer("w", torch.as_tensor(w, dtype=torch.float))
        self._init_epoch = 0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction and volume losses.

        Args:
            x: Input batch tensor.

        Returns:
            Tensor of shape (2,) containing [reconstruction_loss, volume_loss].
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return torch.stack([self.loss_rec(x, x_hat), self.loss_vol(z)])

    def loss_rec(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss (MSE)."""
        return f.mse_loss(x, x_hat)

    def loss_vol(self, z: torch.Tensor) -> torch.Tensor:
        """Compute volume loss as geometric mean of latent standard deviations.

        Volume loss = exp(mean(log(std_i + eta)))

        Args:
            z: Latent codes of shape (batch_size, latent_dim).

        Returns:
            Scalar volume loss.
        """
        s = z.std(0)
        return torch.exp(torch.log(s + self.eta).mean())

    def epoch_hook(self, epoch: int) -> None:
        """Called at the start of each epoch to update weight schedule."""
        if self._w_schedule is not None:
            w = self._w_schedule(epoch)
            self.w = w.to(self.w.device)

    def epoch_report(
        self,
        epoch: int,
        callbacks: list[Callable[..., None]],
        **kwargs: object,
    ) -> None:
        """Called at the end of each epoch for logging/callbacks."""
        for callback in callbacks:
            callback(self, epoch=epoch, **kwargs)

    def fit(
        self,
        dataloader: DataLoader[torch.Tensor],
        epochs: int,
        callbacks: list[Callable[..., None]] | None = None,
    ) -> None:
        """Train the autoencoder.

        Args:
            dataloader: Training data loader.
            epochs: Maximum number of epochs.
            callbacks: Optional list of callback functions.
        """
        if callbacks is None:
            callbacks = []

        with tqdm(
            range(self._init_epoch, epochs),
            initial=self._init_epoch,
            total=epochs,
            bar_format="{l_bar}{bar:20}{r_bar}",
            desc="Training",
        ) as pbar:
            for epoch in pbar:
                self.epoch_hook(epoch=epoch)
                for batch in dataloader:
                    self.optim.zero_grad()
                    loss = self.loss(batch)
                    (loss * self.w).sum().backward()
                    self.optim.step()
                self.epoch_report(epoch=epoch, callbacks=callbacks, batch=batch, loss=loss, pbar=pbar)


class LeastVolumeAE_DynamicPruning(LeastVolumeAE):  # noqa: N801
    """Least-volume autoencoder with dynamic dimension pruning.

    Extends LeastVolumeAE by dynamically pruning low-variance latent dimensions
    during training using either plummet or lognorm pruning strategies.

    Strategies:
        - plummet: Detects sharp drops in sorted variances
        - lognorm: Fits log-normal distribution and prunes below percentile

    Args:
        encoder: Encoder network.
        decoder: Decoder network.
        optimizer: Optimizer instance.
        latent_dim: Total number of latent dimensions.
        weights: Loss weights [reconstruction, volume]. Default: [1.0, 0.001].
        eta: Smoothing parameter for volume loss. Default: 0.
        beta: EMA momentum for latent statistics. Default: 0.9.
        pruning_epoch: Epoch to start pruning. Default: 500.
        pruning_threshold: Threshold for pruning (ratio for plummet, percentile for lognorm). Default: 0.02.
        pruning_strategy: Strategy to use ("plummet" or "lognorm"). Default: "plummet".
        alpha: (lognorm only) Blending factor between reference and current distribution. Default: 0.
    """

    _p: torch.Tensor  # Boolean mask for pruned dimensions
    _z: torch.Tensor  # Frozen mean values for pruned dimensions
    _frozen_std: torch.Tensor  # Frozen std values for volume loss (captured at prune time)

    def __init__(  # noqa: PLR0913
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        optimizer: Optimizer,
        latent_dim: int,
        weights: list[float] | Callable[[int], torch.Tensor] | None = None,
        eta: float = 0,
        beta: float = 0.9,
        pruning_epoch: int = 500,
        pruning_threshold: float = 0.02,
        pruning_strategy: Literal["plummet", "lognorm"] = "plummet",
        alpha: float = 0,
    ) -> None:
        if weights is None:
            weights = [1.0, 0.001]
        super().__init__(encoder, decoder, optimizer, weights, eta)

        self.register_buffer("_p", torch.zeros(latent_dim, dtype=torch.bool))
        self.register_buffer("_z", torch.zeros(latent_dim))
        self.register_buffer("_frozen_std", torch.ones(latent_dim))  # Init to 1.0, overwritten each forward

        self._beta = beta
        self.pruning_epoch = pruning_epoch
        self.pruning_threshold = pruning_threshold
        self.pruning_strategy = pruning_strategy
        self.alpha = alpha

        # EMA statistics (initialized on first batch)
        self._zstd: torch.Tensor | None = None
        self._zmean: torch.Tensor | None = None

        # Reference distribution for lognorm (set at pruning_epoch)
        self._ref_mu: float | None = None
        self._ref_sigma: float | None = None

    def to(self, device: torch.device | str) -> LeastVolumeAE_DynamicPruning:
        """Move model to device."""
        super().to(device)
        self._p = self._p.to(device)
        self._z = self._z.to(device)
        self._frozen_std = self._frozen_std.to(device)
        return self

    @property
    def dim(self) -> int:
        """Number of active (unpruned) latent dimensions."""
        return int((~self._p).sum().item())

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode with pruned dimensions frozen to their mean values."""
        z = z.clone()
        z[:, self._p] = self._z[self._p]
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with pruned dimensions frozen to their mean values."""
        z = self.encoder(x)
        z = z.clone()
        z[:, self._p] = self._z[self._p]
        return z

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute losses and update moving statistics."""
        z = self.encode(x)
        x_hat = self.decode(z)
        self._update_moving_mean(z)

        # Volume loss: pruned dims use frozen std (captured at prune time),
        # active dims use current std. This makes pruning volume-neutral.
        s = self._frozen_std.clone()
        if (~self._p).any():
            s[~self._p] = z[:, ~self._p].std(0)
        vol_loss = torch.exp(torch.log(s).mean())

        return torch.stack([self.loss_rec(x, x_hat), vol_loss])

    @torch.no_grad()
    def _update_moving_mean(self, z: torch.Tensor) -> None:
        """Update exponential moving average of latent statistics."""
        if self._zstd is None or self._zmean is None:
            self._zstd = z.std(0)
            self._zmean = z.mean(0)
        else:
            self._zstd = torch.lerp(self._zstd, z.std(0), 1 - self._beta)
            self._zmean = torch.lerp(self._zmean, z.mean(0), 1 - self._beta)

    @torch.no_grad()
    def _plummet_prune(self, z_std: torch.Tensor) -> torch.Tensor:
        """Plummet-based pruning: detect sharp drops in sorted variances.

        Args:
            z_std: Standard deviation per latent dimension.

        Returns:
            Boolean mask where True indicates dimensions to prune.
        """
        # Sort variances in descending order
        srt, idx = torch.sort(z_std, descending=True)

        # Compute log-space drops
        log_srt = (srt + 1e-12).log()
        d_log = log_srt[1:] - log_srt[:-1]

        # Find the steepest drop (most negative value)
        # d_log[i] = log(srt[i+1]) - log(srt[i]), so argmin gives the index BEFORE the drop
        pidx_sorted = d_log.argmin()

        # Use variance BEFORE the drop as reference (the last "good" dimension)
        ref = srt[pidx_sorted]

        # Prune dimensions with ratio below threshold relative to reference
        ratio = z_std / (ref + 1e-12)
        return ratio < self.pruning_threshold

    @torch.no_grad()
    def _lognorm_prune(self, z_std: torch.Tensor) -> torch.Tensor:
        """Log-normal distribution-based pruning.

        Fits a log-normal distribution to the standard deviations and prunes
        dimensions below a percentile threshold.

        Args:
            z_std: Standard deviation per latent dimension.

        Returns:
            Boolean mask where True indicates dimensions to prune.
        """
        log_std = torch.log(z_std.clamp_min(1e-12))
        mu_current = log_std.mean().item()
        sigma_current = log_std.std().clamp_min(1e-6).item()

        if self._ref_mu is None or self._ref_sigma is None:
            # If no reference set yet, use current distribution
            mu_blend = mu_current
            sigma_blend = sigma_current
        else:
            # Blend between snapshot and current distribution
            mu_blend = (1 - self.alpha) * self._ref_mu + self.alpha * mu_current
            sigma_blend = (1 - self.alpha) * self._ref_sigma + self.alpha * sigma_current

        # Calculate cutoff value using inverse CDF of normal distribution
        # pruning_threshold is used as percentile (e.g., 0.01 = bottom 1%)
        cutoff_val = mu_blend + sigma_blend * float(norm.ppf(self.pruning_threshold))
        cutoff = torch.exp(torch.tensor(cutoff_val, device=z_std.device, dtype=z_std.dtype))
        return z_std < cutoff

    @torch.no_grad()
    def _set_lognorm_reference(self, z_std: torch.Tensor) -> None:
        """Set reference distribution for lognorm pruning at pruning_epoch."""
        log_std = torch.log(z_std.clamp_min(1e-12))
        self._ref_mu = log_std.mean().item()
        self._ref_sigma = log_std.std().item()

    @torch.no_grad()
    def _prune_step(self, _epoch: int) -> None:
        """Execute pruning step if conditions are met."""
        if self._zstd is None or self._zmean is None:
            return

        # Only consider active dimensions
        z_std_active = self._zstd[~self._p]
        if len(z_std_active) == 0:
            return

        # Select pruning strategy
        if self.pruning_strategy == "lognorm":
            cand_active = self._lognorm_prune(z_std_active)
        else:  # default to plummet
            cand_active = self._plummet_prune(z_std_active)

        # Map back to full dimension space
        cand = torch.zeros_like(self._p, dtype=torch.bool)
        cand[~self._p] = cand_active

        # Get indices to prune
        prune_idx = torch.where(cand & (~self._p))[0]
        if len(prune_idx) == 0:
            return

        # Freeze std BEFORE marking as pruned (capture current variance for volume loss)
        self._frozen_std[prune_idx] = self._zstd[prune_idx].clone()

        # Commit pruning
        self._p[prune_idx] = True
        self._z[prune_idx] = self._zmean[prune_idx]

    def epoch_report(
        self,
        epoch: int,
        callbacks: list[Callable[..., None]],
        **kwargs: object,
    ) -> None:
        """Called at end of epoch - triggers pruning if past pruning_epoch."""
        # Set lognorm reference at pruning_epoch
        if epoch == self.pruning_epoch and self.pruning_strategy == "lognorm" and self._zstd is not None:
            self._set_lognorm_reference(self._zstd)

        if epoch >= self.pruning_epoch:
            self._prune_step(epoch)

        super().epoch_report(epoch=epoch, callbacks=callbacks, **kwargs)


class PerfLeastVolumeAE_DP(LeastVolumeAE_DynamicPruning):  # noqa: N801
    """Performance-predicting autoencoder with dynamic pruning.

    Extends LeastVolumeAE_DynamicPruning to include performance prediction
    capabilities alongside reconstruction and volume minimization.

    The predictor takes the full latent code concatenated with conditions
    to predict performance values.

    Args:
        encoder: Encoder network.
        decoder: Decoder network.
        predictor: Performance prediction network (input: [z, conditions]).
        optimizer: Optimizer instance.
        latent_dim: Total number of latent dimensions.
        weights: Loss weights [reconstruction, performance, volume]. Default: [1.0, 1.0, 0.001].
        eta: Smoothing parameter for volume loss. Default: 0.
        beta: EMA momentum for latent statistics. Default: 0.9.
        pruning_epoch: Epoch to start pruning. Default: 500.
        pruning_threshold: Threshold for pruning. Default: 0.02.
        pruning_strategy: Strategy to use ("plummet" or "lognorm"). Default: "plummet".
        alpha: (lognorm only) Blending factor. Default: 0.
    """

    def __init__(  # noqa: PLR0913
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        predictor: nn.Module,
        optimizer: Optimizer,
        latent_dim: int,
        weights: list[float] | Callable[[int], torch.Tensor] | None = None,
        eta: float = 0,
        beta: float = 0.9,
        pruning_epoch: int = 500,
        pruning_threshold: float = 0.02,
        pruning_strategy: Literal["plummet", "lognorm"] = "plummet",
        alpha: float = 0,
    ) -> None:
        if weights is None:
            weights = [1.0, 1.0, 0.001]
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            latent_dim=latent_dim,
            weights=weights,
            eta=eta,
            beta=beta,
            pruning_epoch=pruning_epoch,
            pruning_threshold=pruning_threshold,
            pruning_strategy=pruning_strategy,
            alpha=alpha,
        )
        self.predictor = predictor

    def loss(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute reconstruction, performance, and volume losses.

        Args:
            batch: Tuple of (designs, conditions, performance_targets).

        Returns:
            Tensor of shape (3,) containing [rec_loss, perf_loss, vol_loss].
        """
        x, c, p = batch
        z = self.encode(x)
        x_hat = self.decode(z)

        # Update moving statistics
        self._update_moving_mean(z)

        # Performance prediction using full latent + conditions
        p_hat = self.predictor(torch.cat([z, c], dim=-1))

        # Volume loss: pruned dims use frozen std (captured at prune time),
        # active dims use current std. This makes pruning volume-neutral.
        s = self._frozen_std.clone()
        if (~self._p).any():
            s[~self._p] = z[:, ~self._p].std(0)
        vol_loss = torch.exp(torch.log(s).mean())

        return torch.stack(
            [
                self.loss_rec(x, x_hat),
                self.loss_rec(p, p_hat),
                vol_loss,
            ]
        )


class ConstrainedLeastVolumeAE_DP(LeastVolumeAE_DynamicPruning):  # noqa: N801
    """Constrained least-volume autoencoder with selectable constraint modes.

    Provides three constraint handling strategies for comparing different approaches
    to balancing reconstruction quality and volume compression:

    Modes:
        - **one_sided**: Mutually exclusive optimization. When NMSE > threshold,
          optimize only reconstruction. When NMSE <= threshold, optimize only volume.
          No gradient competition, but may oscillate around threshold.

        - **gated**: Additive with gating. Always optimize reconstruction.
          Add volume loss (weighted by w_vol) only when NMSE <= threshold.
          Stable but volume gradients may be overwhelmed.

        - **gradient_balanced**: Like gated, but scale volume loss by the ratio
          of reconstruction to volume loss magnitudes (EMA-tracked). Self-tuning
          to ensure volume gradients are competitive with reconstruction.

    Uses **Normalized MSE (NMSE)** for problem-independent thresholding:
    - NMSE = MSE / Var(data)
    - Equivalent to R² target: R² = 1 - NMSE

    Args:
        encoder: Encoder network.
        decoder: Decoder network.
        optimizer: Optimizer instance.
        latent_dim: Total number of latent dimensions.
        nmse_threshold: NMSE ceiling. Default: 0.01 (R² = 0.99).
        constraint_mode: "one_sided", "gated", or "gradient_balanced". Default: "one_sided".
        w_vol: Volume loss weight (gated mode only). Default: 1.0.
        ema_beta: EMA smoothing for loss tracking (gradient_balanced mode). Default: 0.9.
        eta: Smoothing parameter for volume loss. Default: 0.
        beta: EMA momentum for latent statistics. Default: 0.9.
        pruning_epoch: Epoch to start pruning. Default: 500.
        pruning_threshold: Threshold for pruning. Default: 0.02.
        pruning_strategy: Strategy to use ("plummet" or "lognorm"). Default: "plummet".
        alpha: (lognorm only) Blending factor. Default: 0.
    """

    _data_var: torch.Tensor  # Buffer for data variance

    def __init__(  # noqa: PLR0913
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        optimizer: Optimizer,
        latent_dim: int,
        nmse_threshold: float = 0.01,
        constraint_mode: Literal["one_sided", "gated", "gradient_balanced"] = "one_sided",
        w_vol: float = 1.0,
        ema_beta: float = 0.9,
        eta: float = 0,
        beta: float = 0.9,
        pruning_epoch: int = 500,
        pruning_threshold: float = 0.02,
        pruning_strategy: Literal["plummet", "lognorm"] = "plummet",
        alpha: float = 0,
    ) -> None:
        # Parent uses weights for its loss computation, but we override loss()
        # so we just pass a dummy value
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            latent_dim=latent_dim,
            weights=[1.0, 1.0],  # Not used - we override loss()
            eta=eta,
            beta=beta,
            pruning_epoch=pruning_epoch,
            pruning_threshold=pruning_threshold,
            pruning_strategy=pruning_strategy,
            alpha=alpha,
        )
        self.nmse_threshold = nmse_threshold
        self.constraint_mode = constraint_mode
        self.w_vol = w_vol
        self._ema_beta = ema_beta

        # Data variance for NMSE computation (must be set via set_data_variance)
        self.register_buffer("_data_var", torch.tensor(1.0))
        self._data_var_set = False

        # EMA tracking for gradient balancing
        self._rec_ema: float = 0.0
        self._vol_ema: float = 0.0

        # Current state for logging
        self._current_nmse: float = 0.0
        self._current_rec_loss: float = 0.0
        self._current_vol_loss: float = 0.0
        self._vol_active: bool = False
        self._balance_factor: float = 1.0

    @property
    def nmse(self) -> float:
        """Current batch NMSE value."""
        return self._current_nmse

    @property
    def vol_active(self) -> bool:
        """Whether volume loss is currently active."""
        return self._vol_active

    @property
    def rec_loss(self) -> float:
        """Current batch reconstruction loss."""
        return self._current_rec_loss

    @property
    def vol_loss(self) -> float:
        """Current batch volume loss (before any scaling)."""
        return self._current_vol_loss

    @property
    def balance_factor(self) -> float:
        """Current gradient balance factor (gradient_balanced mode)."""
        return self._balance_factor

    @property
    def data_var(self) -> float:
        """Data variance used for NMSE normalization."""
        return self._data_var.item()

    def set_data_variance(self, x: torch.Tensor) -> None:
        """Set the data variance from training data for NMSE computation.

        Should be called once before training with the full training dataset
        or a representative sample.

        Args:
            x: Training data tensor of shape (N, ...).
        """
        var = x.var().item()
        if var < 1e-10:
            var = 1.0  # Fallback for constant data
        self._data_var = torch.tensor(var, device=self._data_var.device)
        self._data_var_set = True

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute loss based on selected constraint mode.

        Args:
            x: Input batch tensor.

        Returns:
            Scalar loss tensor for backpropagation.
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        self._update_moving_mean(z)

        rec_loss = self.loss_rec(x, x_hat)

        # Compute volume loss
        s = self._frozen_std.clone()
        if (~self._p).any():
            s[~self._p] = z[:, ~self._p].std(0)
        vol_loss = torch.exp(torch.log(s).mean())

        # Compute NMSE = MSE / Var(data)
        nmse = rec_loss / self._data_var
        self._current_nmse = nmse.item()
        self._current_rec_loss = rec_loss.item()
        self._current_vol_loss = vol_loss.item()

        # Update EMAs for gradient balancing (always update for logging)
        self._rec_ema = self._ema_beta * self._rec_ema + (1 - self._ema_beta) * rec_loss.item()
        self._vol_ema = self._ema_beta * self._vol_ema + (1 - self._ema_beta) * vol_loss.item()
        self._balance_factor = self._rec_ema / (self._vol_ema + 1e-8)

        # Apply constraint mode
        if self.constraint_mode == "one_sided":
            # Mutually exclusive: only one loss active at a time
            if nmse > self.nmse_threshold:
                self._vol_active = False
                return rec_loss
            self._vol_active = True
            return vol_loss

        if self.constraint_mode == "gated":
            # Additive: rec always, vol only when below threshold
            if nmse > self.nmse_threshold:
                self._vol_active = False
                return rec_loss
            self._vol_active = True
            return rec_loss + self.w_vol * vol_loss

        if self.constraint_mode == "gradient_balanced":
            # Additive with auto-scaling based on loss magnitudes
            if nmse > self.nmse_threshold:
                self._vol_active = False
                return rec_loss
            self._vol_active = True
            return rec_loss + self._balance_factor * vol_loss

        raise ValueError(f"Unknown constraint_mode: {self.constraint_mode}")

    @torch.no_grad()
    def _prune_step(self, epoch: int) -> None:
        """Execute pruning step unconditionally after pruning_epoch.

        Pruning is decoupled from _vol_active since the volume loss already
        only engages when the constraint is satisfied. Double-gating would
        introduce a race condition where pruning depends on the last batch's
        constraint state.
        """
        super()._prune_step(epoch)


class InterpretablePerfLeastVolumeAE_DP(LeastVolumeAE_DynamicPruning):  # noqa: N801
    """Interpretable performance-predicting autoencoder with dynamic pruning.

    This variant enforces that the first `perf_dim` latent dimensions are dedicated
    to performance prediction, making them more interpretable.

    The predictor only uses the first `perf_dim` latent dimensions concatenated
    with conditions to predict performance values.

    Args:
        encoder: Encoder network.
        decoder: Decoder network.
        predictor: Performance prediction network (input: [z[:perf_dim], conditions]).
        optimizer: Optimizer instance.
        latent_dim: Total number of latent dimensions.
        perf_dim: Number of latent dimensions dedicated to performance prediction.
        weights: Loss weights [reconstruction, performance, volume]. Default: [1.0, 0.1, 0.001].
        eta: Smoothing parameter for volume loss. Default: 0.
        beta: EMA momentum for latent statistics. Default: 0.9.
        pruning_epoch: Epoch to start pruning. Default: 500.
        pruning_threshold: Threshold for pruning. Default: 0.02.
        pruning_strategy: Strategy to use ("plummet" or "lognorm"). Default: "plummet".
        alpha: (lognorm only) Blending factor. Default: 0.
    """

    def __init__(  # noqa: PLR0913
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        predictor: nn.Module,
        optimizer: Optimizer,
        latent_dim: int,
        perf_dim: int,
        weights: list[float] | Callable[[int], torch.Tensor] | None = None,
        eta: float = 0,
        beta: float = 0.9,
        pruning_epoch: int = 500,
        pruning_threshold: float = 0.02,
        pruning_strategy: Literal["plummet", "lognorm"] = "plummet",
        alpha: float = 0,
    ) -> None:
        if weights is None:
            weights = [1.0, 0.1, 0.001]
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            latent_dim=latent_dim,
            weights=weights,
            eta=eta,
            beta=beta,
            pruning_epoch=pruning_epoch,
            pruning_threshold=pruning_threshold,
            pruning_strategy=pruning_strategy,
            alpha=alpha,
        )
        self.predictor = predictor
        self.perf_dim = perf_dim

    def loss(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute losses using only first perf_dim latents for performance prediction.

        Args:
            batch: Tuple of (designs, conditions, performance_targets).

        Returns:
            Tensor of shape (3,) containing [rec_loss, perf_loss, vol_loss].
        """
        x, c, p = batch
        z = self.encode(x)
        x_hat = self.decode(z)

        # Update moving statistics
        self._update_moving_mean(z)

        # Only first perf_dim dimensions for performance prediction
        pz = z[:, : self.perf_dim]
        p_hat = self.predictor(torch.cat([pz, c], dim=-1))

        # Volume loss: pruned dims use frozen std (captured at prune time),
        # active dims use current std. This makes pruning volume-neutral.
        s = self._frozen_std.clone()
        if (~self._p).any():
            s[~self._p] = z[:, ~self._p].std(0)
        vol_loss = torch.exp(torch.log(s).mean())

        return torch.stack(
            [
                self.loss_rec(x, x_hat),
                self.loss_rec(p, p_hat),
                vol_loss,
            ]
        )


class ConstrainedPerfLeastVolumeAE_DP(LeastVolumeAE_DynamicPruning):  # noqa: N801
    """Constrained performance-predicting LVAE with one-sided constraint handling.

    Extends the one-sided constraint method to handle two constraints:
    1. Reconstruction constraint: NMSE_rec <= threshold_rec
    2. Performance constraint: NMSE_perf <= threshold_perf

    Uses **reconstruction-first priority**: reconstruction must be satisfied before
    performance, and both must be satisfied before volume optimization begins.

    One-sided logic:
        - If rec NMSE > threshold_rec: optimize reconstruction only
        - Elif perf NMSE > threshold_perf: optimize performance only
        - Else: optimize volume only (both constraints satisfied)

    Uses **Normalized MSE (NMSE)** for problem-independent thresholding:
    - NMSE = MSE / Var(data)
    - Equivalent to R² target: R² = 1 - NMSE

    Args:
        encoder: Encoder network.
        decoder: Decoder network.
        predictor: Performance prediction network (input: [z[:perf_dim], conditions]).
        optimizer: Optimizer instance.
        latent_dim: Total number of latent dimensions.
        perf_dim: Number of latent dimensions dedicated to performance prediction.
        nmse_threshold_rec: NMSE ceiling for reconstruction. Default: 0.01 (R² = 0.99).
        nmse_threshold_perf: NMSE ceiling for performance. Default: 0.05 (R² = 0.95).
        eta: Smoothing parameter for volume loss. Default: 0.
        beta: EMA momentum for latent statistics. Default: 0.9.
        pruning_epoch: Epoch to start pruning. Default: 500.
        pruning_threshold: Threshold for pruning. Default: 0.02.
        pruning_strategy: Strategy to use ("plummet" or "lognorm"). Default: "plummet".
        alpha: (lognorm only) Blending factor. Default: 0.
    """

    _data_var: torch.Tensor  # Buffer for design data variance
    _perf_var: torch.Tensor  # Buffer for performance data variance

    def __init__(  # noqa: PLR0913
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        predictor: nn.Module,
        optimizer: Optimizer,
        latent_dim: int,
        perf_dim: int,
        nmse_threshold_rec: float = 0.01,
        nmse_threshold_perf: float = 0.05,
        eta: float = 0,
        beta: float = 0.9,
        pruning_epoch: int = 500,
        pruning_threshold: float = 0.02,
        pruning_strategy: Literal["plummet", "lognorm"] = "plummet",
        alpha: float = 0,
    ) -> None:
        # Parent uses weights for its loss computation, but we override loss()
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            latent_dim=latent_dim,
            weights=[1.0, 1.0],  # Not used - we override loss()
            eta=eta,
            beta=beta,
            pruning_epoch=pruning_epoch,
            pruning_threshold=pruning_threshold,
            pruning_strategy=pruning_strategy,
            alpha=alpha,
        )
        self.predictor = predictor
        self.perf_dim = perf_dim
        self.nmse_threshold_rec = nmse_threshold_rec
        self.nmse_threshold_perf = nmse_threshold_perf

        # Data variances for NMSE computation (must be set via set_* methods)
        self.register_buffer("_data_var", torch.tensor(1.0))
        self.register_buffer("_perf_var", torch.tensor(1.0))
        self._data_var_set = False
        self._perf_var_set = False

        # Current state for logging
        self._current_nmse_rec: float = 0.0
        self._current_nmse_perf: float = 0.0
        self._current_rec_loss: float = 0.0
        self._current_perf_loss: float = 0.0
        self._current_vol_loss: float = 0.0
        self._vol_active: bool = False

    @property
    def nmse_rec(self) -> float:
        """Current batch reconstruction NMSE."""
        return self._current_nmse_rec

    @property
    def nmse_perf(self) -> float:
        """Current batch performance NMSE."""
        return self._current_nmse_perf

    @property
    def vol_active(self) -> bool:
        """Whether volume loss is currently active."""
        return self._vol_active

    @property
    def rec_loss(self) -> float:
        """Current batch reconstruction loss."""
        return self._current_rec_loss

    @property
    def perf_loss(self) -> float:
        """Current batch performance loss."""
        return self._current_perf_loss

    @property
    def vol_loss(self) -> float:
        """Current batch volume loss."""
        return self._current_vol_loss

    @property
    def data_var(self) -> float:
        """Design data variance used for reconstruction NMSE."""
        return self._data_var.item()

    @property
    def perf_var(self) -> float:
        """Performance data variance used for performance NMSE."""
        return self._perf_var.item()

    def set_data_variance(self, x: torch.Tensor) -> None:
        """Set the design data variance for reconstruction NMSE computation.

        Should be called once before training with the full training dataset.

        Args:
            x: Training design tensor of shape (N, ...).
        """
        var = x.var().item()
        if var < 1e-10:
            var = 1.0  # Fallback for constant data
        self._data_var = torch.tensor(var, device=self._data_var.device)
        self._data_var_set = True

    def set_perf_variance(self, p: torch.Tensor) -> None:
        """Set the performance data variance for performance NMSE computation.

        Should be called once before training with scaled performance values.

        Args:
            p: Scaled performance tensor of shape (N, 1) or (N,).
        """
        var = p.var().item()
        if var < 1e-10:
            var = 1.0  # Fallback for constant data
        self._perf_var = torch.tensor(var, device=self._perf_var.device)
        self._perf_var_set = True

    def loss(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute loss with reconstruction-first priority one-sided switching.

        Args:
            batch: Tuple of (designs, conditions, performance_targets).

        Returns:
            Scalar loss tensor for backpropagation.
        """
        x, c, p = batch
        z = self.encode(x)
        x_hat = self.decode(z)

        # Update moving statistics for pruning
        self._update_moving_mean(z)

        # Performance prediction (first perf_dim dims + conditions)
        pz = z[:, : self.perf_dim]
        p_hat = self.predictor(torch.cat([pz, c], dim=-1))

        # Compute individual losses
        rec_loss = self.loss_rec(x, x_hat)
        perf_loss = self.loss_rec(p, p_hat)

        # Volume loss (geometric mean of stds, frozen for pruned dims)
        s = self._frozen_std.clone()
        if (~self._p).any():
            s[~self._p] = z[:, ~self._p].std(0)
        vol_loss = torch.exp(torch.log(s).mean())

        # Compute NMSEs
        nmse_rec = rec_loss / self._data_var
        nmse_perf = perf_loss / self._perf_var

        # Store for logging
        self._current_nmse_rec = nmse_rec.item()
        self._current_nmse_perf = nmse_perf.item()
        self._current_rec_loss = rec_loss.item()
        self._current_perf_loss = perf_loss.item()
        self._current_vol_loss = vol_loss.item()

        # One-sided constraint logic (reconstruction-first priority)
        if nmse_rec > self.nmse_threshold_rec:
            # Reconstruction violated - fix geometry first
            self._vol_active = False
            return rec_loss
        if nmse_perf > self.nmse_threshold_perf:
            # Reconstruction OK, performance violated - fix performance
            self._vol_active = False
            return perf_loss
        # BOTH constraints satisfied - optimize volume
        self._vol_active = True
        return vol_loss

    @torch.no_grad()
    def _prune_step(self, epoch: int) -> None:
        """Execute pruning unconditionally after pruning_epoch."""
        super()._prune_step(epoch)


__all__ = [
    "ConstrainedLeastVolumeAE_DP",
    "ConstrainedPerfLeastVolumeAE_DP",
    "InterpretablePerfLeastVolumeAE_DP",
    "LeastVolumeAE",
    "LeastVolumeAE_DynamicPruning",
    "PerfLeastVolumeAE_DP",
]
