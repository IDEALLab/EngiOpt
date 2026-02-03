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
    """Constrained least-volume autoencoder with threshold-gated volume loss.

    Extends LeastVolumeAE_DynamicPruning with a simple constraint mechanism:
    volume loss is only applied when reconstruction is below a threshold.
    This prevents volume compression from hurting reconstruction quality.

    Uses **Normalized MSE (NMSE)** for the threshold, making it problem-independent:
    - NMSE = MSE / Var(data)
    - Equivalent to R² target: R² = 1 - NMSE
    - nmse_threshold=0.01 means "reconstruction captures 99% of variance"

    The gate transitions smoothly:
    - nmse > threshold: gate = 0 (focus on reconstruction only)
    - nmse < threshold * (1 - margin): gate = comfort_multiplier (push on volume)
    - in between: linear interpolation

    Args:
        encoder: Encoder network.
        decoder: Decoder network.
        optimizer: Optimizer instance.
        latent_dim: Total number of latent dimensions.
        nmse_threshold: Target NMSE threshold for gating (fraction of unexplained variance).
            Default: 0.01 (99% variance explained).
        safety_margin: Fraction below threshold for full volume push. Default: 0.2.
        gate_ema_beta: EMA smoothing for reconstruction tracking. Default: 0.95.
        comfort_multiplier: Volume weight multiplier when in comfort zone. Default: 1.0.
        weights: Loss weights [reconstruction, volume]. Default: [1.0, 1.0].
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
        safety_margin: float = 0.2,
        gate_ema_beta: float = 0.95,
        comfort_multiplier: float = 1.0,
        weights: list[float] | Callable[[int], torch.Tensor] | None = None,
        eta: float = 0,
        beta: float = 0.9,
        pruning_epoch: int = 500,
        pruning_threshold: float = 0.02,
        pruning_strategy: Literal["plummet", "lognorm"] = "plummet",
        alpha: float = 0,
    ) -> None:
        if weights is None:
            weights = [1.0, 1.0]  # Gate handles volume scaling, so base weight is 1.0
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
        self.nmse_threshold = nmse_threshold
        self.safety_margin = safety_margin
        self.gate_ema_beta = gate_ema_beta
        self.comfort_multiplier = comfort_multiplier

        # Data variance for NMSE computation (must be set via set_data_variance)
        self.register_buffer("_data_var", torch.tensor(1.0))
        self._data_var_set = False

        # EMA of NMSE for stable gating
        self._nmse_ema: float | None = None
        self._current_gate: float = 0.0

    @property
    def gate(self) -> float:
        """Current volume gate value (0 to comfort_multiplier)."""
        return self._current_gate

    @property
    def nmse_ema(self) -> float | None:
        """Current EMA of normalized MSE."""
        return self._nmse_ema

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

    def _compute_volume_gate(self, rec_loss: torch.Tensor) -> float:
        """Compute volume gate based on smoothed NMSE.

        Args:
            rec_loss: Current batch reconstruction loss (MSE).

        Returns:
            Gate value from 0 (no volume pressure) to comfort_multiplier (full pressure).
        """
        # Compute NMSE = MSE / Var(data)
        nmse_val = rec_loss.item() / self._data_var.item()

        # Update EMA of NMSE
        if self._nmse_ema is None:
            nmse_ema = nmse_val
        else:
            nmse_ema = self.gate_ema_beta * self._nmse_ema + (1 - self.gate_ema_beta) * nmse_val
        self._nmse_ema = nmse_ema

        # Compute gate based on NMSE thresholds
        upper = self.nmse_threshold
        lower = self.nmse_threshold * (1 - self.safety_margin)

        if nmse_ema > upper:
            return 0.0  # Reconstruction struggling, no volume pressure
        elif nmse_ema < lower:
            return self.comfort_multiplier  # Comfortable, push on volume
        else:
            # Smooth linear interpolation in transition zone
            t = (upper - nmse_ema) / (upper - lower)
            return t * self.comfort_multiplier

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute losses with gated volume loss.

        Args:
            x: Input batch tensor.

        Returns:
            Tensor of shape (2,) containing [reconstruction_loss, gated_volume_loss].
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        self._update_moving_mean(z)

        rec_loss = self.loss_rec(x, x_hat)

        # Compute volume loss (same as parent)
        s = self._frozen_std.clone()
        if (~self._p).any():
            s[~self._p] = z[:, ~self._p].std(0)
        vol_loss = torch.exp(torch.log(s).mean())

        # Compute and apply gate
        self._current_gate = self._compute_volume_gate(rec_loss)

        return torch.stack([rec_loss, self._current_gate * vol_loss])

    @torch.no_grad()
    def _prune_step(self, epoch: int) -> None:
        """Execute pruning step only when reconstruction is acceptable."""
        # Only prune when gate > 0 (reconstruction is acceptable)
        if self._current_gate <= 0:
            return

        # Call parent pruning logic
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


__all__ = [
    "ConstrainedLeastVolumeAE_DP",
    "InterpretablePerfLeastVolumeAE_DP",
    "LeastVolumeAE",
    "LeastVolumeAE_DynamicPruning",
    "PerfLeastVolumeAE_DP",
]
