"""Vanilla LVAE autoencoder implementations with plummet-based dynamic pruning.

This module provides simplified autoencoder classes for EngiOpt's LVAE experiments:
- LeastVolumeAE: Volume-regularized autoencoder (no pruning)
- LeastVolumeAE_DynamicPruning: Volume AE with plummet-based dimension pruning
- PerfLeastVolumeAE_DP: Adds performance prediction to dynamic pruning AE
- InterpretablePerfLeastVolumeAE_DP: Performance prediction using first latent dims

This vanilla version uses only plummet pruning and no safeguards for simplicity.
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

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
    """Least-volume autoencoder with plummet-based dynamic dimension pruning.

    Extends LeastVolumeAE by dynamically pruning low-variance latent dimensions
    during training using the plummet strategy (detects sharp drops in sorted variances).

    Args:
        encoder: Encoder network.
        decoder: Decoder network.
        optimizer: Optimizer instance.
        latent_dim: Total number of latent dimensions.
        weights: Loss weights [reconstruction, volume]. Default: [1.0, 0.001].
        eta: Smoothing parameter for volume loss. Default: 0.
        beta: EMA momentum for latent statistics. Default: 0.9.
        pruning_epoch: Epoch to start pruning. Default: 500.
        plummet_threshold: Ratio threshold for plummet pruning. Default: 0.02.
    """

    _p: torch.Tensor  # Boolean mask for pruned dimensions
    _z: torch.Tensor  # Frozen mean values for pruned dimensions

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
        plummet_threshold: float = 0.02,
    ) -> None:
        if weights is None:
            weights = [1.0, 0.001]
        super().__init__(encoder, decoder, optimizer, weights, eta)

        self.register_buffer("_p", torch.zeros(latent_dim, dtype=torch.bool))
        self.register_buffer("_z", torch.zeros(latent_dim))

        self._beta = beta
        self.pruning_epoch = pruning_epoch
        self.plummet_threshold = plummet_threshold

        # EMA statistics (initialized on first batch)
        self._zstd: torch.Tensor | None = None
        self._zmean: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> LeastVolumeAE_DynamicPruning:
        """Move model to device."""
        super().to(device)
        self._p = self._p.to(device)
        self._z = self._z.to(device)
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

        # Volume loss over active dimensions only
        vol_loss = self.loss_vol(z[:, ~self._p]) if (~self._p).any() else torch.tensor(0.0, device=z.device)

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
        return ratio < self.plummet_threshold

    @torch.no_grad()
    def _prune_step(self, _epoch: int) -> None:
        """Execute pruning step if conditions are met."""
        if self._zstd is None or self._zmean is None:
            return

        # Only consider active dimensions
        z_std_active = self._zstd[~self._p]
        if len(z_std_active) == 0:
            return

        cand_active = self._plummet_prune(z_std_active)

        # Map back to full dimension space
        cand = torch.zeros_like(self._p, dtype=torch.bool)
        cand[~self._p] = cand_active

        # Get indices to prune
        prune_idx = torch.where(cand & (~self._p))[0]
        if len(prune_idx) == 0:
            return

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
        plummet_threshold: Ratio threshold for plummet pruning. Default: 0.02.
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
        plummet_threshold: float = 0.02,
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
            plummet_threshold=plummet_threshold,
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

        # Volume loss over active dimensions only
        vol_loss = self.loss_vol(z[:, ~self._p]) if (~self._p).any() else torch.tensor(0.0, device=z.device)

        return torch.stack(
            [
                self.loss_rec(x, x_hat),
                self.loss_rec(p, p_hat),
                vol_loss,
            ]
        )


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
        plummet_threshold: Ratio threshold for plummet pruning. Default: 0.02.
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
        plummet_threshold: float = 0.02,
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
            plummet_threshold=plummet_threshold,
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

        # Volume loss over active dimensions only
        vol_loss = self.loss_vol(z[:, ~self._p]) if (~self._p).any() else torch.tensor(0.0, device=z.device)

        return torch.stack(
            [
                self.loss_rec(x, x_hat),
                self.loss_rec(p, p_hat),
                vol_loss,
            ]
        )


__all__ = [
    "InterpretablePerfLeastVolumeAE_DP",
    "LeastVolumeAE",
    "LeastVolumeAE_DynamicPruning",
    "PerfLeastVolumeAE_DP",
]
