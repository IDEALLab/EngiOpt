"""Autoencoder and VAE model implementations with least-volume objectives and dynamic pruning.

This module provides a set of autoencoder classes and training utilities used in EngiOpt's
lVAE 2D experiments, including:
- _AutoEncoder / AutoEncoder: basic training loop and reconstruction loss.
- VAE: variational autoencoder with ELBO computation.
- LeastVolumeAE and LeastVolumeAE_DynamicPruning: volume-regularized AEs with pruning policies.
- PruningPolicy: various strategies for selecting latent dimensions to prune.
- DesignLeastVolumeAE_DynamicPruning: extends dynamic pruning AE with performance prediction.

The file also contains helpers and training hooks used by the project.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING

from scipy.stats import norm
import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.independent import Independent
from torch.distributions.kl import kl_divergence
import torch.nn.functional as f
from tqdm import tqdm

if TYPE_CHECKING:
    from engiopt.lvae_core.constraint_handlers import ConstraintHandler, ConstraintLosses, ConstraintThresholds


class _AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, optimizer, weights=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optim = optimizer
        if callable(weights):
            w = weights(0)
            self._w_schedule = weights
        else:
            w = weights
        self.register_buffer("w", torch.as_tensor(w, dtype=torch.float))
        self._init_epoch = 0

    def loss(self, batch, **kwargs):
        """Tensor of loss terms."""
        raise NotImplementedError

    def decode(self, latent_code):
        return self.decoder(latent_code)

    def encode(self, x_batch):
        return self.encoder(x_batch)

    def fit(
        self,
        dataloader,
        epochs,  # maximal epoch
        callbacks=None,
        **kwargs,
    ):
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
                self.epoch_hook(epoch=epoch, pbar=pbar, callbacks=callbacks, **kwargs)
                for batch in dataloader:
                    self.optim.zero_grad()
                    loss = self.loss(batch, **kwargs)
                    (loss * self.w).sum().backward()
                    self.optim.step()
                self.epoch_report(
                    epoch=epoch,
                    batch=batch,
                    loss=loss,
                    pbar=pbar,
                    callbacks=callbacks,
                    **kwargs,
                )

    @torch.no_grad()
    def epoch_hook(self, epoch):
        if hasattr(self, "_w_schedule"):
            w = self._w_schedule(epoch)
            self.w = w.to(self.w.device)

    def epoch_report(self, *args, **kwargs):
        pass

    def save(self, save_dir, epoch, *args):
        file_name = f"{epoch}" + "_".join([str(arg) for arg in args])
        torch.save(
            {
                "params": self.state_dict(),
                "optim": self.optim.state_dict(),
                "epoch": epoch,
            },
            os.path.join(save_dir, file_name + ".tar"),
        )

    def load(self, checkpoint):
        ckp = torch.load(checkpoint)
        self.load_state_dict(ckp["params"])
        self.optim.load_state_dict(ckp["optim"])
        self._init_epoch = ckp["epoch"] + 1


class AutoEncoder(_AutoEncoder):
    def loss(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return self.loss_rec(x, x_hat).reshape(-1)

    def loss_rec(self, x, x_hat):
        return f.mse_loss(x, x_hat)

    def epoch_report(self, callbacks, *args, **kwargs):
        for callback in callbacks:
            callback(self, *args, **kwargs)


class VAE(AutoEncoder):
    def __init__(self, encoder, decoder, optimizer, beta=1):
        super().__init__(encoder, decoder, optimizer, weights=(1, beta))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self.optim.add_param_group({"params": self.log_sigma})

    def encode(self, x_batch):
        return self.encoder(x_batch).chunk(2, dim=-1)

    def _kl_div_qp(self, x):
        z_mean, z_log_std = self.encode(x)
        z_std = torch.exp(z_log_std)
        p_z = Independent(Normal(torch.zeros_like(z_mean), torch.ones_like(z_std)), 1)
        q_zx = Independent(Normal(z_mean, z_std), 1)
        return kl_divergence(q_zx, p_z)  # [batch]

    def _log_pxz(self, x):  # batch: samples of x
        z_mean, z_log_std = self.encode(x)  # [batch, latent_dim]
        z_std = torch.exp(z_log_std)
        q_zx = Independent(Normal(z_mean, z_std), 1)
        x_hat = self.decode(q_zx.rsample())  # [batch, x_dim_0,...]
        p_xz = Independent(Normal(x_hat, torch.exp(self.log_sigma)), len(x_hat.shape[1:]))
        return p_xz.log_prob(x)  # [batch]

    def elbo(self, x, z_num=1):
        expz_log_pxz = torch.stack([self._log_pxz(x) for _ in range(z_num)]).mean(0)
        return torch.stack([expz_log_pxz.mean(), -self._kl_div_qp(x).mean()])

    def loss(self, batch):
        return -self.elbo(batch)


class LeastVolumeAE(AutoEncoder):
    """Autoencoder with volume-regularization loss.

    Minimizes the volume of the latent space (geometric mean of standard deviations)
    in addition to reconstruction error, promoting a compact representation.

    Volume loss is computed as: exp(mean(log(std_i + eta)))
    where std_i is the standard deviation of each latent dimension and eta is a small
    constant for numerical stability.
    """

    def __init__(
        self,
        encoder,
        decoder,
        optimizer,
        weights=None,
        eta=1e-4,
    ):
        """Initialize the least-volume autoencoder.

        Args:
            encoder: Encoder network
            decoder: Decoder network
            optimizer: Optimizer instance
            weights: Loss weights [reconstruction, volume]. Default: [1.0, 0.001]
            eta: Smoothing constant for volume loss computation. Default: 1e-4
        """
        if weights is None:
            weights = [1.0, 0.001]
        super().__init__(encoder, decoder, optimizer, weights)
        self.eta = eta

    def loss(self, x, **kwargs):  # noqa: ARG002
        """Compute reconstruction and volume losses."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return torch.stack([self.loss_rec(x, x_hat), self.loss_vol(z)])

    def loss_vol(self, z):
        """Compute volume loss as geometric mean of latent standard deviations.

        Volume loss = exp(mean(log(std_i + eta)))

        This measures the "volume" of the latent space distribution. Lower values
        indicate more compressed/compact representations.

        Args:
            z: Latent codes of shape (batch_size, latent_dim)

        Returns:
            Scalar volume loss (geometric mean of standard deviations)
        """
        s = z.std(0)  # Standard deviation per dimension
        return torch.exp(torch.log(s + self.eta).mean())


class PruningPolicy:
    """Implements various strategies for pruning latent dimensions.

    Supports multiple pruning strategies:
    - pca_cdf: PCA-inspired cumulative variance thresholding
    - lognorm: Log-normal distribution fitting
    - probabilistic: Temperature-scaled probabilistic pruning
    - plummet: Detects sharp drops in sorted variance values
    """

    def __init__(self, strategy, pruning_params=None):
        """Initialize pruning policy.

        Args:
            strategy: Name of pruning strategy to use
            pruning_params: Dictionary of strategy-specific parameters
        """
        self.strategy = strategy
        self.pruning_params = pruning_params or {}
        self._ref_total_var = None  # Snapshot of total variance at pruning_epoch
        self._ref_mu = None  # Snapshot of log-std mean at pruning_epoch
        self._ref_sigma = None  # Snapshot of log-std std at pruning_epoch

    def __call__(self, z_std):
        """Apply pruning strategy to latent dimension standard deviations.

        Args:
            z_std: Standard deviation of each latent dimension

        Returns:
            Boolean mask where True indicates dimensions to prune
        """
        if self.strategy == "pca_cdf":
            return self._pca_cdf(z_std, self.pruning_params["threshold"])
        if self.strategy == "lognorm":
            return self._lognorm(
                z_std,
                percentile=self.pruning_params["threshold"],
                alpha=self.pruning_params["alpha"],
            )
        if self.strategy == "probabilistic":
            return self._probabilistic(z_std, self.pruning_params["temperature"])
        if self.strategy == "plummet":
            return self._plummet(z_std, self.pruning_params["threshold"])
        raise ValueError(f"Unknown pruning strategy: {self.strategy}")

    def set_reference(self, z_std):
        """Call this at pruning_epoch to snapshot variance/distribution."""
        self._ref_total_var = float((z_std**2).sum().item())
        log_std = torch.log(z_std.clamp_min(1e-12))
        self._ref_mu = log_std.mean().item()
        self._ref_sigma = log_std.std().item()

    def _pca_cdf(self, z_std, thr):
        """PCA-inspired pruning based on cumulative variance explained.

        Keeps dimensions that collectively explain up to threshold fraction
        of total variance. Properly handles cases where variance increases
        during training.

        Args:
            z_std: Standard deviation per dimension
            thr: Cumulative variance threshold (e.g., 0.95 = keep 95% of variance)

        Returns:
            Boolean mask indicating which dimensions to prune
        """
        # Calculate variance contribution of each dimension
        variance = z_std**2
        total_var = variance.sum().clamp_min(1e-12)

        # Normalize contributions to sum to 1.0 (always current distribution)
        contrib = variance / total_var

        # If we have a reference, adjust threshold based on variance change
        adjusted_thr = thr
        if self._ref_total_var is not None:
            # If current variance is higher than reference, be more conservative
            # If current variance is lower, be more aggressive
            variance_ratio = total_var / self._ref_total_var
            # Scale threshold: if variance doubled, need more dims to explain same amount
            adjusted_thr = thr * variance_ratio.item()
            adjusted_thr = min(adjusted_thr, 0.999)  # Cap at 99.9%

        # Sort dimensions by contribution and compute cumulative sum
        vals, idx = torch.sort(contrib, descending=True)
        cdf = torch.cumsum(vals, 0)

        # Keep dimensions needed to reach adjusted threshold
        keep_sorted = cdf <= adjusted_thr
        keep_sorted[0] = True  # Always keep the most important dimension

        # Convert back to original dimension ordering
        keep = torch.zeros_like(keep_sorted, dtype=torch.bool)
        keep[idx] = keep_sorted

        return ~keep  # Return pruning mask (True = prune)

    def _lognorm(self, z_std, percentile=0.05, alpha=1.0):
        """Log-normal distribution-based pruning.

        Fits a log-normal distribution to the standard deviations and prunes
        dimensions below a percentile threshold.

        Args:
            z_std: Standard deviation per dimension
            percentile: Percentile threshold for pruning (e.g., 0.05 = bottom 5%)
            alpha: Blending factor between reference and current distribution (0-1)

        Returns:
            Boolean mask indicating which dimensions to prune
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
            mu_blend = (1 - alpha) * self._ref_mu + alpha * mu_current
            sigma_blend = (1 - alpha) * self._ref_sigma + alpha * sigma_current

        # Calculate cutoff value using inverse CDF of normal distribution
        cutoff_val = mu_blend + sigma_blend * float(norm.ppf(percentile))
        cutoff = torch.exp(torch.tensor(cutoff_val, device=z_std.device, dtype=z_std.dtype))
        return z_std < cutoff

    @staticmethod
    def _probabilistic(z_std, temperature):
        """Probabilistic pruning with temperature-controlled sampling.

        Converts variance contributions to pruning probabilities,
        with temperature controlling the randomness. Higher temperature
        = more random, lower temperature = more deterministic.

        Args:
            z_std: Standard deviation per dimension
            temperature: Controls randomness (higher = more random, like softmax)

        Returns:
            Boolean mask indicating which dimensions to prune
        """
        # Calculate relative contribution of each dimension
        contrib = z_std / z_std.sum().clamp_min(1e-12)

        # Convert to pruning scores (higher score = more likely to prune)
        # Dimensions with low contribution get high pruning scores
        score = 1.0 - (contrib / contrib.max().clamp_min(1e-12))

        # Apply temperature scaling: score / T (standard temperature scaling)
        # Lower T makes high scores even higher (more deterministic)
        # Higher T smooths out differences (more random)
        logits = score / max(temperature, 1e-6)

        # Convert to probabilities using sigmoid
        p = torch.sigmoid(logits)

        # Sample pruning decisions
        return torch.bernoulli(p).bool()

    @staticmethod
    def _plummet(z_std, plummet_threshold):
        """Plummet-based pruning detects sharp drops in sorted variances.

        Finds the steepest drop in log-space sorted variances and uses
        that as a reference point for pruning.

        Args:
            z_std: Standard deviation per dimension
            plummet_threshold: Ratio threshold relative to drop point

        Returns:
            Boolean mask indicating which dimensions to prune
        """
        # Sort variances in descending order
        srt, idx = torch.sort(z_std, descending=True)

        # Compute log-space drops (how much each dimension drops from previous)
        log_srt = (srt + 1e-12).log()
        d_log = log_srt[1:] - log_srt[:-1]  # Negative values = drops

        # Find the steepest drop (most negative value)
        pidx_sorted = d_log.argmin()  # Index of steepest drop (most negative)

        # Use variance at the drop as reference
        ref_idx = idx[pidx_sorted]  # Value at the drop
        ref = z_std[ref_idx]

        # Prune dimensions with ratio below threshold relative to reference
        ratio = z_std / (ref + 1e-12)
        return ratio < plummet_threshold


class LeastVolumeAE_DynamicPruning(LeastVolumeAE):  # noqa: N801
    """Least-volume autoencoder with dynamic dimension pruning.

    Extends LeastVolumeAE by dynamically pruning low-variance latent dimensions
    during training using various pruning strategies.
    """

    def __init__(  # noqa: PLR0913
        self,
        encoder,
        decoder,
        optimizer,
        latent_dim,
        weights=None,
        eta=1,
        beta=0.9,
        ratio_threshold=0.02,  # noqa: ARG002 - kept for backward compatibility, use pruning_params instead
        pruning_epoch=500,
        pruning_strategy="plummet",
        pruning_params=None,
        *,
        min_active_dims: int = 0,
        max_prune_per_epoch: int | None = None,
        cooldown_epochs: int = 0,
        k_consecutive: int = 1,
        recon_tol: float = float("inf"),
    ):
        """Initialize dynamic pruning autoencoder.

        Args:
            encoder: Encoder network
            decoder: Decoder network
            optimizer: Optimizer instance
            latent_dim: Total number of latent dimensions
            weights: Loss weights [reconstruction, volume]
            eta: Smoothing parameter for volume loss
            beta: EMA momentum for latent statistics
            ratio_threshold: DEPRECATED - kept for backward compatibility only.
                Use pruning_params instead to pass strategy-specific parameters.
            pruning_epoch: Epoch to start pruning
            pruning_strategy: Strategy name (plummet, pca_cdf, lognorm, probabilistic)
            pruning_params: Strategy-specific parameters dict (replaces ratio_threshold)
            min_active_dims: Never prune below this many dimensions (default: 0 = no limit)
            max_prune_per_epoch: Max dimensions to prune per epoch (default: None = no limit)
            cooldown_epochs: Epochs to wait between pruning events (default: 0 = no cooldown)
            k_consecutive: Consecutive epochs below threshold required (default: 1 = immediate)
            recon_tol: Relative tolerance to best validation recon (default: inf = no constraint)
        """
        if weights is None:
            weights = [1.0, 0.001]
        super().__init__(encoder, decoder, optimizer, weights, eta)
        self.register_buffer("_p", torch.as_tensor([False] * latent_dim, dtype=torch.bool))
        self.register_buffer("_z", torch.zeros(latent_dim))

        self._beta = beta
        self.pruning_epoch = pruning_epoch

        # Policy --> pca_cdf | lognorm | probabilistic | plummet
        self.policy = PruningPolicy(pruning_strategy, pruning_params)

        # Safeguard configuration (now exposed as init parameters)
        # None means no limit for max_prune_per_epoch
        self.cfg = SimpleNamespace(
            min_active_dims=min_active_dims,
            max_prune_per_epoch=max_prune_per_epoch if max_prune_per_epoch is not None else latent_dim,
            cooldown_epochs=cooldown_epochs,
            K_consecutive=k_consecutive,
            recon_tol=recon_tol,
        )

        # Internal bookkeeping
        self._next_prune_epoch = 0
        self._below_counts = torch.zeros(latent_dim, dtype=torch.long)
        self._best_val_recon = float("inf")

    def to(self, device):
        super().to(device)
        self._p = self._p.to(device)
        self._z = self._z.to(device)
        self._below_counts = self._below_counts.to(device)
        return self

    @property
    def dim(self):
        return (~self._p).sum().item()

    def update_best_val_recon(self, val_rec: float):
        if val_rec < self._best_val_recon:
            self._best_val_recon = float(val_rec)

    def decode(self, z):
        z[:, self._p] = self._z[self._p]
        return self.decoder(z)

    def encode(self, x):
        z = self.encoder(x)
        z[:, self._p] = self._z[self._p]
        return z

    def loss(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        self._update_moving_mean(z)
        m = self.dim / len(self._p)
        return torch.stack([self.loss_rec(x, x_hat), m * self.loss_vol(z[:, ~self._p])])

    @torch.no_grad()
    def _update_moving_mean(self, z):
        if not hasattr(self, "_zstd") or not hasattr(self, "_zmean"):
            self._zstd = z.std(0)
            self._zmean = z.mean(0)
        else:
            self._zstd = torch.lerp(self._zstd, z.std(0), 1 - self._beta)
            self._zmean = torch.lerp(self._zmean, z.mean(0), 1 - self._beta)

    @torch.no_grad()
    def _prune_step(self, epoch, val_recon=None):
        """Core pruning step.

        Applies safeguards + policy to decide whether to prune dims this epoch.
        """
        # --- Safeguards ---
        if self.dim <= self.cfg.min_active_dims:  # stop if we go below min dims
            return
        if epoch < self._next_prune_epoch:  # respect cooldown
            return
        if (
            val_recon is not None
            and self._best_val_recon < float("inf")
            and (val_recon - self._best_val_recon) / self._best_val_recon > self.cfg.recon_tol
        ):
            # Do not prune if recon is already worse than best_val by > tol
            return

        # --- Candidate selection from policy ---
        # Only consider active (unpruned) dimensions for pruning policy
        z_std_active = self._zstd[~self._p]
        cand_active = self.policy(z_std_active).to(self._below_counts.device)

        # Map back to full dimension space
        cand = torch.zeros_like(self._p, dtype=torch.bool)
        cand[~self._p] = cand_active

        # --- Debounce with consecutive evidence ---
        # Keep a counter of how many epochs each dim has been marked as candidate
        # If a dim is not a candidate in the current epoch, reset its count to 0.
        if self.cfg.K_consecutive <= 1:
            stable = cand
            self._below_counts.zero_()
        else:
            # Count how many consecutive epochs each dim is a candidate
            self._below_counts = (self._below_counts + cand.long()) * cand.long()
            stable = self._below_counts >= self.cfg.K_consecutive

        # Filter to *active* dims only (don not re-prune pruned ones)
        candidates = torch.where(stable & (~self._p))[0]
        if len(candidates) == 0:
            return

        # --- Cap how many dims we prune at once ---
        prune_idx = candidates[: self.cfg.max_prune_per_epoch]

        # --- Commit pruning ---
        self._p[prune_idx] = True  # mark as pruned
        self._z[prune_idx] = self._zmean[prune_idx]  # freeze mean for reconstruction
        self._next_prune_epoch = epoch + self.cfg.cooldown_epochs  # set next allowed prune epoch

    # --------------------
    # Training hooks
    # --------------------
    def epoch_report(self, epoch, callbacks, batch=None, loss=None, pbar=None, **kwargs):
        val_recon = kwargs.get("val_recon")
        if isinstance(val_recon, (float, int)):
            self.update_best_val_recon(float(val_recon))

        if epoch == self.pruning_epoch and hasattr(self, "_zstd"):
            self.policy.set_reference(self._zstd)

        if epoch >= self.pruning_epoch and hasattr(self, "_zstd"):
            self._prune_step(epoch, val_recon=val_recon)

        super().epoch_report(
            epoch=epoch,
            callbacks=callbacks,
            batch=batch,
            loss=loss,
            pbar=pbar,
            **kwargs,
        )


class DesignLeastVolumeAE_DP(LeastVolumeAE_DynamicPruning):  # noqa: N801
    """Design autoencoder with performance prediction and dynamic pruning.

    This class extends LeastVolumeAE_DynamicPruning to include performance prediction
    capabilities alongside reconstruction, volume minimization, and pruning.
    """

    _p_mu: torch.Tensor
    _p_std: torch.Tensor
    _perf_epoch_buf: torch.Tensor

    def __init__(  # noqa: PLR0913
        self,
        encoder,
        decoder,
        predictor,
        optimizer,
        latent_dim,
        weights=None,
        eta=1,
        beta=0.9,
        ratio_threshold=0.02,
        pruning_epoch=500,
        pruning_strategy="plummet",
        pruning_params=None,
        *,
        normalize_perf: bool = True,
        perf_ref_warmup_epochs: int = 5,
        perf_ref_momentum: float = 0.9,
        per_dim_perf_ref: bool = True,
        # Safeguard parameters (inherited from parent with same defaults)
        min_active_dims: int = 0,
        max_prune_per_epoch: int | None = None,
        cooldown_epochs: int = 0,
        k_consecutive: int = 1,
        recon_tol: float = float("inf"),
    ):
        """Initialize design autoencoder with performance prediction and dynamic pruning.

        Performance-specific args:
            normalize_perf: Whether to normalize performance values
            perf_ref_warmup_epochs: Epochs to warm up performance normalization
            perf_ref_momentum: EMA momentum for performance stats
            per_dim_perf_ref: Whether to normalize per dimension or globally

        All other args (including safeguards) are inherited from LeastVolumeAE_DynamicPruning.
        """
        if weights is None:
            weights = [1.0, 1.0, 0.001]

        # Build pruning_params from ratio_threshold if not provided
        if pruning_params is None:
            pruning_params = {"threshold": ratio_threshold, "beta": beta}

        # Initialize parent with pruning capabilities
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            latent_dim=latent_dim,
            weights=weights,
            eta=eta,
            beta=beta,
            ratio_threshold=ratio_threshold,
            pruning_epoch=pruning_epoch,
            pruning_strategy=pruning_strategy,
            pruning_params=pruning_params,
            min_active_dims=min_active_dims,
            max_prune_per_epoch=max_prune_per_epoch,
            cooldown_epochs=cooldown_epochs,
            k_consecutive=k_consecutive,
            recon_tol=recon_tol,
        )

        self.predictor = predictor
        self.normalize_perf = normalize_perf
        self.perf_ref_warmup_epochs = perf_ref_warmup_epochs
        self.perf_ref_momentum = perf_ref_momentum
        self.per_dim_perf_ref = per_dim_perf_ref
        self.register_buffer("_p_mu", torch.tensor(0.0))
        self.register_buffer("_p_std", torch.tensor(1.0))
        self.register_buffer("_perf_epoch_buf", torch.zeros((), dtype=torch.long))

    def epoch_hook(self, epoch, *args, **kwargs):
        """Track current epoch for warmup scheduling."""
        super().epoch_hook(epoch, *args, **kwargs)
        self._perf_epoch_buf.fill_(epoch)

    def _norm_perf(self, p: torch.Tensor) -> torch.Tensor:
        """Normalize performance values using exponential moving average statistics."""
        if not self.normalize_perf:
            return p
        # Initialize refs on first use
        if self._p_mu.numel() == 1 and self._p_mu.item() == 0.0:
            if self.per_dim_perf_ref:
                mu = p.mean(0).detach()
                std = p.std(0).detach().clamp_min(1e-8)
            else:
                mu = p.mean().detach().expand_as(p.mean(0))
                std = p.std().detach().clamp_min(1e-8).expand_as(p.std(0))
            self._p_mu = mu
            self._p_std = std

        # Warmup EMA only during training and first K epochs
        if self.training and (self._perf_epoch_buf.item() < self.perf_ref_warmup_epochs):
            with torch.no_grad():
                if self.per_dim_perf_ref:
                    mu = p.mean(0)
                    std = p.std(0).clamp_min(1e-8)
                else:
                    mu = p.mean().expand_as(self._p_mu)
                    std = p.std().clamp_min(1e-8).expand_as(self._p_std)
                a = self.perf_ref_momentum
                self._p_mu = a * self._p_mu + (1 - a) * mu
                self._p_std = a * self._p_std + (1 - a) * std

        return (p - self._p_mu) / self._p_std

    def loss(self, batch, **kwargs):  # noqa: ARG002
        """Compute combined reconstruction, performance prediction, and volume losses."""
        x, c, p = batch
        z = self.encode(x)
        x_hat = self.decode(z)

        # Normalize targets & predictions consistently
        p_hat = self.predictor(torch.cat([z, c], dim=-1))
        p_n = self._norm_perf(p)
        p_hat_n = self._norm_perf(p_hat)

        # Note: _update_moving_mean and pruning logic handled by parent class
        active_ratio = self.dim / len(self._p)  # Scale volume loss by active dimension ratio

        return torch.stack(
            [
                self.loss_rec(x, x_hat),
                self.loss_rec(p_n, p_hat_n),  # normalized prediction loss
                active_ratio * self.loss_vol(z[:, ~self._p]),
            ]
        )


class InterpretableDesignLeastVolumeAE_DP(LeastVolumeAE_DynamicPruning):  # noqa: N801
    """Interpretable design autoencoder where first dimensions predict performance.

    This variant enforces that the first `perf_dim` latent dimensions are dedicated
    to performance prediction, making them more interpretable. Extends
    LeastVolumeAE_DynamicPruning with performance prediction capabilities.

    Performance values should be pre-scaled externally (e.g., using RobustScaler)
    before being passed to the loss function. No runtime normalization is applied.
    """

    def __init__(  # noqa: PLR0913
        self,
        encoder,
        decoder,
        predictor,
        optimizer,
        latent_dim,
        perf_dim,
        weights=None,
        w_r: float | None = None,
        w_p: float | None = None,
        w_v: float | None = None,
        eta=1,
        beta=0.9,
        ratio_threshold=0.02,
        pruning_epoch=500,
        pruning_strategy="plummet",
        pruning_params=None,
        *,
        # Safeguard parameters (inherited from parent with same defaults)
        min_active_dims: int = 0,
        max_prune_per_epoch: int | None = None,
        cooldown_epochs: int = 0,
        k_consecutive: int = 1,
        recon_tol: float = float("inf"),
    ):
        """Initialize interpretable design autoencoder with performance prediction.

        This variant enforces that the first `perf_dim` latent dimensions are dedicated
        to performance prediction, making them more interpretable.

        Performance-specific args:
            perf_dim: Number of latent dimensions dedicated to performance prediction
            w_r: Weight for reconstruction loss (default: 1.0)
            w_p: Weight for performance loss (default: 0.1, reduced to prevent dominance)
            w_v: Weight for volume loss (default: 0.001)

        All other args (including safeguards) are inherited from LeastVolumeAE_DynamicPruning.

        Note: Performance values should be pre-scaled externally (e.g., using RobustScaler)
        before being passed to the loss function.
        """
        # Handle weight specification: either weights array or individual w_r/w_p/w_v
        if weights is None:
            w_r = w_r if w_r is not None else 1.0
            w_p = w_p if w_p is not None else 0.1  # Reduced default to prevent performance loss dominance
            w_v = w_v if w_v is not None else 0.001
            weights = [w_r, w_p, w_v]

        # Build pruning_params from ratio_threshold if not provided
        if pruning_params is None:
            pruning_params = {"threshold": ratio_threshold, "beta": beta}

        # Initialize parent with pruning capabilities
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            latent_dim=latent_dim,
            weights=weights,
            eta=eta,
            beta=beta,
            ratio_threshold=ratio_threshold,
            pruning_epoch=pruning_epoch,
            pruning_strategy=pruning_strategy,
            pruning_params=pruning_params,
            min_active_dims=min_active_dims,
            max_prune_per_epoch=max_prune_per_epoch,
            cooldown_epochs=cooldown_epochs,
            k_consecutive=k_consecutive,
            recon_tol=recon_tol,
        )

        self.predictor = predictor
        self.pdim = perf_dim

    def loss(self, batch, **kwargs):  # noqa: ARG002
        """Compute losses using only first pdim latents for performance prediction.

        Note: Performance values (p) should be pre-scaled externally before training.
        """
        x, c, p = batch  # p is already scaled externally
        z = self.encode(x)
        x_hat = self.decode(z)

        # Update moving mean for pruning statistics
        self._update_moving_mean(z)

        # Only the first pdim dimensions are used for performance prediction
        pz = z[:, : self.pdim]
        p_hat = self.predictor(torch.cat([pz, c], dim=-1))

        # Direct MSE on pre-scaled values (no runtime normalization)
        active_ratio = self.dim / len(self._p)  # Scale volume loss by active dimension ratio

        return torch.stack(
            [
                self.loss_rec(x, x_hat),
                self.loss_rec(p, p_hat),  # Both already scaled
                active_ratio * self.loss_vol(z[:, ~self._p]),
            ]
        )


class ConstrainedDesignLeastVolumeAE_DP(InterpretableDesignLeastVolumeAE_DP):  # noqa: N801
    """Interpretable design LVAE with constraint handling for multi-objective optimization.

    Extends InterpretableDesignLeastVolumeAE_DP to support various constraint optimization
    methods (weighted sum, augmented Lagrangian, log barrier, etc.) for balancing
    reconstruction, performance, and volume objectives.

    This eliminates the need for wrapper classes in training scripts - all constraint
    handling logic is built into this base class.
    """

    def __init__(  # noqa: PLR0913
        self,
        encoder,
        decoder,
        predictor,
        optimizer,
        latent_dim,
        perf_dim,
        constraint_handler,
        reconstruction_threshold: float = 0.001,
        performance_threshold: float = 0.01,
        *,
        conditional_predictor: bool = True,
        weights=None,
        w_r: float | None = None,
        w_p: float | None = None,
        w_v: float | None = None,
        eta=1,
        beta=0.9,
        ratio_threshold=0.02,
        pruning_epoch=500,
        pruning_strategy="plummet",
        pruning_params=None,
        min_active_dims: int = 0,
        max_prune_per_epoch: int | None = None,
        cooldown_epochs: int = 0,
        k_consecutive: int = 1,
        recon_tol: float = float("inf"),
    ):
        """Initialize constrained design LVAE.

        Args:
            encoder: Encoder network
            decoder: Decoder network
            predictor: Performance prediction network
            optimizer: Optimizer instance
            latent_dim: Total latent dimensions
            perf_dim: Number of dimensions for performance prediction
            constraint_handler: ConstraintHandler instance for multi-objective optimization
            reconstruction_threshold: Constraint threshold for reconstruction loss
            performance_threshold: Constraint threshold for performance loss
            conditional_predictor: Whether predictor uses conditions (True) or only latent (False)
            All other args: Same as InterpretableDesignLeastVolumeAE_DP
        """
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            predictor=predictor,
            optimizer=optimizer,
            latent_dim=latent_dim,
            perf_dim=perf_dim,
            weights=weights,
            w_r=w_r,
            w_p=w_p,
            w_v=w_v,
            eta=eta,
            beta=beta,
            ratio_threshold=ratio_threshold,
            pruning_epoch=pruning_epoch,
            pruning_strategy=pruning_strategy,
            pruning_params=pruning_params,
            min_active_dims=min_active_dims,
            max_prune_per_epoch=max_prune_per_epoch,
            cooldown_epochs=cooldown_epochs,
            k_consecutive=k_consecutive,
            recon_tol=recon_tol,
        )

        self.conditional_predictor = conditional_predictor
        self.constraint_handler: ConstraintHandler = constraint_handler
        self.reconstruction_threshold = reconstruction_threshold
        self.performance_threshold = performance_threshold
        self._loss_components: ConstraintLosses | None = None  # Cache for constraint handler

    def epoch_hook(self, epoch, *args, **kwargs):
        """Update weight schedule and constraint handler epoch."""
        super().epoch_hook(epoch, *args, **kwargs)
        self.constraint_handler.epoch_hook(epoch)

    def loss(self, batch, **kwargs):  # noqa: ARG002
        """Compute loss components and cache for constraint handler.

        Returns individual loss components for logging: [reconstruction, performance, volume]
        Call compute_total_loss() after this to get the actual backprop loss.
        """
        x, c, p = batch
        z = self.encode(x)
        x_hat = self.decode(z)

        # Update moving mean for pruning statistics
        self._update_moving_mean(z)

        # Only the first pdim dimensions are used for performance prediction
        pz = z[:, : self.pdim]

        # Conditional: use [z[:perf_dim], c] or just z[:perf_dim]
        if self.conditional_predictor:
            p_hat = self.predictor(torch.cat([pz, c], dim=-1))
        else:
            p_hat = self.predictor(pz)

        # Compute individual loss components
        reconstruction_loss = self.loss_rec(x, x_hat)
        performance_loss = self.loss_rec(p, p_hat)
        active_ratio = self.dim / len(self._p)
        volume_loss = active_ratio * self.loss_vol(z[:, ~self._p])

        # Import here to avoid circular dependency
        from engiopt.lvae_core.constraint_handlers import ConstraintLosses

        # Cache components for constraint handler
        self._loss_components = ConstraintLosses(
            volume=volume_loss,
            reconstruction=reconstruction_loss,
            performance=performance_loss,
        )

        # Return as tensor for logging
        return torch.stack([reconstruction_loss, performance_loss, volume_loss])

    def compute_total_loss(self):
        """Compute total loss using constraint handler.

        Call this after loss() to get the actual loss for backprop.
        """
        from engiopt.lvae_core.constraint_handlers import ConstraintThresholds

        if self._loss_components is None:
            raise RuntimeError("Must call loss() before compute_total_loss()")

        thresholds = ConstraintThresholds(
            reconstruction=self.reconstruction_threshold,
            performance=self.performance_threshold,
        )
        return self.constraint_handler.compute_loss(self._loss_components, thresholds)

    def update_constraint_handler(self):
        """Update constraint handler state (e.g., dual variables, barrier parameter)."""
        from engiopt.lvae_core.constraint_handlers import ConstraintLosses, ConstraintThresholds

        if self._loss_components is None:
            raise RuntimeError("Must call loss() before update_constraint_handler()")

        thresholds = ConstraintThresholds(
            reconstruction=self.reconstruction_threshold,
            performance=self.performance_threshold,
        )
        # Detach to avoid backprop through constraint handler updates
        detached_losses = ConstraintLosses(
            volume=self._loss_components.volume.detach(),
            reconstruction=self._loss_components.reconstruction.detach(),
            performance=self._loss_components.performance.detach(),
        )
        self.constraint_handler.step(detached_losses, thresholds)

    def get_constraint_metrics(self) -> dict[str, float]:
        """Get constraint handler metrics for logging."""
        return self.constraint_handler.get_metrics()
