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

from collections import OrderedDict
import os
from types import SimpleNamespace

from scipy.stats import norm
import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.independent import Independent
from torch.distributions.kl import kl_divergence
import torch.nn.functional as f
from tqdm import tqdm


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
                for i, batch in enumerate(dataloader):
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
    def __init__(
        self,
        encoder,
        decoder,
        optimizer,
        weights=None,
        eta=1,
        vol_ref_warmup_epochs: int = 5,
        vol_ref_momentum: float = 0.9,
        *,
        normalize_vol: bool = True,
        per_dim_ref: bool = True,
    ):
        if weights is None:
            weights = [1.0, 0.001]
        super().__init__(encoder, decoder, optimizer, weights)
        self.eta = eta
        self.normalize_vol = normalize_vol
        self.vol_ref_warmup_epochs = vol_ref_warmup_epochs
        self.vol_ref_momentum = vol_ref_momentum
        self.per_dim_ref = per_dim_ref
        self.register_buffer("_sigma_ref", None)  # shape matches latent dims
        self.register_buffer("_epoch_buf", torch.zeros((), dtype=torch.long))

    def epoch_hook(self, epoch, *args, **kwargs):
        super().epoch_hook(epoch, *args, **kwargs)
        self._epoch_buf.fill_(epoch)

    def loss(self, x, **kwargs):
        z = self.encode(x)
        x_hat = self.decode(z)
        return torch.stack([self.loss_rec(x, x_hat), self.loss_vol(z)])

    def loss_vol(self, z):
        s = z.std(0)

        if not self.normalize_vol:
            return torch.exp(torch.log(s + self.eta).mean())

        if self._sigma_ref is None or self._sigma_ref.shape != s.shape:
            self._sigma_ref = s.detach().clone()
        # --- Normalized volume: exp(mean(log((std+eta)/(ref+eta)))) ~ 1 at start ---
        # init _sigma_ref on first call
        if self._sigma_ref is None:
            if self.per_dim_ref:
                self._sigma_ref = s.detach().clone()
            else:
                scal = s.median().detach()
                self._sigma_ref = scal.expand_as(s)

        # EMA update only during warmup and only in training mode
        if self.training and (self._epoch_buf.item() < self.vol_ref_warmup_epochs):
            with torch.no_grad():
                alpha = self.vol_ref_momentum
                if self.per_dim_ref:
                    self._sigma_ref = alpha * self._sigma_ref + (1 - alpha) * s
                else:
                    scal = s.median()
                    self._sigma_ref = alpha * self._sigma_ref + (1 - alpha) * scal.expand_as(s)

        # compute dimensionless normalized volume on original scale (≈1 initially)
        return torch.exp(torch.log((s + self.eta) / (self._sigma_ref + self.eta)).mean())


class PruningPolicy:
    def __init__(self, strategy, pruning_params=None):
        # Initialize pruning policy with strategy name and keyword arguments
        self.strategy = strategy
        self.pruning_params = pruning_params or {}
        self._ref_total_var = None  # will hold snapshot at pruning_epoch
        self._ref_mu = None  # will hold snapshot at pruning_epoch
        self._ref_sigma = None  # will hold snapshot at pruning_epoch

    def __call__(self, z_std):
        # Select pruning strategy and call appropriate method
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
        raise ValueError(self.strategy)

    def set_reference(self, z_std):
        """Call this at pruning_epoch to snapshot variance/distribution."""
        self._ref_total_var = float((z_std**2).sum().item())
        log_std = torch.log(z_std.clamp_min(1e-12))
        self._ref_mu = log_std.mean().item()
        self._ref_sigma = log_std.std().item()

    def _pca_cdf(self, z_std, thr):
        # PCA-inspired pruning based on cumulative variance
        # 1. Calculate relative contribution of each dimension to total variance
        contrib = z_std**2
        total = contrib.sum().clamp_min(1e-12)

        # fallback to relative mode until reference is set
        ref_total = total if self._ref_total_var is None else self._ref_total_var

        # 2. Normalize contributions relative to reference variance
        contrib /= ref_total

        # 3. Sort dimensions by contribution and compute cumulative sum
        vals, idx = torch.sort(contrib, descending=True)
        cdf = torch.cumsum(vals, 0)

        # 4. Mark dimensions to keep (those needed to reach threshold)
        keep_sorted = cdf <= thr
        keep_sorted[0] = True  # always keep top-1

        # 5. Convert back to original dimension ordering
        keep = torch.zeros_like(keep_sorted)
        keep[idx] = keep_sorted

        return ~keep  # return pruning mask (True = prune)

    def _lognorm(self, z_std, percentile=0.05, alpha=1.0):
        # Log-normal cdf-based pruning fitting reference distribution
        log_std = torch.log(z_std.clamp_min(1e-12))
        mu_current = log_std.mean().item()
        sigma_current = log_std.std().clamp_min(1e-6).item()

        if self._ref_mu is None or self._ref_sigma is None:
            # If no reference set yet, fallback to current fit
            mu_blend = mu_current
            sigma_blend = sigma_current
        else:
            # Blend between snapshot and current distribution
            mu_blend = (1 - alpha) * self._ref_mu + alpha * mu_current
            sigma_blend = (1 - alpha) * self._ref_sigma + alpha * sigma_current

        cutoff_val = mu_blend + sigma_blend * float(norm.ppf(percentile))
        cutoff = torch.exp(torch.tensor(cutoff_val, device=z_std.device, dtype=z_std.dtype))
        return z_std < cutoff

    @staticmethod
    def _probabilistic(z_std, temperature):
        # Probabilistic pruning with temperature control
        # 1. Calculate relative contribution of each dimension
        contrib = (z_std / z_std.sum().clamp_min(1e-12)).clamp_min(1e-12)

        # 2. Convert to pruning scores (higher for low contribution)
        score = 1.0 - (contrib / contrib.max())

        # 3. Apply temperature scaling
        p = (score / score.max().clamp_min(1e-12)) / max(temperature, 1e-6)
        p = p.clamp(0, 1)

        # 4. Sample pruning decisions
        return torch.bernoulli(p).bool()

    @staticmethod
    def _plummet(z_std, plummet_threshold):
        # Ratio-based pruning looking for sudden drops
        # 1. Sort standard deviations
        srt, idx = torch.sort(z_std, descending=True)

        # 2. Calculate log differences between consecutive values
        log = (srt + 1e-12).log()
        d = log[1:] - log[:-1]

        # 3. Find largest drop point
        pidx_sorted = torch.argmin(d)  # keep as tensor index
        ref = srt[pidx_sorted]

        # 4. Prune dimensions with std much smaller than reference
        ratio = z_std / (ref + 1e-12)
        return ratio < plummet_threshold


class LeastVolumeAE_DynamicPruning(LeastVolumeAE):  # noqa: N801
    def __init__(
        self,
        encoder,
        decoder,
        optimizer,
        latent_dim,
        weights=None,
        eta=1,
        beta=0.9,
        pruning_epoch=500,
        pruning_strategy="plummet",
        pruning_params=None,
    ):
        if weights is None:
            weights = [1.0, 0.001]
        super().__init__(encoder, decoder, optimizer, weights, eta)
        self.register_buffer("_p", torch.as_tensor([False] * latent_dim, dtype=torch.bool))
        self.register_buffer("_z", torch.zeros(latent_dim))

        self._beta = beta
        self.pruning_epoch = pruning_epoch

        # Policy --> pca_cdf | lognorm | probabilistic | plummet
        self.policy = PruningPolicy(pruning_strategy, pruning_params)

        # Safeguard configuration
        self.cfg = SimpleNamespace(
            min_active_dims=1,  # Never prune below this many dims
            max_prune_per_epoch=1000,  # Max dims to prune in one epoch
            cooldown_epochs=0,  # Epochs to wait between pruning events
            K_consecutive=0,  # Consecutive epochs a dim must be below threshold to be eligible
            recon_tol=1,  # Relative tolerance to best_val_recon to allow pruning
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
        cand = self.policy(self._zstd).to(self._below_counts.device)  # boolean mask of "low variance" candidates

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


class DesignLeastVolumeAE_DP(LeastVolumeAE):  # noqa: N801
    def __init__(
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
        pruning_epoch=50,
        *,
        normalize_perf: bool = True,
        perf_ref_warmup_epochs: int = 5,
        perf_ref_momentum: float = 0.9,
        per_dim_perf_ref: bool = True,
    ):
        if weights is None:
            weights = [1.0, 1.0, 0.001]
        self.predictor = predictor
        self.normalize_perf = normalize_perf
        self.perf_ref_warmup_epochs = perf_ref_warmup_epochs
        self.perf_ref_momentum = perf_ref_momentum
        self.per_dim_perf_ref = per_dim_perf_ref
        self.register_buffer("_p_mu", None)
        self.register_buffer("_p_std", None)
        self.register_buffer("_epoch_buf", torch.zeros((), dtype=torch.long))

    # track epoch so we only warm up refs for first N epochs ---
    def epoch_hook(self, epoch, *args, **kwargs):
        super().epoch_hook(epoch, *args, **kwargs)
        self._epoch_buf.fill_(epoch)

    # Standardize performance targets/preds with EMA stats
    def _norm_perf(self, p: torch.Tensor) -> torch.Tensor:
        if not self.normalize_perf:
            return p
        # Initialize refs on first use
        if self._p_mu is None or self._p_std is None:
            if self.per_dim_perf_ref:
                mu = p.mean(0).detach()
                std = p.std(0).detach().clamp_min(1e-8)
            else:
                mu = p.mean().detach().expand_as(p.mean(0))
                std = p.std().detach().clamp_min(1e-8).expand_as(p.std(0))
            self._p_mu = mu
            self._p_std = std

        # Warmup EMA only during training and first K epochs
        if self.training and (self._epoch_buf.item() < self.perf_ref_warmup_epochs):
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

    def loss(self, batch, **kwargs):
        x, c, p = batch
        z = self.encode(x)
        x_hat = self.decode(z)

        # Normalize targets & predictions consistently ---
        p_hat = self.predictor(torch.cat([z, c], dim=-1))
        p_n = self._norm_perf(p)
        p_hat_n = self._norm_perf(p_hat)

        self._update_moving_mean(z)
        m = self.dim / len(self._p)  # keep your existing pruning scaling

        return torch.stack(
            [
                self.loss_rec(x, x_hat),
                self.loss_rec(p_n, p_hat_n),  # normalized prediction loss
                m * self.loss_vol(z[:, ~self._p]),
            ]
        )


class InterpretableDesignLeastVolumeAE_DP(LeastVolumeAE):  # noqa: N801
    def __init__(
        self,
        encoder,
        decoder,
        predictor,
        optimizer,
        latent_dim,
        perf_dim,
        weights=None,
        eta=1,
        beta=0.9,
        ratio_threshold=0.02,
        pruning_epoch=50,
        *,
        normalize_perf: bool = True,
        perf_ref_warmup_epochs: int = 5,
        perf_ref_momentum: float = 0.9,
        per_dim_perf_ref: bool = True,
    ):
        if weights is None:
            weights = [1.0, 1.0, 0.001]
        self.predictor = predictor
        self.pdim = perf_dim

        self.normalize_perf = normalize_perf
        self.perf_ref_warmup_epochs = perf_ref_warmup_epochs
        self.perf_ref_momentum = perf_ref_momentum
        self.per_dim_perf_ref = per_dim_perf_ref
        self.register_buffer("_p_mu", None)
        self.register_buffer("_p_std", None)
        self.register_buffer("_epoch_buf", torch.zeros((), dtype=torch.long))

    # --- track current epoch so warmup knows when to stop ---
    def epoch_hook(self, epoch, *args, **kwargs):
        super().epoch_hook(epoch, *args, **kwargs)
        self._epoch_buf.fill_(epoch)

    # --- normalize perf targets/preds with EMA stats ---
    def _norm_perf(self, p: torch.Tensor) -> torch.Tensor:
        if not self.normalize_perf:
            return p

        if self._p_mu is None or self._p_std is None:
            if self.per_dim_perf_ref:
                mu = p.mean(0).detach()
                std = p.std(0).detach().clamp_min(1e-8)
            else:
                mu = p.mean().detach().expand_as(p.mean(0))
                std = p.std().detach().clamp_min(1e-8).expand_as(p.std(0))
            self._p_mu = mu
            self._p_std = std

        if self.training and (self._epoch_buf.item() < self.perf_ref_warmup_epochs):
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

    def loss(self, batch, **kwargs):
        x, c, p = batch
        z = self.encode(x)
        x_hat = self.decode(z)

        # Only the first pdim are for performance prediction
        pz = z[:, : self.pdim]
        p_hat = self.predictor(torch.cat([pz, c], dim=-1))

        # --- normalize targets and preds ---
        p_n = self._norm_perf(p)
        p_hat_n = self._norm_perf(p_hat)

        self._update_moving_mean(z)
        m = self.dim / len(self._p)

        return torch.stack(
            [
                self.loss_rec(x, x_hat),
                self.loss_rec(p_n, p_hat_n),  # normalized prediction loss
                m * self.loss_vol(z[:, ~self._p]),
            ]
        )
