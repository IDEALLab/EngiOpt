"""
DDM_W: Denoising Diffusion Model operating in the LVAE latent space (w).

Architecture
------------
Instead of denoising in BAE latent space [B, S, 3, 30] (2430 dims), this model
denoises in the LVAE's compressed latent space [B, 64] — a 38x reduction.

At training time:
  - Wings are encoded: BAE → z_opt [B, S, 3, 30] → frozen LVAE encoder → w [B, 64]
  - The DDM learns to denoise w vectors, conditioned on flow params and w_init
    (the LVAE encoding of the initial/unoptimised wing)
  - Loss: MSE on denoised w + AoA + optional pressure (via frozen LVAE decoder)

At inference time:
  - Sample noise w ~ N(0, I), denoise with UNet → w_gen [B, 64]
  - Decode: frozen LVAE decoder(w_gen, params) → z_opt_pred [B, S, 3, 30]
  - Decode: frozen BAE decoder(z_opt_pred) → coordinates [B, S, 2, 192]
  - te_shifts come free from LVAE decoder's eta_y_pred output (no GT hack needed)
  - Pressure comes free from LVAE decoder's pressure_pred output
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MLPDenoiser(nn.Module):
    """Small MLP that denoises a flat w vector.

    Input:  [B, w_dim + w_dim + c_dim + 1]  (noisy_w, w_init, params, t_emb)
    Output: [B, w_dim]  (predicted noise in w-space)

    Also predicts AoA noise as a scalar.
    """

    def __init__(self, w_dim: int = 64, c_dim: int = 4,
                 hidden_dims: tuple = (512, 512, 512, 512),
                 t_embed_dim: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        self.w_dim       = w_dim
        self.c_dim       = c_dim
        self.t_embed_dim = t_embed_dim

        # Sinusoidal time embedding
        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, t_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(t_embed_dim * 2, t_embed_dim),
        )

        in_dim = w_dim + w_dim + c_dim + t_embed_dim  # noisy_w + w_init + params + t

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        self.net = nn.Sequential(*layers)
        self.out_w   = nn.Linear(prev, w_dim)   # noise in w
        self.out_aoa = nn.Linear(prev, 1)        # noise in AoA

    def _sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] integer timesteps → [B, t_embed_dim]."""
        half = self.t_embed_dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device, dtype=torch.float32)
            * (np.log(10000) / (half - 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B, t_embed_dim]

    def forward(self, w_noisy: torch.Tensor, aoa_noisy: torch.Tensor,
                params: torch.Tensor, w_init: torch.Tensor,
                t: torch.Tensor):
        """
        w_noisy  : [B, w_dim]
        aoa_noisy: [B, 1]
        params   : [B, c_dim]
        w_init   : [B, w_dim]  — LVAE encoding of initial wing (conditioning)
        t        : [B]         — diffusion timestep
        Returns  : (w_noise_pred [B, w_dim], aoa_noise_pred [B, 1])
        """
        t_emb = self._sinusoidal_embedding(t)         # [B, t_embed_dim]
        t_emb = self.t_mlp(t_emb)                     # [B, t_embed_dim]
        x = torch.cat([w_noisy, w_init, params, t_emb], dim=1)
        h = self.net(x)
        return self.out_w(h), self.out_aoa(h)


class DDM_W:
    """Diffusion model in LVAE w-space.

    Parameters
    ----------
    denoiser      : MLPDenoiser
    lvae_model    : frozen LAE_AoAInit
    bae_model     : frozen BezierAutoencoder
    sampler       : noise schedule (provides T, sqrt_alphas_cumprod, etc.)
    w_dim         : LVAE latent dimension (64)
    c_dim         : condition dimension (4 flow params)
    w_pressure    : weight for pressure loss term
    lvae_params_dim : full condition dim expected by LVAE (4)
    """

    def __init__(
        self,
        denoiser: MLPDenoiser,
        lvae_model,
        bae_model,
        sampler,
        w_dim: int = 64,
        c_dim: int = 4,
        w_pressure: float = 1.0,
        w_aoa: float = 1.0,
        lvae_params_dim: int = 4,
        params_mean_std=None,
        aoas_mean_std=None,
        name: str = "ddm_w_v1",
        opt_lr: float = 1e-3,
    ):
        self.denoiser       = denoiser
        self.lvae_model     = lvae_model
        self.bae_model      = bae_model
        self.sampler        = sampler
        self.w_dim          = w_dim
        self.c_dim          = c_dim
        self.w_pressure     = w_pressure
        self.w_aoa          = w_aoa
        self.lvae_params_dim = lvae_params_dim
        self.name           = name

        # Normalization stats for w (LVAE latent space)
        self.w_mean = None   # [1, w_dim] — set after precomputing training w's
        self.w_std  = None

        # Normalization scalers for conditions and AoA
        from engiopt.data_processing.utils import scaler as make_scaler
        self.scaler_params = make_scaler(params_mean_std) if params_mean_std is not None else None
        self.scaler_aoas   = make_scaler(aoas_mean_std)   if aoas_mean_std   is not None else None

        # Optimizer
        self.optimizer = Adam(self.denoiser.parameters(), lr=opt_lr)
        self.scheduler = None

        self.stats = {'current_loss': 0.0, 'best_loss': float('inf')}

        # Freeze LVAE + BAE completely
        self.lvae_model.encoder.eval()
        self.lvae_model.decoder.eval()
        for p in self.lvae_model.encoder.parameters():
            p.requires_grad_(False)
        for p in self.lvae_model.decoder.parameters():
            p.requires_grad_(False)
        self.bae_model.eval()
        for p in self.bae_model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_index(self, arr: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
        """Index arr at positions t and reshape to broadcast with shape."""
        out = arr.gather(0, t.cpu()).float()
        while out.ndim < len(shape):
            out = out.unsqueeze(-1)
        return out.to(t.device)

    def _forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor):
        """Add noise at timestep t.  Returns (x_noisy, noise)."""
        schedule = self.sampler.schedule_x
        sqrt_ab  = self._get_index(schedule.sqrt_alphas_cumprod,       t, x0.shape)
        sqrt_1ab = self._get_index(schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        noise = torch.randn_like(x0)
        return sqrt_ab * x0 + sqrt_1ab * noise, noise

    def _lvae_params(self, params_norm: torch.Tensor) -> torch.Tensor:
        """Convert DDM-normalised params → LVAE-normalised params."""
        device = params_norm.device
        if self.scaler_params is not None:
            raw_np = self.scaler_params.inverse_transform(params_norm.detach().cpu().numpy())
            raw = torch.tensor(raw_np, dtype=torch.float32, device=device)
        else:
            raw = params_norm
        if (hasattr(self.lvae_model, 'scaler_params')
                and self.lvae_model.scaler_params is not None):
            scaled_np = self.lvae_model.scaler_params.transform(raw.detach().cpu().numpy())
            return torch.tensor(scaled_np, dtype=torch.float32, device=device)
        return raw

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, batch, return_components=False):
        """
        batch: (w_opt, aoa, params, w_init, pressure_gt)
          w_opt      : [B, w_dim]   — LVAE encoding of target wing (normalised)
          aoa        : [B, 1]       — normalised AoA
          params     : [B, c_dim]   — normalised flow params
          w_init     : [B, w_dim]   — LVAE encoding of initial wing (normalised)
          pressure_gt: [B, S, 192]  — normalised pressure (or None)
        """
        w_opt, aoa, params, w_init, pressure_gt = batch
        device = w_opt.device
        B = w_opt.shape[0]

        t = torch.randint(0, self.sampler.T, (B,), device=device)

        w_noisy,   w_noise   = self._forward_diffusion(w_opt, t)
        aoa_noisy, aoa_noise = self._forward_diffusion(aoa,   t)

        w_noise_pred, aoa_noise_pred = self.denoiser(
            w_noisy, aoa_noisy, params, w_init, t
        )

        loss_w   = nn.functional.mse_loss(w_noise_pred,   w_noise)
        loss_aoa = nn.functional.mse_loss(aoa_noise_pred, aoa_noise)

        # Pressure loss via frozen LVAE decoder
        loss_p = torch.tensor(0.0, device=device)
        if self.w_pressure > 0 and pressure_gt is not None:
            # Recover x0 estimate from noisy w
            schedule = self.sampler.schedule_x
            sqrt_ab  = self._get_index(schedule.sqrt_alphas_cumprod,           t, w_noisy.shape)
            sqrt_1ab = self._get_index(schedule.sqrt_one_minus_alphas_cumprod, t, w_noisy.shape)
            w0_est = (w_noisy - sqrt_1ab * w_noise_pred) / sqrt_ab.clamp(min=1e-8)

            # Denormalise w0_est → raw LVAE latent space
            if self.w_mean is not None:
                w0_raw = w0_est * self.w_std.to(device) + self.w_mean.to(device)
            else:
                w0_raw = w0_est

            # Apply active mask so unused dims are zeroed (matches LVAE training)
            w0_masked = self.lvae_model._apply_mask(w0_raw)

            lvae_params = self._lvae_params(params)
            self.lvae_model.encoder.to(device)
            self.lvae_model.decoder.to(device)

            with torch.no_grad():
                _, _, _, pressure_pred, _ = self.lvae_model.decoder(w0_masked, lvae_params)

            # pressure_pred is in LVAE-normalised space; convert to DDM-normalised space
            lvae_ps = getattr(self.lvae_model, 'pressures_mean_std', None)
            if lvae_ps is not None:
                p_lvae_mean = torch.tensor(float(lvae_ps[0]), device=device)
                p_lvae_std  = torch.tensor(float(lvae_ps[1]), device=device)
                p_raw = pressure_pred * p_lvae_std + p_lvae_mean
            else:
                p_raw = pressure_pred

            if hasattr(self, 'p_ddm_mean'):
                p_ddm_mean = torch.tensor(self.p_ddm_mean, device=device)
                p_ddm_std  = torch.tensor(self.p_ddm_std,  device=device)
                pressure_pred_norm = (p_raw - p_ddm_mean) / p_ddm_std.clamp(min=1e-8)
            else:
                pressure_pred_norm = p_raw

            loss_p = nn.functional.mse_loss(pressure_pred_norm, pressure_gt)

        total = loss_w + self.w_aoa * loss_aoa + self.w_pressure * loss_p

        if return_components:
            return total, loss_w, loss_aoa, loss_p
        return total

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, w_init: torch.Tensor, params: torch.Tensor,
                 device: str, T: int = None):
        """
        Denoise from pure noise in w-space, then decode through frozen LVAE + BAE.

        Parameters
        ----------
        w_init   : [B, w_dim]  — LVAE encoding of initial wing (normalised)
        params   : [B, c_dim]  — normalised flow conditions
        device   : str
        T        : number of diffusion steps (default: sampler.T)

        Returns
        -------
        coords      : [B, S, 2, 192]  decoded coordinates
        aoas        : [B]             predicted AoA (raw degrees)
        pressures   : [B, S, 192]     predicted pressure coefficients
        te_shifts   : [B, S]          predicted TE y-shifts (from LVAE eta_y_pred)
        """
        T = T or self.sampler.T
        B = w_init.shape[0]

        w_noisy   = torch.randn(B, self.w_dim, device=device)
        aoa_noisy = torch.randn(B, 1,          device=device)

        schedule = self.sampler.schedule_x
        # DDPM reverse process
        for i in reversed(range(T)):
            t = torch.full((B,), i, device=device, dtype=torch.long)

            w_noise_pred, aoa_noise_pred = self.denoiser(
                w_noisy, aoa_noisy, params, w_init, t
            )

            # DDPM sampling step
            alpha_t      = schedule.alphas[i]
            alpha_bar_t  = schedule.alphas_cumprod[i]
            alpha_bar_t1 = schedule.alphas_cumprod[i - 1] if i > 0 else torch.tensor(1.0)
            beta_t       = 1.0 - alpha_t

            w0_est = (w_noisy - (1 - alpha_bar_t).sqrt() * w_noise_pred) / alpha_bar_t.sqrt().clamp(min=1e-8)
            w0_est = w0_est.clamp(-5, 5)

            if i > 0:
                posterior_var = beta_t * (1 - alpha_bar_t1) / (1 - alpha_bar_t).clamp(min=1e-8)
                posterior_mean = (
                    alpha_bar_t1.sqrt() * beta_t / (1 - alpha_bar_t).clamp(min=1e-8) * w0_est
                    + alpha_t.sqrt() * (1 - alpha_bar_t1) / (1 - alpha_bar_t).clamp(min=1e-8) * w_noisy
                )
                w_noisy = posterior_mean + posterior_var.sqrt() * torch.randn_like(w_noisy)

                aoa0_est  = (aoa_noisy - (1 - alpha_bar_t).sqrt() * aoa_noise_pred) / alpha_bar_t.sqrt().clamp(min=1e-8)
                aoa0_est  = aoa0_est.clamp(-5, 5)
                aoa_mean  = (
                    alpha_bar_t1.sqrt() * beta_t / (1 - alpha_bar_t).clamp(min=1e-8) * aoa0_est
                    + alpha_t.sqrt() * (1 - alpha_bar_t1) / (1 - alpha_bar_t).clamp(min=1e-8) * aoa_noisy
                )
                aoa_noisy = aoa_mean + posterior_var.sqrt() * torch.randn_like(aoa_noisy)
            else:
                w_noisy   = w0_est
                aoa_noisy = (aoa_noisy - (1 - alpha_bar_t).sqrt() * aoa_noise_pred) / alpha_bar_t.sqrt().clamp(min=1e-8)
                aoa_noisy = aoa_noisy.clamp(-5, 5)

        w_gen   = w_noisy    # [B, w_dim]  (normalised)
        aoa_gen = aoa_noisy  # [B, 1]      (normalised)

        # Denormalise w
        if self.w_mean is not None:
            w_raw = w_gen * self.w_std.to(device) + self.w_mean.to(device)
        else:
            w_raw = w_gen

        # Apply active mask
        w_masked = self.lvae_model._apply_mask(w_raw)

        # Decode through frozen LVAE
        lvae_params = self._lvae_params(params)
        self.lvae_model.decoder.to(device)
        z_opt_pred, _, eta_y_pred, pressure_pred, _ = self.lvae_model.decoder(w_masked, lvae_params)
        # z_opt_pred : [B, S, 3, 30]
        # eta_y_pred : [B, S, 1]
        # pressure_pred : [B, S, 192]  (LVAE-normalised)

        # Decode z_opt_pred through frozen BAE slice-by-slice
        S = z_opt_pred.shape[1]
        self.bae_model.to(device)
        coords_list = []
        for s in range(S):
            z_s = z_opt_pred[:, s]  # [B, 3, 30]
            dec, _, _ = self.bae_model.decode_z(
                z_s, z_ae_mode=True, denormalize_output=False, normalized_data=False
            )
            coords_list.append(dec)
        coords = torch.stack(coords_list, dim=1)  # [B, S, 2, 192]

        # Denormalise AoA
        if self.scaler_aoas is not None:
            aoa_np = self.scaler_aoas.inverse_transform(aoa_gen.cpu().numpy())
            aoas   = torch.tensor(aoa_np, dtype=torch.float32).squeeze(1)
        else:
            aoas = aoa_gen.squeeze(1)

        # Denormalise pressure
        lvae_ps = getattr(self.lvae_model, 'pressures_mean_std', None)
        if lvae_ps is not None:
            p_mean = torch.tensor(float(lvae_ps[0]), device=device)
            p_std  = torch.tensor(float(lvae_ps[1]), device=device)
            pressures = pressure_pred * p_std + p_mean
        else:
            pressures = pressure_pred

        te_shifts = eta_y_pred.squeeze(-1)  # [B, S]

        return coords.cpu(), aoas.cpu(), pressures.cpu(), te_shifts.cpu()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, save_dir: str, suffix: str = ""):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{self.name}{suffix}.pth")
        torch.save({
            'denoiser':         self.denoiser.state_dict(),
            'optimizer':        self.optimizer.state_dict(),
            'stats':            self.stats,
            'w_mean':           self.w_mean,
            'w_std':            self.w_std,
            'params_mean_std':  (self.scaler_params.mean, self.scaler_params.std)
                                 if self.scaler_params else None,
            'aoas_mean_std':    (self.scaler_aoas.mean, self.scaler_aoas.std)
                                 if self.scaler_aoas else None,
            'p_ddm_mean':       getattr(self, 'p_ddm_mean', 0.0),
            'p_ddm_std':        getattr(self, 'p_ddm_std',  1.0),
            'w_pressure':       self.w_pressure,
            'w_aoa':            self.w_aoa,
            'lvae_params_dim':  self.lvae_params_dim,
        }, path)
        return path

    def load(self, path: str, train_mode: bool = False):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.denoiser.load_state_dict(ckpt['denoiser'])
        if 'optimizer' in ckpt and train_mode:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        self.stats    = ckpt.get('stats', self.stats)
        self.w_mean   = ckpt.get('w_mean', None)
        self.w_std    = ckpt.get('w_std',  None)
        self.p_ddm_mean = ckpt.get('p_ddm_mean', 0.0)
        self.p_ddm_std  = ckpt.get('p_ddm_std',  1.0)
        if not train_mode:
            self.denoiser.eval()
        print(f"Loaded DDM_W from {path}")
