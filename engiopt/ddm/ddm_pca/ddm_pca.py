"""
DDM_PCA: Denoising Diffusion Model operating in PCA-compressed BAE latent space.

Baseline comparison for DDM_W. Uses the same MLP denoiser and same 64 dimensions,
but the latent space is a linear PCA compression of BAE latents instead of the
nonlinear LVAE w-space. No pressure prediction (PCA is geometry-only).

Architecture
------------
Training:
  coords → BAE encoder → z [S, 3, 30] → flatten → [1350] → PCA → pca_z [64]
  DDM learns p(pca_z_opt | pca_z_init, params)

Inference:
  noise [64] → DDPM reverse → pca_z_gen [64] → PCA inverse → z [1350]
  → reshape [S, 3, 30] → BAE decoder slice-by-slice → coords [S, 2, 192]
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


class MLPDenoiser(nn.Module):
    """Same architecture as DDM_W MLPDenoiser — dedicated AoA head."""

    def __init__(self, z_dim: int = 64, c_dim: int = 4,
                 hidden_dims: tuple = (512, 512, 512, 512),
                 t_embed_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.z_dim       = z_dim
        self.c_dim       = c_dim
        self.t_embed_dim = t_embed_dim

        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, t_embed_dim * 2), nn.SiLU(),
            nn.Linear(t_embed_dim * 2, t_embed_dim),
        )

        in_dim = z_dim + z_dim + c_dim + t_embed_dim
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        self.net   = nn.Sequential(*layers)
        self.out_z = nn.Linear(prev, z_dim)

        aoa_in = prev + 1 + c_dim + t_embed_dim
        self.aoa_head = nn.Sequential(
            nn.Linear(aoa_in, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )

    def _sinusoidal_embedding(self, t):
        half = self.t_embed_dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device, dtype=torch.float32)
            * (np.log(10000) / (half - 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    def forward(self, z_noisy, aoa_noisy, params, z_init, t):
        t_emb = self.t_mlp(self._sinusoidal_embedding(t))
        x = torch.cat([z_noisy, z_init, params, t_emb], dim=1)
        h = self.net(x)
        z_pred   = self.out_z(h)
        aoa_pred = self.aoa_head(torch.cat([h, aoa_noisy, params, t_emb], dim=1))
        return z_pred, aoa_pred


class DDM_PCA:
    """Diffusion model in PCA-compressed BAE latent space."""

    def __init__(
        self,
        denoiser: MLPDenoiser,
        pca,               # fitted sklearn PCA object
        bae_model,
        sampler,
        z_dim: int = 64,
        c_dim: int = 4,
        n_slices: int = 15,
        bae_latent_channels: int = 3,
        bae_latent_length: int = 30,
        w_aoa: float = 1.0,
        params_mean_std=None,
        aoas_mean_std=None,
        name: str = "ddm_pca_v1",
        opt_lr: float = 1e-4,
    ):
        self.denoiser   = denoiser
        self.pca        = pca
        self.bae_model  = bae_model
        self.sampler    = sampler
        self.z_dim      = z_dim
        self.c_dim      = c_dim
        self.n_slices   = n_slices
        self.bae_latent_channels = bae_latent_channels
        self.bae_latent_length   = bae_latent_length
        self.w_aoa      = w_aoa
        self.name       = name

        self.z_mean = None  # [1, z_dim] — set after precomputing
        self.z_std  = None

        from engiopt.data_processing.utils import scaler as make_scaler
        self.scaler_params = make_scaler(params_mean_std) if params_mean_std is not None else None
        self.scaler_aoas   = make_scaler(aoas_mean_std)   if aoas_mean_std   is not None else None

        self.optimizer = Adam(self.denoiser.parameters(), lr=opt_lr)
        self.scheduler = None
        self.stats = {'current_loss': 0.0, 'best_loss': float('inf')}

        self.bae_model.eval()
        for p in self.bae_model.parameters():
            p.requires_grad_(False)

    def _get_index(self, arr, t, shape):
        out = arr.gather(0, t.cpu()).float()
        while out.ndim < len(shape):
            out = out.unsqueeze(-1)
        return out.to(t.device)

    def _forward_diffusion(self, x0, t):
        schedule = self.sampler.schedule_x
        sqrt_ab  = self._get_index(schedule.sqrt_alphas_cumprod,           t, x0.shape)
        sqrt_1ab = self._get_index(schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        noise = torch.randn_like(x0)
        return sqrt_ab * x0 + sqrt_1ab * noise, noise

    def loss(self, batch, return_components=False):
        z_opt, aoa, params, z_init = batch
        device = z_opt.device
        B = z_opt.shape[0]

        t = torch.randint(0, self.sampler.T, (B,), device=device)

        z_noisy,   z_noise   = self._forward_diffusion(z_opt, t)
        aoa_noisy, aoa_noise = self._forward_diffusion(aoa,   t)

        z_noise_pred, aoa_noise_pred = self.denoiser(z_noisy, aoa_noisy, params, z_init, t)

        loss_z   = nn.functional.mse_loss(z_noise_pred, z_noise)
        loss_aoa = nn.functional.mse_loss(aoa_noise_pred, aoa_noise)
        total    = loss_z + self.w_aoa * loss_aoa

        if return_components:
            return total, loss_z, loss_aoa
        return total

    @torch.no_grad()
    def generate(self, z_init, params, device, T=None):
        """
        Returns
        -------
        coords : [B, S, 2, 192]
        aoas   : [B]
        """
        T = T or self.sampler.T
        B = z_init.shape[0]

        z_noisy   = torch.randn(B, self.z_dim, device=device)
        aoa_noisy = torch.randn(B, 1,          device=device)

        schedule = self.sampler.schedule_x
        for i in reversed(range(T)):
            t = torch.full((B,), i, device=device, dtype=torch.long)

            z_noise_pred, aoa_noise_pred = self.denoiser(z_noisy, aoa_noisy, params, z_init, t)

            alpha_t      = schedule.alphas[i]
            alpha_bar_t  = schedule.alphas_cumprod[i]
            alpha_bar_t1 = schedule.alphas_cumprod[i - 1] if i > 0 else torch.tensor(1.0)
            beta_t       = 1.0 - alpha_t

            z0_est = (z_noisy - (1 - alpha_bar_t).sqrt() * z_noise_pred) / alpha_bar_t.sqrt().clamp(min=1e-8)
            z0_est = z0_est.clamp(-5, 5)

            if i > 0:
                posterior_var  = beta_t * (1 - alpha_bar_t1) / (1 - alpha_bar_t).clamp(min=1e-8)
                posterior_mean = (
                    alpha_bar_t1.sqrt() * beta_t / (1 - alpha_bar_t).clamp(min=1e-8) * z0_est
                    + alpha_t.sqrt() * (1 - alpha_bar_t1) / (1 - alpha_bar_t).clamp(min=1e-8) * z_noisy
                )
                z_noisy = posterior_mean + posterior_var.sqrt() * torch.randn_like(z_noisy)

                aoa0_est  = (aoa_noisy - (1 - alpha_bar_t).sqrt() * aoa_noise_pred) / alpha_bar_t.sqrt().clamp(min=1e-8)
                aoa0_est  = aoa0_est.clamp(-5, 5)
                aoa_mean  = (
                    alpha_bar_t1.sqrt() * beta_t / (1 - alpha_bar_t).clamp(min=1e-8) * aoa0_est
                    + alpha_t.sqrt() * (1 - alpha_bar_t1) / (1 - alpha_bar_t).clamp(min=1e-8) * aoa_noisy
                )
                aoa_noisy = aoa_mean + posterior_var.sqrt() * torch.randn_like(aoa_noisy)
            else:
                z_noisy   = z0_est
                aoa_noisy = (aoa_noisy - (1 - alpha_bar_t).sqrt() * aoa_noise_pred) / alpha_bar_t.sqrt().clamp(min=1e-8)
                aoa_noisy = aoa_noisy.clamp(-5, 5)

        # Denormalise PCA latent
        z_gen = z_noisy
        if self.z_mean is not None:
            z_raw = z_gen * self.z_std.to(device) + self.z_mean.to(device)
        else:
            z_raw = z_gen

        # PCA inverse transform → BAE latents [B, S*3*30]
        z_np    = z_raw.cpu().numpy()
        bae_flat = self.pca.inverse_transform(z_np)             # [B, 1350]
        bae_flat = torch.tensor(bae_flat, dtype=torch.float32)

        # Reshape to [B, S, 3, 30] and decode slice-by-slice
        S   = self.n_slices
        ch  = self.bae_latent_channels
        ln  = self.bae_latent_length
        bae_latents = bae_flat.reshape(B, S, ch, ln).to(device)

        self.bae_model.to(device)
        coords_list = []
        for s in range(S):
            z_s = bae_latents[:, s]                             # [B, 3, 30]
            dec, _, _ = self.bae_model.decode_z(
                z_s, z_ae_mode=True, denormalize_output=False, normalized_data=False
            )
            coords_list.append(dec)
        coords = torch.stack(coords_list, dim=1)                # [B, S, 2, 192]

        # Denormalise AoA
        if self.scaler_aoas is not None:
            aoa_np = self.scaler_aoas.inverse_transform(aoa_noisy.cpu().numpy())
            aoas   = torch.tensor(aoa_np, dtype=torch.float32).squeeze(1)
        else:
            aoas = aoa_noisy.squeeze(1)

        return coords.cpu(), aoas.cpu()

    def save(self, save_dir, suffix=""):
        os.makedirs(save_dir, exist_ok=True)
        import pickle
        path = os.path.join(save_dir, f"{self.name}{suffix}.pth")
        torch.save({
            'denoiser':        self.denoiser.state_dict(),
            'optimizer':       self.optimizer.state_dict(),
            'stats':           self.stats,
            'z_mean':          self.z_mean,
            'z_std':           self.z_std,
            'z_dim':           self.z_dim,
            'c_dim':           self.c_dim,
            'n_slices':        self.n_slices,
            'w_aoa':           self.w_aoa,
            'params_mean_std': (self.scaler_params.mean, self.scaler_params.std) if self.scaler_params else None,
            'aoas_mean_std':   (self.scaler_aoas.mean,   self.scaler_aoas.std)   if self.scaler_aoas   else None,
        }, path)
        pca_path = os.path.join(save_dir, f"{self.name}_pca.pkl")
        with open(pca_path, 'wb') as f:
            pickle.dump(self.pca, f)
        print(f"Saved to {path} and {pca_path}")

    def load(self, path, train_mode=True):
        import pickle
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.denoiser.load_state_dict(ckpt['denoiser'])
        self.z_mean = ckpt.get('z_mean')
        self.z_std  = ckpt.get('z_std')
        pca_path = path.replace('_best.pth', '_pca.pkl').replace('.pth', '_pca.pkl')
        with open(pca_path, 'rb') as f:
            self.pca = pickle.load(f)
        if not train_mode:
            self.denoiser.eval()
        print(f"Loaded DDM_PCA from {path}")
