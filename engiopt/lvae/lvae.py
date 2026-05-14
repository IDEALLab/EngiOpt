"""
LAE_AoAInit: deterministic latent autoencoder for 3-D wing geometry, AoA, pressure, and performance.

Architecture
------------
  Encoder: BAE latents + flow conditions -> deterministic z  (LAEEncoder)
  Decoder: z + flow conditions -> BAE latents, AoA, TE-shift, pressure, performance  (LAEDecoder)

Least-volume (LV) penalty
--------------------------
The LV term minimises the volume of the unit hyper-ball after the encoder mapping,
which compresses the effective latent dimensionality.  Dynamic dimension pruning
freezes low-variance dimensions and zeros their contribution to the LV loss.
"""

import os

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from engiopt.data_processing.utils import scaler


class LAE_AoAInit:
    def __init__(
        self,
        encoder,
        decoder,
        sampler,
        bae_model,
        lae_latent_dim: int = 64,
        params_mean_std=None,
        aoas_mean_std=None,
        pressures_mean_std=None,
        perfs_mean_std=None,
        name: str = "lae_aoa_init_3d",
        lambda_lv: float = 7e-5,
        weights=(1000, 1, 9, 1, 1),
        opt_lr: float = 1e-3,
        opt_weight_decay: float = 0.0,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler
        self.bae_model = bae_model
        self.lae_latent_dim = lae_latent_dim
        self.name = name
        self.lambda_lv = lambda_lv
        self.weights = weights
        self.lr_scheduler = 'None'

        self.scaler_params    = scaler(params_mean_std)    if params_mean_std    is not None else None
        self.scaler_aoas      = scaler(aoas_mean_std)      if aoas_mean_std      is not None else None
        self.scaler_pressures = scaler(pressures_mean_std) if pressures_mean_std is not None else None
        self.scaler_perfs     = scaler(perfs_mean_std)     if perfs_mean_std     is not None else None

        self.params_mean_std    = params_mean_std
        self.aoas_mean_std      = aoas_mean_std
        self.pressures_mean_std = pressures_mean_std
        self.perfs_mean_std     = perfs_mean_std

        params = list(encoder.parameters()) + list(decoder.parameters())
        self.optimizer = Adam(params, lr=opt_lr, weight_decay=opt_weight_decay)

        self.active_latent_mask = torch.ones(lae_latent_dim, dtype=torch.bool)

        self._ema_mean = None
        self._ema_std  = None
        self._ema_beta = 0.9

        self.stats = {
            'current_loss':       0.0,
            'train_loss':         np.array([]),
            'train_loss_epoch':   np.array([]),
            'test_loss_recon':    np.array([]),
            'test_loss_alpha':    np.array([]),
            'test_loss_eta_y':    np.array([]),
            'test_loss_pressure': np.array([]),
            'test_loss_perf':     np.array([]),
            'test_loss_lv':       np.array([]),
        }

        self.perf_regressor      = None
        self.perf_regressor_mask = None

    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)
        self.active_latent_mask = self.active_latent_mask.to(device)
        if self._ema_mean is not None:
            self._ema_mean = self._ema_mean.to(device)
        if self._ema_std is not None:
            self._ema_std = self._ema_std.to(device)
        return self

    def encode(self, z_opt, c):
        return self.encoder(z_opt, c)

    def decode(self, z, c):
        return self.decoder(z, c)

    def _apply_mask(self, z):
        mask = self.active_latent_mask.to(z.device)
        return z * mask.float()

    def _forward_pass(self, batch, device):
        z_opt, aoa, params, eta_y, pressure, perf = batch

        z_opt    = z_opt.to(device)
        params   = params.to(device).float()
        aoa      = aoa.to(device).float()
        eta_y    = eta_y.to(device).float()
        pressure = pressure.to(device).float()
        perf     = perf.to(device).float()

        z        = self.encoder(z_opt, params)
        z_masked = self._apply_mask(z)

        z_opt_pred, alpha_pred, eta_y_pred, pressure_pred, perf_pred = self.decoder(z_masked, params)

        return (z_opt, z_opt_pred,
                aoa, alpha_pred,
                eta_y, eta_y_pred,
                pressure, pressure_pred,
                perf, perf_pred,
                z)

    def _loss_LAE(self, z_opt, z_opt_pred, aoa, alpha_pred, eta_y, eta_y_pred,
                  pressure, pressure_pred, perf, perf_pred, z):
        w_x, w_alpha, w_eta, w_pressure, w_perf = self.weights

        recon_loss = nn.functional.mse_loss(z_opt_pred, z_opt)
        alpha_loss = nn.functional.mse_loss(alpha_pred, aoa)
        eta_y_loss = nn.functional.mse_loss(eta_y_pred, eta_y)

        if getattr(self.decoder, 'use_pressure', True):
            pressure_loss = nn.functional.mse_loss(pressure_pred, pressure)
            perf_loss     = nn.functional.mse_loss(perf_pred, perf)
        else:
            pressure_loss = torch.zeros(1, device=z_opt.device)
            perf_loss     = torch.zeros(1, device=z_opt.device)

        loss = (w_x          * recon_loss
                + w_alpha    * alpha_loss
                + w_eta      * eta_y_loss
                + w_pressure * pressure_loss
                + w_perf     * perf_loss)

        if self.lambda_lv > 0:
            active = self.active_latent_mask.to(z.device)
            z_active = z[:, active]
            if z_active.shape[1] > 0:
                std_active = z_active.std(dim=0).clamp(min=1e-8)
                lv_loss = std_active.log().sum()
                loss = loss + self.lambda_lv * lv_loss

        return loss

    def _update_LAE(self, epoch, batch, device):
        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()

        outs = self._forward_pass(batch, device)
        (z_opt, z_opt_pred, aoa, alpha_pred, eta_y, eta_y_pred,
         pressure, pressure_pred, perf, perf_pred, z) = outs

        loss = self._loss_LAE(z_opt, z_opt_pred, aoa, alpha_pred, eta_y, eta_y_pred,
                              pressure, pressure_pred, perf, perf_pred, z)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            batch_std  = z.std(dim=0).detach()
            batch_mean = z.mean(dim=0).detach()
            if self._ema_std is None:
                self._ema_std  = batch_std.clone()
                self._ema_mean = batch_mean.clone()
            else:
                self._ema_std  = self._ema_beta * self._ema_std  + (1 - self._ema_beta) * batch_std
                self._ema_mean = self._ema_beta * self._ema_mean + (1 - self._ema_beta) * batch_mean

        self.stats['current_loss'] = loss.item()

    def update_active_mask(self, loader, device, threshold=0.02):
        self.encoder.eval()
        all_z = []
        with torch.no_grad():
            for batch in loader:
                z_opt, aoa, params, eta_y, pressure, perf = batch
                z_opt  = z_opt.to(device)
                params = params.to(device).float()
                z = self.encoder(z_opt, params)
                all_z.append(z.cpu())
        self.encoder.train()

        all_z = torch.cat(all_z, dim=0)
        std = all_z.std(dim=0)
        self.active_latent_mask = (std >= threshold).to(device)

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        dest = os.path.join(save_dir, self.name + '.pth')
        tmp  = dest + '.tmp'
        payload = {
            'encoder':              self.encoder,
            'decoder':              self.decoder,
            'sampler':              self.sampler,
            'bae_model_state_dict': self.bae_model.state_dict(),
            'optimizer':            self.optimizer.state_dict(),
            'scheduler':            None,
            'stats':                self.stats,
            'params_mean_std':      self.params_mean_std,
            'aoas_mean_std':        self.aoas_mean_std,
            'pressures_mean_std':   self.pressures_mean_std,
            'perfs_mean_std':       self.perfs_mean_std,
            'active_latent_mask':   self.active_latent_mask.cpu(),
            'lambda_lv':            self.lambda_lv,
            'lae_latent_dim':       self.lae_latent_dim,
        }
        for attempt in range(1, 4):
            try:
                torch.save(payload, tmp)
                os.replace(tmp, dest)
                return
            except RuntimeError as e:
                print(f"  [WARN] Checkpoint save failed (attempt {attempt}/3): {e}")
                if os.path.exists(tmp):
                    os.remove(tmp)
        print(f"  [ERROR] All 3 checkpoint save attempts failed for {dest}. Skipping.")

    def load(self, checkpoint_path, train_mode=True):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        self.encoder = ckpt['encoder']
        self.decoder = ckpt['decoder']
        self.sampler = ckpt.get('sampler', self.sampler)
        if 'bae_model_state_dict' in ckpt:
            self.bae_model.load_state_dict(ckpt['bae_model_state_dict'])
        # Rebuild optimizer against the loaded encoder/decoder so parameter
        # group sizes always match the checkpoint, regardless of current Config.
        lr = self.optimizer.param_groups[0]['lr']
        self.optimizer = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr,
        )
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.stats = ckpt.get('stats', self.stats)

        self.params_mean_std    = ckpt.get('params_mean_std',    self.params_mean_std)
        self.aoas_mean_std      = ckpt.get('aoas_mean_std',      self.aoas_mean_std)
        self.pressures_mean_std = ckpt.get('pressures_mean_std', self.pressures_mean_std)
        self.perfs_mean_std     = ckpt.get('perfs_mean_std',     self.perfs_mean_std)

        if self.params_mean_std is not None:
            self.scaler_params = scaler(self.params_mean_std)
        if self.aoas_mean_std is not None:
            self.scaler_aoas = scaler(self.aoas_mean_std)
        if self.pressures_mean_std is not None:
            self.scaler_pressures = scaler(self.pressures_mean_std)
        if self.perfs_mean_std is not None:
            self.scaler_perfs = scaler(self.perfs_mean_std)

        self.active_latent_mask = ckpt.get(
            'active_latent_mask',
            torch.ones(self.lae_latent_dim, dtype=torch.bool),
        )
        self.lambda_lv      = ckpt.get('lambda_lv',      self.lambda_lv)
        self.lae_latent_dim = ckpt.get('lae_latent_dim', self.lae_latent_dim)

        if train_mode:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
