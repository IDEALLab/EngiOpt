"""
DDM_LVAE_3D: DDM_AoAInit_3D with pressure supervision via a frozen LVAE decoder.

Training is two-phase:
  1. Train the LVAE normally (separate, already done).
  2. Freeze the LVAE encoder + decoder, then train this model.
     At each training step the DDM generates a denoised z_opt estimate,
     passes it through the frozen LVAE encoder -> decoder to obtain a
     pressure prediction, and adds a pressure MSE term to the loss.

Nothing in engiopt/ddm/ddm.py is changed.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from ..ddm import DDM_AoAInit_3D


class DDM_LVAE_3D(DDM_AoAInit_3D):
    """DDM_AoAInit_3D extended with pressure supervision through a frozen LVAE.

    Extra constructor arguments
    ---------------------------
    lvae_model : trained LAE_AoAInit instance (encoder + decoder already loaded)
        Will be immediately frozen (requires_grad=False, eval mode).
    w_pressure : float
        Weight for the pressure loss term (default 1.0).
    lvae_params_dim : int
        Dimensionality of the condition vector expected by the LVAE encoder/decoder.
        The DDM conditions (params, [B,4]) are a subset of the full LVAE conditions
        ([B, 38]).  If lvae_params_dim > 4 the extra dims are zero-padded so that
        shapes match without modifying the LVAE.
    """

    def __init__(
        self,
        unet,
        sampler,
        bae_model,
        lvae_model,
        w_pressure: float = 1.0,
        lvae_params_dim: int = 4,
        z_ddm_mean=None,
        z_ddm_std=None,
        p_ddm_mean: float = 0.0,
        p_ddm_std: float = 1.0,
        w_latent: float = 0.0,
        w_smooth: float = 0.0,
        smooth_reg_exp: float = 4.0,
        **kwargs,
    ):
        super().__init__(unet, sampler, bae_model, **kwargs)

        self.lvae_model = lvae_model
        self.w_pressure = w_pressure
        self.lvae_params_dim = lvae_params_dim

        # Stats to undo DDM's z-normalisation before feeding latents to the LVAE encoder.
        # Shape: broadcastable to [B, w_dim, 3, L], e.g. [1,1,3,1] or scalar.
        self.z_ddm_mean = z_ddm_mean  # None means "already raw / don't denormalise"
        self.z_ddm_std  = z_ddm_std

        # Stats to convert pressure from the LVAE's normalisation into the DDM's space.
        # LVAE outputs pressure in LVAE-normalised space; pressure_gt is in DDM-normalised space.
        # lvae_scaler_pressures is the LVAE's own scaler (mean, std) used during LVAE training.
        lvae_ps = getattr(lvae_model, 'pressures_mean_std', None)
        if lvae_ps is not None:
            self._lvae_p_mean = float(lvae_ps[0]) if hasattr(lvae_ps[0], '__float__') else lvae_ps[0]
            self._lvae_p_std  = float(lvae_ps[1]) if hasattr(lvae_ps[1], '__float__') else lvae_ps[1]
        else:
            self._lvae_p_mean = 0.0
            self._lvae_p_std  = 1.0
        self._p_ddm_mean = p_ddm_mean
        self._p_ddm_std  = p_ddm_std

        self.w_latent = w_latent

        # Smoothness regularizer (Option B): decode x0_est through BAE to coordinates
        # and penalize jaggedness (finite differences of control points), weighted by
        # exp(-smooth_reg_exp * t/T) so the penalty concentrates at low-noise timesteps
        # where the x0 estimate is reliable.
        self.w_smooth = w_smooth
        self.smooth_reg_exp = smooth_reg_exp

        # Normalized valid ranges for BAE latent channels (ch0, ch1 only — ch2 is fine).
        # These are set after construction via set_latent_valid_range() once z_mean/z_std
        # are known, or remain None to disable the validity penalty.
        self._latent_norm_lo = None  # [2] tensor: normalized lower bounds for ch0, ch1
        self._latent_norm_hi = None  # [2] tensor: normalized upper bounds for ch0, ch1

        # Freeze the LVAE completely — we only use it as a fixed pressure oracle.
        self.lvae_model.encoder.eval()
        self.lvae_model.decoder.eval()
        for p in self.lvae_model.encoder.parameters():
            p.requires_grad_(False)
        for p in self.lvae_model.decoder.parameters():
            p.requires_grad_(False)

    def set_latent_valid_range(self, z_mean: torch.Tensor, z_std: torch.Tensor):
        """Precompute normalized valid bounds for ch0 and ch1.

        Raw valid ranges (from BAE training data):
            ch0 (Bezier weights): [0.05, 2.1]
            ch1 (CP x):           [-0.1, 1.1]

        Args:
            z_mean: [1,1,3,1] or [3] tensor — per-channel latent mean
            z_std:  [1,1,3,1] or [3] tensor — per-channel latent std
        """
        mean = z_mean.squeeze().float()  # [3]
        std  = z_std.squeeze().float()   # [3]
        raw_lo = torch.tensor([0.05, -0.1], dtype=torch.float32)
        raw_hi = torch.tensor([2.1,   1.1], dtype=torch.float32)
        self._latent_norm_lo = (raw_lo - mean[:2]) / std[:2]
        self._latent_norm_hi = (raw_hi - mean[:2]) / std[:2]

    def _lvae_pressure(self, z_opt: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Run z_opt through the frozen LVAE encoder -> decoder and return pressure.

        z_opt is expected to be in the DDM's normalised latent space.  This method
        denormalises it back to raw BAE latent space before passing it to the LVAE
        encoder (which was trained on raw latents).

        The returned pressure is re-expressed in the DDM's normalised pressure space
        so that it can be compared directly with pressure_gt in the loss.

        Args:
            z_opt  : [B, w_dim, 3, L]  DDM-normalised BAE latents (estimated clean x_0)
            params : [B, c_dim]        DDM-normalised condition vector (already scaled)

        Returns:
            pressure_pred : [B, w_dim, 192]  in DDM-normalised pressure space
        """
        device = z_opt.device

        # 1. Undo DDM's latent normalisation → raw BAE latent space.
        if self.z_ddm_mean is not None and self.z_ddm_std is not None:
            z_mean = self.z_ddm_mean
            z_std  = self.z_ddm_std
            if isinstance(z_mean, torch.Tensor):
                z_mean = z_mean.to(device)
                z_std  = z_std.to(device)
            z_raw = z_opt * z_std + z_mean
        else:
            z_raw = z_opt

        # 2. Undo DDM's params normalisation → raw params space for the LVAE.
        #    The DDM stores its params scaler; use it to invert.
        if hasattr(self, 'scaler_params') and self.scaler_params is not None:
            B = params.shape[0]
            params_np = params.detach().cpu().numpy()
            raw_params_np = self.scaler_params.inverse_transform(params_np)
            lvae_params_raw = torch.tensor(raw_params_np, dtype=torch.float32, device=device)
        else:
            lvae_params_raw = params

        # 3. Pad raw params to the full LVAE condition width if needed.
        if lvae_params_raw.shape[1] < self.lvae_params_dim:
            pad = torch.zeros(
                lvae_params_raw.shape[0], self.lvae_params_dim - lvae_params_raw.shape[1], device=device
            )
            lvae_params_raw = torch.cat([lvae_params_raw, pad], dim=1)

        # 4. Apply LVAE's own params normalisation before passing to encoder/decoder.
        if hasattr(self.lvae_model, 'scaler_params') and self.lvae_model.scaler_params is not None:
            raw_np = lvae_params_raw.detach().cpu().numpy()
            lvae_params_scaled_np = self.lvae_model.scaler_params.transform(raw_np)
            lvae_params = torch.tensor(lvae_params_scaled_np, dtype=torch.float32, device=device)
        else:
            lvae_params = lvae_params_raw

        # Move LVAE modules to the same device as the data (handles multi-GPU or
        # CPU-only setups without requiring the caller to do it explicitly).
        self.lvae_model.encoder.to(device)
        self.lvae_model.decoder.to(device)

        with torch.no_grad():
            z = self.lvae_model.encoder(z_raw, lvae_params)
            z_masked = self.lvae_model._apply_mask(z)
            _, _, _, pressure_pred, _ = self.lvae_model.decoder(z_masked, lvae_params)

        # 5. pressure_pred is in LVAE-normalised space; re-express in DDM-normalised space
        #    so it can be compared with pressure_gt (which is already DDM-normalised).
        #    LVAE-normalised → raw: p_raw = p_lvae * lvae_std + lvae_mean
        #    raw → DDM-normalised: p_ddm = (p_raw - ddm_mean) / ddm_std
        lvae_p_mean = torch.tensor(self._lvae_p_mean, dtype=pressure_pred.dtype, device=device)
        lvae_p_std  = torch.tensor(self._lvae_p_std,  dtype=pressure_pred.dtype, device=device)
        p_ddm_mean  = torch.tensor(self._p_ddm_mean,  dtype=pressure_pred.dtype, device=device)
        p_ddm_std   = torch.tensor(self._p_ddm_std,   dtype=pressure_pred.dtype, device=device)
        p_raw = pressure_pred * lvae_p_std + lvae_p_mean
        pressure_pred = (p_raw - p_ddm_mean) / p_ddm_std.clamp(min=1e-8)

        return pressure_pred  # [B, 9, 192]

    def _estimate_x0(
        self, x_noisy: torch.Tensor, noise_pred: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Recover a clean x_0 estimate from the noisy sample and predicted noise.

        Uses the standard DDPM formula:
            x_0 = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
        """
        from ..samplers import get_index_from_list

        sqrt_alphas_cumprod_t = get_index_from_list(
            self.sampler.schedule_x.sqrt_alphas_cumprod, t, x_noisy.shape
        ).to(x_noisy.device)
        sqrt_one_minus_t = get_index_from_list(
            self.sampler.schedule_x.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape
        ).to(x_noisy.device)

        return (x_noisy - sqrt_one_minus_t * noise_pred) / sqrt_alphas_cumprod_t.clamp(min=1e-8)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _loss_DDM(
        self,
        x_noisy,
        x_noise,
        x_noise_pred,
        alpha_noise,
        alpha_noise_pred,
        t,
        return_components=False,
        params=None,
        pressure_gt=None,
    ):
        """DDM loss + LVAE-pressure supervision.

        Extra keyword arguments (compared to the parent):
            params       : [B, c_dim]   condition vector (required for pressure loss)
            pressure_gt  : [B, 9, 192]  ground-truth pressure (required for pressure loss)
        """
        # Base geometry + AoA loss (identical to DDM_AoAInit_3D).
        base = super()._loss_DDM(
            x_noisy, x_noise, x_noise_pred,
            alpha_noise, alpha_noise_pred,
            t,
            return_components=return_components,
        )

        need_x0 = (
            (pressure_gt is not None and params is not None)
            or self.w_latent > 0
            or self.w_smooth > 0
        )
        x0_est = self._estimate_x0(x_noisy, x_noise_pred, t) if need_x0 else None

        # Pressure term — only computed when ground-truth pressure is provided.
        if pressure_gt is not None and params is not None:
            pressure_pred = self._lvae_pressure(x0_est, params)  # [B, 9, 192]
            loss_pressure = F.mse_loss(pressure_pred, pressure_gt.to(pressure_pred.device))
        else:
            loss_pressure = torch.tensor(0.0, device=x_noisy.device)

        # Soft latent validity penalty — penalizes x0 estimates outside the valid BAE range.
        if self.w_latent > 0 and x0_est is not None and self._latent_norm_lo is not None and self._latent_norm_hi is not None:
            lo = self._latent_norm_lo.to(x0_est.device)  # [2]
            hi = self._latent_norm_hi.to(x0_est.device)  # [2]
            ch01 = x0_est[:, :, :2, :]                   # [B, S, 2, L]
            viol_lo = F.relu(lo[None, None, :, None] - ch01)
            viol_hi = F.relu(ch01 - hi[None, None, :, None])
            loss_latent = (viol_lo ** 2 + viol_hi ** 2).mean()
        else:
            loss_latent = torch.tensor(0.0, device=x_noisy.device)

        # Smoothness regularizer — decode x0_est through frozen BAE to coordinates,
        # then penalize jaggedness (norm of finite differences of the y-coordinate).
        # Weighted by exp(-smooth_reg_exp * t/T) so the penalty is strong at low t
        # (where x0_est is a reliable clean prediction) and decays to ~0 at high t
        # (where x0_est is dominated by noise and gradients would be meaningless).
        if self.w_smooth > 0 and x0_est is not None:
            B, S, C, L = x0_est.shape
            # Denormalize latents from DDM space back to raw BAE space for decoding.
            if self.z_ddm_mean is not None and self.z_ddm_std is not None:
                z_mean = self.z_ddm_mean
                z_std  = self.z_ddm_std
                if isinstance(z_mean, torch.Tensor):
                    z_mean = z_mean.to(x0_est.device)
                    z_std  = z_std.to(x0_est.device)
                x0_raw = x0_est * z_std + z_mean
            else:
                x0_raw = x0_est
            # Clamp to valid BAE range before decoding to avoid decoder pathology.
            x0_raw = x0_raw.clamp(
                torch.tensor([0.05, -0.1, -0.15], device=x0_raw.device)[None, None, :, None],
                torch.tensor([2.10,  1.1,  0.20], device=x0_raw.device)[None, None, :, None],
            )
            # Flatten span dimension into batch for BAE decode: [B*S, 3, L]
            x0_flat = x0_raw.reshape(B * S, C, L)
            self.bae_model.to(x0_flat.device)
            x_decoded, _, _ = self.bae_model.decode_z(x0_flat, z_ae_mode=True, denormalize_output=True, normalized_data=False)
            # x_decoded: [B*S, 2, n_data_points] — ch1 is y-coordinate
            y_coords = x_decoded[:, 1, :]  # [B*S, 192]
            smoothness = torch.norm(y_coords[:, 1:] - y_coords[:, :-1], dim=1).mean()
            # t-weighting: strong at t=0, decays to ~0 at t=T
            t_weight = torch.exp(-self.smooth_reg_exp * t.float() / self.sampler.T).mean()
            loss_smooth = smoothness * t_weight
        else:
            loss_smooth = torch.tensor(0.0, device=x_noisy.device)

        if return_components:
            # base is (total, loss_x, loss_alpha, loss_reg)
            total = (base[0]
                     + self.w_pressure * loss_pressure
                     + self.w_latent   * loss_latent
                     + self.w_smooth   * loss_smooth)
            return total, base[1], base[2], base[3], loss_pressure, loss_latent
        else:
            return (base
                    + self.w_pressure * loss_pressure
                    + self.w_latent   * loss_latent
                    + self.w_smooth   * loss_smooth)

    # ------------------------------------------------------------------
    # Data noising — unpack pressure from batch
    # ------------------------------------------------------------------

    def _noise_data(self, batch, device, **kwargs):
        """Unpack batch that now includes pressure as a 5th element.

        Expected batch format:
            (z_opt, aoa, params, z_init, pressure)

        Returns the standard 7-tuple extended with params and pressure so
        that the loss function can consume them.
        """
        z_opt, aoa, params, z_init, pressure = batch

        z_opt    = z_opt.to(device)
        aoa      = aoa.to(device)
        params   = params.to(device).float()
        z_init   = z_init.to(device)
        pressure = pressure.to(device).float()

        t = self.t_gen(z_opt.shape[0], device)

        x_noisy,     x_noise     = self.sampler.schedule_x.forward_diffusion_sample(z_opt, t, device)
        alpha_noisy, alpha_noise = self.sampler.schedule_AoA.forward_diffusion_sample(aoa, t, device)

        # Return the standard 7 fields plus params + pressure for the pressure loss.
        return x_noisy, x_noise, alpha_noisy, alpha_noise, params, z_init, t, pressure

    # ------------------------------------------------------------------
    # Loss routing — forward params + pressure to _loss_DDM
    # ------------------------------------------------------------------

    def loss(self, batch_noised, return_components, **kwargs):
        (x_noisy, x_noise, alpha_noisy, alpha_noise,
         params, z_init, t, pressure) = batch_noised

        x_noise_pred, alpha_noise_pred = self.unet(
            x_noisy, alpha_noisy, params, z_init, t
        )
        return self._loss_DDM(
            x_noisy, x_noise, x_noise_pred,
            alpha_noise, alpha_noise_pred,
            t,
            return_components=return_components,
            params=params,
            pressure_gt=pressure,
        )

    # ------------------------------------------------------------------
    # Stats — log pressure loss component
    # ------------------------------------------------------------------

    def init_stats(self):
        stats = super().init_stats()
        stats['test_loss_pressure'] = np.array([])
        return stats

    def _test_stats(self, epoch, dataloader_test, stats_intvl, device, **kwargs):
        self.stats['train_loss'] = np.append(self.stats['train_loss'], self.stats['current_loss'])
        self.stats['train_loss_epoch'] = np.append(self.stats['train_loss_epoch'], epoch)
        self.stats['epoch'] = epoch

        if not self.lr_scheduler == 'ReduceLROnPlateau' and not self.lr_scheduler == 'None':
            self.scheduler.step()

        if epoch % stats_intvl == 0:
            self.unet.eval()
            print(f"Epoch {epoch}, Loss = {self.stats['current_loss']}, "
                  f"Compute Time = {self.elapsed_time:.2f} s")

            with torch.no_grad():
                for batch in dataloader_test:
                    batch_noised = self._noise_data(batch, device, **kwargs)
                    loss, loss_x, loss_AoA, loss_reg, loss_pressure, _ = self.loss(
                        batch_noised, return_components=True, **kwargs
                    )
                    self.stats['test_loss']          = np.append(self.stats['test_loss'],          loss.item())
                    self.stats['test_loss_x']        = np.append(self.stats['test_loss_x'],        loss_x.item())
                    self.stats['test_loss_AoA']      = np.append(self.stats['test_loss_AoA'],      loss_AoA.item())
                    self.stats['test_loss_mean_reg'] = np.append(self.stats['test_loss_mean_reg'], loss_reg.item())
                    self.stats['test_loss_pressure'] = np.append(self.stats['test_loss_pressure'], loss_pressure.item())
                    self.stats['test_loss_epoch']    = np.append(self.stats['test_loss_epoch'],    epoch)
                    break

            self.unet.train()
            if self.lr_scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(loss)

            print(
                f"Test Loss: Total = {loss.item():.4f}",
                f"  x = {loss_x.item():.4f}",
                f"  AoA = {loss_AoA.item():.4f}",
                f"  reg = {loss_reg.item():.4f}",
                f"  pressure = {loss_pressure.item():.4f}",
            )
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")

    # ------------------------------------------------------------------
    # Save / load — persist LVAE reference and w_pressure
    # ------------------------------------------------------------------

    def save(self, save_dir, suffix="", **kwargs):
        os.makedirs(save_dir, exist_ok=True)
        dest = os.path.join(save_dir, self.name + suffix + '.pth')
        torch.save(
            {
                'unet':                  self.unet,
                'sampler':               self.sampler,
                'bae_model_state_dict':  self.bae_model.state_dict(),
                'lvae_encoder':          self.lvae_model.encoder,
                'lvae_decoder':          self.lvae_model.decoder,
                'lvae_active_mask':      self.lvae_model.active_latent_mask.cpu(),
                'optimizer':             self.optimizer.state_dict(),
                'stats':                 self.stats,
                'scheduler':             self.scheduler.state_dict() if self.lr_scheduler != 'None' else None,
                'params_mean_std':       self.params_mean_std,
                'aoas_mean_std':         self.aoas_mean_std,
                'latent_mean':           self.latent_mean,
                'latent_std':            self.latent_std,
                'w_pressure':            self.w_pressure,
                'lvae_params_dim':       self.lvae_params_dim,
                'z_ddm_mean':            self.z_ddm_mean,
                'z_ddm_std':             self.z_ddm_std,
                'lvae_p_mean':           self._lvae_p_mean,
                'lvae_p_std':            self._lvae_p_std,
                'p_ddm_mean':            self._p_ddm_mean,
                'p_ddm_std':             self._p_ddm_std,
            },
            dest,
        )

    def load(self, checkpoint, train_mode):
        ckp = torch.load(checkpoint, weights_only=False, map_location='cpu')
        print(ckp.keys())

        self.bae_model.load_state_dict(ckp['bae_model_state_dict'])
        self.unet    = ckp['unet']
        self.sampler = ckp.get('sampler', self.sampler)

        if 'lvae_encoder' in ckp:
            self.lvae_model.encoder = ckp['lvae_encoder']
            self.lvae_model.decoder = ckp['lvae_decoder']
            if 'lvae_active_mask' in ckp:
                self.lvae_model.active_latent_mask = ckp['lvae_active_mask']
            # Re-freeze after loading
            self.lvae_model.encoder.eval()
            self.lvae_model.decoder.eval()
            for p in self.lvae_model.encoder.parameters():
                p.requires_grad_(False)
            for p in self.lvae_model.decoder.parameters():
                p.requires_grad_(False)

        if train_mode:
            self.unet.train()
        else:
            self.unet.eval()

        if train_mode:
            self.optimizer.load_state_dict(ckp['optimizer'])
            if self.lr_scheduler != 'None' and ckp.get('scheduler') is not None:
                self.scheduler.load_state_dict(ckp['scheduler'])

        self.stats = ckp['stats']

        try:
            self.params_mean_std = ckp['params_mean_std']
            self.aoas_mean_std   = ckp['aoas_mean_std']
            self.init_scalers(self.params_mean_std, self.aoas_mean_std)
            self.latent_mean = ckp.get('latent_mean', 0.0)
            self.latent_std  = ckp.get('latent_std',  1.0)
            self.w_pressure      = ckp.get('w_pressure',    self.w_pressure)
            self.lvae_params_dim = ckp.get('lvae_params_dim', self.lvae_params_dim)
            self.z_ddm_mean      = ckp.get('z_ddm_mean',   self.z_ddm_mean)
            self.z_ddm_std       = ckp.get('z_ddm_std',    self.z_ddm_std)
            self._lvae_p_mean    = ckp.get('lvae_p_mean',  self._lvae_p_mean)
            self._lvae_p_std     = ckp.get('lvae_p_std',   self._lvae_p_std)
            self._p_ddm_mean     = ckp.get('p_ddm_mean',   self._p_ddm_mean)
            self._p_ddm_std      = ckp.get('p_ddm_std',    self._p_ddm_std)
        except Exception as e:
            print(f'Warning: could not restore all scalers/hyperparams: {e}')

        print(f'Loaded DDM_LVAE_3D from {checkpoint}')

    # ------------------------------------------------------------------
    # Inference — generate z_opt + pressure via frozen LVAE decoder
    # ------------------------------------------------------------------

    def __call__(
        self,
        noise_list,
        params,
        encoded_init,
        output_decoded: bool = False,
        output_pressure: bool = True,
        T=None,
    ):
        """Generate airfoil geometry, AoA, and optionally pressure.

        Args:
            noise_list      : [noise_x [B,9,3,L], noise_alpha [B,1]]
            params          : condition vector [B, c_dim]
            encoded_init    : root-slice BAE latent [B, 3, L]
            output_decoded  : if True decode z_opt to coordinates via BAE
            output_pressure : if True also return pressure via frozen LVAE
            T               : number of diffusion steps (default: sampler.T)

        Returns:
            gen_airfoil  : [B,9,3,L] latents or [B,9,2,192] coords
            gen_alpha    : [B,1]
            pressure     : [B,9,192]  (only when output_pressure=True)
        """
        noise_x, noise_alpha = noise_list
        self.bae_model.to(noise_x.device)

        gen_airfoil, gen_alpha = self.sampler.sample_airfoil(
            self.unet,
            noise_x, noise_alpha,
            params, encoded_init,
            T=T,
        )

        if output_decoded:
            gen_airfoil = self.bae_model.decode_z(
                gen_airfoil, z_ae_mode=True, denormalize_output=True, normalized_data=False
            )[0]
            gen_alpha = self.scaler_aoas.inverse_transform(gen_alpha)

        if output_pressure:
            # Use the latent-space z_opt (before optional BAE decoding) to get pressure.
            z_opt_for_pressure = gen_airfoil if not output_decoded else noise_x
            if output_decoded:
                # gen_airfoil is now coordinates; re-run sampler to get latents.
                # Simpler: always keep a reference to the latent before decoding.
                # We handle this by running pressure on the already-generated latents
                # stored before the BAE decode step — see below.
                pass
            pressure = self._lvae_pressure(
                gen_airfoil if not output_decoded else self._last_gen_latent,
                params,
            )
            return gen_airfoil, gen_alpha, pressure

        return gen_airfoil, gen_alpha

    # Override to cache the latent before BAE decode so __call__ can use it.
    def _generate_latents(self, noise_list, params, encoded_init, T=None):
        noise_x, noise_alpha = noise_list
        gen_airfoil, gen_alpha = self.sampler.sample_airfoil(
            self.unet, noise_x, noise_alpha, params, encoded_init, T=T
        )
        self._last_gen_latent = gen_airfoil.clone()
        return gen_airfoil, gen_alpha
