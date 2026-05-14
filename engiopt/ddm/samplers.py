"""" 
    File for the diffusion models utilities, such as the linear beta schedule and the forward diffusion sample.
    Contains:
        - beta_schedule
        

"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def beta_schedule(T, start=1e-4, end=0.02, scale= 1.0, cosine=False, exp_biasing=False, exp_bias_factor=1):
    """
    Returns a beta schedule (default: linear) for the diffusion model.
    Args:
        T: Number of timesteps
        start: Starting value of beta
        end: Ending value of beta
        scale: Scaling factor for beta
        cosine: Whether to use a cosine beta schedule
        exp_biasing: Whether to use exponential biasing
        exp_bias_factor: Exponential biasing factor
    """

    beta = torch.linspace(scale*start, scale*end, T)
    if cosine:
        beta = []
        a_func = lambda t_val: math.cos((t_val + 0.008) / 1.008 * np.pi / 2) ** 2
        for i in range(T):
            t1 = i / T
            t2 = (i + 1) / T
            beta.append(min(1 - a_func(t2) / a_func(t1), 0.999))
        
        beta = torch.tensor(beta)
    
    if exp_biasing:
        beta = (torch.flip(torch.exp(-exp_bias_factor*torch.linspace(0, 1, T)), dims=[0]))*beta

    return beta

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def get_modeled_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    out = torch.gather(vals, 0, t.cpu().unsqueeze(-1).unsqueeze(-1)).to(t.device)
    return out

class DiffusionSchedule():
    # Precompute the sqrt alphas and sqrt one minus alphas
    def __init__(self, T, betas):
        self.T = T
        self.betas = betas
        self.alphas = (1. - self.betas)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = (self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod).clamp(min=1e-8)).clamp(min=0.0)

    def forward_diffusion_sample_stoch(self, x_0, eps_t, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise = torch.randn_like(x_0).to(device)
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device)\
        - sqrt_one_minus_alphas_cumprod_t.to(device) * eps_t.to(device), noise.to(device)\
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise = torch.randn_like(x_0).to(device)
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device)\
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

    def diffusion_step_sample(self, noise_pred, x_noisy,  t, device="cpu"):
        """ 
        Takes an image, noise and step; returns denoised image. 
        """
        betas_t = get_index_from_list(self.betas, t, x_noisy.shape).to(device)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape
        ).to(device)
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x_noisy.shape).to(device)
        model_mean = sqrt_recip_alphas_t * (
            x_noisy - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, x_noisy.shape).to(device)

        # mean + variance
        return (model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x_noisy)).to(device)

class ModeledDiffusionSchedule():
    # Precompute the sqrt alphas and sqrt one minus alphas
    def __init__(self, T, betas):
        self.T = T
        self.betas = betas
        self.alphas = (1. - self.betas)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        # if not special:
        #     ones_pad = torch.ones_like(self.alphas_cumprod[0,:,:])
        #     self.alphas_cumprod_prev = self.alphas_cumprod[:-1,:,:]
        #     self.alphas_cumprod_prev = torch.cat((ones_pad.unsqueeze(0), self.alphas_cumprod_prev), dim=0)
        # else:
        gen_pad = torch.ones_like(self.alphas_cumprod[0,:,:]) * self.alphas_cumprod[0,:,:]
        self.alphas_cumprod_prev = self.alphas_cumprod[:-1,:,:]
        self.alphas_cumprod_prev = torch.cat((gen_pad.unsqueeze(0), self.alphas_cumprod_prev), dim=0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = (self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod).clamp(min=1e-8)).clamp(min=0.0)

    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise = torch.randn_like(x_0).to(device)
        sqrt_alphas_cumprod_t = get_modeled_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_modeled_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device)\
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

    def diffusion_step_sample(self, noise_pred, x_noisy,  t, device="cpu"):
        """ 
        Takes an image, noise and step; returns denoised image. 
        """
        betas_t = get_modeled_index_from_list(self.betas, t, x_noisy.shape).to(device)
        sqrt_one_minus_alphas_cumprod_t = get_modeled_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape
        ).to(device)
        sqrt_recip_alphas_t = get_modeled_index_from_list(self.sqrt_recip_alphas, t, x_noisy.shape).to(device)
        model_mean = sqrt_recip_alphas_t * (
            x_noisy - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_modeled_index_from_list(self.posterior_variance, t, x_noisy.shape).to(device)

        # mean + variance
        return (model_mean + torch.sqrt(posterior_variance_t) * noise_pred).to(device)
    
class StandardDiffusionSampler(DiffusionSchedule):
    # Precompute the sqrt alphas and sqrt one minus alphas. Inherit from DiffusionSchedule
    def __init__(self, T, betas):
        super().__init__(T, betas)

    def lossfn_builder(self, reg_factor=0.1, device="cpu"):
        """
        Returns the loss function for the diffusion model.
        """
        def lossfn(x_noisy, noise, noise_pred, t):
            x_denoised = self.diffusion_step_sample(noise_pred, x_noisy, t, device)
            mean_reg = (torch.norm((x_denoised[:, 1:, 1:] - x_denoised[:, 1:, :-1]), dim=1).mean()*torch.exp(-20*(t/self.T))).mean()
            return F.mse_loss(noise_pred, noise) + reg_factor*mean_reg 
        
        return lossfn

    def sample_timestep(self, model, x, t, c=None, t_mask=None):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        model.eval()
        with torch.no_grad():
            betas_t = get_index_from_list(self.betas, t, x.shape)
            sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
                self.sqrt_one_minus_alphas_cumprod, t, x.shape
            )
            sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
            
            # Call model (current image - noise prediction)

            if c is not None:
                model_mean = sqrt_recip_alphas_t * (
                    x - betas_t * model(x, t, c) / sqrt_one_minus_alphas_cumprod_t
                )
            else:
                model_mean = sqrt_recip_alphas_t * (
                    x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
                )

            posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape)
            if t_mask is None:
                device = x.device
               
                t_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))).to(device) 
            
            return model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x) * t_mask

class BaselineSampler_AoA():
    # A modified version of the StandardDiffusionSampler class for the AoA model, but with a modified lossfn_builder and sample_timestep function;
    # Stores two betas: one for the AoA and one for the airfoil (x)
    def __init__(self, T, start_x, end_x, start_alpha, end_alpha, cosine=False):
        betas_x = beta_schedule(T=T, start=start_x, end=end_x,
                            scale= 1.0, cosine=cosine,
                            exp_biasing=False, exp_bias_factor=1.0
                            )
        betas_AoA = beta_schedule(T=T, start=start_alpha, end=end_alpha,
                            scale= 1.0, cosine=cosine,
                            exp_biasing=False, exp_bias_factor=1.0
                            )
        schedule_x = DiffusionSchedule(T, betas_x)
        schedule_AoA = DiffusionSchedule(T, betas_AoA)
        self.schedule_x = schedule_x
        self.schedule_AoA = schedule_AoA
        self.T = T

    def lossfn_builder(self, weights = [1.0, 1.0], apply_reg=False, reg_factor=0.1, reg_exp=20, punish_tail_crossing=False):
        """
        Returns the loss function for the diffusion model.
        weights: (x_weight, alpha_weight); list of weights for the loss function components
        reg_factor: regularization factor; imposes a penalty on poor decoded latent space representations
        reg_exp: time step decay factor for the regularization term (higher values means the term is applied less strongly to later (further from x0) time steps)
        """

        def lossfn(
                x_noisy, x_noise, x_noise_pred, 
                alpha_noise, alpha_noise_pred, t, 
                bae_model, return_components=False,
                   ):
            
            x_denoised = self.schedule_x.diffusion_step_sample(x_noise_pred, x_noisy, t, x_noisy.device)
            x_decoded = bae_model.decode_z(x_denoised, z_ae_mode=True, denormalize_output=True,normalized_data=False)[0]
            # Punish the model if the first points are not above the corresponding last points
            # y_points = x_decoded[:,1,1:x_decoded.shape[2]-1].clone()
            # First 20 points of y_points:
            # y_points_upper = y_points[:,:n_points]
            # Last 20 points of y_points in reverse order:
            # y_points_lower = y_points[:,x_decoded.shape[2]-n_points-1:x_decoded.shape[2]-1].flip(dims=[1])
            # Get the difference between the two
            # y_diff = y_points_upper - y_points_lower
            # Now punish the model for any negative values in y_diff
            # mean_reg = (torch.where(y_diff < 0, -y_diff, torch.zeros_like(y_diff))*torch.exp(-reg_exp*(t.unsqueeze(-1)/self.T))).mean()
            # mean_reg = (torch.where(y_diff < 0, torch.abs(y_diff), torch.zeros_like(y_diff))).mean()

            if apply_reg:
                mean_reg = (torch.norm((x_decoded[:, 1:, 1:] - x_decoded[:, 1:, :-1]), dim=1)*torch.exp(-reg_exp*(t.unsqueeze(-1)/self.T))).mean()
            else:
                mean_reg = torch.tensor(0.0).to(x_noisy.device)
            
            if punish_tail_crossing:
                y_diff = x_decoded[:, 1, 1:int(x_decoded.shape[2]/8)] - torch.flip(x_decoded, [2])[:, 1, 1:int(x_decoded.shape[2]/8)]
                y_diff_loss = torch.where(y_diff < 0, torch.pow(y_diff,2), torch.zeros_like(y_diff)).mean(axis=1)
                y_diff_loss = torch.where(t < 0.25*self.T, y_diff_loss, torch.zeros_like(y_diff_loss)).mean()
                # print(f'Loss ydiff: {y_diff_loss.item()}')
                mean_reg += y_diff_loss

            if return_components:
                # Return the individual components of the loss function
                loss_x = F.mse_loss(x_noise_pred, x_noise)
                loss_alpha = F.mse_loss(alpha_noise_pred, alpha_noise)
                mean_reg = reg_factor*mean_reg
                total_loss = weights[0]*loss_x + weights[1]*loss_alpha + mean_reg 
                return total_loss, loss_x, loss_alpha, mean_reg
            else:
                total_loss = weights[0]*F.mse_loss(x_noise_pred, x_noise) + weights[1]*F.mse_loss(alpha_noise_pred, alpha_noise) + reg_factor*mean_reg
                return total_loss
        
        return lossfn
    
    def sample_timestep(self, model, x, alpha, c, x0, t, t_mask=None):
            """
            Calls the model to predict the noise in the image and returns 
            the denoised image. 
            Applies noise to this image, if we are not in the last step yet.
            """
            # model.eval()
            with torch.no_grad():
                betas_x_t = get_index_from_list(self.schedule_x.betas, t, x.shape)
                sqrt_one_minus_alphas_cumprod_x_t = get_index_from_list(
                    self.schedule_x.sqrt_one_minus_alphas_cumprod, t, x.shape
                )
                sqrt_recip_alphas_x_t = get_index_from_list(self.schedule_x.sqrt_recip_alphas, t, x.shape)
                
                betas_AoA_t = get_index_from_list(self.schedule_AoA.betas, t, alpha.shape)
                sqrt_one_minus_alphas_cumprod_AoA_t = get_index_from_list(
                    self.schedule_AoA.sqrt_one_minus_alphas_cumprod, t, alpha.shape
                )
                sqrt_recip_alphas_AoA_t = get_index_from_list(self.schedule_AoA.sqrt_recip_alphas, t, alpha.shape)

                # Ensure alpha is 2D [batch, 1] for the model
                if alpha.dim() == 3:
                    alpha = alpha.squeeze(1)
                elif alpha.dim() == 1:
                    alpha = alpha.unsqueeze(1)
                elif alpha.dim() == 2 and alpha.shape[1] == 0:
                    # If alpha is empty, create a new tensor
                    alpha = torch.randn(alpha.shape[0], 1).to(alpha.device)

                # Call model (current image - noise prediction)
                x_pred, alpha_pred = model.infer(x, alpha, c, x0, t)

                # Ensure alpha_pred is 2D for the formula
                if alpha_pred.dim() == 3:
                    alpha_pred = alpha_pred.squeeze(1)

                model_mean_x = sqrt_recip_alphas_x_t * (
                    x - betas_x_t * x_pred / sqrt_one_minus_alphas_cumprod_x_t
                )
                model_mean_AoA = sqrt_recip_alphas_AoA_t * (
                    alpha - betas_AoA_t * alpha_pred / sqrt_one_minus_alphas_cumprod_AoA_t
                )

                posterior_variance_x_t = get_index_from_list(self.schedule_x.posterior_variance, t, x.shape)
                posterior_variance_AoA_t = get_index_from_list(self.schedule_AoA.posterior_variance, t, alpha.shape)

                if t_mask is None:
                    device = x.device
                    t_mask_x = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))).to(device)
                    t_mask_AoA = ((t != 0).float().view(-1, *([1] * (len(alpha.shape) - 1)))).to(device)
                
                # Generate new x and alpha
                new_x = model_mean_x + torch.sqrt(posterior_variance_x_t) * torch.randn_like(x) * t_mask_x
                new_alpha = model_mean_AoA + torch.sqrt(posterior_variance_AoA_t) * torch.randn_like(alpha) * t_mask_AoA
                
                # Ensure new_alpha is 2D [batch, 1]
                if new_alpha.dim() == 3:
                    new_alpha = new_alpha.squeeze(1)
                elif new_alpha.dim() == 1:
                    new_alpha = new_alpha.unsqueeze(1)
                
                return new_x, new_alpha
        
    def sample_airfoil(self, 
                        model,
                        noise_x: torch.Tensor, noise_alpha: torch.Tensor, 
                        c: torch.Tensor, x0:torch.Tensor,
                        T=None
                        ):
            """
            Returns a sample generated airfoil and its latent coordinates 
            Args:
                @model: The model to be used for the inference
                @noise_x: Initial noise for the airfoil
                @noise_alpha: Initial noise for the AoA
                @c: The conditioning parameters
                @x0: The initial airfoil
                @T: The number of diffusion steps
            """
            if T is None:
                T = self.T
            device = noise_x.device
            gen_airfoil_latent = noise_x.detach().clone().to(device)
            gen_alpha = noise_alpha.detach().clone().to(device)
            
            # Ensure gen_alpha is 2D [batch, 1] for the model
            if gen_alpha.dim() == 3:
                gen_alpha = gen_alpha.squeeze(1)
            elif gen_alpha.dim() == 1:
                gen_alpha = gen_alpha.unsqueeze(1)  # [batch] -> [batch, 1]
            
            # Make sure it's not empty
            if gen_alpha.shape[1] == 0:
                gen_alpha = torch.randn(gen_alpha.shape[0], 1).to(device)
            
            batch_size = gen_airfoil_latent.shape[0]

            with torch.no_grad():
                for i in range(0,T)[::-1]:
                    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                    gen_airfoil_latent, gen_alpha = self.sample_timestep(
                        model, gen_airfoil_latent, gen_alpha, c, x0, t, t_mask=None
                    )
                    
                    # Ensure gen_alpha stays 2D [batch, 1] for the next iteration
                    if gen_alpha.dim() == 3:
                        gen_alpha = gen_alpha.squeeze(1)
                    elif gen_alpha.dim() == 1:
                        gen_alpha = gen_alpha.unsqueeze(1)
            
            # Final check - ensure it's not empty
            if gen_alpha.dim() == 2 and gen_alpha.shape[1] == 0:
                gen_alpha = torch.randn(gen_alpha.shape[0], 1).to(device)
            
            return gen_airfoil_latent, gen_alpha


class BaselineSampler_AoA_3D(BaselineSampler_AoA):
    """2-schedule sampler for 3D wings: geometry (x) and AoA (alpha).

    Beta schedules:
      schedule_x:   beta 1e-6 -> 0.02
      schedule_AoA: beta 1e-4 -> 0.02
    """

    def __init__(self, T, start_x, end_x, start_alpha, end_alpha, cosine=False):
        super().__init__(T, start_x, end_x, start_alpha, end_alpha, cosine=cosine)

    def sample_timestep(self, model, x, alpha, c, x0, t, t_mask=None):
        """Denoise x and alpha for one DDPM step."""
        with torch.no_grad():
            # Normalise alpha dims
            if alpha.dim() == 3:
                alpha = alpha.squeeze(1)
            elif alpha.dim() == 1:
                alpha = alpha.unsqueeze(1)
            elif alpha.dim() == 2 and alpha.shape[1] == 0:
                alpha = torch.randn(alpha.shape[0], 1).to(alpha.device)

            # Model prediction (2 outputs)
            x_pred, alpha_pred = model.infer(x, alpha, c, x0, t)

            if alpha_pred.dim() == 3:
                alpha_pred = alpha_pred.squeeze(1)

            device = x.device
            if t_mask is None:
                t_mask_x   = ((t != 0).float().view(-1, *([1] * (len(x.shape)     - 1)))).to(device)
                t_mask_AoA = ((t != 0).float().view(-1, *([1] * (len(alpha.shape) - 1)))).to(device)

            def _step(schedule, noisy, noise_pred, t_mask_local):
                betas_t      = get_index_from_list(schedule.betas, t, noisy.shape)
                sqrt_1m_t    = get_index_from_list(schedule.sqrt_one_minus_alphas_cumprod, t, noisy.shape)
                sqrt_recip_t = get_index_from_list(schedule.sqrt_recip_alphas, t, noisy.shape)
                post_var_t   = get_index_from_list(schedule.posterior_variance, t, noisy.shape)
                mean = sqrt_recip_t * (noisy - betas_t * noise_pred / sqrt_1m_t)
                return mean + torch.sqrt(post_var_t) * torch.randn_like(noisy) * t_mask_local

            new_x     = _step(self.schedule_x,   x,     x_pred,     t_mask_x)
            new_alpha = _step(self.schedule_AoA, alpha, alpha_pred, t_mask_AoA)

            if new_alpha.dim() == 3:
                new_alpha = new_alpha.squeeze(1)
            elif new_alpha.dim() == 1:
                new_alpha = new_alpha.unsqueeze(1)

            return new_x, new_alpha

    def sample_airfoil(self, model, noise_x, noise_alpha, c, x0, T=None):
        """Run full reverse diffusion and return (gen_x, gen_alpha)."""
        if T is None:
            T = self.T
        device = noise_x.device
        gen_x     = noise_x.detach().clone().to(device)
        gen_alpha = noise_alpha.detach().clone().to(device)

        if gen_alpha.dim() == 3:
            gen_alpha = gen_alpha.squeeze(1)
        elif gen_alpha.dim() == 1:
            gen_alpha = gen_alpha.unsqueeze(1)
        if gen_alpha.shape[1] == 0:
            gen_alpha = torch.randn(gen_alpha.shape[0], 1).to(device)

        batch_size = gen_x.shape[0]

        with torch.no_grad():
            for i in range(0, T)[::-1]:
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                gen_x, gen_alpha = self.sample_timestep(
                    model, gen_x, gen_alpha, c, x0, t, t_mask=None
                )
                if gen_alpha.dim() == 3:
                    gen_alpha = gen_alpha.squeeze(1)
                elif gen_alpha.dim() == 1:
                    gen_alpha = gen_alpha.unsqueeze(1)

        if gen_alpha.dim() == 2 and gen_alpha.shape[1] == 0:
            gen_alpha = torch.randn(gen_alpha.shape[0], 1).to(device)

        return gen_x, gen_alpha


class EulerSampler_AoA(BaselineSampler_AoA):
    """Sampler that uses a deterministic Euler-style (DDIM-like) update rule for sampling.

    This class is mainly used at inference time; it produces a deterministic backward pass
    (no additional Gaussian noise is added), which is similar to the DDIM solver with eta=0.

    It can be used during training as well by selecting it from the training config.
    """

    def __init__(self, T, start_x, end_x, start_alpha, end_alpha):
        super().__init__(T, start_x, end_x, start_alpha, end_alpha)

    def _ddim_step(self, x, noise_pred, t, schedule):
        """Compute a deterministic DDIM/Euler update step for a single schedule."""
        # x: current noisy latent (batch, ...)
        # noise_pred: predicted noise for x
        alpha_t = get_index_from_list(schedule.alphas_cumprod, t, x.shape)
        alpha_prev = get_index_from_list(schedule.alphas_cumprod_prev, t, x.shape)

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_prev = torch.sqrt(alpha_prev)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        sqrt_one_minus_alpha_prev = torch.sqrt(1.0 - alpha_prev)

        # Predict x0 (denoised latent) using current noise prediction
        x0_pred = (x - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

        # Deterministic update (eta=0)
        x_prev = sqrt_alpha_prev * x0_pred + sqrt_one_minus_alpha_prev * noise_pred
        return x_prev

    def sample_timestep(self, model, x, alpha, c, x0, t, t_mask=None):
        """Calls the model to predict the noise in the image and returns the denoised image.

        This is a deterministic (Euler/ DDIM-like) backward pass.
        """
        with torch.no_grad():
            # Ensure alpha is 2D [batch, 1] for the model
            if alpha.dim() == 3:
                alpha = alpha.squeeze(1)
            elif alpha.dim() == 1:
                alpha = alpha.unsqueeze(1)
            elif alpha.dim() == 2 and alpha.shape[1] == 0:
                alpha = torch.randn(alpha.shape[0], 1).to(alpha.device)

            # Call model (current image - noise prediction)
            x_pred, alpha_pred = model.infer(x, alpha, c, x0, t)

            # Ensure alpha_pred is 2D for the formula
            if alpha_pred.dim() == 3:
                alpha_pred = alpha_pred.squeeze(1)

            new_x = self._ddim_step(x, x_pred, t, self.schedule_x)
            new_alpha = self._ddim_step(alpha, alpha_pred, t, self.schedule_AoA)

            # Ensure new_alpha is 2D [batch, 1]
            if new_alpha.dim() == 3:
                new_alpha = new_alpha.squeeze(1)
            elif new_alpha.dim() == 1:
                new_alpha = new_alpha.unsqueeze(1)

            return new_x, new_alpha


class Sampler3D_AoA(BaselineSampler_AoA):
    def __init__(self, T, start_x, end_x, start_alpha, end_alpha):
        # Inherit from BaselineSampler_AoA
        super().__init__(T, start_x, end_x, start_alpha, end_alpha)

    def lossfn_builder(self, weights = [1.0, 1.0], apply_reg=False, reg_factor=0.1, reg_exp=20, punish_tail_crossing=False):
        """
        Returns the loss function for the diffusion model.
        weights: (x_weight, alpha_weight); list of weights for the loss function components
        reg_factor: regularization factor; imposes a penalty on poor decoded latent space representations
        reg_exp: time step decay factor for the regularization term (higher values means the term is applied less strongly to later (further from x0) time steps)
        """

        def lossfn(
                x_noisy, x_noise, x_noise_pred, 
                alpha_noise, alpha_noise_pred, t, 
                bae_model, return_components=False,
                   ):
            
            x_denoised = self.schedule_x.diffusion_step_sample(x_noise_pred, x_noisy, t, x_noisy.device)
            x_decoded = bae_model.decode_z(x_denoised, z_ae_mode=True, denormalize_output=True,normalized_data=False)[0]

            if apply_reg:
                mean_reg = (torch.norm((x_decoded[:, :, 1:, 1:] - x_decoded[:, :, 1:, :-1]), dim=1)*torch.exp(-reg_exp*(t.unsqueeze(-1)/self.T))).mean()
            else:
                mean_reg = torch.tensor(0.0).to(x_noisy.device)
            
            if punish_tail_crossing:
                y_diff = x_decoded[:, :, 1, 1:int(x_decoded.shape[2]/8)] - torch.flip(x_decoded, [2])[:, :, 1, 1:int(x_decoded.shape[2]/8)]
                y_diff_loss = torch.where(y_diff < 0, torch.pow(y_diff,2), torch.zeros_like(y_diff)).mean(axis=1)
                y_diff_loss = torch.where(t < 0.25*self.T, y_diff_loss, torch.zeros_like(y_diff_loss)).mean()
                # print(f'Loss ydiff: {y_diff_loss.item()}')
                mean_reg += y_diff_loss

            if return_components:
                # Return the individual components of the loss function
                loss_x = F.mse_loss(x_noise_pred, x_noise)
                loss_alpha = F.mse_loss(alpha_noise_pred, alpha_noise)
                mean_reg = reg_factor*mean_reg
                total_loss = weights[0]*loss_x + weights[1]*loss_alpha + mean_reg 
                return total_loss, loss_x, loss_alpha, mean_reg
            else:
                total_loss = weights[0]*F.mse_loss(x_noise_pred, x_noise) + weights[1]*F.mse_loss(alpha_noise_pred, alpha_noise) + reg_factor*mean_reg
                return total_loss
        
        return lossfn
    


class ModeledSampler_AoA():
    # A modified version of the StandardDiffusionSampler class for the AoA model, but with a modified lossfn_builder and sample_timestep function;
    # Stores two betas: one for the AoA and one for the airfoil (x)
    def __init__(self, T, betas_x, start_alpha, end_alpha):
        betas_AoA = beta_schedule(T=T, start=start_alpha, end=end_alpha,
                            scale= 1.0, cosine=False, 
                            exp_biasing=False, exp_bias_factor=1.0
                            )
        schedule_x = ModeledDiffusionSchedule(T, betas_x)
        schedule_AoA = DiffusionSchedule(T, betas_AoA)
        self.schedule_x = schedule_x
        self.schedule_AoA = schedule_AoA
        self.T = T

    def lossfn_builder(self, weights = [1.0, 1.0], apply_reg=False, reg_factor=0.1, reg_exp=20, punish_tail_crossing=False):
        """
        Returns the loss function for the diffusion model.
        weights: (x_weight, alpha_weight); list of weights for the loss function components
        reg_factor: regularization factor; imposes a penalty on poor decoded latent space representations
        reg_exp: time step decay factor for the regularization term (higher values means the term is applied less strongly to later (further from x0) time steps)
        """

        def lossfn(
                x_noisy, x_noise, x_noise_pred, 
                alpha_noise, alpha_noise_pred, t, 
                bae_model, return_components=False,
                   ):
            
            x_denoised = self.schedule_x.diffusion_step_sample(x_noise_pred, x_noisy, t, x_noisy.device)
            x_decoded = bae_model.decode_z(x_denoised, z_ae_mode=True, denormalize_output=True,normalized_data=False)[0]

            if apply_reg:
                mean_reg = (torch.norm((x_decoded[:, 1:, 1:] - x_decoded[:, 1:, :-1]), dim=1)*torch.exp(-reg_exp*(t.unsqueeze(-1)/self.T))).mean()
            else:
                mean_reg = torch.tensor(0.0).to(x_noisy.device)
            
            if punish_tail_crossing:
                y_diff = x_decoded[:, 1, 1:int(x_decoded.shape[2]/8)] - torch.flip(x_decoded, [2])[:, 1, 1:int(x_decoded.shape[2]/8)]
                y_diff_loss = torch.where(y_diff < 0, torch.pow(y_diff,2), torch.zeros_like(y_diff)).mean(axis=1)
                y_diff_loss = torch.where(t < 0.25*self.T, y_diff_loss, torch.zeros_like(y_diff_loss)).mean()
                # print(f'Loss ydiff: {y_diff_loss.item()}')
                mean_reg += y_diff_loss

            if return_components:
                # Return the individual components of the loss function
                loss_x = F.mse_loss(x_noise_pred, x_noise)
                loss_alpha = F.mse_loss(alpha_noise_pred, alpha_noise)
                mean_reg = reg_factor*mean_reg
                total_loss = weights[0]*loss_x + weights[1]*loss_alpha + mean_reg 
                return total_loss, loss_x, loss_alpha, mean_reg
            else:
                total_loss = weights[0]*F.mse_loss(x_noise_pred, x_noise) + weights[1]*F.mse_loss(alpha_noise_pred, alpha_noise) + reg_factor*mean_reg
                return total_loss
        
        return lossfn
    
    def sample_timestep(self, model, x, alpha, c, x0, t, t_mask=None):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        # model.eval()
        with torch.no_grad():
            betas_x_t = get_modeled_index_from_list(self.schedule_x.betas, t, x.shape)
            sqrt_one_minus_alphas_cumprod_x_t = get_modeled_index_from_list(
                self.schedule_x.sqrt_one_minus_alphas_cumprod, t, x.shape
            )
            sqrt_recip_alphas_x_t = get_modeled_index_from_list(self.schedule_x.sqrt_recip_alphas, t, x.shape)

            betas_AoA_t = get_index_from_list(self.schedule_AoA.betas, t, alpha.shape)
            sqrt_one_minus_alphas_cumprod_AoA_t = get_index_from_list(
                self.schedule_AoA.sqrt_one_minus_alphas_cumprod, t, alpha.shape
            )
            sqrt_recip_alphas_AoA_t = get_index_from_list(self.schedule_AoA.sqrt_recip_alphas, t, alpha.shape)

            # Call model (current image - noise prediction)
            x_pred, alpha_pred = model.infer(x, alpha, c, x0, t)

            model_mean_x = sqrt_recip_alphas_x_t * (
                x - betas_x_t * x_pred / sqrt_one_minus_alphas_cumprod_x_t
            )
            model_mean_AoA = sqrt_recip_alphas_AoA_t * (
                alpha - betas_AoA_t * alpha_pred / sqrt_one_minus_alphas_cumprod_AoA_t
            )


            posterior_variance_x_t = get_modeled_index_from_list(self.schedule_x.posterior_variance, t, x.shape)
            posterior_variance_AoA_t = get_index_from_list(self.schedule_AoA.posterior_variance, t, alpha.shape)

            if t_mask is None:
                device = x.device
               
                t_mask_x = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))).to(device)

                t_mask_AoA = ((t != 0).float().view(-1, *([1] * (len(alpha.shape) - 1)))).to(device)
            
            return model_mean_x + torch.sqrt(posterior_variance_x_t) * torch.randn_like(x) * t_mask_x, model_mean_AoA + torch.sqrt(posterior_variance_AoA_t) * torch.randn_like(alpha) * t_mask_AoA

    def sample_airfoil(self, 
                       model,
                       noise_x: torch.Tensor, noise_alpha: torch.Tensor, 
                       c: torch.Tensor, x0:torch.Tensor,
                       T=None
                       ):
        """
        Returns a sample generated airfoil and its latent coordinates 
        Args:
            @model: The model to be used for the inference
            @noise_x: Initial noise for the airfoil
            @noise_alpha: Initial noise for the AoA
            @c: The conditioning parameters
            @x0: The initial airfoil
            @T: The number of diffusion steps
        """
        if T is None:
            T = self.T
        device = noise_x.device
        gen_airfoil_latent = noise_x.detach().clone().to(device)
        gen_alpha = noise_alpha.detach().clone().to(device)
        batch_size = gen_airfoil_latent.shape[0]

        with torch.no_grad():

            for i in range(0,T)[::-1]:
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                gen_airfoil_latent, gen_alpha = self.sample_timestep(model, gen_airfoil_latent, gen_alpha, c, x0, t, t_mask=None)
            
        return gen_airfoil_latent, gen_alpha

class TrajCorrSampler():
    # A modified version of the StandardDiffusionSampler class for the AoA model, but with a modified lossfn_builder and sample_timestep function;
    # Stores two betas: one for the AoA and one for the airfoil (x)
    def __init__(self, T, delta_x = 1, delta_alpha = 1):

        self.T = T
        self.delta_x = delta_x
        self.delta_alpha = delta_alpha
    
    def denoise_sample(self, noise_pred, noisy):
        return noisy - noise_pred
    
    def sample_timestep(self, model, x, alpha, c, x0, t=None, t_mask=None):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        # model.eval()
        with torch.no_grad():

            # Call model (current image - noise prediction)
            x_pred_noise, alpha_pred_noise = model.infer(x, alpha, c, x0)
            x_pred = x - x_pred_noise*self.delta_x
            alpha_pred = alpha - alpha_pred_noise*self.delta_alpha

            return x_pred, alpha_pred

    def sample_airfoil(self, 
                       model,
                       noise_x: torch.Tensor, noise_alpha: torch.Tensor, 
                       c: torch.Tensor, x0:torch.Tensor,
                       T=None
                       ):
        """
        Returns a sample generated airfoil and its latent coordinates 
        Args:
            @model: The model to be used for the inference
            @noise_x: Initial noise for the airfoil
            @noise_alpha: Initial noise for the AoA
            @c: The conditioning parameters
            @x0: The initial airfoil
            @T: The number of diffusion steps
        """
        if T is None:
            T = self.T
        device = noise_x.device
        gen_airfoil_latent = noise_x.detach().clone().to(device)
        gen_alpha = noise_alpha.detach().clone().to(device)

        with torch.no_grad():

            for i in range(0,T)[::-1]:
                # gen_airfoil_latent += torch.randn_like(gen_airfoil_latent)*(1e-5)
                # gen_alpha += torch.randn_like(gen_alpha)*(1e-5)
                gen_airfoil_latent, gen_alpha = self.sample_timestep(model, gen_airfoil_latent, gen_alpha, c, x0)
            
        return gen_airfoil_latent, gen_alpha

class NoiseGenerator:
    def __init__(self, dims=(1,3,60), device='cpu'):
        super().__init__()
        if isinstance(dims, torch.Size):
            dims = list(dims)
        if isinstance(dims, tuple):
            dims = list(dims)
        self.dims = dims
        self.batch = dims[0]
        self.device = device
    
    def change_batch(self, batch):
        self.batch = batch

    def __call__(self, batch_in=None, seed=None):

        if seed is not None:
            torch.manual_seed(seed)

        if batch_in is None:
            batch_in = self.batch
        
        dims = self.dims
        dims[0] = batch_in
        noise = torch.randn(dims, device=self.device)
        return noise
    
class NoiseGeneratorStatic:
    def __init__(self, data, device='cpu'):
        super().__init__()

        # Check if data is an np array
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        self.data = data.to(device)
        dims = data.shape
        self.dims = list(dims)
        self.batch = dims[0]
        self.device = device
    
    def change_data(self, data):
        # Check if data is an np array
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        self.data = data.to(self.device)
        self.dims = list(data.shape)
        self.batch = self.dims[0]

    def __call__(self, batch_in=None, seed=None):
        return self.data

class DualNoiseGenerator:
    def __init__(self, noisegen_x, noisegen_alpha):
        super().__init__()
        self.noisegen_x = noisegen_x
        self.noisegen_alpha = noisegen_alpha

    def __call__(self, batch_in=None, seed=None):
        noise_x = self.noisegen_x(batch_in, seed)
        noise_alpha = self.noisegen_alpha(batch_in, seed)
        return noise_x, noise_alpha