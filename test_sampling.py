import torch
from engiopt.ddm.ddm import DDM_AoAInit_3D
from engiopt.ddm.train_ddm import Config, build_sampler, build_unet, load_bae
from engibench.problems.wings3D.v0 import Wings3D
import os

os.environ['HF_DATASETS_OFFLINE'] = '1'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
cfg = Config()

bae = load_bae(cfg)
unet = build_unet(cfg)
sampler = build_sampler(cfg)
ddm = DDM_AoAInit_3D(unet=unet, sampler=sampler, bae_model=bae, params_mean_std=(0,1), aoas_mean_std=(0,1), name=cfg.model_name, opt_lr=cfg.lr)
ddm.load(f'results/ddm/{cfg.model_name}.pth', train_mode=False)
print('AoA scaler mean:', ddm.scaler_aoas.mean)
print('AoA scaler std:', ddm.scaler_aoas.std)
print('Params scaler mean:', ddm.scaler_params.mean)
print('Params scaler std:', ddm.scaler_params.std)
ddm.unet = ddm.unet.to(device)
ddm.bae_model = ddm.bae_model.to(device)

p = Wings3D(seed=0)
item = p.dataset['test'][0]
coords = torch.tensor(item['coords'], dtype=torch.float32)
x_init = coords[0].permute(1,0).unsqueeze(0).to(device)
with torch.no_grad():
    z_init = bae.encode(x_init, return_z=True, z_ae_mode=True)

params = torch.tensor([[item['mach'], item['reynolds'], item['cl_target'], item['area_case_ratio']]], dtype=torch.float32).to(device)
params_scaled = ddm.scaler_params.transform(params)

noise_x = torch.randn(1, 9, 3, 14, device=device)
noise_alpha = torch.randn(1, 1, device=device)

print('Input noise range:', noise_x.min().item(), noise_x.max().item())
print('z_init range:', z_init.min().item(), z_init.max().item())
print('Real latent mean:', z_init.mean().item(), 'std:', z_init.std().item())

with torch.no_grad():
    gen_z, gen_alpha = ddm.sampler.sample_airfoil(
        model=ddm.unet,
        noise_x=noise_x,
        noise_alpha=noise_alpha,
        c=params_scaled,
        x0=z_init,
        T=None
    )
# Check raw alpha before inverse transform
print('Raw gen_alpha before inverse transform:', gen_alpha.item())
print('Expected normalized range: -2 to 2')
print('AoA scaler: mean=', ddm.scaler_aoas.mean, 'std=', ddm.scaler_aoas.std)

print('Generated latent range:', gen_z.min().item(), gen_z.max().item())
print('Generated latent mean:', gen_z.mean().item(), 'std:', gen_z.std().item())
print('Generated alpha:', gen_alpha.item())

# Decode one slice and check range
z_s = gen_z[0, 0, :, :].unsqueeze(0)  # [1, 3, 14]
with torch.no_grad():
    dec = bae.decode_z(z_s, z_ae_mode=True, denormalize_output=False, normalized_data=False)[0]
print('Decoded slice shape:', dec.shape)
print('Decoded slice range:', dec.min().item(), dec.max().item())

# Compare with real decoded slice
z_real = z_init[0, :, :].unsqueeze(0)  # [1, 3, 14]
with torch.no_grad():
    dec_real = bae.decode_z(z_real, z_ae_mode=True, denormalize_output=False, normalized_data=False)[0]
print('Real decoded slice range:', dec_real.min().item(), dec_real.max().item())
print('MSE between generated and real:', ((dec - dec_real)**2).mean().item())

# Simulate what evaluation does
gt = dec_real  # [1, 2, 192]
gen = dec      # [1, 2, 192]
print('gt device:', gt.device)
print('gen device:', gen.device)
print('Direct MSE:', ((gen - gt)**2).mean().item())
print('gt range:', gt.min().item(), gt.max().item())
print('gen range:', gen.min().item(), gen.max().item())
