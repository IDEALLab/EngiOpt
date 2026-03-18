import sys
import os
from datetime import datetime, timezone

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Aggressively disable all LaTeX
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['text.latex.preamble'] = ''
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['pdf.use14corefonts'] = True
import numpy as np

from .train_ddm import Config, load_bae, build_unet, build_sampler
from engiopt.ddm.ddm import DDM_AoAInit

def generate_airfoils_direct(ddm_model, bae_model, num_samples=10, device='cpu'):
    """Generate new airfoils using trained DDM - direct sampling approach"""
    
    # Create dummy conditions (you can modify these)
    # [mach, reynolds, cl_target, area_case_ratio]
    params = torch.tensor([
        [0.5, 5e6, 0.8, 1.0]  # Example conditions
    ]).repeat(num_samples, 1).to(device).float()
    
    # Scale the parameters using the same scaler as in training
    params_np = params.cpu().numpy()
    params_scaled = ddm_model.scaler_params.transform(params_np)
    params = torch.tensor(params_scaled).to(device).float()
    
    print(f"Params shape: {params.shape}")
    print(f"Scaled params: {params[0].cpu().numpy()}")
    
    # Create random noise for airfoil and alpha
    noise_x = torch.randn(num_samples, 3, 14).to(device)
    print(f"noise_x shape: {noise_x.shape}")
    
    noise_alpha = torch.randn(num_samples, 1).to(device)
    print(f"noise_alpha shape: {noise_alpha.shape}")

    # Use a real initial airfoil (from Wings3D) so sampling starts from a realistic latent state
    from engibench.problems.wings3D.v0 import Wings3D
    problem = Wings3D(seed=0)
    base_dataset = problem.dataset["train"]
    coords = torch.tensor(base_dataset[0]["coords"], dtype=torch.float32)  # [9,192,2]
    x_init = coords[0].permute(1, 0).unsqueeze(0).to(device)  # [1,2,192]

    with torch.no_grad():
        encoded_init_single = bae_model.encode(x_init, return_z=True, z_ae_mode=True)

    encoded_init = encoded_init_single.repeat(num_samples, 1, 1)
    print(f"encoded_init shape: {encoded_init.shape}")

    print("Generating airfoils using sampler directly...")
    with torch.no_grad():
        generated_airfoils, generated_alphas = ddm_model.sampler.sample_airfoil(
            model=ddm_model.unet,
            noise_x=noise_x,
            noise_alpha=noise_alpha,
            c=params,
            x0=encoded_init,
            T=None
        )
    
    print(f"generated_airfoils shape (latent): {generated_airfoils.shape}")
    print(f"generated_alphas shape (normalized): {generated_alphas.shape}")
    
    # Decode the airfoils
    print("Decoding airfoils...")
    with torch.no_grad():
        decoded_airfoils = bae_model.decode_z(
            generated_airfoils,
            z_ae_mode=True,
            denormalize_output=False,
            normalized_data=False
        )[0]
    
    # Check if decoded airfoils are in reasonable range
    print(f"decoded_airfoils shape: {decoded_airfoils.shape}")
    print(f"decoded_airfoils - x range: [{decoded_airfoils[:,0,:].min():.2f}, {decoded_airfoils[:,0,:].max():.2f}]")
    print(f"decoded_airfoils - y range: [{decoded_airfoils[:,1,:].min():.2f}, {decoded_airfoils[:,1,:].max():.2f}]")
    
    # Rescale alpha
    rescaled_alphas = ddm_model.scaler_aoas.inverse_transform(generated_alphas.cpu().numpy())
    rescaled_alphas = torch.tensor(rescaled_alphas).to(device)
    
    print(f"rescaled_alphas shape: {rescaled_alphas.shape}")
    print(f"Alpha values: {rescaled_alphas}")
    
    return decoded_airfoils, rescaled_alphas

def save_airfoils_only(generated_airfoils, generated_alphas):
    """Save airfoils without trying to plot"""
    print("\n" + "="*50)
    print("GENERATION RESULTS:")
    print(f"Generated {len(generated_airfoils)} airfoils")
    print(f"Airfoil shape: {generated_airfoils.shape}")
    print(f"Alpha values: {generated_alphas}")
    print("="*50 + "\n")

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("results", "generated", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    pt_path = os.path.join(save_dir, "generated_airfoils.pt")
    npz_path = os.path.join(save_dir, "generated_airfoils.npz")

    torch.save({
        "airfoils": generated_airfoils.cpu(),
        "alphas": generated_alphas.cpu()
    }, pt_path)
    print(f"Saved generated airfoils to {pt_path}")

    np.savez(npz_path,
             airfoils=generated_airfoils.cpu().numpy(),
             alphas=generated_alphas.cpu().numpy())
    print(f"Also saved as {npz_path}")

    # Print first few points of first airfoil to see what we have
    print("\nFirst airfoil first 10 points (x, y):")
    first_airfoil = generated_airfoils[0].cpu().numpy()
    for i in range(min(10, first_airfoil.shape[1])):
        print(f"  {i}: x={first_airfoil[0,i]:.2f}, y={first_airfoil[1,i]:.2f}")

    return save_dir, timestamp

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    cfg = Config()
    
    print("Loading BAE...")
    bae_model = load_bae(cfg)
    
    print("Building DDM components...")
    unet = build_unet(cfg)
    sampler = build_sampler(cfg)
    
    ddm_model = DDM_AoAInit(
        unet=unet,
        sampler=sampler,
        bae_model=bae_model,
        params_mean_std=(0, 1),
        aoas_mean_std=(0, 1),
        name=cfg.model_name,
        opt_lr=cfg.lr,
    )
    
    checkpoint_path = f"{cfg.save_dir}/{cfg.model_name}.pth"
    print(f"Loading checkpoint from {checkpoint_path}...")
    ddm_model.load(checkpoint_path, train_mode=False)
    ddm_model.unet = ddm_model.unet.to(device)
    ddm_model.bae_model = ddm_model.bae_model.to(device)
    bae_model = ddm_model.bae_model

    # Sanity check: make sure we actually loaded a trained model (non-empty loss history)
    if len(ddm_model.stats.get('train_loss', [])) == 0 or ddm_model.stats.get('current_loss', None) == 0:
        print("\nWARNING: The loaded checkpoint appears to be untrained (no training loss history or loss=0).\n" \
              "Please train the model first by running: \n" \
              "  KMP_DUPLICATE_LIB_OK=TRUE python -m engiopt.ddm.train_ddm\n")
        sys.exit(1)
    
    num_samples = 3
    print(f"Generating {num_samples} airfoils...")
    generated_airfoils, generated_alphas = generate_airfoils_direct(
        ddm_model, bae_model, num_samples, device
    )
    
    # Just save the results without plotting
    save_dir, _ = save_airfoils_only(generated_airfoils, generated_alphas)

    # Plot the first generated airfoil
    plt.figure(figsize=(8, 6))
    x_coords = generated_airfoils[0, 0, :].cpu().numpy()
    y_coords = generated_airfoils[0, 1, :].cpu().numpy()
    plt.plot(x_coords, y_coords, 'b-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Generated Airfoil')
    plt.axis('equal')
    plt.grid(True)
    plot_path = os.path.join(save_dir, "generated_airfoil_plot.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()