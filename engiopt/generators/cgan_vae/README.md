# 3D Conditional VAE-GAN (cgan_vae)

A 3D Conditional Variational Autoencoder-Generative Adversarial Network for generating volumetric engineering designs with multiview training.

## Overview

This implementation combines VAE and GAN architectures to generate 3D engineering designs (e.g., heat conduction topology optimization). The model uses a multiview approach, processing 2D slices from 3D volumes to learn robust latent representations that can both encode existing designs and generate new ones.

## Architecture

### Components
- **Encoder**: 2D CNN that processes cross-sectional slices and outputs latent mean/variance (μ, σ²)
- **Generator3D**: 3D conditional generator that creates volumetric designs from latent codes + conditions
- **Discriminator3D**: 3D conditional discriminator that distinguishes real from generated volumes

### Key Features
- **Multiview Training**: Extracts multiple 2D slices from each 3D volume for robust encoding
- **Conditional Generation**: Uses engineering conditions (e.g., volume fraction) to guide generation
- **VAE Regularization**: KL divergence loss ensures smooth, interpolatable latent space
- **WGAN-GP Training**: Wasserstein GAN with gradient penalty for stable adversarial training

## Quick Start

### Basic Training
```bash
# Train with default parameters
python engiopt/generators/cgan_vae/cgan_vae.py --problem-id heatconduction3d --n-epochs 300

# Save the checkpoint for later evaluation
python engiopt/generators/cgan_vae/cgan_vae.py --save-model --track
```

### Evaluation
```bash
# Score the trained checkpoint under the problem's frozen eval spec
python -m engiopt.evaluate --problem-id heatconduction3d --generators cgan_vae --seeds 1

# Several seeds at once, including the simulator-backed metrics
python -m engiopt.evaluate --problem-id heatconduction3d --generators cgan_vae --seeds 1 2 3 --include-expensive
```

## Key Parameters

### Core Training
- `--n_epochs 300`: Number of training epochs
- `--batch_size 8`: Batch size
- `--lr_gen 0.0025`: Generator learning rate
- `--lr_disc 10e-5`: Discriminator learning rate
- `--lr_enc 0.001`: Encoder learning rate

### VAE-Specific
- `--kl_weight 0.01`: Weight for KL divergence loss
- `--recon_weight 1.0`: Weight for reconstruction loss
- `--n_slices 9`: Number of slices extracted per volume
- `--latent_dim 64`: Dimensionality of latent space

### Training Dynamics
- `--gen_iters 1`: Generator updates per batch
- `--discrim_iters 1`: Discriminator updates per batch



## References:

Original VAE-GAN: https://arxiv.org/pdf/1512.09300
Multiview approach: https://github.com/bryonkucharski/Multiview-3D-VAE-GAN
