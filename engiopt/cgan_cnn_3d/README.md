# 3D Conditional DCGAN (cgan_cnn_3d)

A 3D Conditional Deep Convolutional Generative Adversarial Network for generating volumetric engineering designs using direct 3D CNN operations.

## Overview

This implementation extends the classic 2D cDCGAN architecture to 3D volumetric data for engineering design generation. Unlike multiview approaches, this model processes entire 3D volumes directly using 3D convolutional operations, making it well-suited for applications requiring full 3D spatial coherence like topology optimization and structural design.

## Architecture

### Components
- **Generator3D**: 3D transposed CNN that creates volumetric designs from noise + conditions
- **Discriminator3D**: 3D CNN that classifies real vs. generated volumes with condition awareness
- **Direct 3D Processing**: No slice-based approximations - full volumetric operations

### Key Features
- **Native 3D Operations**: Uses 3D convolutions throughout for true volumetric learning
- **Conditional Generation**: Engineering constraints guide the generation process
- **WGAN-GP Training**: Wasserstein distance with gradient penalty for stable training

## Quick Start

### Basic Training
```bash
# Train with default parameters
python cgan_cnn_3d.py --problem_id heatconduction3d --n_epochs 300

# Save model for later evaluation
python cgan_cnn_3d.py --save_model --track
```

### Evaluation
```bash
# Evaluate trained model using EngiBench metrics
python evaluate_cgan_cnn_3d.py --seed 1 --n_samples 50

# Batch evaluation across multiple seeds
for seed in {1..5}; do
    python evaluate_cgan_cnn_3d.py --seed $seed --n_samples 100
done
```

### Hyperparameter Sweep
```bash
# Run Bayesian optimization sweep
wandb sweep sweep_cgan_cnn_3d.yaml
wandb agent <sweep_id>
```
!! Change the path of cgan_cnn_3d.py to your project path

## Key Parameters

### Core Training
- `--n_epochs 300`: Number of training epochs
- `--batch_size 8`: Batch size
- `--lr_gen 0.0025`: Generator learning rate
- `--lr_disc 10e-5`: Discriminator learning rate
- `--lr_enc 0.001`: Encoder learning rate
- `--latent_dim 64`: Dimensionality of latent space

### Training Dynamics
- `--gen_iters 1`: Generator updates per batch
- `--discrim_iters 1`: Discriminator updates per batch


### References
Original DCGAN: https://arxiv.org/abs/1511.06434
Conditional GAN: https://arxiv.org/abs/1411.1784
WGAN-GP: https://arxiv.org/abs/1704.00028
