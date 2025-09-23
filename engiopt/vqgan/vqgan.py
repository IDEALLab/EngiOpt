"""Vector Quantized Generative Adversarial Network (VQGAN).

Based on https://github.com/dome272/VQGAN-pytorch.

VQGAN is composed of two primary Stages:
    - Stage 1 is similar to an autoencoder (AE) but with a discrete latent space represented by a codebook.
    - Stage 2 is a generative model (a transformer in this case) trained on the latent space of Stage 1.

The transformer now uses nanoGPT (https://github.com/karpathy/nanoGPT) instead of minGPT (https://github.com/karpathy/minGPT) as in the original implementation.

For Stage 2, we take the indices of the codebook vectors and flatten them into a 1D sequence, treating them as training tokens.
The transformer is then trained to autoregressively predict each token in the sequence, after which it is reshaped back to the original 2D latent space and passed through the decoder of Stage 1 to generate an image.
To make VQGAN conditional, we train a separate VQGAN on the conditions only (CVQGAN) and replace the start-of-sequence tokens of the transformer with the CVQGAN latent tokens.

We have updated the transformer architecture, converted VQGAN from a two-stage to a single-stage approach, added several new arguments, switched to wandb for logging, added greyscale support to the perceptual loss, and more.
"""
