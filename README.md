# EngiLearn

This repository contains the code for optimization and machine learning algorithms for engineering design problems. Our goal here is to provide clean example usage of EngiBench and provide strong baselines for future comparisons.

## Coding Philosophy
As much as we can, we follow the [CleanRL](https://github.com/vwxyzjn/cleanrl) philosophy: single-file, high-quality implementations with research-friendly features:
* Single-file implementation: every detail is in one file, so you can easily understand and modify the code.
* High-quality: we use type hints, docstrings, and comments to make the code easy to understand.
* Logging: we use experiment tracking tools like [Weights & Biases](https://wandb.ai/site) to log the results of our experiments.
* Reproducibility: we pin all the versions used in the experiment to ensure reproducibility. We also seed all the random number generators.


## Wishlist
- [x] GAN (vector)
- [x] CGAN (vector)
- [x] Diffusion (vector)
- [ ] BezierGAN, DCGAN (images)
- [ ] Diffusion Models (images)
- [ ] Surrogate Models and Optimization
- [ ] Metrics implementations
