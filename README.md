[![arXiv](https://img.shields.io/badge/arXiv-2508.00831-b31b1b.svg)](https://arxiv.org/abs/2508.00831)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![code style: Ruff](
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](
    https://github.com/astral-sh/ruff)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/main/example_easy_model.ipynb)

# EngiOpt

This repository contains the code for optimization and machine learning algorithms for engineering design problems. Our goal here is to provide clean example usage of [EngiBench](https://github.com/IDEALLab/EngiBench) and provide strong baselines for future comparisons.

## Coding Philosophy
As much as we can, we follow the [CleanRL](https://github.com/vwxyzjn/cleanrl) philosophy: single-file, high-quality implementations with research-friendly features:
* Single-file implementation: every training detail is in one file, so you can easily understand and modify the code. There is usually another file that contains evaluation code.
* High-quality: we use type hints, docstrings, and comments to make the code easy to understand. We also rely on linters for formatting and checking our code.
* Logging: we use experiment tracking tools like [Weights & Biases](https://wandb.ai/site) to log the results of our experiments. All our "official" runs are logged in the [EngiOpt project](https://wandb.ai/engibench/engiopt).
* Reproducibility: we seed all the random number generators, make PyTorch deterministic, report the hyperparameters and code in WandB.

## Implemented algorithms


**Algorithm** | **Class** | **Dimensions** | **Conditional?** | **Model**
--- | --- | --- | --- | ---
[cgan_1d](engiopt/cgan_1d/) | Inverse Design | 1D | ✅ | GAN MLP
[cgan_2d](engiopt/cgan_2d/) | Inverse Design | 2D | ✅ | GAN MLP
[cgan_bezier](engiopt/cgan_bezier/) | Inverse Design | 1D | ✅ | GAN + Bezier layer
[cgan_cnn_2d](engiopt/cgan_cnn_2d/) | Inverse Design | 2D | ✅ | GAN + CNN
[cgan_cnn_3d](engiopt/cgan_cnn_3d/) | Inverse Design | 3D | ✅ | GAN + 3D CNN
[cgan_vae](engiopt/cgan_vae/) | Inverse Design | 3D | ✅ | MultiView GAN + VAE
[diffusion_1d](engiopt/diffusion_1d/) | Inverse Design | 1D | ❌ | Diffusion
[diffusion_2d_cond](engiopt/diffusion_2d_cond/) | Inverse Design | 2D | ✅ | Diffusion
[gan_1d](engiopt/gan_1d/) | Inverse Design | 1D | ❌ | GAN MLP
[gan_2d](engiopt/gan_2d/) | Inverse Design | 2D | ❌ | GAN MLP
[gan_bezier](engiopt/gan_bezier/) | Inverse Design | 1D | ❌ | GAN + Bezier layer
[gan_cnn_2d](engiopt/gan_cnn_2d/) | Inverse Design | 2D | ❌ | GAN + CNN
[surrogate_model](engiopt/surrogate_model/) | Surrogate Model | 1D | ❌ | MLP
[vqgan](engiopt/vqgan) | Inverse Design | 2D | ✅ | VQVAE + Transformer
[pixel_cnn_pp_2d](engiopt/pixel_cnn_pp_2d) | Inverse Design | 2D | ✅ | PixelCNN++ Autoregressive Model

## Dashboards
The integration with WandB allows us to access live dashboards of our runs (on the cluster or not). New checkpoint packages are stored on the Hugging Face Hub by default, while WandB keeps experiment tracking, metadata, and links back to the canonical checkpoint location. Historical WandB model artifacts remain supported for backward compatibility. You can access some of our runs at https://wandb.ai/engibench/engiopt.
<img src="imgs/wandb_dashboard.png" alt="WandB dashboards"/>


## Install
Install EngiOpt dependencies:
```
cd EngiOpt/
pip install -e .
```

You might want to install a specific PyTorch version, e.g., with CUDA on top of it, see [PyTorch install](https://pytorch.org/get-started/locally/).

If you're modifying EngiBench, you can install it from source and as editable:
```
git clone git@github.com:IDEALLab/EngiBench.git
cd EngiBench/
pip install -e ".[all]"
```

## Running the code

First, if you want to use weights and biases, you need to set the `WANDB_API_KEY` environment variable. You can get your API key from [wandb](https://wandb.ai/site). Then, you can run:
```
wandb login
```

If you want to save or load checkpoints from Hugging Face Hub, make sure your environment is authenticated there as well:
```
huggingface-cli login
```

### Inverse design
Usually, we provide two scripts per algorithm: one to train the model, and one to evaluate it.

To train a model, you can run (for example):

```
python engiopt/cgan_cnn_2d/cgan_cnn_2d.py --problem-id "beams2d" --track --wandb-entity None --save-model --n-epochs 200 --seed 1
```

This will run a CGAN 2D using CNN model on the beams2d problem. `--track` will track the run on wandb, `--wandb-entity None` will use the default wandb entity, `--save-model` will save the model, `--n-epochs 200` will run for 200 epochs, and `--seed 1` will set the random seed.

By default, `--save-model` now stores a self-contained checkpoint package on the Hugging Face Hub. The default backend is:
```
--checkpoint-backend hf
```
You can still force legacy or hybrid behavior when needed:
```
--checkpoint-backend wandb
--checkpoint-backend both
--checkpoint-backend none
```

All HF-backed checkpoint packages contain the model files together with `run_config.json` and `metadata.json`, so evaluation does not depend on live WandB run config state.
When W&B tracking is active, the HF package metadata also records the originating W&B run identity, and the W&B run summary records the HF repo, the seed-based convenience path, the exact uploaded HF revision, and an immutable run-specific HF package path.

For reproducible debugging runs, you can additionally enable strict deterministic mode:
```
python engiopt/cgan_cnn_2d/cgan_cnn_2d.py --problem-id "beams2d" --seed 1 --strict-determinism
```
This enables stricter PyTorch deterministic settings and deterministic data shuffling while keeping the default behavior unchanged when the flag is omitted.

You can always check the help for more options:
```
python engiopt/cgan_cnn_2d/cgan_cnn_2d.py -h
```

There are other available models in the `engiopt/` folder.

Then you can restore a trained model and evaluate it:

```
python engiopt/cgan_cnn_2d/evaluate_cgan_cnn_2d.py --problem-id "beams2d" --wandb-entity None --seed 1 --n-samples 10
```
This will generate 10 designs from the trained model and run some [metrics](https://github.com/IDEALLab/EngiOpt/blob/main/engiopt/metrics.py) on them. This is what we used to generate the results in the paper.

Evaluation now defaults to:
```
--model-source auto
```
In `auto` mode, EngiOpt tries to resolve checkpoints in this order:
1. Hugging Face package for the model family, problem, and seed
2. Legacy WandB model artifact
3. Explicit local checkpoint package directory if you pass `--local-model-dir`

For new HF-backed runs, EngiOpt maintains both:
- a seed-based convenience path such as `beams2d/seed_1`
- an immutable run-specific path such as `beams2d/seed_1/run_<wandb_run_id>`

You can force legacy WandB loading for historical runs:
```
python engiopt/cgan_cnn_2d/evaluate_cgan_cnn_2d.py --problem-id "beams2d" --seed 1 --model-source wandb
```

You can also point evaluation at a local package directory:
```
python engiopt/cgan_cnn_2d/evaluate_cgan_cnn_2d.py --problem-id "beams2d" --seed 1 --model-source local --local-model-dir /path/to/package
```

### Surrogate model

The current surrogate model comprises several steps:
- hyperparameter tuning,
- training a (ensemble) model,
- optimization, and
- evaluation.

See this [notebook](https://github.com/IDEALLab/EngiOpt/blob/main/engiopt/surrogate_model/case_study_pe_notebook.ipynb) for an example.

Surrogate-model optimization paths now use the same checkpoint abstraction. For example, the power-electronics optimizer can consume:
* legacy WandB artifact refs
* HF package refs such as `hf://IDEALLab/engiopt-mlp-tabular-only/power_electronics/DcGain/seed_42`
* local checkpoint package directories

For migration guidance on moving historical checkpoint subsets from WandB to the IDEALLab HF organization later, see [docs/checkpoint_migration_playbook.md](docs/checkpoint_migration_playbook.md).




## Colab notebooks
We have some colab notebooks that show how to use some of the EngiBench/EngiOpt features.
* [Example easy model (GAN)](https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/main/example_easy_model.ipynb)
* [Example hard model (Diffusion)](https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/main/example_hard_model.ipynb)


## Citing

If you use EngiBenc/EngiOpt in your research, please cite the following paper:

```bibtex
@misc{felten_engibench_2025,
	title = {{EngiBench}: {A} {Framework} for {Data}-{Driven} {Engineering} {Design} {Research}},
	url = {http://arxiv.org/abs/2508.00831},
	doi = {10.48550/arXiv.2508.00831},
	urldate = {2025-08-07},
	publisher = {arXiv},
	author = {Felten, Florian and Apaza, Gabriel and B\¨aunlich, Gerhard and Diniz, Cashen and Dong, Xuliang and Drake, Arthur and Habibi, Milad and Hoffman, Nathaniel J. and Keeler, Matthew and Massoudi, Soheyl and VanGessel, Francis G. and Fuge, Mark},
	month = jun,
	year = {2025},
}
```
