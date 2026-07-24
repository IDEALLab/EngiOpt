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
[cgan_1d](engiopt/generators/cgan_1d/) | Inverse Design | 1D | ✅ | GAN MLP
[cgan_2d](engiopt/generators/cgan_2d/) | Inverse Design | 2D | ✅ | GAN MLP
[cgan_bezier](engiopt/generators/cgan_bezier/) | Inverse Design | 1D | ✅ | GAN + Bezier layer
[cgan_cnn_2d](engiopt/generators/cgan_cnn_2d/) | Inverse Design | 2D | ✅ | GAN + CNN
[cgan_cnn_3d](engiopt/generators/cgan_cnn_3d/) | Inverse Design | 3D | ✅ | GAN + 3D CNN
[cgan_vae](engiopt/generators/cgan_vae/) | Inverse Design | 3D | ✅ | MultiView GAN + VAE
[diffusion_1d](engiopt/generators/diffusion_1d/) | Inverse Design | 1D | ❌ | Diffusion
[diffusion_2d_cond](engiopt/generators/diffusion_2d_cond/) | Inverse Design | 2D | ✅ | Diffusion
[gan_1d](engiopt/generators/gan_1d/) | Inverse Design | 1D | ❌ | GAN MLP
[gan_2d](engiopt/generators/gan_2d/) | Inverse Design | 2D | ❌ | GAN MLP
[gan_bezier](engiopt/generators/gan_bezier/) | Inverse Design | 1D | ❌ | GAN + Bezier layer
[gan_cnn_2d](engiopt/generators/gan_cnn_2d/) | Inverse Design | 2D | ❌ | GAN + CNN
[surrogate_model](engiopt/surrogate_model/) | Surrogate Model | 1D | ❌ | MLP
[vqgan](engiopt/generators/vqgan) | Inverse Design | 2D | ✅ | VQVAE + Transformer
[pixel_cnn_pp_2d](engiopt/generators/pixel_cnn_pp_2d) | Inverse Design | 2D | ✅ | PixelCNN++ Autoregressive Model

## Dashboards
HuggingFace hosts everything that has to be reloaded or compared -- model weights, run configs, evaluation metrics, and the leaderboard. WandB hosts what you only look at: loss curves and sample images. Nothing in the evaluation path requires WandB, so training with `--track false` produces exactly the same checkpoints and scores. You can access some of our runs at https://wandb.ai/engibench/engiopt.
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
python engiopt/generators/cgan_cnn_2d/cgan_cnn_2d.py --problem-id "beams2d" --track --wandb-entity None --save-model --n-epochs 200 --seed 1
```

This trains a CGAN 2D w/ CNN on `beams2d`. The flags mirror W&B's: `--track` enables W&B logging, `--wandb-entity`/`--wandb-project` say where the run goes, `--save-model` uploads the checkpoint to HuggingFace, and `--hf-entity`/`--hf-repo-prefix` say where the checkpoint goes.

W&B holds media, scalars, and run history. HuggingFace holds the model weights. One log-in per service:

```
wandb login              # for tracking
huggingface-cli login    # for checkpoints (or: export HF_TOKEN=...)
```

The defaults (`--hf-entity IDEALLab --hf-repo-prefix engiopt`) push to `huggingface.co/IDEALLab/engiopt-cgan-cnn-2d/beams2d/cfg_<fingerprint>/seed_1/`, one location per hyperparameter configuration. A run using the script's default hyperparameters additionally claims `beams2d/seed_1/`, which is what the bare model name resolves to. The W&B run summary records the HF path for traceability. Each checkpoint package contains the model files plus `run_config.json` and `metadata.json`, so evaluation needs no live W&B state.

For reproducible debugging runs, you can additionally enable strict deterministic mode:
```
python engiopt/generators/cgan_cnn_2d/cgan_cnn_2d.py --problem-id "beams2d" --seed 1 --strict-determinism
```

For new cGAN density-field runs, you can emit designs natively in the EngiBench `[0, 1]` density range while preserving older `tanh` checkpoint behavior by default:
```
python engiopt/generators/cgan_cnn_2d/cgan_cnn_2d.py --problem-id "beams2d" --generator-output-activation sigmoid
```

Then evaluate:
```
python -m engiopt.evaluate --problem-id "beams2d" --generators cgan_cnn_2d --seeds 1
```
Evaluation pulls the checkpoint from HF automatically. Pass `--hf-entity` / `--hf-repo-prefix` to point at a different HF repo, and `--config-fingerprints` to score specific hyperparameter configurations instead of the default one.

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
