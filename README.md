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
[diffusion_1d](engiopt/diffusion_1d/) | Inverse Design | 1D | ❌ | Diffusion
[diffusion_2d_cond](engiopt/diffusion_2d_cond/) | Inverse Design | 2D | ✅ | Diffusion
[gan_1d](engiopt/gan_1d/) | Inverse Design | 1D | ❌ | GAN MLP
[gan_2d](engiopt/gan_2d/) | Inverse Design | 2D | ❌ | GAN MLP
[gan_bezier](engiopt/gan_bezier/) | Inverse Design | 1D | ❌ | GAN + Bezier layer
[gan_cnn_2d](engiopt/gan_cnn_2d/) | Inverse Design | 2D | ❌ | GAN + CNN
[surrogate_model](engiopt/surrogate_model/) | Surrogate Model | 1D | ❌ | MLP

## Dashboards
The integration with WandB allows us to access live dashboards of our runs (on the cluster or not). We also upload the trained models there. You can access some of our runs at https://wandb.ai/engibench/engiopt.
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

### Inverse design
Usually, we provide two scripts per algorithm: one to train the model, and one to evaluate it.

To train a model, you can run (for example):

```
python engiopt/cgan_cnn_2d/cgan_cnn_2d.py --problem-id "beams2d" --track --wandb-entity None --save-model --n-epochs 200 --seed 1
```

This will run a CGAN 2D using CNN model on the beams2d problem. `--track` will track the run on wandb, `--wandb-entity None` will use the default wandb entity, `--save-model` will save the model, `--n-epochs 200` will run for 200 epochs, and `--seed 1` will set the random seed.

You can always check the help for more options:
```
python engiopt/cgan_cnn_2d/cgan_cnn_2d.py -h
```

There are other available models in the `engiopt/` folder.

Then you can restore a trained model and evaluate it:

```
python engiopt/cgan_cnn_2d/evaluate_cgan_cnn_2d.py --problem-id "beams2d" --wandb-entity None --seed 1 --n-samples 10
```
This will generate 10 designs from the trained model and run some [metrics](https://github.com/IDEALLab/EngiOpt/blob/main/engiopt/metrics.py) on them. This is what we used to generate the results in the paper. This by default will pull the model from wandb. It is possible to restore a model from a local file but is not currently supported.

### Surrogate model

The current surrogate model comprises several steps:
- hyperparameter tuning,
- training a (ensemble) model,
- optimization, and
- evaluation.

See this [notebook](https://github.com/IDEALLab/EngiOpt/blob/main/engiopt/surrogate_model/case_study_pe_notebook.ipynb) for an example.




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
