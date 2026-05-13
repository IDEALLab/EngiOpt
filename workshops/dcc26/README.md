# DCC 2026 Workshop Notebook Suite

This folder contains the DCC'26 workshop notebooks for benchmarking AI methods in engineering design with EngiBench and EngiOpt.

## Start Here: Workshop Notebooks

Use these notebooks during the live workshop. They are guided, narrative notebooks: run cells top to bottom, read the short explanations, and discuss the prompts.

Open notebooks with the `?copy=true` Colab links below. Colab will prompt you to create your own copy before editing, so your changes do not write back to the EngiOpt repository.

| Step | Notebook | Colab |
|---|---|---|
| 00 | Frame an engineering design problem as a benchmark | [Open Simple 00](https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/simple/00_framing_your_design_problem.ipynb?copy=true) |
| 01 | Train a lightweight conditional generator | [Open Simple 01](https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/simple/01_training_a_generative_model.ipynb?copy=true) |
| 02 | Evaluate generated designs with benchmark methods | [Open Simple 02](https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/simple/02_evaluating_your_generated_designs.ipynb?copy=true) |
| 03 | Write a minimal new EngiBench-style problem | [Open Simple 03](https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/simple/03_writing_your_own_problem.ipynb?copy=true) |

Recommended live flow:

1. Run Simple 00 to understand the benchmark contract: design variables, conditions, objectives, constraints, rendering, simulation, and baseline optimization.
2. Run Simple 01 to train a small EngiOpt generator and export generated designs.
3. Run Simple 02 to evaluate visual quality, feasibility, simulation performance, diversity, and warmstarting. If artifacts from Simple 01 are missing, Simple 02 rebuilds them automatically.
4. Run Simple 03 as the capstone for adding a new benchmark problem.

## Optional Extra Exercises

The `participant/` notebooks are more hands-on. They contain `PUBLIC FILL-IN` cells, checkpoints, and deeper metric or implementation exercises. Use these as homework, breakout exercises, or follow-up material after the live workshop.

| Step | Exercise notebook | Colab | Solution |
|---|---|---|---|
| 00 | API warmup with fill-ins | [Participant 00](https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/participant/00_setup_api_warmup.ipynb?copy=true) | [Solution 00](https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/solutions/00_setup_api_warmup.ipynb?copy=true) |
| 01 | Train and generate with fill-ins | [Participant 01](https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/participant/01_train_generate.ipynb?copy=true) | [Solution 01](https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/solutions/01_train_generate.ipynb?copy=true) |
| 02 | Full evaluation metrics exercise | [Participant 02](https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/participant/02_evaluate_metrics.ipynb?copy=true) | [Solution 02](https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/solutions/02_evaluate_metrics.ipynb?copy=true) |
| 03 | Ambitious PyBullet co-design scaffold | [Participant 03](https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/participant/03_add_new_problem_scaffold.ipynb?copy=true) | [Solution 03](https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/solutions/03_add_new_problem_scaffold.ipynb?copy=true) |
| 04 | Heat-exchanger physics-wrapper exercise | [Participant 04](https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/participant/04_heat_exchanger_design_problem.ipynb?copy=true) | [Solution 04](https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/solutions/04_heat_exchanger_design_problem.ipynb?copy=true) |

## What Each Track Is For

- `simple/`: main live-workshop notebooks. Best for participants and first-time readers.
- `participant/`: extra exercises with fill-in cells. Best for homework, small-group work, or deeper practice.
- `solutions/`: completed versions of the extra exercises. Best for facilitators or participants who get stuck.

## Runtime Assumptions

- Primary live problem: `Beams2D`
- No container-dependent problems are required during the main workshop notebooks.
- W&B integration is optional and disabled by default in the exercise track.
- Notebook bootstrap cells install dependencies automatically on Colab.
- On local environments, install cells skip by default unless `FORCE_INSTALL = True`.

## Artifact Flow

Simple 01 writes generated artifacts to:

- Google Colab runtime: `/content/dcc26_artifacts/`
- Local/Jupyter: `workshops/dcc26/artifacts/`

Simple 02 reads those artifacts. If they are missing, it rebuilds the same lightweight Simple 01 train/generate/export path automatically, then continues evaluation.

The key artifacts are:

- `generated_designs.npy`
- `baseline_designs.npy`
- `conditions.json`

The deeper participant/solution evaluation notebooks may additionally write:

- `engiopt_cgan2d_generator_supervised.pt`
- `training_history.csv`
- `training_curve.png`
- `metrics_summary.csv`
- `objective_histogram.png`
- `objective_scatter.png`
- `design_grid.png`

## Suggested Pre-Workshop Checks

1. Open each Simple Colab link in a fresh browser/session.
2. Confirm the install cell succeeds.
3. Run Simple 00 once end-to-end.
4. Run Simple 01 through the artifact export cell.
5. Open Simple 02 in a fresh runtime and confirm the artifact rebuild path works.
6. Run Simple 03 at least through the `Problem` class smoke test.
