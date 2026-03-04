# DCC 2026 Workshop Notebook Suite

This folder contains the DCC'26 hands-on notebook suite for benchmarking AI methods in engineering design with EngiBench and EngiOpt.

It is split into two tracks:

- `participant/`: notebooks with `TODO` cells for attendees
- `solutions/`: fully completed facilitator notebooks

## Workshop flow (3.5h)

- `participant/00_setup_api_warmup.ipynb` and `solutions/00_setup_api_warmup.ipynb` (10-15 min)
  - Environment setup
  - Problem + dataset inspection
  - Rendering and constraint checks

- `participant/01_train_generate.ipynb` and `solutions/01_train_generate.ipynb` (30 min)
  - Lightweight training using `engiopt.cgan_2d.Generator`
  - Deterministic seeds
  - Fallback path with nearest-neighbor generation

- `participant/02_evaluate_metrics.ipynb` and `solutions/02_evaluate_metrics.ipynb` (20 min)
  - Constraint validation
  - Physics simulation
  - Baseline comparison
  - Metric and artifact export

- `participant/03_add_new_problem_scaffold.ipynb` and `solutions/03_add_new_problem_scaffold.ipynb` (25 min)
  - Minimal `Problem` scaffold
  - Toy simulator and optimization loop
  - Mapping to contribution docs

## Runtime assumptions

- Primary live problem: `Beams2D`
- No container-dependent problems are required during workshop exercises
- W&B integration is optional and disabled by default

## Colab setup

Use the pinned requirements in `requirements-colab.txt`.

All notebooks now include a conditional dependency bootstrap cell:

- On Colab: installs required packages automatically.
- On local envs: skips install by default (`FORCE_INSTALL = False`).
- Note: `engiopt` is not installed from PyPI in these notebooks; workshop execution relies on `engibench` + standard ML libraries.

## Open in Colab

Pre-merge (current branch) links:

- Participant 00: https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/participant/00_setup_api_warmup.ipynb
- Participant 01: https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/participant/01_train_generate.ipynb
- Participant 02: https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/participant/02_evaluate_metrics.ipynb
- Participant 03: https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/participant/03_add_new_problem_scaffold.ipynb
- Solution 00: https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/solutions/00_setup_api_warmup.ipynb
- Solution 01: https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/solutions/01_train_generate.ipynb
- Solution 02: https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/solutions/02_evaluate_metrics.ipynb
- Solution 03: https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/solutions/03_add_new_problem_scaffold.ipynb

## Output artifacts

By default, solution notebooks write generated artifacts to:

- Local/Jupyter: `workshops/dcc26/artifacts/`
- Google Colab runtime: `/content/dcc26_artifacts/` (no auth required)

Optional:

- You can enable W&B artifact upload/download in Notebook 01/02 by setting `USE_WANDB_ARTIFACTS = True`.
- W&B is disabled by default so participants can run without account setup.
- Notebook 02 auto-regenerates Notebook 01-style artifacts if they are missing in a fresh Colab runtime.

These include:

- `generated_designs.npy`
- `baseline_designs.npy`
- `conditions.json`
- `metrics_summary.csv`
- `objective_histogram.png`
- `design_grid.png`

## Facilitator fallback policy

If runtime is constrained:

1. Skip long training in `01_train_generate.ipynb` by enabling fallback mode.
2. Continue directly to `02_evaluate_metrics.ipynb` with fallback-generated designs.
3. Keep `03_add_new_problem_scaffold.ipynb` as the capstone for extensibility.

## Suggested pre-workshop checks

1. Run all notebooks once in fresh Colab runtime.
2. Confirm dataset download succeeds.
3. Confirm artifacts are generated in the expected folder.
4. Confirm no cell requires W&B auth unless explicitly enabled.
