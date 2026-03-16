# DCC 2026 Workshop Notebook Suite

This folder contains the DCC'26 hands-on notebook suite for benchmarking AI methods in engineering design with EngiBench and EngiOpt.

It is split into two tracks:

- `participant/`: notebooks with guided `PUBLIC FILL-IN` cells for attendees
- `solutions/`: fully completed facilitator notebooks

## Workshop flow (3.5h)

- `participant/00_setup_api_warmup.ipynb` and `solutions/00_setup_api_warmup.ipynb` (10-15 min)
  - Environment setup
  - Problem + dataset inspection
  - Rendering and constraint checks

- `participant/01_train_generate.ipynb` and `solutions/01_train_generate.ipynb` (30 min)
  - Lightweight training using `engiopt.cgan_2d.Generator`
  - Deterministic seeds
  - Artifact export for downstream evaluation (runtime/W&B optional transport)

- `participant/02_evaluate_metrics.ipynb` and `solutions/02_evaluate_metrics.ipynb` (20 min)
  - Constraint validation
  - Physics simulation
  - Baseline comparison
  - Metric and artifact export

- `participant/03_add_new_problem_scaffold.ipynb` and `solutions/03_add_new_problem_scaffold.ipynb` (25 min)
  - Ambitious `Problem` scaffold (`PlanarManipulatorCoDesignProblem`, not currently in EngiBench)
  - PyBullet-based robotics co-design simulation and optimization loop
  - Mapping to contribution docs

## Runtime assumptions

- Primary live problem: `Beams2D`
- No container-dependent problems are required during workshop exercises
- W&B integration is optional and disabled by default

## Colab setup

Use `requirements-colab.txt` only as a local convenience snapshot.
The notebook bootstrap cells are the runtime source of truth for Colab.

All notebooks now include a conditional dependency bootstrap cell:

- On Colab: installs required packages automatically.
- On local envs: skips install by default (`FORCE_INSTALL = False`).
- Note: notebooks that use EngiOpt install it from the EngiOpt GitHub branch bootstrap.

## Open in Colab

Use these `?copy=true` links for workshop sharing so attendees are prompted to create their own Drive copy first.

- Participant 00: https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/participant/00_setup_api_warmup.ipynb?copy=true
- Participant 01: https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/participant/01_train_generate.ipynb?copy=true
- Participant 02: https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/participant/02_evaluate_metrics.ipynb?copy=true
- Participant 03: https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/participant/03_add_new_problem_scaffold.ipynb?copy=true
- Solution 00: https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/solutions/00_setup_api_warmup.ipynb?copy=true
- Solution 01: https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/solutions/01_train_generate.ipynb?copy=true
- Solution 02: https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/solutions/02_evaluate_metrics.ipynb?copy=true
- Solution 03: https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/dcc26-workshop-notebooks/workshops/dcc26/solutions/03_add_new_problem_scaffold.ipynb?copy=true

## Output artifacts

By default, solution notebooks write generated artifacts to:

- Local/Jupyter: `workshops/dcc26/artifacts/`
- Google Colab runtime: `/content/dcc26_artifacts/` (no Google Drive permission needed)

Optional:

- You can enable W&B artifact upload/download in Notebook 01/02 by setting `USE_WANDB_ARTIFACTS = True`.
- Notebook 01 logs training dynamics (`train/loss`) and can upload checkpoint/history/plots as artifact payload.
- Notebook 02 can log evaluation metrics, tables, and figures to W&B.
- W&B is disabled by default so participants can run without account setup or API keys.
- Notebook 02 auto-builds Notebook 01-style artifacts locally with EngiOpt if they are missing (`AUTO_BUILD_ARTIFACTS_IF_MISSING = True`).

These include:

- `generated_designs.npy`
- `baseline_designs.npy`
- `conditions.json`
- `engiopt_cgan2d_generator_supervised.pt`
- `training_history.csv`
- `training_curve.png`
- `metrics_summary.csv`
- `objective_histogram.png`
- `objective_scatter.png`
- `design_grid.png`

## Facilitator fallback policy

If runtime is constrained:

1. Skip Notebook 01 and run `02_evaluate_metrics.ipynb`; it can build required artifacts automatically.
2. Or reuse a previously saved checkpoint/artifact set from W&B or local runtime files.
3. Keep `03_add_new_problem_scaffold.ipynb` as the capstone for extensibility.

## Suggested pre-workshop checks

1. Run all notebooks once in fresh Colab runtime.
2. Confirm dataset download succeeds.
3. Confirm artifacts are generated in the expected folder.
4. Confirm no cell requires W&B auth unless explicitly enabled.
