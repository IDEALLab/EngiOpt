# DCC 2026 Workshop Notebook Suite

This folder contains the DCC'26 hands-on notebook suite for benchmarking AI methods in engineering design with EngiBench and EngiOpt.

## Workshop flow (3.5h)

- `00_setup_api_warmup.ipynb` (10-15 min)
  - Environment setup
  - Problem + dataset inspection
  - Rendering and constraint checks

- `01_train_generate.ipynb` (30 min)
  - Lightweight conditional generator training
  - Deterministic seeds
  - Fallback path with nearest-neighbor generation

- `02_evaluate_metrics.ipynb` (20 min)
  - Constraint validation
  - Physics simulation
  - Baseline comparison
  - Metric and artifact export

- `03_add_new_problem_scaffold.ipynb` (25 min)
  - Minimal `Problem` scaffold
  - Toy simulator and optimization loop
  - Mapping to contribution docs

## Runtime assumptions

- Primary live problem: `Beams2D`
- No container-dependent problems are required during workshop exercises
- W&B integration is optional and disabled by default

## Colab setup

Use the pinned requirements in `requirements-colab.txt`.

## Output artifacts

By default, notebooks write generated artifacts to:

- `workshops/dcc26/artifacts/`

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
