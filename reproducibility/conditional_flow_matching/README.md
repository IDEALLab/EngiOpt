# Conditional flow-matching reproduction workflow

This directory records the training, checkpoint-selection, evaluation, reporting, and timing protocol
used for the conditional flow-matching study. It is self-contained and does not depend on the private
cluster-orchestration repository used by the authors.

The committed [`manifest.json`](manifest.json) is a sanitized record of the published campaign. It
contains the exact 220-task mapping, source revisions, model configurations, public artifact revisions,
and audit checksums. New runs create their own manifest under the selected results directory.

## Protocol

- Problems: `beams2d` and `heatconduction2d`.
- Training seeds: 1 through 10.
- Maximum training length: 500 epochs.
- Validation and checkpoint interval: 10 epochs.
- Eligible checkpoints: epoch 80 onward.
- Early stopping: 25 validation checks without lower validation MMD.
- Checkpoint selection: shortlist the five lowest validation-MMD checkpoints, then select the lowest
  validation COG on the same 50 fixed validation pairs.
- Final evaluation: 50 matched condition-reference pairs from the test split. Test results do not enter
  checkpoint selection.
- Diffusion: 1,000 denoising steps.
- cGAN: sigmoid generator output.
- Flow matching: Euler `(16, 32, 48)`, midpoint `(8, 16, 24)`, and RK4 `(4, 8, 12)`.
- Generation timing: one warm-up and three timed repeats on an RTX 4090 with CUDA synchronization.

## Required environment

The launchers require an EngiOpt checkout, an EngiBench checkout, and a Python environment containing
their dependencies:

```bash
export ENGIOPT_DIR=/path/to/EngiOpt
export ENGIBENCH_DIR=/path/to/EngiBench
export VENV_ACTIVATE=/path/to/venv/bin/activate
export RESULTS_ROOT=/path/to/persistent/results
```

For `heatconduction2d`, a prebuilt pyadjoint Apptainer image may be supplied once and shared by every
task:

```bash
export PYADJOINT_SIF=/path/to/pyadjoint_master.sif
```

The scripts write checkpoints and results only below `RESULTS_ROOT`. W&B tracking and Hugging Face
publication are disabled by default.

## Smoke test

The smoke workflow trains and evaluates one `beams2d` seed for flow matching, Diffusion, and cGAN. Its
five short validation checks exercise the complete Top-5 selection path.

```bash
./reproducibility/conditional_flow_matching/launch_smoke.sh
./reproducibility/conditional_flow_matching/launch_smoke.sh --submit
```

The first command is a dry run. It creates and validates a manifest but submits no jobs and uploads
nothing. Inspect the manifest before using `--submit`.

## Full campaign

After the smoke audit reports `"ok": true`:

```bash
export SMOKE_AUDIT="$RESULTS_ROOT/conditional-flow-matching-smoke-v1/audit.json"
./reproducibility/conditional_flow_matching/launch_campaign.sh
./reproducibility/conditional_flow_matching/launch_campaign.sh --submit
```

The default full mapping assigns seeds 1--3 to RTX 4090 GPUs with concurrency 6 and seeds 4--10 to
RTX 3090 GPUs with concurrency 20. Generation timing runs separately on RTX 4090 GPUs with concurrency
16. Override these values with `RTX_4090_SEEDS`, `CONCURRENCY_4090`, `CONCURRENCY_3090`, and
`TIMING_CONCURRENCY`.

The final audit waits for evaluation, reports, synchronized timing, qualitative exports, and any
explicitly enabled checkpoint publication. It verifies every expected CSV and selected bundle before
marking the campaign complete.

## Optional tracking and checkpoint publication

To write to W&B, provide a destination owned by the reproducing user:

```bash
export TRACK_WANDB=1
export WANDB_ENTITY=your-entity
export WANDB_PROJECT=your-project
export WANDB_GROUP=conditional-flow-matching-reproduction-v1
export WANDB_API_KEY_FILE="$HOME/.wandb_key"
```

To publish selected checkpoints, explicitly enable publication and provide a writable HF namespace:

```bash
export PUBLISH_CHECKPOINTS=1
export HF_ENTITY=your-hf-user-or-organization
export HF_REPO_PREFIX=engiopt-conditional-flow-matching
export HF_TOKEN_FILE="$HOME/.huggingface_token"
```

No official IDEALLab or author namespace is used as a default. Training checkpoints remain local until
evaluation selects one checkpoint and stages its evidence bundle. Publication jobs are serialized by
model family to avoid concurrent commits to the same HF repository.

See [`ARTIFACTS.md`](ARTIFACTS.md) for the immutable public records from the published experiments.
