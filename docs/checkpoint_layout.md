# Checkpoint layout

HuggingFace is the single home for anything that has to be reloaded or compared:
model weights, the run config needed to rebuild them, evaluation metrics, and
the leaderboard. Weights & Biases, when enabled, hosts what you only look at:
loss curves and sample images. Nothing in the evaluation path requires W&B.

## Where a checkpoint goes

One repo per model family, named from the algorithm:

```
{hf_entity}/{hf_repo_prefix}-{algo}        e.g. IDEALLab/engiopt-cgan-cnn-2d
```

Inside it, one package per hyperparameter configuration and seed:

```
{problem_id}/cfg_{fingerprint}/seed_{seed}/     every configuration, always written
{problem_id}/seed_{seed}/                       the canonical default, see below
```

`fingerprint` is a short hash of the run config with infrastructure settings
(seed, tracking flags, HF/W&B destinations) excluded, so a sweep of fifty
configurations produces fifty independently addressable packages rather than
overwriting one. Computed by `engiopt.core.config_fingerprint`.

The canonical `{problem_id}/seed_{seed}` path is what a bare model name resolves
to. It is written **only** when a run used the training script's default
hyperparameters, so a sweep can never redefine what `cgan_cnn_2d` means.

Each package contains the model files plus:

| file | contents |
|---|---|
| `run_config.json` | the training script's `Args`, enough to rebuild the model |
| `metadata.json` | problem, algo, seed, fingerprint, primary files, W&B run URL |
| `metrics.json` | evaluation scores, when `--attach-metrics` was used |

## Writing one

Training scripts call `save_checkpoint_package` and splat
`engiopt.core.checkpoint_identity(args)`, which supplies the fingerprint and
whether the run used default hyperparameters:

```python
save_checkpoint_package(
    checkpoint_backend="hf",
    hf_entity=args.hf_entity,
    hf_repo_prefix=args.hf_repo_prefix,
    hf_private=False,
    problem_id=args.problem_id,
    algo=args.algo,
    seed=args.seed,
    checkpoint_files={"generator.pth": "generator.pth"},
    run_config=vars(args),
    **checkpoint_identity(args),
)
```

## Reading one

```python
# the default-hyperparameter checkpoint -- what the bare model name means
gen = BUILTIN_GENERATORS["cgan_cnn_2d"].from_pretrained(problem, problem_id="beams2d", seed=1)

# one specific configuration from a sweep
gen = BUILTIN_GENERATORS["cgan_cnn_2d"].from_pretrained(
    problem, problem_id="beams2d", seed=1, config_fingerprint="3ab84748"
)
```

## The leaderboard

Published as a HuggingFace **dataset** repo holding `leaderboard.csv`, one row
per `(problem_id, algo_id, config_fingerprint, seed, spec_version)`. Pushing
merges into the existing board rather than replacing it, so adding one model
never recomputes or disturbs anyone else's rows:

```bash
python -m engiopt.evaluate --problem-id beams2d --generators my_model \
    --push-to IDEALLab/engiopt-leaderboard --attach-metrics
```
