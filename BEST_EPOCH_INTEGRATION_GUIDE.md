# Best epoch selection integration guide

This guide describes the current best-epoch selection flow for `flow_matching_2d_cond`, `diffusion_2d_cond`, and `cgan_cnn_2d`.

The current implementation does not stop training early. It keeps training for the full `--n-epochs` budget, and then loads the best validation checkpoint before the final `--save-model` save.

## What is supported now

- `--enable-best-epoch-selection true`
- `--best-validation-metric` with:
  - `mmd`
  - `iog`
  - `fog`
  - `viol`
- `--validation-batch-size` to control validation sample count
- `--validation-sigma` to control MMD bandwidth
- `--validation-interval-epochs` to control validation frequency
- `--checkpoint-interval-epochs` to control periodic checkpoint saving

## How it works

1. During initialization, the training script samples a fixed validation batch from the dataset test split.
2. Every `validation_interval_epochs`, it generates designs for those sampled conditions.
3. It computes all available validation metrics via `engiopt.metrics.metrics(...)`.
4. It updates `BestEpochTracker` for the chosen `best_validation_metric`.
5. If `--save-model` is used, the script loads the best validation checkpoint before saving the final model.

## Key differences from the old guide

- The code now tracks more metrics than just `mmd`.
- The validation step computes:
  - `mmd`
  - `iog`
  - `fog`
  - `viol`
- `best_validation_metric` chooses which one is used to select the best epoch.
- Checkpoints are saved on validation epochs when best-epoch selection is enabled, even if `checkpoint_interval_epochs == 0`.

## Example Args to use in the script

```python
@dataclass
class Args:
    # existing args...
    enable_best_epoch_selection: bool = True
    best_validation_metric: str = "mmd"
    validation_batch_size: int = 50
    validation_sigma: float = 1.0
    validation_interval_epochs: int = 5
    checkpoint_interval_epochs: int = 5
```

## Example validation flow

```python
if (
    args.enable_best_epoch_selection
    and best_epoch_tracker is not None
    and validation_conditions_tensor is not None
    and validation_sampled_conditions is not None
    and validation_sampled_designs_np is not None
    and (epoch + 1) % args.validation_interval_epochs == 0
):
    model.eval()
    with th.no_grad():
        gen_designs = generate_validation_designs(...)
        gen_designs_np = gen_designs.detach().cpu().numpy().reshape(...)
        gen_designs_np = np.clip(gen_designs_np, 0.0, 1.0)

        metrics_dict = metrics.metrics(
            problem,
            gen_designs_np,
            validation_sampled_designs_np,
            validation_sampled_conditions,
            sigma=args.validation_sigma,
        )
        validation_metric_value = float(metrics_dict[args.best_validation_metric])
    model.train()

    is_best = best_epoch_tracker.update(epoch, validation_metric_value)
```

The training script then logs the full metric set and whether the epoch was best.

## Final model save behavior

At the end of training, if `--save-model` is enabled, the script will:

- look up the best epoch from `BestEpochTracker`
- load the checkpoint saved for that epoch
- save that loaded best model to the final checkpoint path

If no validation checkpoints were recorded, it falls back to the final trained model.

## Model-specific notes

### flow_matching_2d_cond

- Validation uses sampled test-set conditions and optimal designs.
- Best-epoch selection works with `mmd`, `iog`, `fog`, and `viol`.
- Best checkpoint is reloaded before final save.

### diffusion_2d_cond

- Validation uses the diffusion model sampler and test-set conditions.
- It supports all current metrics and best-epoch selection.
- The model will auto-select CUDA if available, even when `--device` is not explicitly passed.

### cgan_cnn_2d

- Validation uses the conditional generator and sampled test-set conditions.
- The best model loader restores both generator and discriminator checkpoints.

### gan_2d

- `gan_2d` was not updated in this patch set and should be avoided if you only want the new best-epoch flow.

## Recommended command setup

For the `flow_matching` profile with only `beams2d` and `heatconduction2d` and 3 seeds:

```bash
PROBLEMS=("beams2d" "heatconduction2d")
#SBATCH --array=0-17
```

Use a submit command like:

```bash
TRAIN_JOB_ID=$(sbatch --parsable --array=0-17 \
  --export=ALL,WORKFLOW_PROFILE=flow_matching,ENGIOPT_2D_PROFILE=flow_matching,\
  WANDB_PROJECT=engiopt-flow-matching-dev,\
  WANDB_ENTITY=feldej-eth-z-rich,\
  N_SEEDS=3,N_EPOCHS=200 \
  run_2d_models.slurm)
```

## Important runtime note

- `--n-epochs` is still a fixed training budget.
- Training does not stop early by default.
- Best-epoch selection only chooses the best checkpoint after training completes.

## If you want early stopping

The current scripts do not implement automatic early stopping.
If you need it, add a separate patience counter around the validation update logic and break from the epoch loop manually.

## Testing checklist

1. Run one model with `--n-epochs 20` and `--validation-interval-epochs 5`.
2. Confirm the script logs `validation/mmd`, `validation/iog`, `validation/fog`, and `validation/viol`.
3. Confirm the best checkpoint path is written under `args.checkpoint_dir`.
4. Confirm the final saved model was reloaded from the best checkpoint before save.

## Summary

This guide now matches the current code:
- multiple validation metrics are supported
- the best epoch is selected by the chosen metric
- the checkpoint save/load flow is handled automatically
- training still runs for the full configured epoch budget

