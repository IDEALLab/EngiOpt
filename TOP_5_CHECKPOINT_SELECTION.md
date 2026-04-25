# Top-5 Checkpoint Auto-Selection

## Overview

The training scripts now automatically track and save the **5 best validation checkpoints** by lowest MMD score, eliminating manual selection. At the end of training, the script prints which 5 epochs were selected.

## What Changed

### 1. Enhanced `BestEpochTracker` class
**File**: `engiopt/best_epoch_selection.py`

Added:
- `top_k` parameter (default: 5) to track multiple best checkpoints
- `get_top_k_epochs()` - Returns list of top-K epochs with their MMD values
- `get_top_k_checkpoint_paths()` - Returns file paths to top-K checkpoints
- `print_top_k_summary()` - Prints a clean summary of selected epochs
- Updated `_save_metrics()` to include top-K info in `validation_metrics.json`

### 2. Updated Training Scripts

**Files Modified**:
- `engiopt/flow_matching_2d_cond/flow_matching_2d_cond.py`
- `engiopt/diffusion_2d_cond/diffusion_2d_cond.py`
- `engiopt/cgan_cnn_2d/cgan_cnn_2d.py`

Changes:
- Pass `top_k=5` when initializing `BestEpochTracker`
- At end of training, call `print_top_k_summary()` to report selected epochs

## Usage

No additional flags needed! Training works exactly the same, but now you get automatic top-5 tracking.

### Example Output

At the end of training, you'll see:

```
🎯 Top-5 Validation Checkpoints (Best MMD):
  1. Epoch 150: mmd=0.0124567890 (epoch_0150.pth)
  2. Epoch 140: mmd=0.0125123456 (epoch_0140.pth)
  3. Epoch 160: mmd=0.0126789012 (epoch_0160.pth)
  4. Epoch 130: mmd=0.0127345678 (epoch_0130.pth)
  5. Epoch 170: mmd=0.0128901234 (epoch_0170.pth)
```

## Next Steps: Using Top-5 Checkpoints

The workflow is now:

1. **Train** (`flow_matching_2d_cond.py` etc.)
   - Validates every 10 epochs (after warm start)
   - Saves validation checkpoints
   - Prints top-5 at end

2. **Evaluate top-5 on Validation COG/FOG** (next script to implement)
   - Load each of the 5 best checkpoints
   - Generate samples on validation split
   - Compute full `metrics.metrics()` including COG/FOG
   - Rank by COG (primary), FOG (secondary)

3. **Test Final Checkpoint** (post-evaluation)
   - Load selected checkpoint
   - Generate samples on test split
   - Compute final metrics

## Data Files

The `validation_metrics.json` now includes:
```json
{
  "best_epoch": 150,
  "best_metric_value": 0.0124567890,
  "top_k": 5,
  "top_k_epochs": [
    {"epoch": 150, "metric_value": 0.0124567890},
    {"epoch": 140, "metric_value": 0.0125123456},
    ...
  ],
  "all_metrics": {...}
}
```

This allows you to programmatically access the top-5 for automated evaluation.
