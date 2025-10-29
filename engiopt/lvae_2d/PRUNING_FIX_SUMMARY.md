# D-LVAE Pruning Fix: Loss Weight Rebalancing Approach

## Problem Summary

**Observed Issue:**
- Even with `perf_dim=1` or `perf_dim=0`, almost all dimensions (close to 250) were preserved
- Regular `LeastVolumeAE_DynamicPruning` prunes aggressively with same hyperparameters
- This indicates performance loss prevents proper pruning

**Root Cause:**
The performance loss (weight = 1.0) was too strong relative to reconstruction/volume losses. Even though only the first `perf_dim` dimensions are directly used for performance prediction, gradients flow back through the shared encoder architecture, causing performance-relevant features to spread across all dimensions. This prevents the volume loss from shrinking "unimportant" dimensions.

## Solution Implemented: Stage 1 (Loss Rebalancing)

### Changes Made

1. **Reverted gradient masking commit (d63bfcf)**
   - Previous attempt detached `z_recon` from reconstruction loss
   - This broke reconstruction learning for dimensions beyond `perf_dim`
   - Reverted to original working implementation

2. **Added configurable `w_p` parameter**
   - Modified `InterpretableDesignLeastVolumeAE_DP.__init__` to accept `w_r`, `w_p`, `w_v` parameters
   - Default changed from `w_p=1.0` to `w_p=0.1` (10x reduction)
   - Updated `d_lvae_2d.py` Args class to match

3. **Created new sweep configuration**
   - `sweep_d_lvae_2d_weight_tuning.yaml`: Tests `w_p ∈ [0.01, 0.05, 0.1, 0.5]`
   - Tests `perf_dim ∈ [1, 10]` with `percentile ∈ [0.05, 0.10]`
   - Single seed for fast iteration
   - Updated original sweep to use `w_p=0.1` default

### Rationale

By reducing the performance loss weight:
- Volume loss has more influence to shrink low-variance dimensions
- Encoder training is dominated by reconstruction (primary objective)
- Performance prediction becomes secondary guidance rather than dominant signal
- Dimensions beyond `perf_dim` can be pruned if they only encode low-variance details

### Expected Behavior

With reduced `w_p`:
- First `perf_dim` dimensions still learn performance (due to predictor)
- But performance signal is weaker, allowing pruning to work
- Trade-off: May slightly reduce performance prediction accuracy
- Goal: Find `w_p` that balances pruning and prediction quality

## Testing Strategy

### Quick Test Sweep
```bash
wandb sweep engiopt/lvae_2d/sweep_d_lvae_2d_weight_tuning.yaml
wandb agent <sweep_id>
```

**Metrics to monitor:**
- Number of active dimensions at end of training
- Reconstruction MSE (should remain good)
- Performance prediction MSE (acceptable to be slightly higher)
- Variance per dimension (verify pruning behavior)

**Success criteria for Stage 1:**
- With `perf_dim=1`: Achieve 5-30 active dimensions (vs current ~250)
- With `perf_dim=10`: Achieve 15-50 active dimensions
- Reconstruction quality comparable to regular LVAE
- Performance prediction R² > 0.7 (acceptable trade-off)

### Analysis Questions

1. **Does reducing `w_p` enable pruning?**
   - Compare active dims across different `w_p` values
   - Check if `w_p=0.01` prunes more aggressively than `w_p=0.5`

2. **What's the performance prediction quality trade-off?**
   - Plot performance MSE vs `w_p`
   - Find minimum `w_p` that maintains acceptable prediction quality

3. **Does more aggressive pruning help?**
   - Compare `percentile=0.05` vs `percentile=0.10`
   - Check if higher percentile enables more pruning

## Stage 2: If Loss Rebalancing Fails

If `w_p` reduction doesn't enable sufficient pruning (even with `w_p=0.01`), we'll need architectural changes:

### Split-Head Encoder Architecture

**Concept:** Separate the encoder output into two paths:
```python
class SplitHeadEncoder(nn.Module):
    def __init__(self, perf_dim, recon_dim):
        self.shared_features = nn.Sequential(...)  # Conv layers
        self.to_perf = nn.Conv2d(512, perf_dim, ...)     # Performance path
        self.to_recon = nn.Conv2d(512, recon_dim, ...)   # Reconstruction path

    def forward(self, x):
        h = self.shared_features(x)
        h_for_perf = h  # Gets both perf and recon gradients
        h_for_recon = h.detach()  # Only gets recon gradients

        z_perf = self.to_perf(h_for_perf).flatten(1)
        z_recon = self.to_recon(h_for_recon).flatten(1)
        return torch.cat([z_perf, z_recon], dim=1)
```

**Benefits:**
- Architecturally separates performance and reconstruction pathways
- Performance loss cannot update `to_recon` weights
- Guarantees pruning can work on `z_recon` dimensions

**Trade-offs:**
- More complex architecture
- `z_recon` cannot benefit from performance-relevant features
- May slightly hurt reconstruction quality

**When to implement:** Only if Stage 1 shows insufficient pruning across all tested `w_p` values.

## Implementation Timeline

### Completed ✓
1. Reverted broken gradient masking
2. Added `w_p` parameter with new default (0.1)
3. Created weight tuning sweep config
4. Updated documentation

### Next Steps
1. **Run weight tuning sweep** on heatconduction2d problem
2. **Analyze results** to find optimal `w_p`
3. **If successful:** Run full sweep with best `w_p` on multiple problems
4. **If unsuccessful:** Implement Stage 2 (split-head architecture)

## Files Modified

- `engiopt/lvae_2d/aes.py`: Added `w_r`, `w_p`, `w_v` parameters to `InterpretableDesignLeastVolumeAE_DP`
- `engiopt/lvae_2d/d_lvae_2d.py`: Changed default `w_p=1.0` → `w_p=0.1`
- `engiopt/lvae_2d/sweep_d_lvae_2d_weight_tuning.yaml`: New sweep for testing `w_p` values
- `engiopt/lvae_2d/sweep_d_lvae_2d_lognorm_uncond.yaml`: Updated to use `w_p=0.1`

## Key Commits

- `241ba00`: Revert gradient masking implementation
- `57c2746`: Add configurable w_p parameter for performance loss weight
- `f475b54`: Add sweep config for w_p tuning and update default sweep

## Notes

- Performance loss already only uses `z[:, :perf_dim]` (no direct gradient to `z_recon`)
- Issue is **indirect** gradients through shared encoder affecting all dimensions
- Interpretability requirement: First `perf_dim` dimensions MUST encode performance
- This rules out approaches that detach encoder from performance loss entirely
