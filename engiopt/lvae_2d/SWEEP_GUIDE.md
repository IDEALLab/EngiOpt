# LVAE Sweep Experimental Guide

This guide outlines the scientific approach to understanding LVAE pruning methods and spectral normalization.

## Experimental Philosophy

**Goal**: Isolate and understand the effects of:
1. Pruning method hyperparameters (plummet vs lognorm)
2. Spectral normalization (on/off, future: tunable Lipschitz bound)
3. Safeguards (tertiary concern, only after understanding 1 & 2)

**Approach**: No safeguards initially. We want to see failures, edge cases, and bad runs to truly understand the parameter space.

## Available Sweep Configurations

### 1. `sweep_lvae_2d.yaml` - Grid Search (Recommended First)
**Purpose**: Complete coverage of pruning hyperparameters × spectral norm combinations

**What it tests**:
- Both plummet and lognorm strategies
- Wide range of method-specific hyperparameters:
  - Plummet: thresholds from 0.005 to 0.2
  - Lognorm: alpha (0.0-0.5) × percentile (0.01-0.2)
- Spectral norm on/off
- 3 seeds for reliability

**Pros**:
- Comprehensive view of parameter space
- Easy to visualize interactions
- See all failure modes

**Cons**:
- Large number of runs (for plummet: 6 thresholds × 2 spec_norm × 3 seeds = 36 runs per problem)
- (for lognorm: 5 alphas × 6 percentiles × 2 spec_norm × 3 seeds = 180 runs per problem)

**Usage**:
```bash
# For each problem
wandb sweep engiopt/lvae_2d/sweep_lvae_2d.yaml
# Edit problem_id in yaml or override with --problem-id
wandb agent <sweep-id>
```

### 2. `sweep_lvae_2d_bayes.yaml` - Bayesian Optimization
**Purpose**: Efficient exploration after initial grid search

**What it tests**:
- One pruning method at a time (change `pruning_strategy` value)
- Continuous hyperparameter ranges
- Also explores lr, w_v, eta, latent_dim sensitivity

**Pros**:
- Efficient - learns from previous runs
- Good for fine-tuning around promising regions
- Hyperband early termination saves compute

**Cons**:
- Less interpretable than grid
- Can miss multimodal regions
- Need separate sweeps for plummet vs lognorm

**Usage**:
```bash
# Edit pruning_strategy to 'plummet' or 'lognorm'
wandb sweep engiopt/lvae_2d/sweep_lvae_2d_bayes.yaml
wandb agent <sweep-id>
```

### 3. `sweep_lvae_2d_spectral_norm.yaml` - Spectral Norm Isolation
**Purpose**: Clean A/B test of spectral normalization effect

**What it tests**:
- Spectral norm on vs off ONLY
- Fixed "good" pruning parameters (update after initial sweeps)
- 5 seeds for statistical power

**Pros**:
- Clean isolation of one variable
- Fast (only 10 runs per problem)
- Easy to analyze statistically

**Cons**:
- Requires knowing good pruning params first

**Usage**:
```bash
# After finding good pruning params, update the yaml
# Then run for each pruning method
wandb sweep engiopt/lvae_2d/sweep_lvae_2d_spectral_norm.yaml
wandb agent <sweep-id>
```

## Recommended Experimental Workflow

### Phase 1: Understand Pruning Methods (Use Grid Search)

For EACH problem (beams2d, heatconduction2d, etc.):

1. **Edit `sweep_lvae_2d.yaml`** - set `problem_id` to your target problem

2. **Launch sweep**:
   ```bash
   wandb sweep engiopt/lvae_2d/sweep_lvae_2d.yaml
   wandb agent <sweep-id>
   ```

3. **Analyze results**:
   - Plot val_rec vs plummet_threshold (grouped by use_spectral_norm)
   - Plot val_rec vs alpha/percentile for lognorm
   - Plot active_dims over training for different params
   - Identify failure modes: complete pruning, instability, poor reconstruction

4. **Document findings**:
   - Which pruning method works better for this problem?
   - What's the sensitivity to hyperparameters?
   - Are there sweet spots or cliff edges?
   - Does spectral norm help or hurt?

### Phase 2: Spectral Normalization Deep Dive

After finding good pruning parameters from Phase 1:

1. **Update `sweep_lvae_2d_spectral_norm.yaml`**:
   - Set `pruning_strategy` to the winner from Phase 1
   - Set pruning hyperparameters to good values found

2. **Run isolation sweep**:
   ```bash
   wandb sweep engiopt/lvae_2d/sweep_lvae_2d_spectral_norm.yaml
   wandb agent <sweep-id>
   ```

3. **Statistical analysis**:
   - t-test: spectral norm on vs off for val_rec
   - Plot training curves: convergence speed, stability
   - Analyze final active dimensions
   - Check for interactions with problem type

### Phase 3: Fine-tuning with Bayesian (Optional)

If you want to optimize further:

1. **Edit `sweep_lvae_2d_bayes.yaml`**:
   - Set `pruning_strategy` to your chosen method
   - Narrow hyperparameter ranges based on Phase 1 findings
   - Set `use_spectral_norm` based on Phase 2 findings (or keep as categorical)

2. **Run Bayesian sweep**:
   ```bash
   wandb sweep engiopt/lvae_2d/sweep_lvae_2d_bayes.yaml
   wandb agent <sweep-id> --count 50  # or however many runs you want
   ```

### Phase 4: Safeguards (Future)

Once you understand pruning behavior:

1. Choose pruning method + params for each problem
2. Add safeguards to prevent known failure modes
3. Test that safeguards don't hurt performance in good cases

## Key Metrics to Track

- **val_rec**: Primary metric - validation reconstruction loss
- **active_dims**: How many dimensions remain unpruned
- **vol_loss**: Volume loss over training
- **rec_loss**: Reconstruction loss over training
- **Pruning timeline**: When do dimensions get pruned? (via active_dims plot)

## Problem-Specific Considerations

Different problems may have different optimal configurations:

- **beams2d**: Structure optimization, might benefit from more aggressive pruning
- **heatconduction2d**: Smooth fields, might need more dims for detail
- **Others**: Document as you go

Keep notes on problem-specific behavior to inform safeguard design later.

## Notes on Spectral Normalization

Current implementation: `use_spectral_norm` is binary (1-Lipschitz bound)

**Future extension**: Tunable Lipschitz bound
- Modify `TrueSNDecoder` to accept `lipschitz_bound` parameter
- Scale spectral norm: instead of σ=1, enforce σ≤L for tunable L
- Add to sweeps as continuous parameter (e.g., [1.0, 2.0, 5.0, 10.0])

## Common Pitfalls to Watch For

1. **Complete pruning**: All dims pruned → decoder sees only mean values → poor reconstruction
2. **No pruning**: Threshold too conservative → defeats purpose of volume loss
3. **Instability**: Spectral norm can sometimes slow learning or cause oscillations
4. **Problem-specific sensitivity**: What works for one problem may not transfer

## Example Analysis Commands

```python
import wandb
api = wandb.Api()

# Get all runs from a sweep
sweep = api.sweep("engibench/lvae/<sweep-id>")
runs = sweep.runs

# Compare plummet vs lognorm
import pandas as pd
df = pd.DataFrame([{
    'pruning_strategy': r.config['pruning_strategy'],
    'val_rec': r.summary.get('val_rec'),
    'active_dims': r.summary.get('active_dims'),
    'use_spectral_norm': r.config['use_spectral_norm'],
} for r in runs])

# Plot results
import seaborn as sns
sns.boxplot(data=df, x='pruning_strategy', y='val_rec', hue='use_spectral_norm')
```

## Questions to Answer Through Sweeps

1. **Plummet vs Lognorm**: Which is more robust? More efficient? Problem-dependent?
2. **Hyperparameter Sensitivity**: Are there safe defaults? Or must we tune per-problem?
3. **Spectral Norm**: Does it help stability? Hurt expressiveness? Interact with pruning?
4. **Timing**: Does pruning_epoch matter? Should we prune earlier/later?
5. **Volume Weight**: How does w_v affect pruning behavior?

Document answers as you go to build understanding.
