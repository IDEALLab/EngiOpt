# WandB Sweep Configuration for LVAE Dynamic Pruning

This directory contains scientifically-designed hyperparameter sweeps for studying the Least Volume Autoencoder (LVAE) with dynamic pruning across three engineering design problems.

## Scientific Philosophy

These sweeps are designed for **scientific understanding**, not just optimization. Each sweep:
- Tests one main hypothesis at a time
- Varies only relevant parameters while holding others fixed
- Uses sufficient seeds for statistical significance
- Enables interpretable, generalizable conclusions

## Sweep Hierarchy

Run sweeps in order for maximum insight:

### Phase 1: Broad Comparisons

**01_pruning_strategies.yaml** - Which pruning strategy works best?
- **Independent Variable**: Pruning strategy (plummet, pca_cdf, lognorm, probabilistic)
- **Controlled**: Safeguards (moderate), training hyperparameters
- **Varied**: Strategy-specific hyperparameters
- **Outcome**: Best strategy per problem + sensitivity analysis

**02_safeguard_impact.yaml** - Are safeguards necessary?
- **Independent Variable**: Safeguard configuration (none, individual, combined)
- **Controlled**: Pruning strategy (use best from 01), training hyperparameters
- **Varied**: Individual and combined safeguard profiles
- **Outcome**: Stability vs. performance trade-offs

**03_hyperparameter_sensitivity.yaml** - What else matters?
- **Method**: Bayesian optimization (efficient exploration)
- **Independent Variables**: Architecture (latent_dim), training (lr, batch_size, w_v), timing (pruning_epoch)
- **Controlled**: Pruning setup (best from 01), moderate safeguards
- **Outcome**: Hyperparameter importance ranking

### Phase 2: Problem-Specific Deep Dives

Run these after analyzing Phase 1 results to focus on promising regions.

**04_beams2d_deep_dive.yaml** - Structural optimization
- **Hypothesis**: Beams benefit from aggressive pruning (sparse structures)
- **Focus**: Fine-grained pruning timing and volume weight
- **Expected**: Lower optimal dimensionality, higher w_v

**05_heatconduction2d_deep_dive.yaml** - Physics simulation
- **Hypothesis**: Heat conduction needs higher dimensionality (smooth fields)
- **Focus**: Conservative pruning, later timing
- **Expected**: Higher optimal dimensionality, lower w_v

**06_photonics2d_deep_dive.yaml** - Photonic design
- **Hypothesis**: Photonics has sharp features, benefits from early pruning
- **Focus**: Aggressive early pruning
- **Expected**: Moderate dimensionality, early pruning_epoch

## How to Run

### 1. Initialize a Sweep

```bash
wandb sweep sweeps/01_pruning_strategies.yaml
```

This returns a sweep ID like `username/project/sweep_id`.

### 2. Launch Agents

```bash
# Single agent
wandb agent username/project/sweep_id

# Multiple agents (parallel)
wandb agent username/project/sweep_id --count 5
```

### 3. Monitor Results

```bash
wandb sweep username/project/sweep_id
```

Or view at: `https://wandb.ai/username/project/sweeps/sweep_id`

## Key Metrics to Track

### Primary Metrics
- `val_rec`: Validation reconstruction loss (primary objective)
- `val_total_loss`: Combined loss (includes volume term)
- `active_dims`: Final number of active dimensions

### Secondary Metrics (for analysis)
- `rec_loss`: Training reconstruction loss
- `vol_loss`: Volume regularization loss
- Training curves over epochs

## Interpreting Results

### For Pruning Strategy Comparison (Sweep 01)
- **Compare**: Final `val_rec` across strategies per problem
- **Look for**: Consistent winner across seeds
- **Analyze**: Correlation between `active_dims` and performance

### For Safeguard Impact (Sweep 02)
- **Compare**: Baseline (none) vs. individual safeguards vs. combined
- **Look for**: Which safeguards prevent instability without hurting performance
- **Analyze**: Training curves (do safeguards smooth training?)

### For Hyperparameter Sensitivity (Sweep 03)
- **Use**: Bayesian optimization's importance scores
- **Look for**: Which parameters have highest impact on `val_rec`
- **Analyze**: Interaction effects (e.g., latent_dim × w_v)

### For Problem-Specific Sweeps (04-06)
- **Compare**: Optimal settings across problems
- **Look for**: Problem-specific patterns (e.g., beams prefer high w_v)
- **Generalize**: Can you predict optimal settings for new problems?

## Expected Compute

Rough estimates per sweep:

- **Sweep 01**: 3 problems × 4 strategies × 3 seeds = 36 runs × 2500 epochs ≈ 18-36 GPU hours
- **Sweep 02**: 3 problems × 8 profiles × 5 seeds = 120 runs × 2500 epochs ≈ 60-120 GPU hours
- **Sweep 03**: Bayesian (early termination) ≈ 50-100 runs × variable epochs ≈ 30-60 GPU hours
- **Sweeps 04-06**: ~200 runs each × 3000 epochs ≈ 150-300 GPU hours each

**Total**: ~500-1000 GPU hours for complete scientific study

## Tips for Success

1. **Start Small**: Run sweep 01 on one problem first to validate setup
2. **Use Early Termination**: Enable hyperband for sweep 03 to save compute
3. **Monitor Actively**: Check first few runs to catch configuration errors
4. **Save Checkpoints**: Enable `save_model=True` for best runs
5. **Document Insights**: Use WandB notes to record observations per sweep

## Citation

If these sweeps contribute to your research, please cite:

```bibtex
@software{lvae_sweeps_2025,
  title={Systematic Hyperparameter Sweeps for LVAE Dynamic Pruning},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo}
}
```

## Questions?

- **Sweep syntax**: https://docs.wandb.ai/guides/sweeps
- **Bayesian optimization**: https://docs.wandb.ai/guides/sweeps/configuration#bayesian
- **Early termination**: https://docs.wandb.ai/guides/sweeps/configuration#stopping
