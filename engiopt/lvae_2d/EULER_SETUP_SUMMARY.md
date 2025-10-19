# Euler HPC Setup Complete ✓

Your LVAE sweep infrastructure for Euler is ready to use!

## What Was Created

### Core Sweep Configurations
- ✅ **sweep_lvae_2d.yaml** - Grid search (plummet + lognorm hyperparameters)
- ✅ **sweep_lvae_2d_bayes.yaml** - Bayesian optimization for fine-tuning
- ✅ **sweep_lvae_2d_spectral_norm.yaml** - Spectral norm isolation experiment

### Euler HPC Scripts
- ✅ **euler_sweep.slurm** - SLURM batch script (200 agents, GPU, 24h)
- ✅ **euler_launch_sweep.sh** - One-command launcher (creates sweep + submits jobs)
- ✅ **euler_sweep_monitor.sh** - Job monitoring and management tool

### Analysis & Documentation
- ✅ **analyze_sweeps.py** - Automated analysis with plots
- ✅ **README_SWEEPS.md** - Comprehensive reference guide
- ✅ **SWEEP_GUIDE.md** - Scientific methodology guide
- ✅ **EULER_QUICK_START.md** - Quick reference for Euler

## Key Design Features

### Scientific Rigor (As You Requested)
- **No safeguards initially** - See failures to understand behavior
- **Isolated variables** - Grid search separates effects cleanly
- **Complete coverage** - Both good and bad runs for full understanding
- **Statistical rigor** - Multiple seeds (3-5) for reliability

### Sweep Focus
1. **Primary**: Pruning methods (plummet vs lognorm) + their hyperparameters
2. **Secondary**: Spectral normalization (on/off, ready for tunable Lipschitz)
3. **Tertiary**: Safeguards (after understanding 1 & 2)

### Euler Optimization
- **Parameterized SWEEP_ID** - No more manual editing!
- **Automatic sweep creation** - One command does everything
- **Saved sweep info** - Never lose track of sweep IDs
- **Smart resource allocation** - 200 agents, max 50 concurrent
- **Efficient caching** - Uses $TMPDIR and pre-loads datasets

## Immediate Next Steps

### 1. Copy to Euler
```bash
# On your local machine
rsync -avz --exclude wandb --exclude '*.pyc' \
  ~/Desktop/Work/ETHZ/Research/EngiLearn/EngiOpt/ \
  <nethz-id>@euler.ethz.ch:~/projects/EngiOpt/
```

### 2. Test on Euler
```bash
# SSH to Euler
ssh euler.ethz.ch
cd ~/projects/EngiOpt

# Make scripts executable
chmod +x engiopt/lvae_2d/euler_*.sh

# Test with 10 runs first
./engiopt/lvae_2d/euler_launch_sweep.sh grid heatconduction2d 10
```

### 3. Monitor
```bash
# Check status
./engiopt/lvae_2d/euler_sweep_monitor.sh status

# Watch logs
./engiopt/lvae_2d/euler_sweep_monitor.sh logs <job-id>
```

### 4. Full Sweep (When Test Works)
```bash
# Launch 200-run grid search
./engiopt/lvae_2d/euler_launch_sweep.sh grid beams2d 200
```

## Parameter Ranges Configured

### Grid Search (sweep_lvae_2d.yaml)

**Plummet method:**
- `plummet_threshold`: [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
- 6 thresholds × 2 spectral_norm × 3 seeds = **36 runs per problem**

**Lognorm method:**
- `alpha`: [0.0, 0.1, 0.2, 0.3, 0.5]
- `percentile`: [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
- 5 alphas × 6 percentiles × 2 spectral_norm × 3 seeds = **180 runs per problem**

**Total: ~216 runs per problem** (both methods combined)

### Spectral Normalization
- `use_spectral_norm`: [false, true]
- Currently 1-Lipschitz when true
- **Future**: Add tunable `lipschitz_bound` parameter (documented in README)

### Fixed Parameters (for isolation)
- `latent_dim`: 250
- `w_v`: 0.01
- `eta`: 0.0001
- `pruning_epoch`: 500
- `beta`: 0.9
- `batch_size`: 128
- `lr`: 0.0001

### Safeguards (DISABLED for Phase 1)
- `min_active_dims`: 0 (allow complete pruning)
- `cooldown_epochs`: 0 (no cooldown)
- `k_consecutive`: 1 (immediate pruning)
- No `max_prune_per_epoch` limit
- No `recon_tol` constraint

## Expected Results & Timeline

### Grid Search (200 agents on Euler)
- **Per problem**: ~216 runs
- **Wall time**: ~10-12 hours (with 50 concurrent jobs)
- **GPU hours**: ~200 × 2500 epochs × 0.5 min/epoch ≈ 4,200 GPU-hours
- **What you'll learn**:
  - Which pruning method works better for each problem
  - Hyperparameter sensitivity curves
  - Failure modes (complete pruning, instability)
  - Spectral norm effect on both methods

### What to Look For in Results

**Good runs:**
- `val_rec` decreases steadily
- `active_dims` stabilizes at 20-100 (not 0, not 250)
- `vol_loss` decreases over time
- Training curves are smooth

**Bad runs (informative!):**
- `active_dims` → 0 (complete pruning) - threshold too aggressive
- `active_dims` = 250 (no pruning) - threshold too conservative
- `val_rec` increases or oscillates - instability
- Early divergence - learning rate or spectral norm issue

## Analysis Workflow

### After Sweep Completes

```bash
# 1. Get sweep ID from saved info
cat engiopt/lvae_2d/sweep_info_beams2d_grid.txt

# 2. Run analysis (on local machine or Euler)
python engiopt/lvae_2d/analyze_sweeps.py <sweep-id> --entity mkeeler43-eth

# 3. Check outputs
ls sweep_analysis/
# - sweep_data.csv (raw data)
# - pruning_method_comparison.png
# - plummet_sensitivity.png
# - lognorm_sensitivity.png
# - spectral_norm_effect.png
```

### Key Questions to Answer

1. **Plummet vs Lognorm**: Which is more robust? More efficient?
2. **Hyperparameter Sensitivity**: Smooth degradation or cliff edges?
3. **Spectral Norm**: Does it help stability? Hurt performance?
4. **Problem-Specific**: Do different problems need different methods?

## Customization Points

### To Change Problem Set
Edit sweep YAML files, change `problem_id` value, or use command-line:
```bash
./engiopt/lvae_2d/euler_launch_sweep.sh grid YOUR_PROBLEM_ID 200
```

### To Adjust Resource Limits
Edit `euler_sweep.slurm`:
```bash
#SBATCH --time=48:00:00      # For longer runs
#SBATCH --mem-per-cpu=16384  # For larger models
#SBATCH --gpus=rtx_3090:1    # For specific GPU
```

### To Tune Hyperparameter Ranges
Edit sweep YAML files:
```yaml
plummet_threshold:
  values: [0.01, 0.02, 0.04, 0.08]  # Your custom range
```

### To Add Tunable Lipschitz Bound
See section "Next Steps: Tunable Lipschitz Bound" in README_SWEEPS.md

## Files Reference

| File | Purpose | When to Use |
|------|---------|-------------|
| `euler_launch_sweep.sh` | Launch sweep + submit jobs | Every new sweep |
| `euler_sweep_monitor.sh` | Monitor running jobs | While jobs run |
| `euler_sweep.slurm` | SLURM batch script | Auto-used by launcher |
| `sweep_lvae_2d.yaml` | Grid search config | Main experiments |
| `sweep_lvae_2d_bayes.yaml` | Bayesian search config | Fine-tuning |
| `sweep_lvae_2d_spectral_norm.yaml` | Spectral norm isolation | After Phase 1 |
| `analyze_sweeps.py` | Generate plots & stats | After sweep completes |
| `EULER_QUICK_START.md` | Quick reference | When you forget syntax |
| `README_SWEEPS.md` | Full documentation | For detailed info |
| `SWEEP_GUIDE.md` | Experimental methodology | Scientific planning |

## Common Issues & Solutions

### "SWEEP_ID not set" error
**Solution**: Use `euler_launch_sweep.sh` instead of submitting `euler_sweep.slurm` directly

### Jobs stay pending
**Solution**: Check with `squeue -j <job-id> -o "%.18i %.9P %.8T %.10r"` - might be waiting for resources

### Out of disk space
**Solution**: Clean up old logs `rm logs/sweep_*.{out,err}` and use `$SCRATCH` for datasets

### Import errors on Euler
**Solution**: `source ~/venv/engibench/bin/activate` and check `module list`

### Sweep results look wrong
**Solution**: Check first few logs to ensure correct parameters: `head -200 logs/sweep_*_1.out`

## Success Checklist

Before your first full sweep:

- [ ] Code works locally: `python engiopt/lvae_2d/lvae_2d.py --n-epochs 10 --track False`
- [ ] Code copied to Euler: `rsync -avz EngiOpt/ euler:~/projects/EngiOpt/`
- [ ] Scripts are executable: `chmod +x engiopt/lvae_2d/euler_*.sh`
- [ ] Virtual env activated: `source ~/venv/engibench/bin/activate`
- [ ] WandB logged in: `wandb login` (one-time)
- [ ] Test run completed: `./euler_launch_sweep.sh grid heatconduction2d 10`
- [ ] Logs look good: `./euler_sweep_monitor.sh logs <job-id>`
- [ ] WandB dashboard shows runs: Check URL in sweep_info file

If all checkmarks, you're ready for full sweeps!

## Get Started Now

```bash
# Simplest possible start:
ssh euler.ethz.ch
cd ~/projects/EngiOpt
./engiopt/lvae_2d/euler_launch_sweep.sh grid beams2d 200

# That's it! Monitor with:
./engiopt/lvae_2d/euler_sweep_monitor.sh status
```

## Questions?

- Check [EULER_QUICK_START.md](EULER_QUICK_START.md) for commands
- Check [README_SWEEPS.md](README_SWEEPS.md) for detailed info
- Check [SWEEP_GUIDE.md](SWEEP_GUIDE.md) for scientific methodology
- Check saved sweep info: `cat engiopt/lvae_2d/sweep_info_*.txt`

Good luck with your experiments! 🚀
