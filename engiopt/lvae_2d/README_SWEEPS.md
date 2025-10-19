# LVAE Sweep Configuration

Scientific exploration of LVAE pruning methods and spectral normalization effects.

## Quick Start

### Option A: Euler HPC (Recommended for large sweeps)

```bash
# 1. Launch sweep (creates sweep and submits 200 SLURM jobs)
cd ~/projects/EngiOpt/
./engiopt/lvae_2d/euler_launch_sweep.sh grid beams2d 200

# 2. Monitor progress
./engiopt/lvae_2d/euler_sweep_monitor.sh status
./engiopt/lvae_2d/euler_sweep_monitor.sh logs <job-id>

# 3. Analyze results (when complete)
python engiopt/lvae_2d/analyze_sweeps.py <sweep-id>
```

### Option B: Local Machine (Small sweeps or testing)

```bash
# 1. Launch sweep
cd EngiOpt/
./engiopt/lvae_2d/launch_sweeps.sh grid beams2d

# 2. Run agents (manually or in parallel)
wandb agent <sweep-id>

# Or multiple agents in parallel
for i in {1..4}; do
  wandb agent <sweep-id> &
done

# 3. Analyze results
python engiopt/lvae_2d/analyze_sweeps.py <sweep-id>
```

## Available Sweep Configurations

| File | Method | Purpose | Runs |
|------|--------|---------|------|
| `sweep_lvae_2d.yaml` | Grid | Complete parameter space exploration | ~200 per problem |
| `sweep_lvae_2d_bayes.yaml` | Bayesian | Efficient hyperparameter tuning | ~50 per problem |
| `sweep_lvae_2d_spectral_norm.yaml` | Grid | Isolated spectral norm A/B test | 10 per problem |

## Sweep Parameters

### Grid Search (`sweep_lvae_2d.yaml`)

**Fixed parameters** (consistent across all runs):
- `n_epochs: 2500`
- `batch_size: 128`
- `lr: 0.0001`
- `latent_dim: 250`
- `w_v: 0.01`
- `eta: 0.0001`
- `pruning_epoch: 500`
- `beta: 0.9`

**Variables tested**:
- `pruning_strategy`: [plummet, lognorm]
- `use_spectral_norm`: [false, true]
- **Plummet**: `threshold` ∈ [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
- **Lognorm**: `alpha` ∈ [0.0, 0.1, 0.2, 0.3, 0.5] × `percentile` ∈ [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
- `seed`: [1, 2, 3]

**Safeguards**: ALL DISABLED (no min_active_dims, no cooldown, no reconstruction tolerance)

### Bayesian Search (`sweep_lvae_2d_bayes.yaml`)

**Continuous optimization** of:
- `lr`: log-uniform [5e-5, 5e-4]
- `latent_dim`: categorical [150, 200, 250, 300]
- `w_v`: log-uniform [0.005, 0.05]
- `eta`: log-uniform [5e-5, 5e-4]
- `pruning_epoch`: uniform [300, 700]
- `beta`: uniform [0.85, 0.95]
- Pruning hyperparameters (method-specific ranges)

**Note**: Set `pruning_strategy` to one method at a time (edit yaml before launching).

### Spectral Norm Isolation (`sweep_lvae_2d_spectral_norm.yaml`)

**Only varies**: `use_spectral_norm` (true/false)

**Fixed pruning parameters** (update after Phase 1):
- `pruning_strategy: plummet` (or lognorm)
- `plummet_threshold: 0.02` (update to best value found)
- `alpha: 0.2`, `percentile: 0.05` (for lognorm)

**Seeds**: [1, 2, 3, 4, 5] for statistical power

## Experimental Phases

### Phase 1: Pruning Method Exploration (Use Grid Search)

**Goal**: Understand which pruning method works for each problem and what hyperparameters matter.

**Steps**:
1. Edit `sweep_lvae_2d.yaml` - set your `problem_id`
2. Launch sweep: `wandb sweep engiopt/lvae_2d/sweep_lvae_2d.yaml`
3. Run agents (parallelize for speed): `wandb agent <sweep-id> &`
4. Wait for completion (~200 runs)
5. Analyze: `python engiopt/lvae_2d/analyze_sweeps.py <sweep-id>`

**What to look for**:
- Which pruning method (plummet vs lognorm) achieves better val_rec?
- How sensitive is each method to its hyperparameters?
- Do runs fail catastrophically (complete pruning, divergence)?
- What's the sweet spot for number of active dimensions?

**Expected observations**:
- **Plummet**: May be more sensitive to threshold (log-scale sensitivity)
- **Lognorm**: Alpha controls how much it adapts (0.0 = frozen reference, higher = adaptive)
- Some hyperparameter combinations may cause complete pruning (active_dims → 0)
- Spectral norm effect may be positive or negative depending on problem

### Phase 2: Spectral Normalization Deep Dive

**Goal**: Cleanly isolate spectral norm effect with known-good pruning parameters.

**Steps**:
1. Based on Phase 1, identify best pruning method + hyperparameters for your problem
2. Edit `sweep_lvae_2d_spectral_norm.yaml`:
   - Set `pruning_strategy` to winner from Phase 1
   - Set pruning hyperparameters to best values found
3. Launch: `wandb sweep engiopt/lvae_2d/sweep_lvae_2d_spectral_norm.yaml`
4. Run agents: `wandb agent <sweep-id>`
5. Statistical analysis (only 10 runs - fast!)

**Analysis**:
```python
import pandas as pd
from scipy import stats

df = pd.read_csv("sweep_analysis/sweep_data.csv")
finished = df[df["state"] == "finished"]

spec_on = finished[finished["use_spectral_norm"] == True]["val_rec"]
spec_off = finished[finished["use_spectral_norm"] == False]["val_rec"]

# t-test
t_stat, p_val = stats.ttest_ind(spec_on, spec_off)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
print(f"Mean with spec norm: {spec_on.mean():.6f} ± {spec_on.std():.6f}")
print(f"Mean without: {spec_off.mean():.6f} ± {spec_off.std():.6f}")
```

### Phase 3: Fine-tuning (Optional)

**Goal**: Squeeze out last bit of performance.

**Steps**:
1. Edit `sweep_lvae_2d_bayes.yaml`:
   - Set `pruning_strategy` to your chosen method
   - Narrow hyperparameter ranges around Phase 1 sweet spots
   - Set or remove `use_spectral_norm` based on Phase 2
2. Launch: `wandb sweep engiopt/lvae_2d/sweep_lvae_2d_bayes.yaml`
3. Run: `wandb agent <sweep-id> --count 50`
4. Pick best run for final model

## Problem-Specific Workflows

### Running Sweeps for All Problems

```bash
# Grid search for all 2D problems
for problem in beams2d heatconduction2d; do
  echo "Starting sweep for $problem"
  # Edit sweep_lvae_2d.yaml to set problem_id: $problem
  sed -i '' "s/value: .*/value: $problem/" engiopt/lvae_2d/sweep_lvae_2d.yaml
  sweep_id=$(wandb sweep engiopt/lvae_2d/sweep_lvae_2d.yaml 2>&1 | grep -oP 'wandb agent \K[^\s]+')
  echo "Sweep ID for $problem: $sweep_id"
  # Launch multiple agents
  for i in {1..4}; do
    wandb agent $sweep_id &
  done
  wait  # Wait for all agents to finish before next problem
done
```

### Comparing Across Problems

```python
import pandas as pd
import matplotlib.pyplot as plt

# Collect results from multiple sweeps
results = []
for problem, sweep_id in [("beams2d", "xxx"), ("heatconduction2d", "yyy")]:
    df = pd.read_csv(f"sweep_analysis_{problem}/sweep_data.csv")
    df["problem"] = problem
    results.append(df)

all_results = pd.concat(results, ignore_index=True)
finished = all_results[all_results["state"] == "finished"]

# Compare best pruning method per problem
import seaborn as sns
sns.boxplot(data=finished, x="problem", y="val_rec", hue="pruning_strategy")
plt.title("Pruning Method Performance by Problem")
plt.show()
```

## Key Metrics to Track

| Metric | What it tells you | Good vs Bad |
|--------|-------------------|-------------|
| `val_rec` | Reconstruction quality | Lower = better |
| `active_dims` | How many dims remain | Too low (0-5) = over-pruning, too high (>200) = under-pruning |
| `val_vol_loss` | Latent space volume | Lower = more compressed |
| `rec_loss` (training) | Convergence behavior | Should decrease monotonically |

## Troubleshooting

### Complete Pruning (active_dims = 0)

**Symptom**: Model prunes all dimensions, reconstruction fails.

**Causes**:
- Plummet threshold too high
- Lognorm percentile too high
- Pruning started too early (low `pruning_epoch`)

**Solutions for Phase 4 (safeguards)**:
- Set `min_active_dims: 10` to prevent complete pruning
- Increase `cooldown_epochs` to slow pruning
- Increase `k_consecutive` to require stable evidence

### No Pruning (active_dims = latent_dim)

**Symptom**: No dimensions get pruned at all.

**Causes**:
- Plummet threshold too low (< 0.005)
- Lognorm percentile too low (< 0.01)
- Volume loss weight `w_v` too small
- `beta` too high (moving average too slow)

**Solutions**:
- Increase thresholds
- Increase `w_v` to 0.05 or higher
- Decrease `beta` to 0.8 or lower

### Unstable Training

**Symptom**: Loss oscillates, doesn't converge.

**Possible causes**:
- Learning rate too high
- Spectral norm interfering with learning
- Pruning too aggressive

**Debug**:
- Check training curves in wandb (plot `rec_loss` and `vol_loss` over epochs)
- Try without spectral norm
- Increase `pruning_epoch` to let model stabilize first

## Euler HPC Specific Instructions

### Setup (One-time)

1. **Prepare your environment on Euler:**
   ```bash
   # SSH to Euler
   ssh <nethz-id>@euler.ethz.ch

   # Create project directory
   mkdir -p ~/projects
   cd ~/projects

   # Clone/copy your code
   git clone <your-repo> EngiOpt
   # or: rsync -avz /path/to/local/EngiOpt euler:~/projects/

   # Set up virtual environment (if not done)
   module load python_cuda/3.11.6
   python -m venv ~/venv/engibench
   source ~/venv/engibench/bin/activate
   pip install -e EngiOpt/
   pip install wandb

   # Login to wandb (one-time)
   wandb login
   ```

2. **Update paths in scripts if needed:**
   - Check `PROJECT_DIR` in [euler_sweep.slurm](euler_sweep.slurm) (line 47)
   - Check `WANDB_ENTITY` in [euler_launch_sweep.sh](euler_launch_sweep.sh) (line 15)

### Running Sweeps on Euler

#### Method 1: Automated Launch (Recommended)

```bash
cd ~/projects/EngiOpt

# Launch grid sweep for a problem with 200 agents
./engiopt/lvae_2d/euler_launch_sweep.sh grid beams2d 200

# This will:
# 1. Create the wandb sweep
# 2. Submit SLURM job array automatically
# 3. Save sweep info to sweep_info_*.txt
```

#### Method 2: Manual Launch

```bash
# 1. Create sweep on your local machine or Euler
wandb sweep --project lvae --entity mkeeler43-eth engiopt/lvae_2d/sweep_lvae_2d.yaml
# Note the sweep ID: abc123xyz

# 2. On Euler, submit jobs
mkdir -p logs
sbatch --export=SWEEP_ID=abc123xyz --array=1-200%50 engiopt/lvae_2d/euler_sweep.slurm
```

### Monitoring Sweeps on Euler

```bash
# Check job status
./engiopt/lvae_2d/euler_sweep_monitor.sh status
./engiopt/lvae_2d/euler_sweep_monitor.sh status <job-id>

# View logs (live tail)
./engiopt/lvae_2d/euler_sweep_monitor.sh logs <job-id>

# Check for errors
./engiopt/lvae_2d/euler_sweep_monitor.sh errors <job-id>

# Summary of all sweeps
./engiopt/lvae_2d/euler_sweep_monitor.sh summary

# List sweep info files
./engiopt/lvae_2d/euler_sweep_monitor.sh sweeps

# Or use SLURM directly
squeue -u $USER                    # All your jobs
squeue -j <job-id>                # Specific job
scancel <job-id>                  # Cancel job
```

### Euler Resource Configuration

Current settings in `euler_sweep.slurm`:
- **Time**: 24 hours per job
- **Memory**: 8GB total (4GB per CPU × 2 CPUs)
- **CPUs**: 2 per task
- **GPUs**: 1 per task
- **Array**: 1-200 jobs, max 50 concurrent

**Memory breakdown for LVAE:**
- Model parameters: ~100MB
- Dataset in RAM: ~1GB
- PyTorch CUDA cache: ~2GB
- Overhead: ~1GB
- **Total needed: ~4GB, allocated: 8GB** (comfortable headroom)

Adjust if needed:
```bash
#SBATCH --time=48:00:00           # Increase if runs take >24h
#SBATCH --mem-per-cpu=8192        # Increase for 3D models (16GB total)
#SBATCH --cpus-per-task=4         # If using num_workers>2 in DataLoader
#SBATCH --gpus=rtx_3090:1         # Request specific GPU type
```

### Troubleshooting on Euler

**Problem**: Jobs fail with "module not found"
```bash
# Solution: Check module versions match your setup
module spider python_cuda
module spider cuda
```

**Problem**: Dataset downloads are slow
```bash
# Solution: Pre-cache datasets to $SCRATCH
mkdir -p $SCRATCH/datasets
python -c "from engibench.utils.all_problems import BUILTIN_PROBLEMS; BUILTIN_PROBLEMS['beams2d']()"
# Then rsync from $SCRATCH in SLURM script (already configured)
```

**Problem**: Sweep ID keeps changing
```bash
# Solution: Use saved sweep info files
cat engiopt/lvae_2d/sweep_info_*.txt
# Or check wandb dashboard
```

**Problem**: Out of GPU memory
```bash
# Reduce batch size or request more GPU memory
# Edit sweep yaml: batch_size: 64 (instead of 128)
```

## Files in This Directory

```
engiopt/lvae_2d/
├── lvae_2d.py                         # Main training script
├── aes.py                             # LVAE models and pruning logic
├── utils.py                           # Utilities (spectral norm, schedules)
├── sweep_lvae_2d.yaml                 # Grid search configuration
├── sweep_lvae_2d_bayes.yaml           # Bayesian search configuration
├── sweep_lvae_2d_spectral_norm.yaml   # Spectral norm isolation sweep
├── launch_sweeps.sh                   # Helper script to launch sweeps (local)
├── euler_sweep.slurm                  # SLURM batch script for Euler
├── euler_launch_sweep.sh              # Automated sweep launcher for Euler
├── euler_sweep_monitor.sh             # Monitor/manage Euler sweep jobs
├── analyze_sweeps.py                  # Analysis script with plotting
├── SWEEP_GUIDE.md                     # Detailed experimental guide
└── README_SWEEPS.md                   # This file
```

## Expected Timeline

**Per problem** (with 4 parallel agents):

- **Phase 1 (Grid)**: ~200 runs × 2500 epochs × ~30s/epoch ≈ 42 hours wall-clock / 4 = ~10.5 hours
- **Phase 2 (Spectral)**: ~10 runs × 2500 epochs × ~30s/epoch ≈ 2 hours
- **Phase 3 (Bayesian)**: ~50 runs × variable (early stopping) ≈ 5-10 hours

**Total per problem**: ~15-20 hours wall-clock with 4 agents

**Pro tip**: Use GPU instances and increase parallelism (8+ agents) to finish faster.

## Citation

If these sweeps help you find good configurations, document them in your paper's hyperparameter table!

```latex
\begin{table}
\caption{LVAE hyperparameters found via grid search}
\begin{tabular}{lcccc}
Problem & Pruning & Threshold/Alpha & Spectral Norm & Final Dims \\
\hline
beams2d & plummet & 0.02 & Yes & 42 \\
heatconduction2d & lognorm & α=0.2, p=0.05 & No & 68 \\
\end{tabular}
\end{table}
```

## Next Steps: Tunable Lipschitz Bound

Currently `use_spectral_norm` is binary (1-Lipschitz). To test tunable bounds:

1. **Modify `TrueSNDecoder`** in [lvae_2d.py](lvae_2d.py:216-296):
   ```python
   class TrueSNDecoder(nn.Module):
       def __init__(self, latent_dim, design_shape, lipschitz_bound=1.0):
           # ... existing code ...
           # Scale spectral norm by lipschitz_bound
           self.lipschitz_bound = lipschitz_bound
   ```

2. **Add to Args**:
   ```python
   lipschitz_bound: float = 1.0
   """Lipschitz bound for spectral normalization (only used if use_spectral_norm=True)"""
   ```

3. **Create new sweep**:
   ```yaml
   use_spectral_norm:
     value: true
   lipschitz_bound:
     values: [0.5, 1.0, 2.0, 5.0, 10.0]  # Test different bounds
   ```

This would test: Does relaxing the 1-Lipschitz constraint help or hurt?
