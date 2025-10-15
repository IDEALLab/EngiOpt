# Quick Start Guide for LVAE Sweeps

## Prerequisites

```bash
# Install wandb
pip install wandb

# Login to WandB
wandb login

# Verify installation
cd EngiOpt/
python -c "from engiopt.lvae_2d.lvae_2d import Args; print('✓ Setup OK')"
```

## Running Your First Sweep (5 minutes)

### Step 1: Test Run (Locally)

Before launching a sweep, test that your code works:

```bash
python engiopt/lvae_2d/lvae_2d.py \
  --problem-id beams2d \
  --n-epochs 10 \
  --track False
```

If this completes without errors, you're ready!

### Step 2: Initialize Sweep

```bash
wandb sweep sweeps/01_pruning_strategies.yaml
```

**Output:**
```
wandb: Creating sweep from: sweeps/01_pruning_strategies.yaml
wandb: Created sweep with ID: abc123xyz
wandb: View sweep at: https://wandb.ai/username/lvae/sweeps/abc123xyz
wandb: Run sweep agent with: wandb agent username/lvae/abc123xyz
```

Copy the agent command!

### Step 3: Launch Agent

```bash
# Single agent (one run at a time)
wandb agent username/lvae/abc123xyz

# Or: Run N runs then stop
wandb agent username/lvae/abc123xyz --count 5
```

### Step 4: Monitor Progress

Open the URL from Step 2 in your browser. You'll see:
- Live metrics (val_rec, active_dims, losses)
- Parallel coordinates plot (which params matter?)
- Hyperparameter importance

## Running Sweeps in Parallel

### On a Single Machine (Multi-GPU)

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 wandb agent username/lvae/sweep_id

# Terminal 2
CUDA_VISIBLE_DEVICES=1 wandb agent username/lvae/sweep_id

# Terminal 3 (CPU fallback)
CUDA_VISIBLE_DEVICES="" wandb agent username/lvae/sweep_id
```

### On a Cluster (SLURM Example)

```bash
#!/bin/bash
#SBATCH --job-name=lvae_sweep
#SBATCH --array=0-9          # 10 parallel agents
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

source activate engiopt
wandb agent username/lvae/sweep_id
```

Submit with: `sbatch sweep_job.sh`

## Sweep Order Recommendation

For first-time users, run in this order:

### Week 1: Understanding the Landscape
1. **01_pruning_strategies.yaml** (2-3 days)
   - Smallest sweep, fastest results
   - Teaches you which strategy works best
   - ~36 runs

2. **02_safeguard_impact.yaml** (3-4 days)
   - Moderate size
   - Shows stability vs. performance trade-offs
   - ~120 runs

### Week 2: Optimization
3. **03_hyperparameter_sensitivity.yaml** (3-4 days)
   - Bayesian optimization (smart search)
   - Uses early termination (saves time)
   - ~50-100 runs

### Week 3: Deep Dives (Pick Your Problem)
4. **04_beams2d_deep_dive.yaml** OR
5. **05_heatconduction2d_deep_dive.yaml** OR
6. **06_photonics2d_deep_dive.yaml**
   - Run the one most relevant to your research
   - ~200 runs, 3-5 days each

### Optional: Dynamics Study
7. **07_pruning_dynamics.yaml**
   - For understanding temporal behavior
   - Great for visualization/paper figures
   - ~200 runs

## Stopping a Sweep

### Graceful Stop (After Current Run)
```bash
# Ctrl+C in the agent terminal
# The current run completes, then agent stops
```

### Immediate Stop (Kill Run)
```bash
# Ctrl+C twice
# Current run is marked as "crashed"
```

### Pause Sweep (Stop All Agents)
In WandB UI: Sweep page → "Pause Sweep" button
- All agents finish current runs then stop
- Resume anytime with `wandb agent ...`

## Analyzing Results

### Quick Analysis (WandB UI)

1. **Parallel Coordinates Plot**
   - Shows all parameter combinations
   - Color by val_rec (best runs = green)
   - Identify patterns

2. **Hyperparameter Importance**
   - Shows which params correlate with val_rec
   - Higher importance = more impactful

3. **Line Plots**
   - Click "Charts" → "Add Chart" → "Line Plot"
   - X-axis: epoch, Y-axis: val_rec, Group: seed
   - Shows training dynamics

### Export for Analysis

```python
import wandb

api = wandb.Api()
sweep = api.sweep("username/lvae/sweep_id")

# Get all runs
runs = sweep.runs

# Extract data
data = []
for run in runs:
    data.append({
        "name": run.name,
        "config": run.config,
        "val_rec": run.summary.get("val_rec"),
        "active_dims": run.summary.get("active_dims"),
    })

import pandas as pd
df = pd.DataFrame(data)
df.to_csv("sweep_results.csv")
```

## Common Issues

### Issue: "No module named 'engiopt'"
**Solution**: Make sure you installed EngiOpt in editable mode:
```bash
cd EngiOpt/
pip install -e .
```

### Issue: Sweep runs but doesn't log to WandB
**Solution**: Check that `--track True` is set (it's default):
```bash
python engiopt/lvae_2d/lvae_2d.py --problem-id beams2d --track True
```

### Issue: Out of memory errors
**Solution**: Reduce batch size or latent_dim:
```bash
# Edit the YAML file
batch_size:
  value: 64  # Instead of 128
```

### Issue: Runs are too slow
**Solution**:
1. Enable early termination (sweep 03 already has it)
2. Reduce n_epochs for testing
3. Use fewer seeds initially

### Issue: Sweep fills up disk
**Solution**: Disable model saving for sweeps:
```bash
# In YAML, add:
save_model:
  value: false
```

## Tips for Efficient Sweeps

1. **Test with `--count 3` first**: Run just 3 trials to catch bugs
2. **Use `--dry-run` to validate config**: Check YAML without running
3. **Monitor first 5 runs**: Catch configuration errors early
4. **Use short runs for debugging**: Set n_epochs=100 for testing
5. **Leverage parallel agents**: Run 5-10 agents simultaneously if you have GPUs

## Next Steps

After completing sweeps:
1. **Write up findings**: Document insights in wandb Reports
2. **Share results**: Use WandB's shareable links
3. **Iterate**: Create new sweeps based on findings
4. **Publish**: Include wandb links in papers

Good luck! 🚀
