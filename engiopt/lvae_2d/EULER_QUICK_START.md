# Euler HPC Quick Start for LVAE Sweeps

## One-Line Commands

```bash
# Launch sweep (all-in-one)
./engiopt/lvae_2d/euler_launch_sweep.sh grid beams2d 200

# Monitor
./engiopt/lvae_2d/euler_sweep_monitor.sh status

# Cancel if needed
./engiopt/lvae_2d/euler_sweep_monitor.sh cancel <job-id>
```

## Step-by-Step First Time Setup

### 1. On Your Local Machine

```bash
# Test sweep configuration locally first
cd EngiOpt/
python engiopt/lvae_2d/lvae_2d.py --problem-id beams2d --n-epochs 10 --track False

# If that works, commit and push your code
git add engiopt/lvae_2d/
git commit -m "Add LVAE sweep configurations"
git push
```

### 2. On Euler

```bash
# SSH to Euler
ssh <your-nethz-id>@euler.ethz.ch

# Navigate to your projects directory
cd ~/projects

# Clone/update code
git clone git@github.com:YourUsername/EngiOpt.git
# Or if already cloned: cd EngiOpt && git pull

cd EngiOpt

# Make scripts executable
chmod +x engiopt/lvae_2d/euler_*.sh

# Test the SLURM script syntax
sbatch --test-only engiopt/lvae_2d/euler_sweep.slurm
# Should say "Job would submit successfully" (ignore SWEEP_ID error for now)
```

### 3. Launch Your First Sweep

```bash
# Start with a small test (just 10 runs)
./engiopt/lvae_2d/euler_launch_sweep.sh grid heatconduction2d 10

# You'll see output like:
# Sweep ID: abc123xyz
# Job ID: 12345678
# URL: https://wandb.ai/mkeeler43-eth/lvae/sweeps/abc123xyz

# Monitor it
./engiopt/lvae_2d/euler_sweep_monitor.sh status 12345678

# Watch logs live
./engiopt/lvae_2d/euler_sweep_monitor.sh logs 12345678
```

### 4. If Test Works, Launch Full Sweep

```bash
# Full grid search (200 runs)
./engiopt/lvae_2d/euler_launch_sweep.sh grid beams2d 200

# Bayesian optimization (100 runs)
./engiopt/lvae_2d/euler_launch_sweep.sh bayes beams2d 100

# Spectral norm isolation (just 10 runs)
./engiopt/lvae_2d/euler_launch_sweep.sh spectral beams2d 10
```

## Common Workflows

### Scenario: Run sweeps for all problems

```bash
cd ~/projects/EngiOpt

# Loop through problems
for problem in beams2d heatconduction2d; do
  echo "Launching sweep for $problem"
  ./engiopt/lvae_2d/euler_launch_sweep.sh grid $problem 200
  sleep 5  # Brief pause between submissions
done

# Check all jobs
./engiopt/lvae_2d/euler_sweep_monitor.sh summary
```

### Scenario: A job failed, need to resubmit

```bash
# Check what failed
./engiopt/lvae_2d/euler_sweep_monitor.sh errors <job-id>

# Fix the issue (e.g., edit code, increase memory, etc.)

# Get the sweep ID from saved info
cat engiopt/lvae_2d/sweep_info_beams2d_grid.txt

# Resubmit with same sweep ID
sbatch --export=SWEEP_ID=<sweep-id> --array=1-50 engiopt/lvae_2d/euler_sweep.slurm
```

### Scenario: Need to download results for local analysis

```bash
# On Euler: Check sweep info
cat engiopt/lvae_2d/sweep_info_*.txt

# On local machine: Download and analyze
python engiopt/lvae_2d/analyze_sweeps.py <sweep-id> --entity mkeeler43-eth

# This saves results to sweep_analysis/
# - sweep_data.csv (raw data)
# - *.png (plots)
```

## Monitoring Commands Reference

```bash
# Job status
squeue -u $USER                           # All your jobs
squeue -j <job-id>                        # Specific job
squeue -j <job-id> -t RUNNING             # Only running tasks
squeue -j <job-id> -t PENDING             # Only pending tasks

# Job details
scontrol show job <job-id>                # Full job info
sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed  # Accounting info

# Cancel jobs
scancel <job-id>                          # Cancel entire job array
scancel <job-id>_<array-index>            # Cancel single array task
scancel -u $USER                          # Cancel ALL your jobs (careful!)

# Logs
ls logs/sweep_<job-id>_*.out              # List output logs
ls logs/sweep_<job-id>_*.err              # List error logs
tail -f logs/sweep_<job-id>_*.out         # Follow logs
grep "ERROR\|error" logs/sweep_*.err      # Search for errors

# Disk usage (important on Euler!)
du -sh $TMPDIR/*                          # Space used on compute node
du -sh $SCRATCH/*                         # Space used in $SCRATCH
quota -s                                  # Your home directory quota
```

## Resource Adjustments

Edit `engiopt/lvae_2d/euler_sweep.slurm` if you need to change resources:

```bash
# Increase time limit (default: 24h)
#SBATCH --time=48:00:00

# Increase memory (default: 8GB per CPU)
#SBATCH --mem-per-cpu=16384

# Request specific GPU
#SBATCH --gpus=rtx_3090:1

# Change array size/concurrency
#SBATCH --array=1-500%100    # 500 total, max 100 concurrent

# Request more CPUs
#SBATCH --cpus-per-task=8
```

## Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| Job pending forever | `squeue -j <job-id> -o "%.18i %.9P %.8T %.10r"` then check reason |
| Out of memory | Increase `--mem-per-cpu` or reduce batch size in sweep yaml |
| GPU out of memory | Reduce `batch_size` or `latent_dim` in sweep yaml |
| Can't find sweep ID | `cat engiopt/lvae_2d/sweep_info_*.txt` |
| Wrong Python version | Check `module list` matches `module load python_cuda/3.11.6` |
| Import errors | `source ~/venv/engibench/bin/activate` and `pip install -e .` |
| WandB login fails | `wandb login` or set `WANDB_API_KEY` in slurm script |
| Logs not appearing | Jobs might be queued, check with `squeue` |
| All jobs fail immediately | Check first error: `head -50 logs/sweep_<job-id>_1.err` |

## Performance Tips

1. **Pre-cache datasets** (first time only):
   ```bash
   mkdir -p $SCRATCH/datasets
   export HF_DATASETS_CACHE=$SCRATCH/datasets
   python -c "from engibench.utils.all_problems import BUILTIN_PROBLEMS; \
              [BUILTIN_PROBLEMS[p]() for p in ['beams2d', 'heatconduction2d']]"
   ```

2. **Use $SCRATCH for large files**, not $HOME:
   ```bash
   # $HOME has small quota (16GB)
   # $SCRATCH has large quota (2.5TB) but is temporary
   # $WORK is persistent but slower
   ```

3. **Monitor resource usage**:
   ```bash
   # While job is running, SSH to compute node
   squeue -j <job-id> -o "%N"  # Get node name
   ssh <node-name>
   nvidia-smi  # Check GPU usage
   htop        # Check CPU/memory
   ```

4. **Optimize concurrency**:
   ```bash
   # Grid search: High concurrency OK (different hyperparams)
   #SBATCH --array=1-200%100

   # Bayesian: Lower concurrency better (learns from previous runs)
   #SBATCH --array=1-100%20
   ```

## Emergency Commands

```bash
# Cancel everything and start over
scancel -u $USER
wandb sweep --stop <sweep-id>

# Clean up logs
rm -rf logs/sweep_*.{out,err}

# Clean up TMPDIR if job crashed and left files
# (Only safe when no jobs running!)
ssh <compute-node>
rm -rf $TMPDIR/*

# Check Euler system status
sinfo                    # Cluster status
sinfo -p gpu.4           # GPU partition status
squeue --start           # Estimated start times
```

## Getting Help

- **Euler docs**: https://scicomp.ethz.ch/wiki/Euler
- **WandB docs**: https://docs.wandb.ai/guides/sweeps
- **Your sweep info**: `cat engiopt/lvae_2d/sweep_info_*.txt`
- **Sweep dashboard**: https://wandb.ai/mkeeler43-eth/lvae/sweeps

## Example: Complete First Run

```bash
# 1. SSH to Euler
ssh euler.ethz.ch
cd ~/projects/EngiOpt

# 2. Quick test run (10 jobs, 5 concurrent)
./engiopt/lvae_2d/euler_launch_sweep.sh grid heatconduction2d 10

# 3. Note the job ID from output (e.g., 12345678)

# 4. Monitor (wait ~5 min for jobs to start)
watch -n 30 './engiopt/lvae_2d/euler_sweep_monitor.sh status 12345678'

# 5. Check one log to ensure it's working
./engiopt/lvae_2d/euler_sweep_monitor.sh logs 12345678 | head -100

# 6. If all looks good, check WandB dashboard
# URL shown in sweep_info file

# 7. If test worked, launch full sweep
./engiopt/lvae_2d/euler_launch_sweep.sh grid beams2d 200

# 8. Monitor summary of all sweeps
./engiopt/lvae_2d/euler_sweep_monitor.sh summary
```

Done! Your sweeps are now running. Come back in ~12-24 hours to analyze results.
