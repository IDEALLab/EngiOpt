#!/bin/bash
#SBATCH --job-name=lvae_sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --array=1-200%50
# Resource notes:
# - mem-per-cpu=4096 (4GB) × cpus-per-task=2 = 8GB total RAM
# - LVAE memory breakdown: ~3-4GB (model ~100MB, dataset ~1GB, PyTorch cache ~2GB)
# - 8GB provides comfortable headroom; increase if training larger 3D models
# - array=1-200%50 means 200 total jobs, max 50 running concurrently

#-------------------------------------------------------------------------------
# USAGE:
#   sbatch --export=SWEEP_ID=<sweep-id> engiopt/lvae_2d/euler_sweep.slurm
#
# Example:
#   sbatch --export=SWEEP_ID=abc123xyz engiopt/lvae_2d/euler_sweep.slurm
#
# Or set environment variable before submitting:
#   export SWEEP_ID=abc123xyz
#   sbatch engiopt/lvae_2d/euler_sweep.slurm
#-------------------------------------------------------------------------------

set -e  # Exit on error

# Check if SWEEP_ID is set
if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: SWEEP_ID not set!"
    echo "Usage: sbatch --export=SWEEP_ID=<your-sweep-id> $0"
    exit 1
fi

echo "=========================================="
echo "LVAE Sweep Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $HOSTNAME"
echo "Sweep ID: $SWEEP_ID"
echo "=========================================="

#-------------------------------------------------------------------------------
# Load modules & activate virtual environment
#-------------------------------------------------------------------------------
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.4.1
module load eth_proxy
source "$HOME/venv/engibench/bin/activate"

echo "Using Python: $(which python) ($(python -c 'import sys; print(sys.version)'))"

#-------------------------------------------------------------------------------
# Copy code to node-local scratch and cd into it
#-------------------------------------------------------------------------------
PROJECT_DIR="EngiOpt"  # Adjust if your project has a different name
echo "Copying code from $HOME/projects/$PROJECT_DIR to $TMPDIR/$PROJECT_DIR"

cp -r "$HOME/projects/$PROJECT_DIR" "$TMPDIR/$PROJECT_DIR"
cd "$TMPDIR/$PROJECT_DIR" || { echo "cd failed"; exit 1; }

echo "Working directory: $(pwd)"

#-------------------------------------------------------------------------------
# Redirect Hugging Face, PyTorch, and WandB caches to node-local scratch
#-------------------------------------------------------------------------------
export HF_HOME="$TMPDIR/models"
export HF_DATASETS_CACHE="$TMPDIR/datasets"
export TORCH_HOME="$TMPDIR/torch_cache"
export WANDB_DIR="$TMPDIR/wandb"
export WANDB_CACHE_DIR="$TMPDIR/wandb_cache"
export WANDB_CONFIG_DIR="$TMPDIR/wandb_config"

# Make sure these directories exist
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# Preload HF dataset (if available)
if [ -d "$SCRATCH/datasets" ]; then
    echo "Preloading datasets from $SCRATCH/datasets"
    rsync -a "$SCRATCH/datasets/"* "$HF_DATASETS_CACHE/"
fi

#-------------------------------------------------------------------------------
# WandB configuration
#-------------------------------------------------------------------------------
# Option 1: Use wandb login (recommended - stores key securely)
# Run once: wandb login
# Then comment out the WANDB_API_KEY line below

# Option 2: Set API key directly (less secure but works)
export WANDB_API_KEY="${WANDB_API_KEY:-867b0410bb40edb1a14cd1b8afd23fad3663e906}"

# These can be overridden via --export when submitting
export WANDB_PROJECT="${WANDB_PROJECT:-lvae}"
export WANDB_ENTITY="${WANDB_ENTITY:-mkeeler43-eth}"

echo "WandB Entity: $WANDB_ENTITY"
echo "WandB Project: $WANDB_PROJECT"
echo "Sweep ID: $SWEEP_ID"

#-------------------------------------------------------------------------------
# Launch one W&B agent per array task (each pulls one set of hyperparams)
#-------------------------------------------------------------------------------
echo "=========================================="
echo "Starting wandb agent at $(date)"
echo "=========================================="

# Run exactly 1 sweep run for this array task
wandb agent "$WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID" --count 1

echo "=========================================="
echo "Completed at $(date)"
echo "=========================================="
