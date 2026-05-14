#!/bin/bash
#SBATCH --job-name=lae_memory_sweep
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.24h
#SBATCH --output=logs/lae_memory_sweep_%A_%a.out
#SBATCH --array=0-1

# Job 0: lae_memory_baseline  — lambda_lv=0, prune=0, w_eta=9,    w_x=1000, pressure ON
#         Goal: prove 64 dims fixes shape + pressure, no spatial learning pressure.
# Job 1: lae_strict_alignment — lambda_lv=0, prune=0, w_eta=1000,  w_x=1000, pressure ON
#         Goal: force simultaneous shape + physics + exact 3D placement.

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    python -u -m engiopt.lvae.train_lvae \
        --lambda_lv 0.0 \
        --prune_threshold 0.0 \
        --w_x 1000 \
        --w_eta 9 \
        --model_name lae_memory_baseline \
        --wandb
else
    python -u -m engiopt.lvae.train_lvae \
        --lambda_lv 0.0 \
        --prune_threshold 0.0 \
        --w_x 1000 \
        --w_eta 1000 \
        --model_name lae_strict_alignment \
        --wandb
fi
