#!/bin/bash
#SBATCH --job-name=lvae_eval_memory_sweep
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.4h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/eval_memory_sweep_%A_%a.out
#SBATCH --array=0-1

# Job 0: lae_memory_baseline  — forced alignment ON  (cheat: use GT te_shifts)
# Job 1: lae_strict_alignment — forced alignment OFF (real test: decoder must predict placement)

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    python -u -m engiopt.lvae.evaluate_lvae \
        --checkpoint results/lvae/lae_memory_baseline_best.pth \
        --refit
else
    python -u -m engiopt.lvae.evaluate_lvae \
        --checkpoint results/lvae/lae_strict_alignment_best.pth \
        --no_force_align \
        --refit
fi
