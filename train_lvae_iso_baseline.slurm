#!/bin/bash
#SBATCH --job-name=lae_iso_baseline
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.24h
#SBATCH --output=logs/lae_iso_baseline_%j.out

# Isolation baseline: geometry + pressure only, no cd/cl loss
#   w_x=1000, w_eta=1000, w_pressure=1, w_perf=0
#   lambda_lv=1e-5, prune_threshold=0.01

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

python -u -m engiopt.lvae.train_lvae \
    --lambda_lv 1e-5 \
    --prune_threshold 0.01 \
    --w_x 1000 \
    --w_eta 1000 \
    --w_perf 0 \
    --model_name lae_iso_baseline \
    --wandb
