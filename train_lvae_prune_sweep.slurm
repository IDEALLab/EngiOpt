#!/bin/bash
#SBATCH --job-name=lae_prune_sweep
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --exclude=eu-lo-g3-047,eu-lo-g3-048
#SBATCH --partition=gpu.24h
#SBATCH --output=logs/lae_prune_sweep_%j.out
#SBATCH --array=0-3

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

THRESHOLDS=(0.0 0.005 0.01 0.05)
THRESHOLD=${THRESHOLDS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="lae_prune_thr${THRESHOLD}"

python -u -m engiopt.lvae.train_lvae \
    --prune_threshold ${THRESHOLD} \
    --model_name ${MODEL_NAME} \
    --wandb
