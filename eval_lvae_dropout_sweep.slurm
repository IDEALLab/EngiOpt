#!/bin/bash
#SBATCH --job-name=lae_drop_sweep_eval
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.4h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/eval_lae_drop_sweep_%a_%j.out
#SBATCH --array=0-2

DROPOUTS=(0.05 0.15 0.25)
DROPOUT=${DROPOUTS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="lae_dropout_${DROPOUT}"

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
python -u -m engiopt.lvae.evaluate_lvae \
    --checkpoint results/lvae/${MODEL_NAME}_best.pth \
    --no_force_align \
    --refit
