#!/bin/bash
#SBATCH --job-name=lae_drop_sweep
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.24h
#SBATCH --output=logs/lae_drop_sweep_%a_%j.out
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

python -u -m engiopt.lvae.train_lvae \
    --lambda_lv 1e-5 \
    --prune_threshold 0.01 \
    --w_x 1000 \
    --w_eta 1000 \
    --dropout ${DROPOUT} \
    --model_name ${MODEL_NAME} \
    --wandb
