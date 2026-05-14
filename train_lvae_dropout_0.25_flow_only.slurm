#!/bin/bash
#SBATCH --job-name=lae_dropout_0.25_flow_only
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.24h
#SBATCH --output=logs/lae_dropout_0.25_flow_only_%j.out

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
    --dropout 0.25 \
    --flow_only \
    --model_name lae_dropout_0.25_flow_only \
    --wandb
