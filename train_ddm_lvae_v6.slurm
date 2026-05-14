#!/bin/bash
#SBATCH --job-name=ddm_lvae_v6
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.120h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/train_ddm_lvae_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
python -u -m engiopt.ddm.ddm_lvae.train_ddm_lvae \
    --lvae_checkpoint results/lvae/lae_dropout_0.25_best.pth \
    --model_name ddm_lvae_v6 \
    --w_pressure 1e-5 \
    --cosine_schedule \
    --seed 0 \
    --wandb
