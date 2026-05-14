#!/bin/bash
#SBATCH --job-name=lae_dropout_0.25_flow_only_eval
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.4h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/eval_lae_dropout_0.25_flow_only_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
python -u -m engiopt.lvae.evaluate_lvae \
    --checkpoint results/lvae/lae_dropout_0.25_flow_only_best.pth \
    --no_force_align \
    --refit
