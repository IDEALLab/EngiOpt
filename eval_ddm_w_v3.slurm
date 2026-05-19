#!/bin/bash
#SBATCH --job-name=eval_ddm_w_v3
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.4h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/eval_ddm_w_v3_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
python -u -m engiopt.ddm.ddm_w.evaluate_ddm_w \
    --checkpoint results/ddm_w/ddm_w_v3_best.pth \
    --lvae_checkpoint results/lvae/lae_dropout_0.25_flow_only_best.pth \
    --n_passes 10 \
    --seed 0 \
    --T 1000
