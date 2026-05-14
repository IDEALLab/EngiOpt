#!/bin/bash
#SBATCH --job-name=train_ddm
#SBATCH --output=logs/train_ddm_%j.out
#SBATCH --error=logs/train_ddm_%j.err
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
HF_DATASETS_OFFLINE=1 python -m engiopt.ddm.train_ddm
