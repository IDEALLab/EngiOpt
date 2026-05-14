#!/bin/bash
#SBATCH --job-name=generate_airfoils
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=4G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

HF_DATASETS_OFFLINE=1 python -m engiopt.ddm.generate_airfoils
