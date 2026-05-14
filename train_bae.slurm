#!/bin/bash
#SBATCH --job-name=train_bae
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.4h
#SBATCH --output=logs/bae_%j.out
module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
python -u -m engiopt.bezier_ae.train_bezier_ae
