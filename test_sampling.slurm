#!/bin/bash
#SBATCH --job-name=test_sampling
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.4h
#SBATCH --output=logs/test_sampling_%j.out
module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
python -u test_sampling.py
