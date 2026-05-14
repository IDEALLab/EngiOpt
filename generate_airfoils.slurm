#!/bin/bash
#SBATCH --job-name=engiopt_gen
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.8h
#SBATCH --output=logs/generate_airfoils_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
python -u -m engiopt.ddm.generate_airfoils
