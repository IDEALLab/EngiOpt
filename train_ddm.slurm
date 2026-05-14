#!/bin/bash
#SBATCH --job-name=engiopt_ddm_3d
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --exclude=eu-lo-g3-047,eu-lo-g3-048
#SBATCH --partition=gpu.4h
#SBATCH --output=logs/ddm_3d_%j.out

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

python -u -m engiopt.ddm.train_ddm



