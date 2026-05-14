#!/bin/bash
#SBATCH --job-name=engiopt_lvae_eval_v10
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.8h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/eval_lvae_v10_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
# v10 = v1 + lambda_lv=4e-5
python -u -m engiopt.lvae.evaluate_lvae --checkpoint results/lvae/lae_aoa_init_3d_v10.pth --refit --seed 0
