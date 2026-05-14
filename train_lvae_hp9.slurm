#!/bin/bash
#SBATCH --job-name=lvae_hp9
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --exclude=eu-lo-g3-047,eu-lo-g3-048
#SBATCH --partition=gpupr.4h
#SBATCH --output=logs/lvae_hp9_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1

# combined best: w_perf=10 (best cd) + lambda_lv=1e-4 (best latent structure)
python -u -m engiopt.lvae.train_lvae --model_name lae_hp9_combined --w_perf 10 --lambda_lv 1e-4
