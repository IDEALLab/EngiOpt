#!/bin/bash
#SBATCH --job-name=lae_no_perf
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --exclude=eu-lo-g3-047,eu-lo-g3-048
#SBATCH --partition=gpu.4h
#SBATCH --output=logs/lae_no_perf_%j.out

# Ablation: lambda_lv=7e-6, w_perf=0 — isolate whether Cd/Cl prediction
# interferes with reconstruction quality.

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

python -u -m engiopt.lvae.train_lvae --lambda_lv 7e-6 --w_perf 0 --model_name lae_no_perf_lv7e6 --wandb
