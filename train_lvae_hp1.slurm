#!/bin/bash
#SBATCH --job-name=lvae_hp1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --exclude=eu-lo-g3-047,eu-lo-g3-048
#SBATCH --partition=gpupr.4h
#SBATCH --output=logs/lvae_hp1_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1

# w_perf sweep: lower (baseline=50)
python -u -m engiopt.lvae.train_lvae --model_name lae_hp1_wperf20 --w_perf 20
