#!/bin/bash
#SBATCH --job-name=engiopt_lae_baseline
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --exclude=eu-lo-g3-047,eu-lo-g3-048
#SBATCH --partition=gpu.4h
#SBATCH --output=logs/lae_baseline_%j.out

# Non-LV baseline: same architecture as the LVAE but with lambda_lv=0 and
# no pruning.  Run this to check whether the volume penalty (and aggressive
# pruning) is the cause of high cd MSE, or whether the issue is elsewhere.

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

python -u -m engiopt.lvae.train_lvae --no_lv
