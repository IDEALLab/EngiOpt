#!/bin/bash
#SBATCH --job-name=engiopt_lvae_v2_weighting
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --exclude=eu-lo-g3-047,eu-lo-g3-048
#SBATCH --partition=gpu.4h
#SBATCH --output=logs/lvae_v2_weighting_%j.out

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

# v2_weighting: pure (x*, cp*) autoencoder (x0 removed) with w_perf=50
# Shape weight is 500, so w_perf=50 gives a 10:1 ratio instead of 500:1
# Goal: force stronger gradient signal for cd prediction

python -u -m engiopt.lvae.train_lvae --model_name lae_aoa_init_3d_v2_weighting --w_perf 10
