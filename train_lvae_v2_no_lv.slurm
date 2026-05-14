#!/bin/bash
#SBATCH --job-name=engiopt_lvae_v2_no_lv
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --exclude=eu-lo-g3-047,eu-lo-g3-048
#SBATCH --partition=gpu.4h
#SBATCH --output=logs/lvae_v2_no_lv_%j.out

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

# v2_no_lv: pure (x*, cp*) autoencoder, no LV penalty, no pruning (all 64 dims active)
# Goal: remove mask instability; give max reconstruction capacity
python -u -m engiopt.lvae.train_lvae --model_name lae_aoa_init_3d_v2_no_lv --lambda_lv 0 --prune_every 0
