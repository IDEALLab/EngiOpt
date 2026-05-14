#!/bin/bash
#SBATCH --job-name=engiopt_lvae_v9
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --exclude=eu-lo-g3-047,eu-lo-g3-048
#SBATCH --partition=gpu.4h
#SBATCH --output=logs/lvae_%j.out

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

# v9: v1 hyperparams (lambda_lv=7e-5, linear perf head) + deep pressure head only
python -u -m engiopt.lvae.train_lvae \
    --lambda_lv 7e-5 \
    --model_name lae_aoa_init_3d_v9 \
    --no_deep_perf_head
