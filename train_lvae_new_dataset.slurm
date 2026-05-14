#!/bin/bash
#SBATCH --job-name=engiopt_lvae_new_dataset
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --exclude=eu-lo-g3-047,eu-lo-g3-048
#SBATCH --partition=gpu.12h
#SBATCH --output=logs/lvae_new_dataset_%j.out

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

mkdir -p logs results/lvae_new_dataset

python -u -m engiopt.lvae.train_lvae \
    --model_name lae_aoa_init_3d_new_dataset_v9 \
    --save_dir   results/lvae_new_dataset
