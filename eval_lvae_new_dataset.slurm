#!/bin/bash
#SBATCH --job-name=engiopt_lvae_eval_new_dataset
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.8h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/eval_lvae_new_dataset_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1

python -u -m engiopt.lvae.evaluate_lvae \
    --checkpoint results/lvae_new_dataset/lae_aoa_init_3d_new_dataset_v7_ep6000.pth \
    --save_dir   results/lvae_new_dataset \
    --no_x_norm
