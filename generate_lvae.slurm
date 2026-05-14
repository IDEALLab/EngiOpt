#!/bin/bash
#SBATCH --job-name=engiopt_lvae_gen
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.8h
#SBATCH --output=logs/generate_lvae_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
python -u -m engiopt.lvae.generate_airfoils --checkpoint results/lvae/lae_aoa_init_3d_v8.pth --mach 0.5 --reynolds 5e6 --cl_target 0.8 --area 1.0

