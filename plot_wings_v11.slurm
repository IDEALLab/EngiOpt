#!/bin/bash
#SBATCH --job-name=plot_wings_v11
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.4h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/plot_wings_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
python -u -m engiopt.ddm.ddm_lvae.plot_wings \
    --checkpoint results/ddm_lvae/ddm_lvae_v11_best.pth \
    --n_wings 6 \
    --seed 0 \
    --T 980
