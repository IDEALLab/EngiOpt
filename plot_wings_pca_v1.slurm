#!/bin/bash
#SBATCH --job-name=plot_wings_pca_v1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.4h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/plot_wings_pca_v1_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
python -u -m engiopt.ddm.ddm_pca.plot_wings_pca \
    --checkpoint results/ddm_pca/ddm_pca_v1_best.pth \
    --n_wings 6 \
    --seed 0 \
    --T 1000
