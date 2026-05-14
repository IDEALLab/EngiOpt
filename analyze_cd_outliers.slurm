#!/bin/bash
#SBATCH --job-name=cd_outlier_analysis
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=normal.4h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/cd_outlier_analysis_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
python -u analyze_cd_outliers.py --checkpoint results/lvae/lae_aoa_init_3d_v2_weighting.pth
