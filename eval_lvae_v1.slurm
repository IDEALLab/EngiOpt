#!/bin/bash
#SBATCH --job-name=engiopt_lvae_eval_v1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.8h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/eval_lvae_v1_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
# seed=0: matched x0 (standard evaluation)
python -u -m engiopt.lvae.evaluate_lvae --checkpoint results/lvae/lae_aoa_init_3d_v1.pth --refit --seed 0
