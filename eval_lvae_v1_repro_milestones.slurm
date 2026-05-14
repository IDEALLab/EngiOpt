#!/bin/bash
#SBATCH --job-name=engiopt_eval_milestones
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.8h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/eval_milestones_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1

# Evaluate each 1000-epoch milestone checkpoint
for ep in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000; do
    ckpt="results/lvae/lae_aoa_init_3d_v1_repro_ep${ep}.pth"
    echo "======================================================"
    echo "Evaluating epoch ${ep}: ${ckpt}"
    echo "======================================================"
    python -u -m engiopt.lvae.evaluate_lvae --checkpoint "${ckpt}" --refit --seed 0
done
