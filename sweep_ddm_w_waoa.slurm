#!/bin/bash
#SBATCH --job-name=sweep_ddm_w_waoa
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.24h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/sweep_ddm_w_waoa_%j.out
#SBATCH --array=0-3

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1

W_AOA_VALUES=(3 5 7 9)
W_AOA=${W_AOA_VALUES[$SLURM_ARRAY_TASK_ID]}

python -u -m engiopt.ddm.ddm_w.train_ddm_w \
    --lvae_checkpoint results/lvae/lae_dropout_0.25_flow_only_best.pth \
    --model_name ddm_w_waoa_${W_AOA} \
    --n_epochs 20000 \
    --w_pressure 1.0 \
    --lr 1e-4 \
    --grad_clip 1.0 \
    --w_aoa ${W_AOA} \
    --wandb
