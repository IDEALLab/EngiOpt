#!/bin/bash
#SBATCH --job-name=engiopt_lvae_v7_sweep
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.8h
#SBATCH --array=1-10
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/eval_lvae_v7_sweep_%a_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1

EPOCH=$(( SLURM_ARRAY_TASK_ID * 1000 ))

if [ "$SLURM_ARRAY_TASK_ID" -eq 10 ]; then
    CKPT="results/lvae_new_dataset/lae_aoa_init_3d_new_dataset_v7.pth"
else
    CKPT="results/lvae_new_dataset/lae_aoa_init_3d_new_dataset_v7_ep${EPOCH}.pth"
fi

echo "Evaluating checkpoint: $CKPT"

python -u -m engiopt.lvae.evaluate_lvae \
    --checkpoint "$CKPT" \
    --save_dir   results/lvae_new_dataset \
    --no_x_norm
