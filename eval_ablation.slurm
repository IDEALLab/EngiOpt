#!/bin/bash
#SBATCH --job-name=ddm_ablation_eval
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.8h
#SBATCH --output=logs/ablation_eval_%A_%a.out
#SBATCH --array=0-15

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

# Must match the arrays in train_ablation.slurm exactly
# Max is 543 (total final items in Wings3D train split)
N_SAMPLES=(50 100 150 200 300 400 500 543)
SEEDS=(0 1)

N_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
S_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

N=${N_SAMPLES[$N_IDX]}
S=${SEEDS[$S_IDX]}

CHECKPOINT="results/ablation/n${N}_s${S}/ddm_ablation_n${N}_s${S}.pth"

echo "Array task $SLURM_ARRAY_TASK_ID → checkpoint=$CHECKPOINT"

python -u -m engiopt.ddm.evaluate_ddm \
    --checkpoint $CHECKPOINT \
    --n_samples $N \
    --seed $S
