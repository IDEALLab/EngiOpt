#!/bin/bash
#SBATCH --job-name=ddm_ablation_train
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --exclude=eu-lo-g3-047,eu-lo-g3-048
#SBATCH --partition=gpu.4h
#SBATCH --output=logs/ablation_train_%A_%a.out
#SBATCH --array=0-15

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

# All (n_samples, seed) combinations — 8 sizes x 2 seeds = 16 jobs
# Max is 543 (total final items in Wings3D train split)
N_SAMPLES=(50 100 150 200 300 400 500 543)
SEEDS=(0 1)

# Map SLURM array index to (n_samples, seed)
N_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
S_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

N=${N_SAMPLES[$N_IDX]}
S=${SEEDS[$S_IDX]}

echo "Array task $SLURM_ARRAY_TASK_ID → n_samples=$N, seed=$S"

python -u -m engiopt.ddm.train_ddm --n_samples $N --seed $S
