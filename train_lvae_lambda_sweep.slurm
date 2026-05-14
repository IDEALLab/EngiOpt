#!/bin/bash
#SBATCH --job-name=engiopt_lvae_lambda_sweep
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --exclude=eu-lo-g3-047,eu-lo-g3-048
#SBATCH --partition=gpu.8h
#SBATCH --array=1-5
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/lvae_lambda_sweep_%a_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1

# lambda_lv values: 7e-06, 2e-05, 7e-05 (current v7), 2e-04, 7e-04
LAMBDAS=(7e-06 2e-05 7e-05 2e-04 7e-04)
LAMBDA=${LAMBDAS[$SLURM_ARRAY_TASK_ID - 1]}
MODEL_NAME="lae_new_dataset_lambda_sweep_${LAMBDA}"
SAVE_DIR="results/lvae_lambda_sweep/lambda_${LAMBDA}"

echo "Array task: $SLURM_ARRAY_TASK_ID  lambda_lv=$LAMBDA"

python -u -m engiopt.lvae.train_lvae \
    --lambda_lv   "$LAMBDA" \
    --model_name  "$MODEL_NAME" \
    --save_dir    "$SAVE_DIR" \
    --wandb \
    --wandb_project engiopt-lvae-lambda-sweep
