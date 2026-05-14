#!/bin/bash
#SBATCH --job-name=lae_geo_sweep
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --exclude=eu-lo-g3-047,eu-lo-g3-048
#SBATCH --partition=gpu.24h
#SBATCH --output=logs/lae_geo_sweep_%A_%a.out
#SBATCH --array=0-1

# Job 0: lae_geo_pressure  — pressure head ON  + geo conditioning (c_dim=38)
#         Tests whether aerodynamics regularizes geometry representation.
# Job 1: lae_geo_no_prune  — pressure head OFF + geo conditioning (c_dim=38) + prune_threshold=0.0
#         Tests whether keeping all 64 latent dims breaks the 0.20 val loss floor.

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    python -u -m engiopt.lvae.train_lvae \
        --w_x 100 \
        --model_name lae_geo_pressure_wx100 \
        --wandb
else
    python -u -m engiopt.lvae.train_lvae \
        --no_pressure \
        --prune_threshold 0.0 \
        --model_name lae_geo_no_prune \
        --wandb
fi
