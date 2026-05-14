#!/bin/bash
#SBATCH --job-name=lae_geo_cond
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --exclude=eu-lo-g3-047,eu-lo-g3-048
#SBATCH --partition=gpu.24h
#SBATCH --output=logs/lae_geo_cond_%j.out

# Isolation test: geometry-only with full geometric conditioning (c_dim=38).
# No pressure head. 4 flow params + 34 geometric scalars (sweep, chord, twist, taper, dihedral).
# Revert: remove --no_pressure and reset c_dim=4 in Config to restore baseline.

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

python -u -m engiopt.lvae.train_lvae \
    --no_pressure \
    --model_name lae_geo_cond \
    --wandb
