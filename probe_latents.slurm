#!/bin/bash
#SBATCH --job-name=engiopt_probe
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.8h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/probe_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1

# Linear probe: does v1's z contain enough info for cd/cl/pressure?
# Compare GBR test MSE vs. v1 decoder head test MSE to identify bottleneck.
python -u -m engiopt.lvae.probe_latents --checkpoint results/lvae/lae_aoa_init_3d_v1.pth
