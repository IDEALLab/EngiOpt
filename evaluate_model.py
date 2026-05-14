#!/bin/bash
#SBATCH --job-name=eval_ddm
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu.4h

# Load modules (uncomment and adjust if needed)
# module load python/3.10
# module load cuda/11.8

# Activate conda environment if you use one (adjust path as needed)
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate engiopt

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Navigate to project directory
cd ~/EngiOpt

# Create logs directory if it doesn't exist
mkdir -p logs

# Print start time
echo "Starting evaluation at $(date)"

# Run evaluation
python evaluate_model.py

# Print end time
echo "Evaluation completed at $(date)"
