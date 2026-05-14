#!/bin/bash
#SBATCH --job-name=eval_ddm
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu.4h

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Navigate to project directory
cd ~/EngiOpt

# Create logs directory if it doesn't exist
mkdir -p logs

# Print start time
echo "Starting evaluation at $(date)"

# Run evaluation (using evaluate_ddm.py)
python evaluate_ddm.py

# Print end time
echo "Evaluation completed at $(date)"
