#!/bin/bash
# Helper script to launch LVAE sweeps for different problems
# Usage: ./launch_sweeps.sh [sweep_type] [problem_id]
# Example: ./launch_sweeps.sh grid beams2d

set -e

SWEEP_TYPE=${1:-grid}
PROBLEM_ID=${2:-heatconduction2d}

# Map sweep type to yaml file
case $SWEEP_TYPE in
  grid)
    SWEEP_FILE="engiopt/lvae_2d/sweep_lvae_2d.yaml"
    ;;
  bayes)
    SWEEP_FILE="engiopt/lvae_2d/sweep_lvae_2d_bayes.yaml"
    ;;
  spectral)
    SWEEP_FILE="engiopt/lvae_2d/sweep_lvae_2d_spectral_norm.yaml"
    ;;
  *)
    echo "Unknown sweep type: $SWEEP_TYPE"
    echo "Valid types: grid, bayes, spectral"
    exit 1
    ;;
esac

echo "=========================================="
echo "Launching LVAE Sweep"
echo "=========================================="
echo "Sweep type: $SWEEP_TYPE"
echo "Config file: $SWEEP_FILE"
echo "Problem: $PROBLEM_ID"
echo "=========================================="

# Create a temporary config with the problem_id updated
TEMP_SWEEP=$(mktemp /tmp/sweep_XXXXXX.yaml)
sed "s/problem_id:.*/problem_id:\n    value: $PROBLEM_ID/" "$SWEEP_FILE" > "$TEMP_SWEEP"

echo ""
echo "Initializing sweep with wandb..."
SWEEP_ID=$(wandb sweep --project lvae "$TEMP_SWEEP" 2>&1 | grep -oP 'wandb agent \K[^\s]+' || echo "")

rm "$TEMP_SWEEP"

if [ -z "$SWEEP_ID" ]; then
  echo "ERROR: Failed to create sweep"
  exit 1
fi

echo ""
echo "=========================================="
echo "Sweep created successfully!"
echo "=========================================="
echo "Sweep ID: $SWEEP_ID"
echo ""
echo "To start an agent, run:"
echo "  wandb agent $SWEEP_ID"
echo ""
echo "To start multiple agents (parallel execution):"
echo "  wandb agent $SWEEP_ID &  # Repeat in multiple terminals"
echo ""
echo "Monitor at: https://wandb.ai/YOUR_ENTITY/lvae/sweeps/${SWEEP_ID##*/}"
echo "=========================================="
