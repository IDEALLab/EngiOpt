#!/bin/bash
# Unified script to create and launch LVAE sweeps on Euler HPC
# Usage: ./euler_launch_sweep.sh [sweep_type] [problem_id] [n_agents]
# Example: ./euler_launch_sweep.sh grid beams2d 100

set -e

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------
SWEEP_TYPE=${1:-grid}
PROBLEM_ID=${2:-heatconduction2d}
N_AGENTS=${3:-200}

WANDB_ENTITY="mkeeler43-eth"
WANDB_PROJECT="lvae"

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
    echo "ERROR: Unknown sweep type: $SWEEP_TYPE"
    echo "Valid types: grid, bayes, spectral"
    exit 1
    ;;
esac

echo "=========================================="
echo "LVAE Sweep Setup for Euler HPC"
echo "=========================================="
echo "Sweep type:   $SWEEP_TYPE"
echo "Config file:  $SWEEP_FILE"
echo "Problem:      $PROBLEM_ID"
echo "Agents:       $N_AGENTS"
echo "WandB Entity: $WANDB_ENTITY"
echo "WandB Project: $WANDB_PROJECT"
echo "=========================================="
echo ""

#-------------------------------------------------------------------------------
# Assume always running on HPC
#-------------------------------------------------------------------------------
ON_EULER=true
echo "Running on Euler HPC"

#-------------------------------------------------------------------------------
# Create temporary sweep config with problem_id set
#-------------------------------------------------------------------------------
TEMP_SWEEP=$(mktemp /tmp/sweep_XXXXXX.yaml)

# Update problem_id in the yaml
sed "s/value: heatconduction2d/value: $PROBLEM_ID/" "$SWEEP_FILE" > "$TEMP_SWEEP"
# Also update any other hardcoded problem references
sed -i.bak "s/problem_id: .*/problem_id:\n    value: $PROBLEM_ID/" "$TEMP_SWEEP" 2>/dev/null || true

echo "Creating sweep with wandb..."
echo ""

#-------------------------------------------------------------------------------
# Initialize sweep
#-------------------------------------------------------------------------------
SWEEP_OUTPUT=$(wandb sweep --project "$WANDB_PROJECT" --entity "$WANDB_ENTITY" "$TEMP_SWEEP" 2>&1)
echo "$SWEEP_OUTPUT"

# Extract sweep ID (format: entity/project/sweep_id)
SWEEP_FULL_ID=$(echo "$SWEEP_OUTPUT" | grep -oE "${WANDB_ENTITY}/${WANDB_PROJECT}/[a-z0-9]+" | head -1)
SWEEP_ID=$(echo "$SWEEP_FULL_ID" | awk -F'/' '{print $NF}')

rm -f "$TEMP_SWEEP" "$TEMP_SWEEP.bak"

if [ -z "$SWEEP_ID" ]; then
    echo ""
    echo "ERROR: Failed to extract sweep ID"
    exit 1
fi

echo ""
echo "=========================================="
echo "Sweep Created Successfully!"
echo "=========================================="
echo "Full ID: $SWEEP_FULL_ID"
echo "Sweep ID: $SWEEP_ID"
echo "URL: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT/sweeps/$SWEEP_ID"
echo "=========================================="
echo ""

#-------------------------------------------------------------------------------
# Save sweep info for later reference
#-------------------------------------------------------------------------------
SWEEP_INFO_FILE="engiopt/lvae_2d/sweep_info_${PROBLEM_ID}_${SWEEP_TYPE}.txt"
cat > "$SWEEP_INFO_FILE" <<EOF
Sweep Information
=================
Created: $(date)
Type: $SWEEP_TYPE
Problem: $PROBLEM_ID
Agents: $N_AGENTS

Sweep ID: $SWEEP_ID
Full ID: $SWEEP_FULL_ID
URL: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT/sweeps/$SWEEP_ID

To submit on Euler:
  sbatch --export=SWEEP_ID=$SWEEP_ID --array=1-$N_AGENTS%50 engiopt/lvae_2d/euler_sweep.slurm

To run locally:
  wandb agent $SWEEP_FULL_ID
EOF

echo "Sweep info saved to: $SWEEP_INFO_FILE"
echo ""

#-------------------------------------------------------------------------------
# Launch on Euler (always true now)
#-------------------------------------------------------------------------------
echo "Submitting SLURM job array..."
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Determine max concurrent jobs (50% of total for safety)
MAX_CONCURRENT=$((N_AGENTS / 2))
if [ $MAX_CONCURRENT -lt 10 ]; then
    MAX_CONCURRENT=10
fi
if [ $MAX_CONCURRENT -gt 100 ]; then
    MAX_CONCURRENT=100
fi

# Submit the job array
JOB_ID=$(sbatch \
    --export=SWEEP_ID=$SWEEP_ID,WANDB_PROJECT=$WANDB_PROJECT,WANDB_ENTITY=$WANDB_ENTITY \
    --array=1-${N_AGENTS}%${MAX_CONCURRENT} \
    engiopt/lvae_2d/euler_sweep.slurm 2>&1 | grep -oP 'Submitted batch job \K[0-9]+')

if [ -n "$JOB_ID" ]; then
    echo "=========================================="
    echo "SLURM Job Submitted!"
    echo "=========================================="
    echo "Job ID: $JOB_ID"
    echo "Array size: $N_AGENTS"
    echo "Max concurrent: $MAX_CONCURRENT"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo "  squeue -j $JOB_ID"
    echo ""
    echo "Check logs:"
    echo "  tail -f logs/sweep_${JOB_ID}_*.out"
    echo ""
    echo "Cancel if needed:"
    echo "  scancel $JOB_ID"
    echo "=========================================="

    # Append job info to sweep info file
    cat >> "$SWEEP_INFO_FILE" <<EOF

SLURM Job: $JOB_ID
Submitted: $(date)
Command: sbatch --export=SWEEP_ID=$SWEEP_ID --array=1-${N_AGENTS}%${MAX_CONCURRENT} engiopt/lvae_2d/euler_sweep.slurm
EOF
else
    echo "ERROR: Failed to submit SLURM job"
    exit 1
fi
