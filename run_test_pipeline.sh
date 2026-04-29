#!/bin/bash
#SBATCH --job-name=engiopt-test-pipeline
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --array=0-1
#SBATCH --output=logs/test_pipeline_%a_%A.log
#SBATCH --error=logs/test_pipeline_%a_%A.err

# Quick test: flow-matching euler 16 & 32 steps for 1 seed, both problems
# Shorter training (50 epochs) for faster validation of pipeline

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "$SCRIPT_DIR/common_flow_matching_env.sh"

ensure_flow_layout
load_flow_modules
activate_flow_venv
load_flow_secrets
configure_flow_caches

# Test configs: euler with 16 and 32 steps
declare -a STEPS=(16 32)

FLOW_INTEGRATION_STEPS=${STEPS[$SLURM_ARRAY_TASK_ID]}
SEED=1
FLOW_METHOD=euler

echo "=========================================="
echo "Starting test pipeline: Seed=$SEED, Method=$FLOW_METHOD, Steps=$FLOW_INTEGRATION_STEPS"
echo "=========================================="

export SEED
export PYTHONPATH="$FLOW_ENGIBENCH_DIR:$FLOW_ENGIOPT_DIR:${PYTHONPATH:-}"
cd "$FLOW_ENGIOPT_DIR"

RESULTS_ROOT="${FLOW_RESULTS_DIR}/test_seed_${SEED}"
QUALITATIVE_ROOT="${RESULTS_ROOT}/qualitative_bundles"
REPORT_DIR="${RESULTS_ROOT}/reports"
CSV_SHARD_DIR="${RESULTS_ROOT}/csv_shards"

mkdir -p logs "$RESULTS_ROOT" "$QUALITATIVE_ROOT" "$REPORT_DIR" "$CSV_SHARD_DIR"

# ============================================================================
# 1. TRAIN FLOW-MATCHING ONLY (for quick test)
# ============================================================================
echo "[$(date)] ========== PHASE 1: TRAINING (Flow-Matching only, 50 epochs) =========="

echo "[$(date)] Training Flow-Matching for both problems..."
python -m engiopt.flow_matching_2d_cond.flow_matching_2d_cond \
  --seed "$SEED" \
  --num-epochs 50 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --eval-every 5 \
  --problems beams2d,heatconduction2d

echo "[$(date)] ========== Training complete =========="

# ============================================================================
# 2. MMD-BASED CHECKPOINT SELECTION (flow-matching per solver config)
# ============================================================================
echo "[$(date)] ========== PHASE 2: MMD EVALUATION (per solver config) =========="

# Test configs: euler with 16 and 32 steps
declare -a TEST_STEPS=(16 32)

for STEP in "${TEST_STEPS[@]}"; do
  echo "[$(date)] MMD selection for $FLOW_METHOD with $STEP steps..."
  
  python -m engiopt.flow_matching_2d_cond.evaluate_flow_matching_2d_cond \
    --checkpoint-dir "$RESULTS_ROOT/checkpoints/flow_matching" \
    --seed "$SEED" \
    --top-k 5 \
    --problems beams2d,heatconduction2d \
    --flow-method "$FLOW_METHOD" \
    --flow-integration-steps "$STEP" \
    --output-rankings "rankings_${FLOW_METHOD}_steps${STEP}.json"
done

echo "[$(date)] MMD evaluation complete for test solver configs."

# ============================================================================
# 3. EVALUATION: Flow-matching euler only on test set
# ============================================================================
echo "[$(date)] ========== PHASE 3: TEST EVALUATION =========="

# Flow-matching test configs only
echo "[$(date)] Evaluating Flow-Matching: $FLOW_METHOD steps=$FLOW_INTEGRATION_STEPS"

python -m engiopt.report_metrics \
  --problems beams2d,heatconduction2d \
  --seed "$SEED" \
  --split test \
  --checkpoint-dir "$RESULTS_ROOT/checkpoints/flow_matching" \
  --flow-method "$FLOW_METHOD" \
  --flow-integration-steps "$FLOW_INTEGRATION_STEPS" \
  --output-dir "$CSV_SHARD_DIR" \
  --run-name "test_flow_${FLOW_METHOD}_steps${FLOW_INTEGRATION_STEPS}_seed${SEED}"

echo "[$(date)] Test evaluation complete."

# ============================================================================
# 4. QUALITATIVE EXPORT: Flow-matching only
# ============================================================================
echo "[$(date)] ========== PHASE 4: QUALITATIVE EXPORT =========="

WANDB_ARGS=(--wandb-project "${WANDB_PROJECT:-engiopt}")
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

# Flow-matching test configs using solver-specific top-5 rankings
declare -a TEST_STEPS=(16 32)

for STEP in "${TEST_STEPS[@]}"; do
  echo "[$(date)] Exporting Flow-Matching qualitative: $FLOW_METHOD steps=$STEP"
  
  BUNDLE_NAME="test_flow_${FLOW_METHOD}_steps${STEP}_seed${SEED}"
  RANKINGS_FILE="rankings_${FLOW_METHOD}_steps${STEP}.json"
  
  python -m engiopt.export_qualitative_bundle_2d \
    --problems beams2d,heatconduction2d \
    --seed "$SEED" \
    --checkpoint-dir "$RESULTS_ROOT/checkpoints/flow_matching" \
    --select-best-of-top-k \
    --top-k-rankings "$RANKINGS_FILE" \
    --flow-method "$FLOW_METHOD" \
    --flow-integration-steps "$STEP" \
    "${WANDB_ARGS[@]}" \
    --output-dir "$QUALITATIVE_ROOT/$BUNDLE_NAME" \
    --upload-wandb
done

echo "[$(date)] Qualitative export complete."

# ============================================================================
# 5. REPORT: Flow-matching only (euler 16 & 32)
# ============================================================================
echo "[$(date)] ========== PHASE 5: TEST REPORT =========="

python -m engiopt.report_metrics \
  --shard-dir "$CSV_SHARD_DIR" \
  --output-dir "$REPORT_DIR" \
  --problem-id beams2d,heatconduction2d \
  "${WANDB_ARGS[@]}" \
  --upload-wandb \
  --run-name "test_report_euler_seed${SEED}" \
  --artifact-name "test_report_euler_seed${SEED}"

echo "[$(date)] Test report generated at: $REPORT_DIR"

# ============================================================================
# 6. SUMMARY
# ============================================================================
echo "[$(date)] =========================================="
echo "[$(date)] Test pipeline COMPLETE for Seed=$SEED"
echo "[$(date)] Model: Flow-Matching Euler (16 & 32 steps)"
echo "[$(date)] Results: $RESULTS_ROOT"
echo "[$(date)] Report:  $REPORT_DIR"
echo "[$(date)] =========================================="
