#!/bin/bash
#SBATCH --job-name=engiopt-test-pipeline
#SBATCH --partition=cuda13pr.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --array=0-1
#SBATCH --output=logs/test_pipeline_%a_%A.log
#SBATCH --error=logs/test_pipeline_%a_%A.err

# Quick test: flow-matching euler 16 & 32 steps for 1 seed, both problems
# Shorter training (50 epochs) for faster validation of pipeline

set -euo pipefail

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
declare -a PROBLEMS=(beams2d heatconduction2d)
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
for PROBLEM_ID in "${PROBLEMS[@]}"; do
  CHECKPOINT_DIR="$RESULTS_ROOT/checkpoints/flow_matching/$PROBLEM_ID"
  mkdir -p "$CHECKPOINT_DIR"
  python -m engiopt.flow_matching_2d_cond.flow_matching_2d_cond \
    --problem-id "$PROBLEM_ID" \
    --seed "$SEED" \
    --n-epochs 50 \
  --batch-size 32 \
    --lr 1e-3 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --checkpoint-interval-epochs 5 \
    --validation-interval-epochs 5 \
    --track
done

echo "[$(date)] ========== Training complete =========="

# ============================================================================
# 2. CHECKPOINTS ARE SELECTED DURING EVALUATION
# ============================================================================
echo "[$(date)] ========== PHASE 2: TOP-K CHECKPOINTS WILL BE SELECTED IN EVALUATION =========="

# ============================================================================
# 3. EVALUATION: Flow-matching Euler on test set
# ============================================================================
echo "[$(date)] ========== PHASE 3: TEST EVALUATION =========="

for PROBLEM_ID in "${PROBLEMS[@]}"; do
  CHECKPOINT_DIR="$RESULTS_ROOT/checkpoints/flow_matching/$PROBLEM_ID"
  for STEP in "${STEPS[@]}"; do
    echo "[$(date)] Evaluating Flow-Matching: $PROBLEM_ID $FLOW_METHOD steps=$STEP"

    python -m engiopt.flow_matching_2d_cond.evaluate_flow_matching_2d_cond \
      --checkpoint-dir "$CHECKPOINT_DIR" \
      --problem-id "$PROBLEM_ID" \
      --seed "$SEED" \
      --method "$FLOW_METHOD" \
      --integration-steps "$STEP" \
      --output-csv "$CSV_SHARD_DIR/test_${PROBLEM_ID}_${FLOW_METHOD}_steps${STEP}_metrics.csv" \
      --track
  done
done

echo "[$(date)] Test evaluation complete."

# ============================================================================
# 4. QUALITATIVE EXPORT: Flow-matching only
# ============================================================================
echo "[$(date)] ========== PHASE 4: QUALITATIVE EXPORT =========="

WANDB_ARGS=(--wandb-project "${WANDB_PROJECT:-engiopt}")
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

for PROBLEM_ID in "${PROBLEMS[@]}"; do
  CHECKPOINT_DIR="$RESULTS_ROOT/checkpoints/flow_matching/$PROBLEM_ID"
  for STEP in "${STEPS[@]}"; do
    echo "[$(date)] Exporting Flow-Matching qualitative: $PROBLEM_ID $FLOW_METHOD steps=$STEP"

    BUNDLE_NAME="test_${PROBLEM_ID}_flow_${FLOW_METHOD}_steps${STEP}_seed${SEED}"

    python -m engiopt.export_qualitative_bundle_2d \
      --problems "$PROBLEM_ID" \
      --seed "$SEED" \
      --checkpoint-dir "$CHECKPOINT_DIR" \
      --select-best-of-top-k \
      --top-k 5 \
      --flow-method "$FLOW_METHOD" \
      --flow-integration-steps "$STEP" \
      --track \
      "${WANDB_ARGS[@]}" \
      --output-dir "$QUALITATIVE_ROOT/$BUNDLE_NAME"
  done
done

echo "[$(date)] Qualitative export complete."

# ============================================================================
# 5. REPORT: Aggregate all test shards
# ============================================================================
echo "[$(date)] ========== PHASE 5: TEST REPORT =========="

python -m engiopt.report_metrics \
  --shard-dir "$CSV_SHARD_DIR" \
  --output-dir "$REPORT_DIR" \
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
