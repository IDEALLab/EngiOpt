#!/bin/bash
#SBATCH --job-name=engiopt-full-pipeline
#SBATCH --partition=cuda13pr.24h
#SBATCH --time=600:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --array=0-9
#SBATCH --output=logs/pipeline_seed_%a_%A.log
#SBATCH --error=logs/pipeline_seed_%a_%A.err

# Full pipeline: 10 seeds
# Each seed trains: flow-matching (9 solver configs), diffusion, cGAN for both problems
# Then evaluates all and generates master report table per seed

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

WANDB_PROJECT_VALUE="${WANDB_PROJECT:-engiopt}"
WANDB_ENTITY_ARGS=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_ENTITY_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

# Seeds
declare -a SEEDS=(1 2 3 4 5 6 7 8 9 10)
declare -a PROBLEMS=(beams2d heatconduction2d)

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Starting full pipeline for Seed=$SEED"
echo "=========================================="

export SEED
export PYTHONPATH="$FLOW_ENGIBENCH_DIR:$FLOW_ENGIOPT_DIR:${PYTHONPATH:-}"
cd "$FLOW_ENGIOPT_DIR"

RESULTS_ROOT="${FLOW_RESULTS_DIR}/seed_${SEED}"
QUALITATIVE_ROOT="${RESULTS_ROOT}/qualitative_bundles"
REPORT_DIR="${RESULTS_ROOT}/reports"
CSV_SHARD_DIR="${RESULTS_ROOT}/csv_shards"

mkdir -p logs "$RESULTS_ROOT" "$QUALITATIVE_ROOT" "$REPORT_DIR" "$CSV_SHARD_DIR"

# ============================================================================
# 1. TRAIN ALL MODELS (flow-matching, diffusion, cGAN) for both problems
# ============================================================================
echo "[$(date)] ========== PHASE 1: TRAINING =========="

for PROBLEM_ID in "${PROBLEMS[@]}"; do
  FLOW_CHECKPOINT_DIR="$RESULTS_ROOT/checkpoints/flow_matching/$PROBLEM_ID"
  DIFFUSION_CHECKPOINT_DIR="$RESULTS_ROOT/checkpoints/diffusion/$PROBLEM_ID"
  CGAN_CHECKPOINT_DIR="$RESULTS_ROOT/checkpoints/cgan/$PROBLEM_ID"
  mkdir -p "$FLOW_CHECKPOINT_DIR" "$DIFFUSION_CHECKPOINT_DIR" "$CGAN_CHECKPOINT_DIR"

  echo "[$(date)] Training Flow-Matching for $PROBLEM_ID..."
  python -m engiopt.flow_matching_2d_cond.flow_matching_2d_cond \
    --problem-id "$PROBLEM_ID" \
    --seed "$SEED" \
    --n-epochs 200 \
    --batch-size 32 \
    --lr 1e-3 \
    --validation-interval-epochs 5 \
    --checkpoint-interval-epochs 5 \
    --checkpoint-dir "$FLOW_CHECKPOINT_DIR" \
    --track \
    --wandb-project "$WANDB_PROJECT_VALUE" \
    "${WANDB_ENTITY_ARGS[@]}"

  echo "[$(date)] Training Diffusion for $PROBLEM_ID..."
  python -m engiopt.diffusion_2d_cond.diffusion_2d_cond \
    --problem-id "$PROBLEM_ID" \
    --seed "$SEED" \
    --n-epochs 200 \
    --batch-size 32 \
    --lr 1e-3 \
    --validation-interval-epochs 5 \
    --checkpoint-interval-epochs 5 \
    --checkpoint-dir "$DIFFUSION_CHECKPOINT_DIR" \
    --save-model \
    --checkpoint-path "$DIFFUSION_CHECKPOINT_DIR/model.pth" \
    --track \
    --wandb-project "$WANDB_PROJECT_VALUE" \
    "${WANDB_ENTITY_ARGS[@]}"

  echo "[$(date)] Training cGAN for $PROBLEM_ID..."
  python -m engiopt.cgan_2d.cgan_2d \
    --problem-id "$PROBLEM_ID" \
    --seed "$SEED" \
    --n-epochs 200 \
    --batch-size 32 \
    --lr-gen 1e-4 \
    --lr-disc 4e-4 \
    --checkpoint-interval-epochs 5 \
    --checkpoint-dir "$CGAN_CHECKPOINT_DIR" \
    --save-model \
    --generator-checkpoint-path "$CGAN_CHECKPOINT_DIR/generator.pth" \
    --discriminator-checkpoint-path "$CGAN_CHECKPOINT_DIR/discriminator.pth" \
    --track \
    --wandb-project "$WANDB_PROJECT_VALUE" \
    "${WANDB_ENTITY_ARGS[@]}"
done

echo "[$(date)] ========== Training complete =========="

# ============================================================================
# 2. MMD-BASED CHECKPOINT SELECTION (flow-matching per solver config)
# ============================================================================
echo "[$(date)] ========== PHASE 2: MMD EVALUATION (per solver config) =========="

declare -a SOLVERS=(euler euler euler midpoint midpoint midpoint rk4 rk4 rk4)
declare -a STEPS=(16 32 48 8 16 24 4 8 12)

# Evaluate top-5 checkpoints for EACH solver configuration
for PROBLEM_ID in "${PROBLEMS[@]}"; do
  FLOW_CHECKPOINT_DIR="$RESULTS_ROOT/checkpoints/flow_matching/$PROBLEM_ID"
  for i in "${!SOLVERS[@]}"; do
    FLOW_METHOD=${SOLVERS[$i]}
    FLOW_INTEGRATION_STEPS=${STEPS[$i]}

    echo "[$(date)] MMD selection for $PROBLEM_ID $FLOW_METHOD with $FLOW_INTEGRATION_STEPS steps..."

    python -m engiopt.flow_matching_2d_cond.evaluate_flow_matching_2d_cond \
      --checkpoint-dir "$FLOW_CHECKPOINT_DIR" \
      --problem-id "$PROBLEM_ID" \
      --seed "$SEED" \
      --top-k 5 \
      --method "$FLOW_METHOD" \
      --integration-steps "$FLOW_INTEGRATION_STEPS" \
      --output-csv "$CSV_SHARD_DIR/flow_${PROBLEM_ID}_${FLOW_METHOD}_steps${FLOW_INTEGRATION_STEPS}_metrics.csv" \
      --track \
      --wandb-project "$WANDB_PROJECT_VALUE" \
      "${WANDB_ENTITY_ARGS[@]}"
  done
done

echo "[$(date)] MMD evaluation complete for all solver configs."

# ============================================================================
# 3. EVALUATION: All models with all solver configs on test set
# ============================================================================
echo "[$(date)] ========== PHASE 3: TEST EVALUATION =========="

for PROBLEM_ID in "${PROBLEMS[@]}"; do
  FLOW_CHECKPOINT_DIR="$RESULTS_ROOT/checkpoints/flow_matching/$PROBLEM_ID"
  DIFFUSION_CHECKPOINT_DIR="$RESULTS_ROOT/checkpoints/diffusion/$PROBLEM_ID"
  CGAN_CHECKPOINT_PATH="$RESULTS_ROOT/checkpoints/cgan/$PROBLEM_ID/generator.pth"

  for i in "${!SOLVERS[@]}"; do
    FLOW_METHOD=${SOLVERS[$i]}
    FLOW_INTEGRATION_STEPS=${STEPS[$i]}

    echo "[$(date)] Evaluating Flow-Matching: $PROBLEM_ID $FLOW_METHOD steps=$FLOW_INTEGRATION_STEPS"

    python -m engiopt.flow_matching_2d_cond.evaluate_flow_matching_2d_cond \
      --checkpoint-dir "$FLOW_CHECKPOINT_DIR" \
      --problem-id "$PROBLEM_ID" \
      --seed "$SEED" \
      --method "$FLOW_METHOD" \
      --integration-steps "$FLOW_INTEGRATION_STEPS" \
      --output-csv "$CSV_SHARD_DIR/${PROBLEM_ID}_flow_${FLOW_METHOD}_steps${FLOW_INTEGRATION_STEPS}_metrics.csv" \
      --track \
      --wandb-project "$WANDB_PROJECT_VALUE" \
      "${WANDB_ENTITY_ARGS[@]}"
  done

  echo "[$(date)] Evaluating Diffusion for $PROBLEM_ID..."
  python -m engiopt.diffusion_2d_cond.evaluate_diffusion_2d_cond \
    --problem-id "$PROBLEM_ID" \
    --seed "$SEED" \
    --checkpoint-dir "$DIFFUSION_CHECKPOINT_DIR" \
    --select-best-of-top-k \
    --top-k 5 \
    --output-csv "$CSV_SHARD_DIR/${PROBLEM_ID}_diffusion_metrics.csv" \
    --track \
    --wandb-project "$WANDB_PROJECT_VALUE" \
    "${WANDB_ENTITY_ARGS[@]}"

  echo "[$(date)] Evaluating cGAN for $PROBLEM_ID..."
  python -m engiopt.cgan_2d.evaluate_cgan_2d \
    --problem-id "$PROBLEM_ID" \
    --seed "$SEED" \
    --checkpoint-path "$CGAN_CHECKPOINT_PATH" \
    --output-csv "$CSV_SHARD_DIR/${PROBLEM_ID}_cgan_metrics.csv"
done

echo "[$(date)] Test evaluation complete."

# ============================================================================
# 4. QUALITATIVE EXPORT: All models with all solver configs
# ============================================================================
echo "[$(date)] ========== PHASE 4: QUALITATIVE EXPORT =========="

WANDB_ARGS=(--wandb-project "${WANDB_PROJECT:-engiopt}")
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

# Flow-matching with all solver configs using solver-specific top-5 rankings
for PROBLEM_ID in "${PROBLEMS[@]}"; do
  FLOW_CHECKPOINT_DIR="$RESULTS_ROOT/checkpoints/flow_matching/$PROBLEM_ID"
  DIFFUSION_CHECKPOINT_DIR="$RESULTS_ROOT/checkpoints/diffusion/$PROBLEM_ID"
  CGAN_CHECKPOINT_PATH="$RESULTS_ROOT/checkpoints/cgan/$PROBLEM_ID/generator.pth"

  for i in "${!SOLVERS[@]}"; do
    FLOW_METHOD=${SOLVERS[$i]}
    FLOW_INTEGRATION_STEPS=${STEPS[$i]}

    echo "[$(date)] Exporting Flow-Matching qualitative: $PROBLEM_ID $FLOW_METHOD steps=$FLOW_INTEGRATION_STEPS"

    BUNDLE_NAME="${PROBLEM_ID}_flow_${FLOW_METHOD}_steps${FLOW_INTEGRATION_STEPS}_seed${SEED}"

    python -m engiopt.export_qualitative_bundle_2d \
      --problems "$PROBLEM_ID" \
      --seed "$SEED" \
      --checkpoint-dir "$FLOW_CHECKPOINT_DIR" \
      --select-best-of-top-k \
      --top-k 5 \
      --flow-method "$FLOW_METHOD" \
      --flow-integration-steps "$FLOW_INTEGRATION_STEPS" \
      --track \
      "${WANDB_ARGS[@]}" \
      --output-dir "$QUALITATIVE_ROOT/$BUNDLE_NAME"
  done

  echo "[$(date)] Exporting Diffusion qualitative for $PROBLEM_ID..."
  python -m engiopt.export_qualitative_bundle_2d \
    --problems "$PROBLEM_ID" \
    --seed "$SEED" \
    --diffusion-checkpoint-path "$DIFFUSION_CHECKPOINT_DIR/model.pth" \
    --track \
    "${WANDB_ARGS[@]}" \
      --output-dir "$QUALITATIVE_ROOT/${PROBLEM_ID}_diffusion_seed${SEED}"

  echo "[$(date)] Exporting cGAN qualitative for $PROBLEM_ID..."
  python -m engiopt.export_qualitative_bundle_2d \
    --problems "$PROBLEM_ID" \
    --seed "$SEED" \
    --cgan-checkpoint-path "$CGAN_CHECKPOINT_DIR/generator.pth" \
    --track \
    "${WANDB_ARGS[@]}" \
      --output-dir "$QUALITATIVE_ROOT/${PROBLEM_ID}_cgan_seed${SEED}"
done
echo "[$(date)] Qualitative export complete."

# ============================================================================
# 5. MASTER REPORT AGGREGATION: All models × all configs in one table per seed
# ============================================================================
echo "[$(date)] ========== PHASE 5: MASTER REPORT AGGREGATION =========="

python -m engiopt.report_metrics \
  --shard-dir "$CSV_SHARD_DIR" \
  --output-dir "$REPORT_DIR" \
  "${WANDB_ARGS[@]}" \
  --upload-wandb \
  --run-name "master_report_seed${SEED}" \
  --artifact-name "master_report_seed${SEED}" \
  --aggregate-all-models \
  --seed "$SEED"

echo "[$(date)] Master report generated at: $REPORT_DIR"

# ============================================================================
# 6. SUMMARY
# ============================================================================
echo "[$(date)] =========================================="
echo "[$(date)] Full pipeline COMPLETE for Seed=$SEED"
echo "[$(date)] Results: $RESULTS_ROOT"
echo "[$(date)] Report:  $REPORT_DIR"
echo "[$(date)] =========================================="
