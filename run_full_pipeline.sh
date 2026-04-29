#!/bin/bash
#SBATCH --job-name=engiopt-full-pipeline
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=600:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --array=0-9
#SBATCH --output=logs/pipeline_seed_%a_%A.log
#SBATCH --error=logs/pipeline_seed_%a_%A.err

# Full pipeline: 10 seeds
# Each seed trains: flow-matching (9 solver configs), diffusion, cGAN for both problems
# Then evaluates all and generates master report table per seed

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

# Seeds
declare -a SEEDS=(1 2 3 4 5 6 7 8 9 10)

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

# Flow-matching
echo "[$(date)] Training Flow-Matching for both problems..."
python -m engiopt.flow_matching_2d_cond.flow_matching_2d_cond \
  --seed "$SEED" \
  --num-epochs 200 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --eval-every 5 \
  --problems beams2d,heatconduction2d

# Diffusion
echo "[$(date)] Training Diffusion for both problems..."
python -m engiopt.diffusion_2d_cond.diffusion_2d_cond \
  --seed "$SEED" \
  --num-epochs 200 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --eval-every 5 \
  --problems beams2d,heatconduction2d

# cGAN
echo "[$(date)] Training cGAN for both problems..."
python -m engiopt.cgan_cnn_2d.cgan_cnn_2d \
  --seed "$SEED" \
  --num-epochs 200 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --eval-every 5 \
  --problems beams2d,heatconduction2d

echo "[$(date)] ========== Training complete =========="

# ============================================================================
# 2. MMD-BASED CHECKPOINT SELECTION (flow-matching per solver config)
# ============================================================================
echo "[$(date)] ========== PHASE 2: MMD EVALUATION (per solver config) =========="

declare -a SOLVERS=(euler euler euler midpoint midpoint midpoint rk4 rk4 rk4)
declare -a STEPS=(16 32 48 8 16 24 4 8 12)

# Evaluate top-5 checkpoints for EACH solver configuration
for i in "${!SOLVERS[@]}"; do
  FLOW_METHOD=${SOLVERS[$i]}
  FLOW_INTEGRATION_STEPS=${STEPS[$i]}
  
  echo "[$(date)] MMD selection for $FLOW_METHOD with $FLOW_INTEGRATION_STEPS steps..."
  
  python -m engiopt.flow_matching_2d_cond.evaluate_flow_matching_2d_cond \
    --checkpoint-dir "$RESULTS_ROOT/checkpoints/flow_matching" \
    --seed "$SEED" \
    --top-k 5 \
    --problems beams2d,heatconduction2d \
    --flow-method "$FLOW_METHOD" \
    --flow-integration-steps "$FLOW_INTEGRATION_STEPS" \
    --output-rankings "rankings_${FLOW_METHOD}_steps${FLOW_INTEGRATION_STEPS}.json"
done

echo "[$(date)] MMD evaluation complete for all solver configs."

# ============================================================================
# 3. EVALUATION: All models with all solver configs on test set
# ============================================================================
echo "[$(date)] ========== PHASE 3: TEST EVALUATION =========="

declare -a SOLVERS=(euler euler euler midpoint midpoint midpoint rk4 rk4 rk4)
declare -a STEPS=(16 32 48 8 16 24 4 8 12)

# Flow-matching with all solver configs
for i in "${!SOLVERS[@]}"; do
  FLOW_METHOD=${SOLVERS[$i]}
  FLOW_INTEGRATION_STEPS=${STEPS[$i]}
  
  echo "[$(date)] Evaluating Flow-Matching: $FLOW_METHOD steps=$FLOW_INTEGRATION_STEPS"
  
  python -m engiopt.report_metrics \
    --problems beams2d,heatconduction2d \
    --seed "$SEED" \
    --split test \
    --checkpoint-dir "$RESULTS_ROOT/checkpoints/flow_matching" \
    --flow-method "$FLOW_METHOD" \
    --flow-integration-steps "$FLOW_INTEGRATION_STEPS" \
    --output-dir "$CSV_SHARD_DIR" \
    --run-name "flow_${FLOW_METHOD}_steps${FLOW_INTEGRATION_STEPS}_seed${SEED}"
done

# Diffusion
echo "[$(date)] Evaluating Diffusion..."
python -m engiopt.report_metrics \
  --problems beams2d,heatconduction2d \
  --seed "$SEED" \
  --split test \
  --checkpoint-dir "$RESULTS_ROOT/checkpoints/diffusion" \
  --output-dir "$CSV_SHARD_DIR" \
  --run-name "diffusion_seed${SEED}"

# cGAN
echo "[$(date)] Evaluating cGAN..."
python -m engiopt.report_metrics \
  --problems beams2d,heatconduction2d \
  --seed "$SEED" \
  --split test \
  --checkpoint-dir "$RESULTS_ROOT/checkpoints/cgan" \
  --output-dir "$CSV_SHARD_DIR" \
  --run-name "cgan_seed${SEED}"

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
for i in "${!SOLVERS[@]}"; do
  FLOW_METHOD=${SOLVERS[$i]}
  FLOW_INTEGRATION_STEPS=${STEPS[$i]}
  
  echo "[$(date)] Exporting Flow-Matching qualitative: $FLOW_METHOD steps=$FLOW_INTEGRATION_STEPS"
  
  BUNDLE_NAME="flow_${FLOW_METHOD}_steps${FLOW_INTEGRATION_STEPS}_seed${SEED}"
  RANKINGS_FILE="rankings_${FLOW_METHOD}_steps${FLOW_INTEGRATION_STEPS}.json"
  
  python -m engiopt.export_qualitative_bundle_2d \
    --problems beams2d,heatconduction2d \
    --seed "$SEED" \
    --checkpoint-dir "$RESULTS_ROOT/checkpoints/flow_matching" \
    --select-best-of-top-k \
    --top-k-rankings "$RANKINGS_FILE" \
    --flow-method "$FLOW_METHOD" \
    --flow-integration-steps "$FLOW_INTEGRATION_STEPS" \
    "${WANDB_ARGS[@]}" \
    --output-dir "$QUALITATIVE_ROOT/$BUNDLE_NAME" \
    --upload-wandb
done

# Diffusion
echo "[$(date)] Exporting Diffusion qualitative..."
python -m engiopt.export_qualitative_bundle_2d \
  --problems beams2d,heatconduction2d \
  --seed "$SEED" \
  --checkpoint-dir "$RESULTS_ROOT/checkpoints/diffusion" \
  "${WANDB_ARGS[@]}" \
  --output-dir "$QUALITATIVE_ROOT/diffusion_seed${SEED}" \
  --upload-wandb

# cGAN
echo "[$(date)] Exporting cGAN qualitative..."
python -m engiopt.export_qualitative_bundle_2d \
  --problems beams2d,heatconduction2d \
  --seed "$SEED" \
  --checkpoint-dir "$RESULTS_ROOT/checkpoints/cgan" \
  "${WANDB_ARGS[@]}" \
  --output-dir "$QUALITATIVE_ROOT/cgan_seed${SEED}" \
  --upload-wandb

echo "[$(date)] Qualitative export complete."

# ============================================================================
# 5. MASTER REPORT AGGREGATION: All models × all configs in one table per seed
# ============================================================================
echo "[$(date)] ========== PHASE 5: MASTER REPORT AGGREGATION =========="

python -m engiopt.report_metrics \
  --shard-dir "$CSV_SHARD_DIR" \
  --output-dir "$REPORT_DIR" \
  --problem-id beams2d,heatconduction2d \
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
