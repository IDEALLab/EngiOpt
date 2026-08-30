#!/usr/bin/env bash
#SBATCH --job-name=cfm_train
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7G
#SBATCH --gpus=rtx_4090:1

set -euo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPRO_DIR/common.sh"
setup_reproduction_env

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task}"
CAMPAIGN_MODE="${CAMPAIGN_MODE:-full}"
N_CPU="${SLURM_CPUS_PER_TASK:-8}"

if [[ "$CAMPAIGN_MODE" == "smoke" ]]; then
    N_EPOCHS="${N_EPOCHS:-5}"
    VALIDATION_INTERVAL_EPOCHS="${VALIDATION_INTERVAL_EPOCHS:-1}"
    CHECKPOINT_INTERVAL_EPOCHS="${CHECKPOINT_INTERVAL_EPOCHS:-1}"
    MIN_EPOCH_FOR_SELECTION="${MIN_EPOCH_FOR_SELECTION:-1}"
    EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
    VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-2}"
    case "$TASK_ID" in
        0) MODEL_ID=flow_matching_2d_cond; PROBLEM_ID=beams2d; SEED=1; METHOD=euler; STEPS=32 ;;
        1) MODEL_ID=diffusion_2d_cond; PROBLEM_ID=beams2d; SEED=1; METHOD=na; STEPS=0 ;;
        2) MODEL_ID=cgan_cnn_2d; PROBLEM_ID=beams2d; SEED=1; METHOD=na; STEPS=0 ;;
        *) echo "ERROR: Smoke training task must be in 0..2, got $TASK_ID" >&2; exit 1 ;;
    esac
else
    N_EPOCHS="${N_EPOCHS:-500}"
    VALIDATION_INTERVAL_EPOCHS="${VALIDATION_INTERVAL_EPOCHS:-10}"
    CHECKPOINT_INTERVAL_EPOCHS="${CHECKPOINT_INTERVAL_EPOCHS:-10}"
    MIN_EPOCH_FOR_SELECTION="${MIN_EPOCH_FOR_SELECTION:-80}"
    EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-25}"
    VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-50}"
    if (( TASK_ID < 0 || TASK_ID >= 220 )); then
        echo "ERROR: Full training task must be in 0..219, got $TASK_ID" >&2
        exit 1
    fi
    if (( TASK_ID < 180 )); then
        CONFIGS=("euler:16" "euler:32" "euler:48" "midpoint:8" "midpoint:16" "midpoint:24" "rk4:4" "rk4:8" "rk4:12")
        PROBLEMS=(beams2d heatconduction2d)
        CONFIG_INDEX=$(( TASK_ID / 20 ))
        PROBLEM_INDEX=$(( (TASK_ID / 10) % 2 ))
        SEED=$(( TASK_ID % 10 + 1 ))
        IFS=: read -r METHOD STEPS <<< "${CONFIGS[$CONFIG_INDEX]}"
        MODEL_ID=flow_matching_2d_cond
        PROBLEM_ID="${PROBLEMS[$PROBLEM_INDEX]}"
    else
        BASELINE_TASK=$(( TASK_ID - 180 ))
        MODELS=(diffusion_2d_cond cgan_cnn_2d)
        PROBLEMS=(beams2d heatconduction2d)
        MODEL_INDEX=$(( BASELINE_TASK / 20 ))
        PROBLEM_INDEX=$(( (BASELINE_TASK / 10) % 2 ))
        SEED=$(( BASELINE_TASK % 10 + 1 ))
        MODEL_ID="${MODELS[$MODEL_INDEX]}"
        PROBLEM_ID="${PROBLEMS[$PROBLEM_INDEX]}"
        METHOD=na
        STEPS=0
    fi
fi

if [[ "$PROBLEM_ID" == "heatconduction2d" ]]; then
    VALIDATION_SIGMA=10.0
else
    VALIDATION_SIGMA=1.0
fi

if [[ "$MODEL_ID" == "flow_matching_2d_cond" ]]; then
    PACKAGE_LABEL="${METHOD}_${STEPS}"
    CHECKPOINT_DIR="$FLOW_CHECKPOINT_ROOT/$PACKAGE_LABEL/$MODEL_ID/$PROBLEM_ID/seed_$SEED"
else
    CHECKPOINT_DIR="$CHECKPOINT_ROOT/$MODEL_ID/$PROBLEM_ID/seed_$SEED"
fi
mkdir -p "$CHECKPOINT_DIR"

build_wandb_args
COMMON_ARGS=(
    --problem-id "$PROBLEM_ID"
    --seed "$SEED"
    "${WANDB_ARGS[@]}"
    --save-model
    --n-epochs "$N_EPOCHS"
    --n-cpu "$N_CPU"
    --enable-best-epoch-selection
    --checkpoint-dir "$CHECKPOINT_DIR"
    --checkpoint-interval-epochs "$CHECKPOINT_INTERVAL_EPOCHS"
    --validation-interval-epochs "$VALIDATION_INTERVAL_EPOCHS"
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
    --min-epoch-for-best-selection "$MIN_EPOCH_FOR_SELECTION"
    --validation-log-precision 10
    --validation-batch-size "$VALIDATION_BATCH_SIZE"
    --validation-sigma "$VALIDATION_SIGMA"
    --checkpoint-backend none
    --checkpoint-archive-mode eval
)

echo "Training task=$TASK_ID model=$MODEL_ID problem=$PROBLEM_ID seed=$SEED method=$METHOD steps=$STEPS"
cd "$ENGIOPT_DIR"
case "$MODEL_ID" in
    flow_matching_2d_cond)
        python engiopt/flow_matching_2d_cond/flow_matching_2d_cond.py \
            "${COMMON_ARGS[@]}" \
            --method "$METHOD" \
            --integration-steps "$STEPS" \
            --checkpoint-path "$CHECKPOINT_DIR/final_model.pth" \
            --checkpoint-package-label "${METHOD}_${STEPS}" \
            --device cuda
        ;;
    diffusion_2d_cond)
        python engiopt/diffusion_2d_cond/diffusion_2d_cond.py \
            "${COMMON_ARGS[@]}" \
            --num-timesteps 1000 \
            --checkpoint-path "$CHECKPOINT_DIR/final_model.pth" \
            --device cuda
        ;;
    cgan_cnn_2d)
        python engiopt/cgan_cnn_2d/cgan_cnn_2d.py \
            "${COMMON_ARGS[@]}" \
            --generator-output-activation sigmoid \
            --generator-checkpoint-path "$CHECKPOINT_DIR/final_generator.pth" \
            --discriminator-checkpoint-path "$CHECKPOINT_DIR/final_discriminator.pth" \
            --device cuda
        ;;
    *)
        echo "ERROR: Unsupported model: $MODEL_ID" >&2
        exit 1
        ;;
esac
