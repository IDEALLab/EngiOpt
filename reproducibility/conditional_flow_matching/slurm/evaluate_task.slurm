#!/usr/bin/env bash
#SBATCH --job-name=cfm_eval
#SBATCH --time=16:00:00
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
TOP_K="${TOP_K:-5}"
SELECTED_CHECKPOINT_RELEASE="${SELECTED_CHECKPOINT_RELEASE:-v1}"

if [[ "$CAMPAIGN_MODE" == "smoke" ]]; then
    N_SAMPLES="${N_SAMPLES:-2}"
    SELECTION_BATCH_SIZE="${SELECTION_BATCH_SIZE:-2}"
    case "$TASK_ID" in
        0) MODEL_ID=flow_matching_2d_cond; PROBLEM_ID=beams2d; SEED=1; METHOD=euler; STEPS=32 ;;
        1) MODEL_ID=diffusion_2d_cond; PROBLEM_ID=beams2d; SEED=1; METHOD=na; STEPS=0 ;;
        2) MODEL_ID=cgan_cnn_2d; PROBLEM_ID=beams2d; SEED=1; METHOD=na; STEPS=0 ;;
        *) echo "ERROR: Smoke evaluation task must be in 0..2, got $TASK_ID" >&2; exit 1 ;;
    esac
else
    N_SAMPLES="${N_SAMPLES:-50}"
    SELECTION_BATCH_SIZE="${SELECTION_BATCH_SIZE:-50}"
    if (( TASK_ID < 0 || TASK_ID >= 220 )); then
        echo "ERROR: Full evaluation task must be in 0..219, got $TASK_ID" >&2
        exit 1
    fi
    if (( TASK_ID < 180 )); then
        CONFIGS=("euler:16" "euler:32" "euler:48" "midpoint:8" "midpoint:16" "midpoint:24" "rk4:4" "rk4:8" "rk4:12")
        PROBLEMS=(beams2d heatconduction2d)
        CONFIG_INDEX=$(( TASK_ID % 9 ))
        PROBLEM_INDEX=$(( (TASK_ID / 9) % 2 ))
        SEED=$(( TASK_ID / 18 + 1 ))
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
    SIGMA=10.0
    stage_pyadjoint_image "$ENGIOPT_DIR"
else
    SIGMA=1.0
fi

if [[ "$MODEL_ID" == "flow_matching_2d_cond" ]]; then
    PACKAGE_LABEL="${METHOD}_${STEPS}"
    CHECKPOINT_DIR="$FLOW_CHECKPOINT_ROOT/$PACKAGE_LABEL/$MODEL_ID/$PROBLEM_ID/seed_$SEED"
    if [[ "$CAMPAIGN_MODE" == "smoke" ]]; then
        OUTPUT_CSV="$CSV_DIR/smoke_${MODEL_ID}_${PROBLEM_ID}_seed${SEED}.csv"
    else
        OUTPUT_CSV="$CSV_DIR/tourn_${PROBLEM_ID}_${METHOD}_${STEPS}_s${SEED}.csv"
    fi
else
    CHECKPOINT_DIR="$CHECKPOINT_ROOT/$MODEL_ID/$PROBLEM_ID/seed_$SEED"
    PACKAGE_LABEL=timesteps_1000
    if [[ "$MODEL_ID" == "cgan_cnn_2d" ]]; then
        PACKAGE_LABEL=activation_sigmoid
    fi
    if [[ "$CAMPAIGN_MODE" == "smoke" ]]; then
        OUTPUT_CSV="$CSV_DIR/smoke_${MODEL_ID}_${PROBLEM_ID}_seed${SEED}.csv"
    else
        OUTPUT_CSV="$CSV_DIR/metrics_${MODEL_ID}_na_${PROBLEM_ID}_seed${SEED}.csv"
    fi
fi

build_wandb_args
COMMON_ARGS=(
    --problem-id "$PROBLEM_ID"
    --seed "$SEED"
    "${WANDB_ARGS[@]}"
    --n-samples "$N_SAMPLES"
    --sigma "$SIGMA"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --checkpoint-source local
    --select-best-of-top-k
    --top-k "$TOP_K"
    --selection-batch-size "$SELECTION_BATCH_SIZE"
    --selected-checkpoint-backend local
    --selected-checkpoint-release "$SELECTED_CHECKPOINT_RELEASE"
    --selected-checkpoint-staging-root "$SELECTED_STAGING_ROOT"
    --device cuda
    --output-csv "$OUTPUT_CSV"
    --no-append-output
)

echo "Evaluating task=$TASK_ID model=$MODEL_ID problem=$PROBLEM_ID seed=$SEED method=$METHOD steps=$STEPS"
cd "$ENGIOPT_DIR"
case "$MODEL_ID" in
    flow_matching_2d_cond)
        python engiopt/flow_matching_2d_cond/evaluate_flow_matching_2d_cond.py \
            "${COMMON_ARGS[@]}" \
            --method "$METHOD" \
            --integration-steps "$STEPS" \
            --checkpoint-package-label "$PACKAGE_LABEL"
        ;;
    diffusion_2d_cond)
        python engiopt/diffusion_2d_cond/evaluate_diffusion_2d_cond.py \
            "${COMMON_ARGS[@]}" \
            --checkpoint-package-label "$PACKAGE_LABEL"
        ;;
    cgan_cnn_2d)
        python engiopt/cgan_cnn_2d/evaluate_cgan_cnn_2d.py \
            "${COMMON_ARGS[@]}" \
            --checkpoint-package-label "$PACKAGE_LABEL"
        ;;
    *)
        echo "ERROR: Unsupported model: $MODEL_ID" >&2
        exit 1
        ;;
esac
