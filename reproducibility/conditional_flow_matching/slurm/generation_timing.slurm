#!/usr/bin/env bash
#SBATCH --job-name=cfm_timing
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --gpus=rtx_4090:1
#SBATCH --array=0-219%16

set -euo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPRO_DIR/common.sh"
setup_reproduction_env

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task}"
if (( TASK_ID < 0 || TASK_ID >= 220 )); then
    echo "ERROR: Timing task must be in 0..219, got $TASK_ID" >&2
    exit 1
fi

SELECTED_CHECKPOINT_RELEASE="${SELECTED_CHECKPOINT_RELEASE:-v1}"
N_SAMPLES="${N_SAMPLES:-50}"
WARMUP_REPEATS="${WARMUP_REPEATS:-1}"
TIMED_REPEATS="${TIMED_REPEATS:-3}"

if (( TASK_ID < 180 )); then
    CONFIGS=("euler:16" "euler:32" "euler:48" "midpoint:8" "midpoint:16" "midpoint:24" "rk4:4" "rk4:8" "rk4:12")
    PROBLEMS=(beams2d heatconduction2d)
    CONFIG_INDEX=$(( TASK_ID / 20 ))
    PROBLEM_INDEX=$(( (TASK_ID / 10) % 2 ))
    SEED=$(( TASK_ID % 10 + 1 ))
    IFS=: read -r METHOD STEPS <<< "${CONFIGS[$CONFIG_INDEX]}"
    MODEL_ID=flow_matching_2d_cond
    PROBLEM_ID="${PROBLEMS[$PROBLEM_INDEX]}"
    PACKAGE_LABEL="${METHOD}_${STEPS}"
    CHECKPOINT_FILENAME=model.pth
    METHOD_SUFFIX="$PACKAGE_LABEL"
else
    BASELINE_TASK=$(( TASK_ID - 180 ))
    MODELS=(diffusion_2d_cond cgan_cnn_2d)
    PROBLEMS=(beams2d heatconduction2d)
    MODEL_INDEX=$(( BASELINE_TASK / 20 ))
    PROBLEM_INDEX=$(( (BASELINE_TASK / 10) % 2 ))
    SEED=$(( BASELINE_TASK % 10 + 1 ))
    MODEL_ID="${MODELS[$MODEL_INDEX]}"
    PROBLEM_ID="${PROBLEMS[$PROBLEM_INDEX]}"
    METHOD_SUFFIX=na
    if [[ "$MODEL_ID" == "diffusion_2d_cond" ]]; then
        PACKAGE_LABEL=timesteps_1000
        CHECKPOINT_FILENAME=model.pth
    else
        PACKAGE_LABEL=activation_sigmoid
        CHECKPOINT_FILENAME=generator.pth
    fi
fi

CHECKPOINT_PATH="$SELECTED_STAGING_ROOT/$MODEL_ID/$PROBLEM_ID/selected/$SELECTED_CHECKPOINT_RELEASE/$PACKAGE_LABEL/seed_$SEED/$CHECKPOINT_FILENAME"
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "ERROR: Selected checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
fi

OUTPUT_CSV="$TIMING_CSV_DIR/generation_timing_${MODEL_ID}_${METHOD_SUFFIX}_${PROBLEM_ID}_seed${SEED}.csv"
BENCHMARK_ARGS=(
    --model-id "$MODEL_ID"
    --problem-id "$PROBLEM_ID"
    --seed "$SEED"
    --n-samples "$N_SAMPLES"
    --warmup-repeats "$WARMUP_REPEATS"
    --timed-repeats "$TIMED_REPEATS"
    --checkpoint-path "$CHECKPOINT_PATH"
    --device cuda
)
if [[ "$MODEL_ID" == "flow_matching_2d_cond" ]]; then
    BENCHMARK_ARGS+=(--flow-method "$METHOD" --flow-integration-steps "$STEPS")
fi
BENCHMARK_ARGS+=(--output-csv "$OUTPUT_CSV" --no-append-output)
cd "$ENGIOPT_DIR"
python -m engiopt.benchmark_generation_2d "${BENCHMARK_ARGS[@]}"
