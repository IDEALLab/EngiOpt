#!/usr/bin/env bash
#SBATCH --job-name=cfm_report
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-1%1

set -euo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPRO_DIR/common.sh"
setup_reproduction_env

PROBLEMS=(beams2d heatconduction2d)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PROBLEM_ID="${PROBLEMS[$TASK_ID]:?Report task must be in 0..1}"
CAMPAIGN_NAME="${CAMPAIGN_NAME:-conditional-flow-matching-reproduction-v1}"
OUTPUT_DIR="$REPORT_ROOT/$PROBLEM_ID"

args=(
    --shard-dir "$CSV_DIR"
    --output-dir "$OUTPUT_DIR"
    --problem-id "$PROBLEM_ID"
    --no-latest-only
)
if is_true "${TRACK_WANDB:-0}"; then
    args+=(
        --wandb-project "$WANDB_PROJECT"
        --wandb-entity "$WANDB_ENTITY"
        --upload-wandb
        --run-name "${CAMPAIGN_NAME}_${PROBLEM_ID}_summary"
        --artifact-name "${CAMPAIGN_NAME}_${PROBLEM_ID}_summary"
    )
fi

cd "$ENGIOPT_DIR"
python -m engiopt.report_metrics "${args[@]}"
