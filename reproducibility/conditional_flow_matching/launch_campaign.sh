#!/usr/bin/env bash

set -euo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$REPRO_DIR/common.sh"

SUBMIT=0
if [[ "${1:-}" == "--submit" ]]; then
    SUBMIT=1
elif (( $# > 0 )); then
    echo "Usage: $0 [--submit]" >&2
    exit 2
fi

setup_reproduction_env
CAMPAIGN_NAME="${CAMPAIGN_NAME:-conditional-flow-matching-reproduction-v1}"
CAMPAIGN_ROOT="$RESULTS_ROOT/$CAMPAIGN_NAME"
CAMPAIGN_MANIFEST="$CAMPAIGN_ROOT/manifest.json"
RTX_4090_SEEDS="${RTX_4090_SEEDS:-1,2,3}"
CONCURRENCY_4090="${CONCURRENCY_4090:-6}"
CONCURRENCY_3090="${CONCURRENCY_3090:-20}"
TIMING_CONCURRENCY="${TIMING_CONCURRENCY:-16}"
mkdir -p "$CAMPAIGN_ROOT/logs"

if [[ -f "$CAMPAIGN_MANIFEST" ]]; then
    RECORDED_JOBS="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("jobs", {})))' "$CAMPAIGN_MANIFEST")"
    if (( RECORDED_JOBS > 0 )); then
        echo "ERROR: This campaign manifest already records submitted jobs: $CAMPAIGN_MANIFEST" >&2
        echo "Use a new CAMPAIGN_NAME or resume the recorded jobs instead of launching duplicates." >&2
        exit 1
    fi
fi

MANIFEST_ARGS=(
    manifest
    --mode full
    --campaign "$CAMPAIGN_NAME"
    --release "${SELECTED_CHECKPOINT_RELEASE:-v1}"
    --output "$CAMPAIGN_MANIFEST"
    --engiopt-commit "$(git -C "$ENGIOPT_DIR" rev-parse HEAD)"
    --engibench-commit "$(git -C "$ENGIBENCH_DIR" rev-parse HEAD)"
    --rtx-4090-seeds "$RTX_4090_SEEDS"
    --concurrency-4090 "$CONCURRENCY_4090"
    --concurrency-3090 "$CONCURRENCY_3090"
)
if is_true "${TRACK_WANDB:-0}"; then
    MANIFEST_ARGS+=(
        --wandb-enabled
        --wandb-entity "$WANDB_ENTITY"
        --wandb-project "$WANDB_PROJECT"
        --wandb-group "${WANDB_GROUP:-$CAMPAIGN_NAME}"
    )
fi
if is_true "${PUBLISH_CHECKPOINTS:-0}"; then
    MANIFEST_ARGS+=(
        --checkpoint-publication-enabled
        --hf-entity "$HF_ENTITY"
        --hf-repo-prefix "$HF_REPO_PREFIX"
    )
fi

python "$REPRO_DIR/campaign.py" "${MANIFEST_ARGS[@]}"

python "$REPRO_DIR/campaign.py" preflight \
    --manifest "$CAMPAIGN_MANIFEST" \
    --engiopt-repo "$ENGIOPT_DIR" \
    --engibench-repo "$ENGIBENCH_DIR"

TRAIN_4090="$(python "$REPRO_DIR/campaign.py" task-ids --manifest "$CAMPAIGN_MANIFEST" --phase train --gpu rtx_4090)"
TRAIN_3090="$(python "$REPRO_DIR/campaign.py" task-ids --manifest "$CAMPAIGN_MANIFEST" --phase train --gpu rtx_3090)"
EVAL_4090="$(python "$REPRO_DIR/campaign.py" task-ids --manifest "$CAMPAIGN_MANIFEST" --phase eval --gpu rtx_4090)"
EVAL_3090="$(python "$REPRO_DIR/campaign.py" task-ids --manifest "$CAMPAIGN_MANIFEST" --phase eval --gpu rtx_3090)"

echo "Manifest: $CAMPAIGN_MANIFEST"
echo "RTX 4090 train tasks: $TRAIN_4090 (limit $CONCURRENCY_4090)"
echo "RTX 3090 train tasks: $TRAIN_3090 (limit $CONCURRENCY_3090)"
echo "RTX 4090 eval tasks: $EVAL_4090 (limit $CONCURRENCY_4090)"
echo "RTX 3090 eval tasks: $EVAL_3090 (limit $CONCURRENCY_3090)"
echo "W&B tracking: ${TRACK_WANDB:-0}"
echo "HF publication: ${PUBLISH_CHECKPOINTS:-0}"
if (( SUBMIT == 0 )); then
    echo "Dry run complete. No jobs were submitted and no artifacts were uploaded."
    echo "Run '$0 --submit' only after a successful smoke audit."
    exit 0
fi

require_var SMOKE_AUDIT
if [[ ! -f "$SMOKE_AUDIT" ]]; then
    echo "ERROR: SMOKE_AUDIT does not exist: $SMOKE_AUDIT" >&2
    exit 1
fi
if [[ "$(python -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["ok"])).lower())' "$SMOKE_AUDIT")" != "true" ]]; then
    echo "ERROR: Smoke audit did not pass: $SMOKE_AUDIT" >&2
    exit 1
fi

EXPORTS="ALL,CAMPAIGN_MODE=full,CAMPAIGN_NAME=$CAMPAIGN_NAME,CAMPAIGN_MANIFEST=$CAMPAIGN_MANIFEST,ENGIOPT_DIR=$ENGIOPT_DIR,ENGIBENCH_DIR=$ENGIBENCH_DIR,VENV_ACTIVATE=${VENV_ACTIVATE:-},RESULTS_ROOT=$CAMPAIGN_ROOT,CHECKPOINT_ROOT=$CAMPAIGN_ROOT/checkpoints,FLOW_CHECKPOINT_ROOT=$CAMPAIGN_ROOT/checkpoints_by_config,SELECTED_STAGING_ROOT=$CAMPAIGN_ROOT/selected_checkpoint_staging,CSV_DIR=$CAMPAIGN_ROOT/csv_shards,REPORT_ROOT=$CAMPAIGN_ROOT/reports,TIMING_CSV_DIR=$CAMPAIGN_ROOT/generation_timing,QUALITATIVE_ROOT=$CAMPAIGN_ROOT/qualitative,LOG_DIR=$CAMPAIGN_ROOT/logs,TRACK_WANDB=${TRACK_WANDB:-0},WANDB_ENTITY=${WANDB_ENTITY:-},WANDB_PROJECT=${WANDB_PROJECT:-},WANDB_GROUP=${WANDB_GROUP:-$CAMPAIGN_NAME},PUBLISH_CHECKPOINTS=${PUBLISH_CHECKPOINTS:-0},HF_ENTITY=${HF_ENTITY:-},HF_REPO_PREFIX=${HF_REPO_PREFIX:-},HF_PRIVATE=${HF_PRIVATE:-0},SELECTED_CHECKPOINT_RELEASE=${SELECTED_CHECKPOINT_RELEASE:-v1},PYADJOINT_SIF=${PYADJOINT_SIF:-},N_EPOCHS=500,VALIDATION_INTERVAL_EPOCHS=10,CHECKPOINT_INTERVAL_EPOCHS=10,MIN_EPOCH_FOR_SELECTION=80,EARLY_STOPPING_PATIENCE=25,VALIDATION_BATCH_SIZE=50,N_SAMPLES=50,SELECTION_BATCH_SIZE=50,WARMUP_REPEATS=1,TIMED_REPEATS=3"

TRAIN_4090_JOB="$(sbatch --parsable --time="${TRAIN_TIME_4090:-06:00:00}" --gpus=rtx_4090:1 --array="$TRAIN_4090%$CONCURRENCY_4090" --export="$EXPORTS" --output="$CAMPAIGN_ROOT/logs/train_4090_%A_%a.log" --error="$CAMPAIGN_ROOT/logs/train_4090_%A_%a.err" "$REPRO_DIR/slurm/train_task.slurm")"
TRAIN_4090_JOB="${TRAIN_4090_JOB%%;*}"
TRAIN_3090_JOB="$(sbatch --parsable --time="${TRAIN_TIME_3090:-12:00:00}" --gpus=rtx_3090:1 --array="$TRAIN_3090%$CONCURRENCY_3090" --export="$EXPORTS" --output="$CAMPAIGN_ROOT/logs/train_3090_%A_%a.log" --error="$CAMPAIGN_ROOT/logs/train_3090_%A_%a.err" "$REPRO_DIR/slurm/train_task.slurm")"
TRAIN_3090_JOB="${TRAIN_3090_JOB%%;*}"
TRAIN_DEP="afterok:$TRAIN_4090_JOB:$TRAIN_3090_JOB"

EVAL_4090_JOB="$(sbatch --parsable --dependency="$TRAIN_DEP" --gpus=rtx_4090:1 --array="$EVAL_4090%$CONCURRENCY_4090" --export="$EXPORTS" --output="$CAMPAIGN_ROOT/logs/eval_4090_%A_%a.log" --error="$CAMPAIGN_ROOT/logs/eval_4090_%A_%a.err" "$REPRO_DIR/slurm/evaluate_task.slurm")"
EVAL_4090_JOB="${EVAL_4090_JOB%%;*}"
EVAL_3090_JOB="$(sbatch --parsable --dependency="$TRAIN_DEP" --gpus=rtx_3090:1 --array="$EVAL_3090%$CONCURRENCY_3090" --export="$EXPORTS" --output="$CAMPAIGN_ROOT/logs/eval_3090_%A_%a.log" --error="$CAMPAIGN_ROOT/logs/eval_3090_%A_%a.err" "$REPRO_DIR/slurm/evaluate_task.slurm")"
EVAL_3090_JOB="${EVAL_3090_JOB%%;*}"
EVAL_DEP="afterok:$EVAL_4090_JOB:$EVAL_3090_JOB"

REPORT_JOB="$(sbatch --parsable --dependency="$EVAL_DEP" --export="$EXPORTS" --output="$CAMPAIGN_ROOT/logs/report_%A_%a.log" --error="$CAMPAIGN_ROOT/logs/report_%A_%a.err" "$REPRO_DIR/slurm/report.slurm")"
REPORT_JOB="${REPORT_JOB%%;*}"
TIMING_JOB="$(sbatch --parsable --dependency="$EVAL_DEP" --gpus=rtx_4090:1 --array="0-219%$TIMING_CONCURRENCY" --export="$EXPORTS" --output="$CAMPAIGN_ROOT/logs/timing_%A_%a.log" --error="$CAMPAIGN_ROOT/logs/timing_%A_%a.err" "$REPRO_DIR/slurm/generation_timing.slurm")"
TIMING_JOB="${TIMING_JOB%%;*}"
QUALITATIVE_JOB="$(sbatch --parsable --dependency="$EVAL_DEP" --gpus=rtx_4090:1 --export="$EXPORTS" --output="$CAMPAIGN_ROOT/logs/qualitative_%A_%a.log" --error="$CAMPAIGN_ROOT/logs/qualitative_%A_%a.err" "$REPRO_DIR/slurm/export_qualitative.slurm")"
QUALITATIVE_JOB="${QUALITATIVE_JOB%%;*}"

AUDIT_DEP="$EVAL_4090_JOB:$EVAL_3090_JOB:$REPORT_JOB:$TIMING_JOB:$QUALITATIVE_JOB"
JOB_ARGS=(
    --job "train_4090=$TRAIN_4090_JOB"
    --job "train_3090=$TRAIN_3090_JOB"
    --job "eval_4090=$EVAL_4090_JOB"
    --job "eval_3090=$EVAL_3090_JOB"
    --job "report=$REPORT_JOB"
    --job "generation_timing=$TIMING_JOB"
    --job "qualitative=$QUALITATIVE_JOB"
)
if is_true "${PUBLISH_CHECKPOINTS:-0}"; then
    PUBLISH_JOB="$(sbatch --parsable --dependency="$EVAL_DEP" --export="$EXPORTS" --output="$CAMPAIGN_ROOT/logs/publish_%A_%a.log" --error="$CAMPAIGN_ROOT/logs/publish_%A_%a.err" "$REPRO_DIR/slurm/publish_selected.slurm")"
    PUBLISH_JOB="${PUBLISH_JOB%%;*}"
    AUDIT_DEP="$AUDIT_DEP:$PUBLISH_JOB"
    JOB_ARGS+=(--job "publish=$PUBLISH_JOB")
fi

AUDIT_JOB="$(sbatch --parsable --dependency="afterok:$AUDIT_DEP" --export="$EXPORTS" --output="$CAMPAIGN_ROOT/logs/audit_%j.log" --error="$CAMPAIGN_ROOT/logs/audit_%j.err" "$REPRO_DIR/slurm/audit.slurm")"
AUDIT_JOB="${AUDIT_JOB%%;*}"
JOB_ARGS+=(--job "audit=$AUDIT_JOB")

python "$REPRO_DIR/campaign.py" record-jobs --manifest "$CAMPAIGN_MANIFEST" "${JOB_ARGS[@]}"
echo "Submitted full workflow. Final audit: $CAMPAIGN_ROOT/audit.json"
