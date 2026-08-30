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
CAMPAIGN_NAME="${CAMPAIGN_NAME:-conditional-flow-matching-smoke-v1}"
CAMPAIGN_ROOT="$RESULTS_ROOT/$CAMPAIGN_NAME"
CAMPAIGN_MANIFEST="$CAMPAIGN_ROOT/manifest.json"
mkdir -p "$CAMPAIGN_ROOT"

if [[ -f "$CAMPAIGN_MANIFEST" ]]; then
    RECORDED_JOBS="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("jobs", {})))' "$CAMPAIGN_MANIFEST")"
    if (( RECORDED_JOBS > 0 )); then
        echo "ERROR: This smoke manifest already records submitted jobs: $CAMPAIGN_MANIFEST" >&2
        echo "Use a new CAMPAIGN_NAME or resume the recorded jobs instead of launching duplicates." >&2
        exit 1
    fi
fi

MANIFEST_ARGS=(
    manifest
    --mode smoke
    --campaign "$CAMPAIGN_NAME"
    --release "${SELECTED_CHECKPOINT_RELEASE:-v1}"
    --output "$CAMPAIGN_MANIFEST"
    --engiopt-commit "$(git -C "$ENGIOPT_DIR" rev-parse HEAD)"
    --engibench-commit "$(git -C "$ENGIBENCH_DIR" rev-parse HEAD)"
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

echo "Smoke manifest: $CAMPAIGN_MANIFEST"
echo "W&B tracking: ${TRACK_WANDB:-0}"
echo "HF publication: ${PUBLISH_CHECKPOINTS:-0}"
if (( SUBMIT == 0 )); then
    echo "Dry run complete. No jobs were submitted and no artifacts were uploaded."
    echo "Run '$0 --submit' after reviewing the manifest."
    exit 0
fi

EXPORTS="ALL,CAMPAIGN_MODE=smoke,CAMPAIGN_NAME=$CAMPAIGN_NAME,CAMPAIGN_MANIFEST=$CAMPAIGN_MANIFEST,ENGIOPT_DIR=$ENGIOPT_DIR,ENGIBENCH_DIR=$ENGIBENCH_DIR,VENV_ACTIVATE=${VENV_ACTIVATE:-},RESULTS_ROOT=$CAMPAIGN_ROOT,CHECKPOINT_ROOT=$CAMPAIGN_ROOT/checkpoints,FLOW_CHECKPOINT_ROOT=$CAMPAIGN_ROOT/checkpoints_by_config,SELECTED_STAGING_ROOT=$CAMPAIGN_ROOT/selected_checkpoint_staging,CSV_DIR=$CAMPAIGN_ROOT/csv_shards,REPORT_ROOT=$CAMPAIGN_ROOT/reports,LOG_DIR=$CAMPAIGN_ROOT/logs,TRACK_WANDB=${TRACK_WANDB:-0},WANDB_ENTITY=${WANDB_ENTITY:-},WANDB_PROJECT=${WANDB_PROJECT:-},WANDB_GROUP=${WANDB_GROUP:-$CAMPAIGN_NAME},PUBLISH_CHECKPOINTS=${PUBLISH_CHECKPOINTS:-0},HF_ENTITY=${HF_ENTITY:-},HF_REPO_PREFIX=${HF_REPO_PREFIX:-},HF_PRIVATE=${HF_PRIVATE:-0},SELECTED_CHECKPOINT_RELEASE=${SELECTED_CHECKPOINT_RELEASE:-v1},N_EPOCHS=5,VALIDATION_INTERVAL_EPOCHS=1,CHECKPOINT_INTERVAL_EPOCHS=1,MIN_EPOCH_FOR_SELECTION=1,EARLY_STOPPING_PATIENCE=10,VALIDATION_BATCH_SIZE=2,N_SAMPLES=2,SELECTION_BATCH_SIZE=2"

TRAIN_JOB="$(sbatch --parsable --array=0-2%3 --export="$EXPORTS" --output="$CAMPAIGN_ROOT/logs/train_%A_%a.log" --error="$CAMPAIGN_ROOT/logs/train_%A_%a.err" "$REPRO_DIR/slurm/train_task.slurm")"
TRAIN_JOB="${TRAIN_JOB%%;*}"
EVAL_JOB="$(sbatch --parsable --dependency="afterok:$TRAIN_JOB" --array=0-2%3 --export="$EXPORTS" --output="$CAMPAIGN_ROOT/logs/eval_%A_%a.log" --error="$CAMPAIGN_ROOT/logs/eval_%A_%a.err" "$REPRO_DIR/slurm/evaluate_task.slurm")"
EVAL_JOB="${EVAL_JOB%%;*}"

AUDIT_DEP="$EVAL_JOB"
JOB_ARGS=(--job "train=$TRAIN_JOB" --job "eval=$EVAL_JOB")
if is_true "${PUBLISH_CHECKPOINTS:-0}"; then
    PUBLISH_JOB="$(sbatch --parsable --dependency="afterok:$EVAL_JOB" --export="$EXPORTS" --output="$CAMPAIGN_ROOT/logs/publish_%A_%a.log" --error="$CAMPAIGN_ROOT/logs/publish_%A_%a.err" "$REPRO_DIR/slurm/publish_selected.slurm")"
    PUBLISH_JOB="${PUBLISH_JOB%%;*}"
    AUDIT_DEP="$PUBLISH_JOB"
    JOB_ARGS+=(--job "publish=$PUBLISH_JOB")
fi

REPORT_JOB="$(sbatch --parsable --dependency="afterok:$EVAL_JOB" --array=0 --export="$EXPORTS" --output="$CAMPAIGN_ROOT/logs/report_%A_%a.log" --error="$CAMPAIGN_ROOT/logs/report_%A_%a.err" "$REPRO_DIR/slurm/report.slurm")"
REPORT_JOB="${REPORT_JOB%%;*}"
AUDIT_JOB="$(sbatch --parsable --dependency="afterok:$AUDIT_DEP:$REPORT_JOB" --export="$EXPORTS" --output="$CAMPAIGN_ROOT/logs/audit_%j.log" --error="$CAMPAIGN_ROOT/logs/audit_%j.err" "$REPRO_DIR/slurm/audit.slurm")"
AUDIT_JOB="${AUDIT_JOB%%;*}"
JOB_ARGS+=(--job "report=$REPORT_JOB" --job "audit=$AUDIT_JOB")

python "$REPRO_DIR/campaign.py" record-jobs --manifest "$CAMPAIGN_MANIFEST" "${JOB_ARGS[@]}"
echo "Submitted smoke workflow. Final audit: $CAMPAIGN_ROOT/audit.json"
