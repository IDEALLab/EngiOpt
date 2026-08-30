#!/usr/bin/env bash

# Shared, public environment setup for the reproduction launchers.

set -euo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

require_var() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: Set $name before running this workflow." >&2
        return 1
    fi
}

is_true() {
    [[ "${1:-0}" == "1" || "${1:-0}" == "true" ]]
}

load_optional_secret() {
    local variable_name="$1"
    local file_name="$2"
    local file_path="${!file_name:-}"
    if [[ -z "${!variable_name:-}" && -n "$file_path" && -f "$file_path" ]]; then
        export "$variable_name=$(tr -d '\r\n' < "$file_path")"
    fi
}

setup_reproduction_env() {
    require_var ENGIOPT_DIR
    require_var ENGIBENCH_DIR

    if [[ -n "${VENV_ACTIVATE:-}" ]]; then
        if [[ ! -f "$VENV_ACTIVATE" ]]; then
            echo "ERROR: VENV_ACTIVATE does not exist: $VENV_ACTIVATE" >&2
            return 1
        fi
        # shellcheck disable=SC1090
        source "$VENV_ACTIVATE"
    fi

    export PYTHONPATH="$ENGIBENCH_DIR:$ENGIOPT_DIR:${PYTHONPATH:-}"
    export RESULTS_ROOT="${RESULTS_ROOT:-$PWD/outputs/conditional-flow-matching}"
    export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RESULTS_ROOT/checkpoints}"
    export FLOW_CHECKPOINT_ROOT="${FLOW_CHECKPOINT_ROOT:-$RESULTS_ROOT/checkpoints_by_config}"
    export SELECTED_STAGING_ROOT="${SELECTED_STAGING_ROOT:-$RESULTS_ROOT/selected_checkpoint_staging}"
    export CSV_DIR="${CSV_DIR:-$RESULTS_ROOT/csv_shards}"
    export REPORT_ROOT="${REPORT_ROOT:-$RESULTS_ROOT/reports}"
    export TIMING_CSV_DIR="${TIMING_CSV_DIR:-$RESULTS_ROOT/generation_timing}"
    export QUALITATIVE_ROOT="${QUALITATIVE_ROOT:-$RESULTS_ROOT/qualitative}"
    export LOG_DIR="${LOG_DIR:-$RESULTS_ROOT/logs}"

    mkdir -p \
        "$RESULTS_ROOT" \
        "$CHECKPOINT_ROOT" \
        "$FLOW_CHECKPOINT_ROOT" \
        "$SELECTED_STAGING_ROOT" \
        "$CSV_DIR" \
        "$REPORT_ROOT" \
        "$TIMING_CSV_DIR" \
        "$QUALITATIVE_ROOT" \
        "$LOG_DIR"

    load_optional_secret WANDB_API_KEY WANDB_API_KEY_FILE
    load_optional_secret HF_TOKEN HF_TOKEN_FILE
    export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

    if is_true "${TRACK_WANDB:-0}"; then
        require_var WANDB_ENTITY
        require_var WANDB_PROJECT
        if [[ -z "${WANDB_API_KEY:-}" ]]; then
            echo "ERROR: TRACK_WANDB=1 requires WANDB_API_KEY or WANDB_API_KEY_FILE." >&2
            return 1
        fi
    fi

    if is_true "${PUBLISH_CHECKPOINTS:-0}"; then
        require_var HF_ENTITY
        require_var HF_REPO_PREFIX
        if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
            echo "ERROR: PUBLISH_CHECKPOINTS=1 requires HF_TOKEN or HF_TOKEN_FILE." >&2
            return 1
        fi
    fi
}

stage_pyadjoint_image() {
    local workdir="$1"
    if [[ -z "${PYADJOINT_SIF:-}" ]]; then
        return 0
    fi
    if [[ ! -f "$PYADJOINT_SIF" ]]; then
        echo "ERROR: PYADJOINT_SIF does not exist: $PYADJOINT_SIF" >&2
        return 1
    fi
    ln -sfn "$PYADJOINT_SIF" "$workdir/pyadjoint_master.sif"
}

build_wandb_args() {
    if is_true "${TRACK_WANDB:-0}"; then
        WANDB_ARGS=(--track --wandb-project "$WANDB_PROJECT" --wandb-entity "$WANDB_ENTITY")
        if [[ -n "${WANDB_GROUP:-}" ]]; then
            WANDB_ARGS+=(--wandb-group "$WANDB_GROUP")
        fi
    else
        WANDB_ARGS=(--no-track)
    fi
}
