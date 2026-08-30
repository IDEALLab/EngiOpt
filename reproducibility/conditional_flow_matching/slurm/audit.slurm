#!/usr/bin/env bash
#SBATCH --job-name=cfm_audit
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G

set -euo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPRO_DIR/common.sh"
setup_reproduction_env

require_var CAMPAIGN_MANIFEST
AUDIT_OUTPUT="${AUDIT_OUTPUT:-$(dirname "$CAMPAIGN_MANIFEST")/audit.json}"
args=(
    audit-local
    --manifest "$CAMPAIGN_MANIFEST"
    --staging-root "$SELECTED_STAGING_ROOT"
    --csv-dir "$CSV_DIR"
    --output "$AUDIT_OUTPUT"
)
if is_true "${PUBLISH_CHECKPOINTS:-0}"; then
    args+=(--require-published)
fi
if [[ "${CAMPAIGN_MODE:-full}" == "full" ]]; then
    args+=(
        --require-timing
        --timing-csv-dir "$TIMING_CSV_DIR"
        --require-qualitative
        --qualitative-root "$QUALITATIVE_ROOT"
    )
fi

python "$REPRO_DIR/campaign.py" "${args[@]}"
