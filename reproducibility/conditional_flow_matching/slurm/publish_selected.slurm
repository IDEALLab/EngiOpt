#!/usr/bin/env bash
#SBATCH --job-name=cfm_publish
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-2%1

set -euo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPRO_DIR/common.sh"

if ! is_true "${PUBLISH_CHECKPOINTS:-0}"; then
    echo "Checkpoint publication is disabled. Set PUBLISH_CHECKPOINTS=1 explicitly to upload."
    exit 0
fi
setup_reproduction_env

MODELS=(flow_matching_2d_cond diffusion_2d_cond cgan_cnn_2d)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_ID="${MODELS[$TASK_ID]:?Publication task must be in 0..2}"
MODEL_ROOT="$SELECTED_STAGING_ROOT/$MODEL_ID"

if [[ ! -d "$MODEL_ROOT" ]]; then
    echo "ERROR: No staged selected checkpoints for $MODEL_ID at $MODEL_ROOT" >&2
    exit 1
fi

mapfile -t BUNDLES < <(find "$MODEL_ROOT" -type f -name metadata.json -print | sort)
if (( ${#BUNDLES[@]} == 0 )); then
    echo "ERROR: No selected checkpoint bundles found under $MODEL_ROOT" >&2
    exit 1
fi

cd "$ENGIOPT_DIR"
for metadata_path in "${BUNDLES[@]}"; do
    bundle_dir="$(dirname "$metadata_path")"
    args=(
        --bundle-dir "$bundle_dir"
        --hf-entity "$HF_ENTITY"
        --hf-repo-prefix "$HF_REPO_PREFIX"
    )
    if is_true "${HF_PRIVATE:-0}"; then
        args+=(--hf-private)
    fi
    python -m engiopt.publish_selected_checkpoint "${args[@]}"
done

echo "Published ${#BUNDLES[@]} selected $MODEL_ID checkpoint bundles."
