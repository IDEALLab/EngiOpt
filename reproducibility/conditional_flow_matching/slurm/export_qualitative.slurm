#!/usr/bin/env bash
#SBATCH --job-name=cfm_qualitative
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:1
#SBATCH --array=0-1%2

set -euo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPRO_DIR/common.sh"
setup_reproduction_env

PROBLEMS=(beams2d heatconduction2d)
SEEDS=("${BEAMS_SEED:-8}" "${HEAT_SEED:-5}")
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PROBLEM_ID="${PROBLEMS[$TASK_ID]:?Qualitative task must be in 0..1}"
SEED="${SEEDS[$TASK_ID]}"
SELECTED_CHECKPOINT_RELEASE="${SELECTED_CHECKPOINT_RELEASE:-v1}"

FLOW_CHECKPOINT="$SELECTED_STAGING_ROOT/flow_matching_2d_cond/$PROBLEM_ID/selected/$SELECTED_CHECKPOINT_RELEASE/euler_32/seed_$SEED/model.pth"
DIFFUSION_CHECKPOINT="$SELECTED_STAGING_ROOT/diffusion_2d_cond/$PROBLEM_ID/selected/$SELECTED_CHECKPOINT_RELEASE/timesteps_1000/seed_$SEED/model.pth"
CGAN_CHECKPOINT="$SELECTED_STAGING_ROOT/cgan_cnn_2d/$PROBLEM_ID/selected/$SELECTED_CHECKPOINT_RELEASE/activation_sigmoid/seed_$SEED/generator.pth"

for checkpoint in "$FLOW_CHECKPOINT" "$DIFFUSION_CHECKPOINT" "$CGAN_CHECKPOINT"; do
    if [[ ! -f "$checkpoint" ]]; then
        echo "ERROR: Missing selected checkpoint: $checkpoint" >&2
        exit 1
    fi
done

if [[ "$PROBLEM_ID" == "heatconduction2d" ]]; then
    stage_pyadjoint_image "$ENGIOPT_DIR"
fi

cd "$ENGIOPT_DIR"
python -m engiopt.export_qualitative_bundle_2d \
    --problems "$PROBLEM_ID" \
    --seed "$SEED" \
    --condition-seed "$SEED" \
    --sample-index 0 \
    --n-samples 3 \
    --flow-checkpoint-path "$FLOW_CHECKPOINT" \
    --diffusion-checkpoint-path "$DIFFUSION_CHECKPOINT" \
    --cgan-checkpoint-path "$CGAN_CHECKPOINT" \
    --flow-method euler \
    --flow-integration-steps 32 \
    --device cuda \
    --output-dir "$QUALITATIVE_ROOT/${PROBLEM_ID}_seed${SEED}" \
    --image-dpi 300 \
    --image-min-pixels 600
