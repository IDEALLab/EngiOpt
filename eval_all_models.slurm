#!/bin/bash
#SBATCH --job-name=eval_all_models
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7G
#SBATCH --gpus=rtx_4090:1
#SBATCH --array=0-99
#SBATCH --output=/cluster/scratch/%u/logs/engiopt/eval_%A_%a.log
#SBATCH --error=/cluster/scratch/%u/logs/engiopt/eval_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common_flow_matching_env.sh"

ensure_flow_layout
load_flow_modules
activate_flow_venv
load_flow_secrets
configure_flow_caches

CSV_DIR="${CSV_DIR:-$FLOW_RESULTS_DIR/csv_shards}"
mkdir -p "$CSV_DIR"
N_SAMPLES="${N_SAMPLES:-50}"
SIGMA="${SIGMA:-10.0}"
FLOW_INTEGRATION_STEPS="${FLOW_INTEGRATION_STEPS:-50}"
N_SEEDS="${N_SEEDS:-10}"
WANDB_PROJECT="${WANDB_PROJECT:-engiopt}"

echo "Running on node-local scratch: $TMPDIR"
stage_flow_repos_to_tmpdir "$TMPDIR"
cd "$TMPDIR/EngiOpt"

ENGIOPT_2D_PROFILE="${ENGIOPT_2D_PROFILE:-${WORKFLOW_PROFILE:-main}}"
case "$ENGIOPT_2D_PROFILE" in
  main)
    MODELS=(diffusion_2d_cond cgan_cnn_2d gan_cnn_2d)
    ;;
  flow_matching|flow-matching)
    MODELS=(flow_matching_2d_cond diffusion_2d_cond cgan_cnn_2d)
    ;;
  *)
    echo "ERROR: Unsupported ENGIOPT_2D_PROFILE=$ENGIOPT_2D_PROFILE (use main or flow_matching)"
    exit 1
    ;;
esac

PROBLEMS=(beams2d heatconduction2d)
N_PROBS=${#PROBLEMS[@]}

WANDB_ARGS=(--wandb-project "$WANDB_PROJECT")
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

# Option B mapping: build only the actually needed model/method combinations.
# flow_matching_2d_cond runs across multiple methods, other models run once.
FLOW_METHODS=(euler midpoint dpm)
COMBO_MODELS=()
COMBO_METHODS=()

for model in "${MODELS[@]}"; do
  if [[ "$model" == "flow_matching_2d_cond" ]]; then
    for method in "${FLOW_METHODS[@]}"; do
      COMBO_MODELS+=("$model")
      COMBO_METHODS+=("$method")
    done
  else
    COMBO_MODELS+=("$model")
    COMBO_METHODS+=("na")
  fi
done

N_COMBOS=${#COMBO_MODELS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=$(( N_COMBOS * N_PROBS * N_SEEDS ))
if (( TASK_ID >= TOTAL )); then
  echo "Task $TASK_ID outside valid range 0..$((TOTAL - 1)); exiting cleanly."
  exit 0
fi

C_IDX=$(( TASK_ID / (N_PROBS * N_SEEDS) ))
P_IDX=$(( (TASK_ID / N_SEEDS) % N_PROBS ))
SEED=$(( TASK_ID % N_SEEDS + 1 ))

MODEL=${COMBO_MODELS[$C_IDX]}
METHOD=${COMBO_METHODS[$C_IDX]}
PROBLEM=${PROBLEMS[$P_IDX]}

if [[ "$METHOD" == "midpoint" ]]; then
  ACTUAL_STEPS=$(( FLOW_INTEGRATION_STEPS / 2 ))
else
  ACTUAL_STEPS=$FLOW_INTEGRATION_STEPS
fi

echo "Task $TASK_ID -> Model=$MODEL, Method=$METHOD, Problem=$PROBLEM, Seed=$SEED"
SHARD_CSV="$CSV_DIR/metrics_${MODEL}_${METHOD}_${PROBLEM}_seed${SEED}.csv"
EVAL_SCRIPT="engiopt/${MODEL}/evaluate_${MODEL}.py"

APPEND_FLAGS=()
if python "$EVAL_SCRIPT" --help 2>&1 | grep -q -- "--append-output"; then
  APPEND_FLAGS+=(--no-append-output)
fi

if [[ "$PROBLEM" == "heatconduction2d" ]]; then
  stage_shared_apptainer_image "$FLOW_PYADJOINT_IMAGE" "$PWD"
fi

case "$MODEL" in
  flow_matching_2d_cond)
    python "$EVAL_SCRIPT" \
      --problem-id "$PROBLEM" \
      --seed "$SEED" \
      "${WANDB_ARGS[@]}" \
      --n-samples "$N_SAMPLES" \
      --sigma "$SIGMA" \
      --method "$METHOD" \
      --integration-steps "$ACTUAL_STEPS" \
      --device cuda \
      --output-csv "$SHARD_CSV" \
      "${APPEND_FLAGS[@]}"
    ;;
  diffusion_2d_cond|cgan_cnn_2d|gan_cnn_2d)
    python "$EVAL_SCRIPT" \
      --problem-id "$PROBLEM" \
      --seed "$SEED" \
      "${WANDB_ARGS[@]}" \
      --n-samples "$N_SAMPLES" \
      --sigma "$SIGMA" \
      --output-csv "$SHARD_CSV" \
      "${APPEND_FLAGS[@]}"
    ;;
  *)
    echo "ERROR: Unsupported model $MODEL"
    exit 1
    ;;
esac
