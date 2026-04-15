#!/bin/bash
#SBATCH --job-name=engiopt_2d_train_array
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7G
#SBATCH --gpus=rtx_4090:1
#SBATCH --array=0-5
#SBATCH --output=/cluster/scratch/%u/logs/engiopt/train_%A_%a.log
#SBATCH --error=/cluster/scratch/%u/logs/engiopt/train_%A_%a.err
#SBATCH --mail-type=END,FAIL

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

echo "Running on node-local scratch: $TMPDIR"
echo "Using persistent scratch EngiOpt clone: $FLOW_ENGIOPT_DIR"
echo "Using persistent scratch EngiBench clone: $FLOW_ENGIBENCH_DIR"
echo "Current scratch EngiOpt branch: $(git -C "$FLOW_ENGIOPT_DIR" rev-parse --abbrev-ref HEAD)"

stage_flow_repos_to_tmpdir "$TMPDIR"
cd "$TMPDIR/EngiOpt"

ENGIOPT_2D_PROFILE="${ENGIOPT_2D_PROFILE:-${WORKFLOW_PROFILE:-flow_matching}}"
case "$ENGIOPT_2D_PROFILE" in
  main)
    MODELS=("diffusion_2d_cond" "cgan_cnn_2d" "gan_cnn_2d")
    ;;
  flow_matching|flow-matching)
    MODELS=("flow_matching_2d_cond" "diffusion_2d_cond" "cgan_cnn_2d")
    ;;
  *)
    echo "ERROR: Unsupported ENGIOPT_2D_PROFILE=$ENGIOPT_2D_PROFILE (use main or flow_matching)"
    exit 1
    ;;
esac

PROBLEMS=("beams2d" "heatconduction2d")
N_PROBS=${#PROBLEMS[@]}
N_SEEDS="${N_SEEDS:-1}"
N_EPOCHS="${N_EPOCHS:-500}"
FLOW_INTEGRATION_STEPS="${FLOW_INTEGRATION_STEPS:-50}"
CHECKPOINT_INTERVAL_EPOCHS="${CHECKPOINT_INTERVAL_EPOCHS:-5}"
VALIDATION_INTERVAL_EPOCHS="${VALIDATION_INTERVAL_EPOCHS:-10}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-25}"
WANDB_PROJECT="${WANDB_PROJECT:-engiopt-best-epoch-training}"

WANDB_ARGS=(--wandb-project "$WANDB_PROJECT")
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=$(( ${#MODELS[@]} * N_PROBS * N_SEEDS ))
if (( TASK_ID >= TOTAL )); then
  echo "Invalid task ID $TASK_ID (>= $TOTAL)"
  exit 1
fi

M_IDX=$(( TASK_ID / (N_PROBS * N_SEEDS) ))
P_IDX=$(( (TASK_ID / N_SEEDS) % N_PROBS ))
SEED=$(( TASK_ID % N_SEEDS + 1 ))

MODEL=${MODELS[$M_IDX]}
PROBLEM=${PROBLEMS[$P_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID -> Model=$MODEL, Problem=$PROBLEM, Seed=$SEED"
echo "2D profile: $ENGIOPT_2D_PROFILE"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
nvidia-smi -L

case "$MODEL" in
  flow_matching_2d_cond)
    python "engiopt/${MODEL}/${MODEL}.py" \
      --problem-id "$PROBLEM" \
      --seed "$SEED" \
      "${WANDB_ARGS[@]}" \
      --save-model \
      --n-epochs "$N_EPOCHS" \
      --integration-steps "$FLOW_INTEGRATION_STEPS" \
      --n-cpu "$SLURM_CPUS_PER_TASK" \
      --enable-best-epoch-selection \
      --checkpoint-interval-epochs "$CHECKPOINT_INTERVAL_EPOCHS" \
      --validation-interval-epochs "$VALIDATION_INTERVAL_EPOCHS" \
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
      --validation-batch-size 50 \
      --validation-sigma 1.0 \
      --device cuda
    ;;
  diffusion_2d_cond|cgan_cnn_2d|gan_cnn_2d)
    python "engiopt/${MODEL}/${MODEL}.py" \
      --problem-id "$PROBLEM" \
      --seed "$SEED" \
      "${WANDB_ARGS[@]}" \
      --save-model \
      --n-epochs "$N_EPOCHS" \
      --n-cpu "$SLURM_CPUS_PER_TASK" \
      --enable-best-epoch-selection \
      --checkpoint-interval-epochs "$CHECKPOINT_INTERVAL_EPOCHS" \
      --validation-interval-epochs "$VALIDATION_INTERVAL_EPOCHS" \
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
      --validation-batch-size 50 \
      --validation-sigma 1.0
    ;;
  *)
    echo "ERROR: Unsupported model $MODEL"
    exit 1
    ;;
esac
