#!/bin/bash
# Train the deconvolutional regression baseline on every 2D problem.
#
# This is the comparator from Habibi et al. -- a supervised conditions-to-design
# network -- and the bank currently carries only the kNN half of that
# comparison. One array task per problem.
#
# A GPU is requested because this one actually trains: ~3.7M parameters over a
# few hundred epochs of the full design split. It is small, so a single GPU for
# an hour covers every problem.
#SBATCH --job-name=deconvreg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --output=logs/deconvreg_%A_%a.log
#SBATCH --error=logs/deconvreg_%A_%a.err
set -uo pipefail
PROJECT_DIR="${ENGIOPT_DIR:-$HOME/projects/EngiOpt-eval}"
module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy
source "$HOME/venv/engibench/bin/activate"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.engiopt_secrets" ]; then source "$HOME/.engiopt_secrets"; fi
export HF_TOKEN
export HF_HOME="$TMPDIR/hf"
export HF_HUB_CACHE="$TMPDIR/hf/hub"
export HF_DATASETS_CACHE="$TMPDIR/datasets"
export TORCH_HOME="$TMPDIR/torch"
export XDG_CACHE_HOME="$TMPDIR/xdg"
export MPLCONFIGDIR="$TMPDIR/mpl"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
cd "$PROJECT_DIR" || exit 1

# One problem per array index. Keep in step with train_knn_k1.sh.
# thermoelastic2d is deliberately absent: 4 of its 7 conditions are 65x65
# boundary matrices, which `condition_keys` drops, so a scalar-conditioned model
# cannot see what determines the design. The deconv regressor flatlined at the
# mean design there (val MSE 0.144, unmoved from epoch 20) and every other
# scalar-conditioned model in the bank has the same blind spot. It is out of
# workshop scope until image conditions are wired through.
PROBLEMS=(beams2d heatconduction2d photonics2d)
PROBLEM="${PROBLEMS[${SLURM_ARRAY_TASK_ID:-0}]}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-200}"

echo "=== deconv_regression on $PROBLEM, seed $SEED, $EPOCHS epochs"
# Each run writes deconv_regression.pth into its own directory: the array tasks
# share a working directory, and the filename is fixed by the training script.
RUN_DIR="$TMPDIR/deconvreg_${PROBLEM}_${SEED}"
mkdir -p "$RUN_DIR" && cd "$RUN_DIR" || exit 1

python "$PROJECT_DIR/engiopt/generators/deconv_regression/deconv_regression.py" \
  --problem-id "$PROBLEM" \
  --seed "$SEED" \
  --n-epochs "$EPOCHS" \
  --save-model \
  --checkpoint-backend "${BACKEND:-hf}"
echo "exit=$?"
