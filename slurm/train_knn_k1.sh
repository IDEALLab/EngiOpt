#!/bin/bash
# Publish the k=1 retrieval baseline for every 2D problem.
#
# `k = 1` is pure retrieval: it can only return designs that are already in the
# training set, which is what makes it a lookup table and what makes the
# memorization argument land -- `mmd` near zero and `novelty` near zero at the
# same time. The published default is k=5, which averages five neighbours and so
# returns a blend that exists nowhere in the data; that is a different model
# making a different point, and both can live in the pool.
#
# CPU-only, and minutes: "training" a kNN is storing the training split.
#SBATCH --job-name=knnk1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0:30:00
#SBATCH --output=logs/knnk1_%A_%a.log
#SBATCH --error=logs/knnk1_%A_%a.err
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

# Keep in step with train_deconv_regression.sh.
# thermoelastic2d is deliberately absent: 4 of its 7 conditions are 65x65
# boundary matrices, which `condition_keys` drops, so a scalar-conditioned model
# cannot see what determines the design. The deconv regressor flatlined at the
# mean design there (val MSE 0.144, unmoved from epoch 20) and every other
# scalar-conditioned model in the bank has the same blind spot. It is out of
# workshop scope until image conditions are wired through.
PROBLEMS=(beams2d heatconduction2d photonics2d)
PROBLEM="${PROBLEMS[${SLURM_ARRAY_TASK_ID:-0}]}"
SEED="${SEED:-1}"

echo "=== knn_retrieval k=1 on $PROBLEM, seed $SEED"
RUN_DIR="$TMPDIR/knnk1_${PROBLEM}_${SEED}"
mkdir -p "$RUN_DIR" && cd "$RUN_DIR" || exit 1

python "$PROJECT_DIR/engiopt/generators/knn_retrieval/knn_retrieval.py" \
  --problem-id "$PROBLEM" \
  --seed "$SEED" \
  --k 1 \
  --save-model \
  --checkpoint-backend "${BACKEND:-hf}"
echo "exit=$?"
