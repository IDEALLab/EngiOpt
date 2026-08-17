#!/bin/bash
# Simulator-backed board over the WHOLE published pool for one problem, sharded
# across a job array.
#
# The single-node version is fine for beams2d (~1.5 min/package) and hopeless
# everywhere else: heatconduction2d measures ~36 min/package and photonics2d
# more than 1.7 h, so their 217- and 211-package pools are ~130 h and >360 h of
# serial work. Each task strides through the sorted pool and appends to its own
# CSV, so tasks never contend for a file and a task that dies resumes where it
# stopped.
#
# `--skip-published` drops every package whose checkpoint already carries these
# metrics on the Hub, so a rerun costs only the genuine remainder -- 323 of the
# 426 packages across heat and photonics are already done. `--publish` puts each
# new row beside the weights it describes, so the next sweep can skip it too.
#
# CPU-only by default. The optimizer is CPU-bound on all three problems, so a
# GPU buys only faster sampling -- a few minutes against a multi-hour physics
# cost -- while competing for the 14-GPU student limit that the shards need to
# run wide. Pass GPUS=1 for beams2d, where sampling is most of the cost.
#
#SBATCH --job-name=physarr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7G
#SBATCH --time=24:00:00
#SBATCH --output=logs/physarr_%A_%a.log
#SBATCH --error=logs/physarr_%A_%a.err
set -uo pipefail

PROJECT_DIR="${ENGIOPT_DIR:-$HOME/projects/EngiOpt-eval}"
PROBLEM="${PROBLEM:?set PROBLEM}"
NUM_SHARDS="${NUM_SHARDS:?set NUM_SHARDS}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
# Outputs to scratch (2.5 TB) rather than home (50 GB hard quota). A full home
# killed a physics task at the 84-minute mark, after it had done all the work.
OUT_DIR="${OUT_DIR:-/cluster/scratch/$USER/physpool}"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/${OUT_PREFIX:-physboard_${PROBLEM}}_shard${SHARD}.csv"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy
source "$HOME/venv/engibench/bin/activate"
[ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.engiopt_secrets" ] && source "$HOME/.engiopt_secrets"
export HF_TOKEN

# Caches to node-local scratch. Home is a 50 GB quota and a few hundred model
# downloads fill it; a job that dies on "Disk quota exceeded" takes every other
# running job with it.
TMPDIR="${TMPDIR:-/tmp}"
export HF_HOME="$TMPDIR/hf"
export HF_HUB_CACHE="$TMPDIR/hf/hub"
export HF_DATASETS_CACHE="$TMPDIR/datasets"
export TORCH_HOME="$TMPDIR/torch"
export XDG_CACHE_HOME="$TMPDIR/xdg"
export MPLCONFIGDIR="$TMPDIR/mpl"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

# The optimizer is single-threaded per design; letting BLAS fan out over the
# same 8 cores oversubscribes them and slows the sweep down.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$PROJECT_DIR" || exit 1
echo "problem=$PROBLEM shard=$SHARD/$NUM_SHARDS out=$OUT host=$(hostname)"
python -m engiopt.evaluation.physics_board \
  --problem-id "$PROBLEM" \
  ${SPEC:+--spec "$SPEC"} \
  ${ALGOS:+--algos $ALGOS} \
  --num-shards "$NUM_SHARDS" \
  --shard "$SHARD" \
  --skip-published \
  --publish \
  --out "$OUT"
echo "exit=$?"
