#!/bin/bash
# Fill the physics gaps in the IDETC'26 line-ups, and publish them to the Hub.
#
# The full-pool sweeps of 2026-08-14 reached 198/198 beams2d, 146/217
# heatconduction2d and 174/211 photonics2d packages. The unfinished remainder
# happens to contain six packages the workshop line-ups name, so rather than
# re-run two pools this scores exactly those six -- one array task each, because
# a photonics package is >1.7 h on its own and there is no reason to serialize
# them.
#
#   sbatch --array=0-5 slurm/physics_gapfill.sh
#
# Each task appends to its own CSV *and* writes the result into the checkpoint
# package on the Hub (`--publish`), so the numbers live beside the weights they
# describe rather than only in a home directory. That is the whole point: five
# hours of optimizer time should not survive as a file somebody can overwrite.
#
# CPU-only. The optimizer is CPU-bound on all three problems; a GPU buys only
# faster sampling, which is minutes against a multi-hour physics cost.
#
#SBATCH --job-name=physgap
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7G
#SBATCH --time=12:00:00
#SBATCH --output=logs/physgap_%A_%a.log
#SBATCH --error=logs/physgap_%A_%a.err
set -uo pipefail

PROJECT_DIR="${ENGIOPT_DIR:-$HOME/projects/EngiOpt-eval}"

# (problem, package key) pairs, in array-index order. The fingerprints are the
# ones the workshop configs pin -- note vqgan differs between problems because
# that family hashes `cond_dim` into its fingerprint and heatconduction2d has two
# conditions where the others have three.
TARGETS=(
  "heatconduction2d knn_retrieval/73df04e1/s1"
  "heatconduction2d vqgan/a6fb3f0b/s1"
  "heatconduction2d gan_cnn_2d/6293adb3/s1"
  "photonics2d      knn_retrieval/73df04e1/s1"
  "photonics2d      vqgan/1151406c/s1"
  "photonics2d      gan_cnn_2d/6293adb3/s1"
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$INDEX" -ge "${#TARGETS[@]}" ]; then
  echo "no target at index $INDEX (have ${#TARGETS[@]})"
  exit 0
fi
read -r PROBLEM KEY <<< "${TARGETS[$INDEX]}"
OUT="physgap_${PROBLEM}_$(echo "$KEY" | tr '/' '_').csv"

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

# The optimizer is single-threaded per design; letting BLAS fan out over the same
# 8 cores oversubscribes them and slows the run down.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$PROJECT_DIR" || exit 1
echo "task=$INDEX problem=$PROBLEM key=$KEY out=$OUT host=$(hostname)"
# Pin the spec: this checkout still carries both v1 and v2, whose conditions are
# identical (same condition_digest -- v2 was renamed to v1), but the version is
# stamped onto every published row and must name the one the repo still has.
python -m engiopt.evaluation.physics_board \
  --problem-id "$PROBLEM" \
  --spec "${PROBLEM}/v1" \
  --only "$KEY" \
  --publish \
  --out "$OUT"
echo "exit=$?"
