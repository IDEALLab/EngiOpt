#!/bin/bash
# Pre-sample every line-up member for the IDETC'26 workshop, one task per problem.
#
#   sbatch --array=0-2 slurm/build_design_cache.sh
#
# The session opens by showing designs from eight models. Drawing them live costs
# minutes -- a diffusion model on a Colab CPU runtime is the worst of it -- and
# produces the same designs for every team every time, so they are sampled once
# here and shipped inside the package. Without this the workshop's first ten
# minutes are a progress bar.
#
# GPU, unlike the physics jobs: this is pure sampling, which is exactly what a
# GPU helps. The 1.3 GB VQGAN transformer is the bulk of it.
#
#SBATCH --job-name=dcache
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dcache_%A_%a.log
#SBATCH --error=logs/dcache_%A_%a.err
set -uo pipefail

PROJECT_DIR="${ENGIOPT_DIR:-$HOME/projects/EngiOpt-eval}"
PROBLEMS=(beams2d heatconduction2d photonics2d)
PROBLEM="${PROBLEMS[${SLURM_ARRAY_TASK_ID:-0}]}"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy
source "$HOME/venv/engibench/bin/activate"
[ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.engiopt_secrets" ] && source "$HOME/.engiopt_secrets"
export HF_TOKEN

# Everything cacheable goes to node-local scratch. Home is a 50 GB quota and this
# job downloads every checkpoint in the line-up, including a 1.3 GB VQGAN
# transformer per problem -- which is precisely how a full home killed a
# physics task mid-run on 2026-08-17.
TMPDIR="${TMPDIR:-/tmp}"
export HF_HOME="$TMPDIR/hf"
export HF_HUB_CACHE="$TMPDIR/hf/hub"
export HF_DATASETS_CACHE="$TMPDIR/datasets"
export TORCH_HOME="$TMPDIR/torch"
export XDG_CACHE_HOME="$TMPDIR/xdg"
export MPLCONFIGDIR="$TMPDIR/mpl"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

cd "$PROJECT_DIR" || exit 1
echo "problem=$PROBLEM host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

python workshops/idetc26/tools/build_design_cache.py --problem-id "$PROBLEM" --into package
echo "exit=$?"
du -sh engiopt/workshops/idetc26/cache 2>/dev/null
