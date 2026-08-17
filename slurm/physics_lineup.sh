#!/bin/bash
# Score every member of one problem's IDETC'26 line-up, from one evaluation, and
# publish each row to its checkpoint package.
#
#   PROBLEM=beams2d sbatch slurm/physics_lineup.sh
#
# Why this exists rather than harvesting: the archived sweeps disagree with each
# other. Sampling is seeded but not reproducible across hardware, and the
# optimizer amplifies a small difference in a starting design into a large
# difference in the gap -- on beams2d, 262 of 263 packages scored by two sweeps
# differ, one of them flipping sign. Both are valid evaluations; neither is
# wrong. But a sealed board must come from ONE of them, and a row assembled from
# two (means here, medians there) describes an evaluation that never happened.
#
# So for a problem cheap enough to redo -- beams2d is 2-3 min per package -- the
# honest move is to score the whole line-up once, together, and publish that.
#
# The keys come from the workshop config, so they cannot drift from the bank the
# notebook actually loads.
#
#SBATCH --job-name=physline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7G
#SBATCH --time=08:00:00
#SBATCH --output=logs/physline_%j.log
#SBATCH --error=logs/physline_%j.err
set -uo pipefail

PROJECT_DIR="${ENGIOPT_DIR:-$HOME/projects/EngiOpt-eval}"
PROBLEM="${PROBLEM:?set PROBLEM=beams2d|heatconduction2d|photonics2d}"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy
source "$HOME/venv/engibench/bin/activate"
[ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.engiopt_secrets" ] && source "$HOME/.engiopt_secrets"
export HF_TOKEN

TMPDIR="${TMPDIR:-/tmp}"
export HF_HOME="$TMPDIR/hf"
export HF_HUB_CACHE="$TMPDIR/hf/hub"
export HF_DATASETS_CACHE="$TMPDIR/datasets"
export TORCH_HOME="$TMPDIR/torch"
export XDG_CACHE_HOME="$TMPDIR/xdg"
export MPLCONFIGDIR="$TMPDIR/mpl"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$PROJECT_DIR" || exit 1

# Derive the package keys from the workshop config rather than repeating them.
KEYS=$(python - "$PROBLEM" <<'PYEOF'
import json, pathlib, sys
problem = sys.argv[1]
cfg = json.loads(pathlib.Path(f"engiopt/workshops/idetc26/problems/{problem}.json").read_text())
print(" ".join(
    f"{e['algo']}/{e.get('config_fingerprint') or 'default'}/s{e.get('seed', 1)}"
    for e in cfg["bank"] if e.get("kind", "pretrained") == "pretrained"
))
PYEOF
)
echo "problem=$PROBLEM host=$(hostname)"
echo "line-up keys: $KEYS"

python -m engiopt.evaluation.physics_board \
  --problem-id "$PROBLEM" \
  --spec "${PROBLEM}/v1" \
  --only $KEYS \
  --publish \
  --out "physline_${PROBLEM}.csv"
echo "exit=$?"
