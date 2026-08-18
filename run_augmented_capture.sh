#!/bin/bash
#SBATCH --job-name=augmented_capture
#SBATCH --time=06:00:00                # slowest task is a heatconduction2d shard (~35 min)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7G
#                                      # NO --gpus on purpose: building and simulating are
#                                      # pure numpy/scipy on CPU, nothing loads an LVAE.
#                                      # Dropping the GPU request skips the GPU queue.
#SBATCH --array=0-12                   # 1 beams + 2 photonics + 8 heatcond + 2 thermo = 13
#SBATCH --output=logs/augcap_%A_%a.log
#SBATCH --error=logs/augcap_%A_%a.err

# =============================================================================
# Build the augmented test datasets for the LV-metrics paper and simulate them.
#
# Each task does both halves for one (problem, shard):
#   1. build     the frozen augmented dataset into node-local $TMPDIR
#                (clean + gaussian_noise + blur + intensity_shift + condition_jumble,
#                 5 severities x 3 random draws)
#   2. simulate  its shard of that dataset, writing one CSV into $CSV_DIR
#
# Rebuilding per task instead of chaining a dependency job is deliberate: the build
# is a couple of minutes of numpy next to ~35 minutes of simulation, and building
# into $TMPDIR means no two tasks ever touch the same file. It is only safe because
# the builder is now reproducible from --seed (it used to seed off hash(family),
# which Python salts per process, so every rebuild drew different noise).
#
# Coverage/diversity failures (mode_drop, mode_invent, collapse) need no task here:
# they are built downstream by subsetting and repeating these rows by condition band.
#
# Submit:
#   mkdir -p logs && sbatch run_augmented_capture.sh
#
# Merge the shards afterwards, on the login node (run it under bash — an unmatched
# glob aborts the loop in zsh):
#   cd "$SCRATCH/engiopt_augmented_sim"
#   for p in beams2d photonics2d heatconduction2d thermoelastic2d; do
#     shards=$(find . -maxdepth 1 -name "${p}_test_simulated_shard*.csv" | sort)
#     [ -z "$shards" ] && continue
#     head -1 $(echo "$shards" | head -1) > ${p}_test_simulated.csv
#     for s in $shards; do tail -n +2 "$s" >> ${p}_test_simulated.csv; done
#     echo "$p: $(( $(wc -l < ${p}_test_simulated.csv) - 1 )) rows"
#   done
# =============================================================================

# where to drop the CSV shards (one per task — no race on append)
CSV_DIR="$SCRATCH/engiopt_augmented_sim"
mkdir -p "$CSV_DIR"

# hyper-parameters
N_LEVELS=6          # severity levels including 0 (the uncorrupted baseline)
BUILD_REPEATS=3     # random draws per (corruption, severity) stored in the dataset
SIM_REPEATS=2       # draws actually simulated: 2 gives error bars, and simulation is
                    # the only expensive half. The 3rd draw stays in the dataset for
                    # the LV-metric side, which costs nothing to run over it.
SEED=0

# load your environment
module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy
module load apptainer 2>/dev/null || true   # heatconduction2d simulates inside a container
source "$HOME/venv/engibench_test2/bin/activate"

echo "Running on node-local scratch: $TMPDIR"

# copy & cd
cp -r "$HOME/projects/EngiOpt" "$TMPDIR/EngiOpt"
cd "$TMPDIR/EngiOpt"

# redirect caches into TMPDIR
export HF_HOME="$TMPDIR/models"
export HF_DATASETS_CACHE="$TMPDIR/datasets"
export TORCH_HOME="$TMPDIR/torch_cache"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME"
rsync -a "$SCRATCH/datasets/"* "$HF_DATASETS_CACHE/"

# No W&B block: neither the builder nor the simulator touches W&B.
# Belt-and-braces on top of the FAMILY_SEED_OFFSET fix, so string hashing can never
# perturb a build again.
export PYTHONHASHSEED=0

# apptainer settings for heatconduction2d (see scicomp.ethz.ch/wiki/Apptainer)
export APPTAINER_CACHEDIR="$SCRATCH/.apptainer"
export APPTAINER_TMPDIR="$TMPDIR"
HEAT_SIF="pyadjoint_master.sif"

# sweep definition
#   problem            sources  test split   simulate cost per design
#   beams2d            243      243 (all)    ~0.08 s
#   photonics2d        200      200 (all)    ~0.18 s
#   heatconduction2d    40       40 (all)    ~9    s   <- container round-trip, hence 8 shards
#   thermoelastic2d    256     1800          ~0.18 s
PROBLEMS=(beams2d photonics2d heatconduction2d thermoelastic2d)
N_SOURCES=(243 200 40 256)
N_SHARDS=(1 2 8 2)

# Flatten (problem, shard) into the array index. Shard counts differ per problem, so
# this walks the table rather than dividing.
TASK_ID=$SLURM_ARRAY_TASK_ID
TOTAL=0
for c in "${N_SHARDS[@]}"; do TOTAL=$(( TOTAL + c )); done
if (( TASK_ID >= TOTAL )); then
  echo "Invalid task ID $TASK_ID (>= $TOTAL)"
  exit 1
fi

CURSOR=0
for i in "${!PROBLEMS[@]}"; do
  if (( TASK_ID < CURSOR + ${N_SHARDS[$i]} )); then
    P_IDX=$i
    SHARD_IDX=$(( TASK_ID - CURSOR ))
    break
  fi
  CURSOR=$(( CURSOR + ${N_SHARDS[$i]} ))
done

PROBLEM=${PROBLEMS[$P_IDX]}
SOURCES=${N_SOURCES[$P_IDX]}
SHARDS=${N_SHARDS[$P_IDX]}

echo "Task $TASK_ID → Problem=$PROBLEM, Sources=$SOURCES, Shard=$SHARD_IDX of $SHARDS"

# --- 1. build the augmented dataset (node-local, deterministic from $SEED) ----
AUG_DIR="$TMPDIR/augmented"
mkdir -p "$AUG_DIR"

python -m engiopt.build_augmented_datasets \
  --problem-id "$PROBLEM" \
  --split test \
  --n-samples "$SOURCES" \
  --n-levels "$N_LEVELS" \
  --n-repeats "$BUILD_REPEATS" \
  --seed "$SEED" \
  --output-dir "$AUG_DIR"

# Keep one copy of the dataset and its manifest for the record. Shard 0 wins the race;
# every shard built byte-identical data, so whichever lands first is the right one.
if (( SHARD_IDX == 0 )); then
  mkdir -p "$SCRATCH/engiopt_augmented"
  cp -r "$AUG_DIR/${PROBLEM}_test_augmented" "$SCRATCH/engiopt_augmented/" 2>/dev/null || true
  cp "$AUG_DIR/${PROBLEM}_test_augmented_manifest.json" "$SCRATCH/engiopt_augmented/" 2>/dev/null || true
fi

# heatconduction2d only: the container runtime looks for the .sif in the working
# directory and bind-mounts the working directory, so the image has to sit next to
# the code. Reuse a pre-pulled copy if there is one instead of pulling 8 times.
if [[ "$PROBLEM" == "heatconduction2d" ]]; then
  if [[ -f "$SCRATCH/engiopt_augmented/$HEAT_SIF" ]]; then
    cp "$SCRATCH/engiopt_augmented/$HEAT_SIF" "$TMPDIR/EngiOpt/$HEAT_SIF"
    echo "using pre-pulled $HEAT_SIF"
  else
    echo "no pre-pulled $HEAT_SIF — this task will pull it itself."
    echo "To avoid 8 concurrent pulls, run once on a login node:"
    echo "  cd \$SCRATCH/engiopt_augmented && apptainer pull docker://quay.io/dolfinadjoint/pyadjoint:master"
  fi
fi

# --- 2. simulate this shard ---------------------------------------------------
# --resume is on by default: if a task hits the wall clock, resubmitting just that
# array index picks up where its shard CSV left off.
python -m engiopt.simulate_augmented_datasets \
  --problem-id "$PROBLEM" \
  --split test \
  --dataset-dir "$AUG_DIR" \
  --output-dir "$CSV_DIR" \
  --repeats "$SIM_REPEATS" \
  --n-shards "$SHARDS" \
  --shard-idx "$SHARD_IDX"
