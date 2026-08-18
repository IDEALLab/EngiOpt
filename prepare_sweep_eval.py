"""Fetch run IDs from WandB HP-tuning sweeps and write sweep_runs.txt for SLURM array eval.

Usage:
    python prepare_sweep_eval.py

Edit SWEEP_IDS below with your actual sweep IDs, then submit:
    sbatch --array=0-<N-1> run_eval_cgan_sweep.sh
(The script prints the exact sbatch command at the end.)
"""

import wandb

ENTITY = "engibench"
PROJECT = "engiopt"

# Fill in your sweep IDs for each problem
SWEEP_IDS = {
    "beams2d": "<beams2d_sweep_id>",
    "heatconduction2d": "<heatconduction2d_sweep_id>",
    "thermoelastic2d": "<thermoelastic2d_sweep_id>",
}

api = wandb.Api()
rows = []
for problem_id, sweep_id in SWEEP_IDS.items():
    if sweep_id.startswith("<"):
        print(f"Skipping {problem_id}: no sweep ID set")
        continue
    sweep = api.sweep(f"{ENTITY}/{PROJECT}/{sweep_id}")
    for run in sweep.runs:
        rows.append(f"{run.id}\t{problem_id}")
    print(f"  {problem_id}: {len(sweep.runs)} runs")

with open("sweep_runs.txt", "w") as f:
    f.write("\n".join(rows) + "\n")

print(f"\nWrote {len(rows)} runs to sweep_runs.txt")
print(f"Submit with:\n  sbatch --array=0-{len(rows) - 1} run_eval_cgan_sweep.sh")
