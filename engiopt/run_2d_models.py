"""Run all 2D models with different problems and seeds."""

import itertools
import os
import subprocess

# Define your models, problems, and seeds
models = {
    "cgan_2d": os.path.join("engiopt/cgan_2d", "cgan_2d.py"),
    "gan_2d": os.path.join("engiopt/gan_2d", "gan_2d.py"),
    "cgan_cnn_2d": os.path.join("engiopt/cgan_cnn_2d", "cgan_cnn_2d.py"),
    "gan_cnn_2d": os.path.join("engiopt/gan_cnn_2d", "gan_cnn_2d.py"),
    "diffusion_2d_cond": os.path.join("engiopt/diffusion_2d_cond", "diffusion_2d_cond.py"),
}

problems = ["beams2d", "heatconduction2d", "thermoelastic2d"]
seeds = list(range(1, 11))  # Seeds 1 to 10

# Loop through all combinations
for model_name, problem_id, seed in itertools.product(models.keys(), problems, seeds):
    script = models[model_name]

    # Construct CLI args
    args = [
        "python",
        script,
        "--problem_id",
        problem_id,
        "--seed",
        str(seed),
    ]

    # Show and run
    print("=" * 80)
    print(f"Running {script} | problem_id={problem_id} | seed={seed}")
    print("Command:", " ".join(args))
    print("=" * 80)

    subprocess.run(args, check=False)
