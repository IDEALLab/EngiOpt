"""Run every 2D generative model on every problem/seed combination."""

from datetime import datetime
import itertools
import os
import subprocess
import sys

# ----------------------------------------------------------------------
# 1.  Paths to your training scripts
# ----------------------------------------------------------------------
HOME = os.environ["HOME"]
MODELS = {
    "cgan_2d": os.path.join(HOME, "projects/EngiOpt/engiopt/cgan_2d", "cgan_2d.py"),
    "gan_2d": os.path.join(HOME, "projects/EngiOpt/engiopt/gan_2d", "gan_2d.py"),
    "cgan_cnn_2d": os.path.join(HOME, "projects/EngiOpt/engiopt/cgan_cnn_2d", "cgan_cnn_2d.py"),
    "gan_cnn_2d": os.path.join(HOME, "projects/EngiOpt/engiopt/gan_cnn_2d", "gan_cnn_2d.py"),
    "diffusion_2d_cond": os.path.join(HOME, "projects/EngiOpt/engiopt/diffusion_2d_cond", "diffusion_2d_cond.py"),
}

PROBLEMS = ["thermoelastic2d", "heatconduction2d", "beams2d"]
SEEDS = range(1, 11)


# ----------------------------------------------------------------------
def main() -> None:
    """Run all 2D generative models on each problem/seed combination and track failures."""
    failures: list[tuple[str, str, int, int]] = []  # (model, problem, seed, exitcode)

    for model, problem, seed in itertools.product(MODELS, PROBLEMS, SEEDS):
        script = MODELS[model]
        cmd = [
            sys.executable,
            script,
            "--problem_id",
            problem,
            "--seed",
            str(seed),
            "--save_model",
            "--n_epochs=1000",
        ]

        banner = f"[{datetime.now():%Y‑%m‑%d %H:%M:%S}] {model:<15} | {problem:<17} | seed={seed}"
        print("=" * len(banner))
        print(banner)
        print("CMD:", " ".join(cmd))
        print("=" * len(banner), flush=True)

        try:
            subprocess.run(cmd, check=True)  # raise on nonzero return code
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            exitcode = e.returncode if isinstance(e, subprocess.CalledProcessError) else -1
            print(f"❌  FAILED (exit {exitcode}) — continuing with next combo…\n", flush=True)
            failures.append((model, problem, seed, exitcode))

    # ------------------------------------------------------------------
    #  Summary
    # ------------------------------------------------------------------
    if failures:
        print("\nSummary of failures:")
        for m, p, s, code in failures:
            print(f"  {m} | {p} | seed={s} → exit{code}")
        sys.exit(1)  # flag *overall* run as failed
    else:
        print("\n🎉  All combinations finished without errors!")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
