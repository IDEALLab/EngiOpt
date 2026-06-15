"""run_pe_optimization.py.

Command-line script to run the multi-objective power-electronics design
optimization using surrogate pipelines together with pymoo.

The script mirrors the style of your other utilities (dataclass + tyro CLI)
and logs every generation to Weights & Biases so you can explore:
  • an interactive scatter of the Pareto front with a generation slider
  • a parallel-coordinates plot of the decision variables per generation

Example usage
-------------
python run_pe_optimization.py \
    --model_gain_path "my_entity/engiopt/your_run_name_model:latest" \
    --model_ripple_path "my_entity/engiopt/your_other_run_name_model:latest" \
    --device mps \
    --pop_size 500 \
    --n_gen 100 \
    --seed 1 \
    --track \
    --wandb_project engibench
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import tyro
import wandb

from engiopt.checkpoint_store import resolve_checkpoint_reference
from engiopt.surrogate_model.model_pipeline import ModelPipeline
from engiopt.surrogate_model.pymoo_pe_problem import PymooPowerElecProblem
from engiopt.surrogate_model.training_utils import get_device

if TYPE_CHECKING:
    from pymoo.core.algorithm import Algorithm
    from pymoo.core.result import Result


# ---------------------------------------------------------------------------
#  CLI arguments
# ---------------------------------------------------------------------------
@dataclass
class Args:
    # Surrogate pipelines
    model_gain_path: str
    """Model ref for the gain surrogate (W&B artifact, HF package ref, or local package directory)."""
    model_ripple_path: str
    """Model ref for the ripple surrogate (W&B artifact, HF package ref, or local package directory)."""

    # Optimisation hyperparameters
    seed: int
    """Random seed for the optimization, must match the seed used to train the models."""
    pop_size: int = 500
    n_gen: int = 100

    # Hardware
    device: Literal["cpu", "cuda", "mps"] = "cpu"

    # Logging / I/O
    track: bool = True
    wandb_project: str = "engiopt"
    wandb_entity: str | None = None
    output_dir_prefix: str = "results"
    save_csv: bool = True
    log_every: int = 1  # gens between logs
    algo: str = os.path.basename(__file__)[: -len(".py")]
    problem_id: str = "power_electronics"


# -------------------------   W&B callback  ------------------------------
class WandbLogCallback:
    """Log pareto history table and slider-ready scatter each generation."""

    def __init__(self, log_every: int, *, track: bool) -> None:
        self.log_every = max(1, log_every)
        self.track = track
        self.history: wandb.Table | None = None
        self.columns: list[str] | None = None

    def _init_table(self, n_obj: int, n_var: int) -> None:
        # Columns: generation, f0..f{n_obj-1}, x0..x{n_var-1}
        self.columns = ["generation"] + [f"f{i}" for i in range(n_obj)] + [f"x{i}" for i in range(n_var)]
        self.history = wandb.Table(columns=self.columns)

    def __call__(self, algorithm: Algorithm) -> None:
        """Callback invoked by pymoo to log data each generation."""
        if not self.track:
            return
        gen = algorithm.n_gen
        # guard against Pyright thinking this might be Optional[int]
        if gen is None or gen % self.log_every != 0:
            return

        # Classic SolutionSet API
        sol_set = algorithm.opt
        assert sol_set is not None, "SolutionSet is None"
        f_vals = sol_set.get("F")  # returns an (n_points, n_obj) array
        x_vals = sol_set.get("X")  # returns an (n_points, n_var) array

        # Initialize table columns once
        if self.columns is None:
            self._init_table(f_vals.shape[1], x_vals.shape[1])

        # Prepare data for current generation
        data = [[gen, *f.tolist(), *x.tolist()] for f, x in zip(f_vals, x_vals)]

        # Create a table and scatter plot
        current_table = wandb.Table(columns=self.columns, data=data)
        scatter = wandb.plot.scatter(
            current_table,
            x="f0",
            y="f1",
            title="Pareto Front",
        )

        # Log metrics and visualizations
        wandb.log(
            {
                "generation": gen,
                "pareto_front": scatter,
                "min_f1": f_vals[:, 0].min(),
                "min_f2": f_vals[:, 1].min(),
            },
            step=gen,
        )


# -------------------------   CSV & TXT persistence  -----------------------------
def save_front(res: Result, output_dir: str) -> tuple[str, str, str, str, str]:
    """Save Pareto front objectives and decision variables to CSV and TXT, and return file paths."""
    os.makedirs(output_dir, exist_ok=True)

    # Pareto front csv
    evals_csv = os.path.join(output_dir, "pareto_front.csv")
    np.savetxt(
        evals_csv,
        res.F,
        delimiter=",",
        header="Objective_r,Objective_abs_g_minus_0.25",
        comments="",
    )

    # Pareto front txt
    evals_txt = os.path.join(output_dir, "pareto_front.txt")
    np.savetxt(evals_txt, res.F, fmt="%.6e", delimiter=",")

    # Designs csv -- Pareto set
    designs_csv = os.path.join(output_dir, "pareto_set.csv")
    cols = [f"x{i}" for i in range(res.X.shape[1])]
    pd.DataFrame(res.X, columns=cols).to_csv(designs_csv, index=False)

    # Pareto set txt
    designs_txt = os.path.join(output_dir, "pareto_set.txt")
    np.savetxt(designs_txt, res.X, fmt="%.6e", delimiter=",")

    # combined front and set
    pareto_csv = os.path.join(output_dir, "pareto_both.csv")
    pareto_df = pd.read_csv(designs_csv)
    pareto_df[["f0", "f1"]] = res.F
    pareto_df.to_csv(pareto_csv, index=False)

    return evals_csv, designs_csv, pareto_csv, evals_txt, designs_txt


def load_model_from_reference(
    model_ref: str,
    *,
    active_wandb_run: wandb.sdk.wandb_run.Run | None,
) -> ModelPipeline:
    """Load a model pipeline from a W&B artifact, HF package, or local directory.

    Args:
        model_ref: Reference to the stored model package or artifact.
        active_wandb_run: Optional active W&B run for artifact access.

    Returns:
        Loaded model pipeline.
    """
    resolved = resolve_checkpoint_reference(
        model_source="auto",
        model_ref=model_ref,
        required_files=[],
        active_wandb_run=active_wandb_run,
    )
    model_file = next(file_path for file_path in resolved.files.values() if file_path.endswith(".pkl"))
    return ModelPipeline.load(model_file)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main(args: Args) -> None:
    """Parse arguments, run the optimization, and handle logging and persistence."""
    device = get_device(args.device)
    print(f"[INFO] Device: {device}")

    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            save_code=True,
        )
        wandb.define_metric("generation")
        wandb.define_metric("*", step_metric="generation")

    active_wandb_run = wandb.run if args.track else None
    pipeline_g = load_model_from_reference(
        args.model_gain_path,
        active_wandb_run=active_wandb_run,
    )
    pipeline_r = load_model_from_reference(
        args.model_ripple_path,
        active_wandb_run=active_wandb_run,
    )

    problem = PymooPowerElecProblem(
        pipeline_r=pipeline_r,
        pipeline_g=pipeline_g,
        device=device,
    )
    algo = NSGA2(pop_size=args.pop_size)
    term = get_termination("n_gen", args.n_gen)

    res = minimize(
        problem,
        algo,
        term,
        seed=args.seed,
        verbose=True,
        callback=WandbLogCallback(args.log_every, track=args.track),
    )

    if args.save_csv:
        _, _, _, f_txt, x_txt = save_front(res, f"{args.output_dir_prefix}_{run_name}")
        print(f"[INFO] Saved CSV and TXT to {args.output_dir_prefix}_{run_name}")
        if args.track:
            # Only upload the TXT versions as a W&B artifact
            txt_art = wandb.Artifact(f"{run_name}_pareto_txt", type="pymoo_results")
            txt_art.add_file(f_txt)
            txt_art.add_file(x_txt)
            wandb.log_artifact(txt_art)

    if args.track:
        wandb.finish()


if __name__ == "__main__":
    try:
        main(tyro.cli(Args))
    except Exception:
        print("[ERROR] run_pe_optimization.py failed")
        raise
