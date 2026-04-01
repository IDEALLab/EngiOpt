"""Run flow-matching evaluation across multiple integration-step settings."""

from __future__ import annotations

import dataclasses
from pathlib import Path
import time
from typing import Any

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd
import torch as th
import tyro
import wandb

from engiopt import metrics
from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.flow_matching_2d_cond.core import build_model
from engiopt.flow_matching_2d_cond.core import generate_samples
from engiopt.flow_matching_2d_cond.core import load_local_checkpoint
from engiopt.flow_matching_2d_cond.evaluate_flow_matching_2d_cond import checkpoint_config
from engiopt.flow_matching_2d_cond.evaluate_flow_matching_2d_cond import load_artifact_checkpoint
from engiopt.flow_matching_2d_cond.evaluate_flow_matching_2d_cond import select_device
from engiopt.reporting import write_metrics_csv


@dataclasses.dataclass
class Args:
    """Command-line arguments for flow-matching integration-step sweeps."""

    problem_id: str = "beams2d"
    """Problem identifier."""
    seed: int = 1
    """Random seed to run."""
    wandb_project: str = "engiopt"
    """W&B project name."""
    wandb_entity: str | None = None
    """W&B entity name."""
    track: bool = True
    """Log sweep metrics and metadata to W&B."""
    run_name: str | None = None
    """Optional W&B run name override."""
    wandb_job_type: str = "ablation-evaluation"
    """W&B job type for ablation runs."""
    n_samples: int = 50
    """Number of generated samples per step setting."""
    sigma: float = 10.0
    """Kernel bandwidth for MMD and DPP metrics."""
    integration_steps_list: str = "5,10,20,35,50"
    """Comma-separated list of Euler integration steps."""
    output_dir: str = "ablation_csv_shards"
    """Output directory for per-step metric shards."""
    checkpoint_path: str | None = None
    """Optional local checkpoint path. Preferred over WandB artifacts when set."""
    device: str = "auto"
    """Device selection for local smoke runs and evaluation."""
    clip_min: float = 1e-3
    """Minimum value used when clipping generated designs."""
    clip_max: float = 1.0
    """Maximum value used when clipping generated designs."""
    condition_seed: int | None = None
    """Seed used for condition sampling. Defaults to the main seed."""
    method: str = "euler"
    """Integration method: euler, midpoint, heun, rk4, rk45, dpm."""
    atol: float = 1e-3
    """Absolute tolerance for adaptive solvers (RK45)."""
    rtol: float = 1e-3
    """Relative tolerance for adaptive solvers (RK45)."""


def parse_steps(steps_text: str) -> list[int]:
    """Parse and validate integration-step values."""
    parsed: list[int] = []
    for token in steps_text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"integration steps must be positive; got {value}")
        parsed.append(value)
    if not parsed:
        raise ValueError("No integration-step values were provided")
    return sorted(set(parsed))


if __name__ == "__main__":
    args = tyro.cli(Args)
    eval_start = time.perf_counter()
    integration_steps_values = parse_steps(args.integration_steps_list)

    # If using an adaptive solver, the number of steps is irrelevant. 
    # We force the list to [0] so the loop runs exactly once.
    if args.method == 'rk45':
        print("Adaptive solver detected: Running single evaluation (ignoring step sweep).")
        integration_steps_values = [0]

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    device = select_device(args.device)
    condition_seed = args.seed if args.condition_seed is None else args.condition_seed
    conditions_tensor, sampled_conditions, sampled_designs_np, sampled_indices = sample_conditions(
        problem=problem,
        n_samples=args.n_samples,
        device=device,
        seed=condition_seed,
    )
    conditions_tensor = conditions_tensor.unsqueeze(1)

    if args.checkpoint_path is not None:
        checkpoint = load_local_checkpoint(args.checkpoint_path, device)
        run_config: dict[str, Any] = dict(checkpoint.get("args", {}))
        checkpoint_source = "local_checkpoint"
    else:
        checkpoint, run_config = load_artifact_checkpoint(
            problem_id=args.problem_id,
            seed=args.seed,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            device=device,
        )
        checkpoint_source = "wandb_artifact"

    layers_per_block = int(checkpoint_config(checkpoint, "layers_per_block", run_config.get("layers_per_block", 2)))
    num_train_timesteps = int(
        checkpoint_config(checkpoint, "num_train_timesteps", run_config.get("num_train_timesteps", 1000))
    )

    model = build_model(
        design_shape=problem.design_space.shape,
        encoder_hid_dim=len(problem.conditions_keys),
        layers_per_block=layers_per_block,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for integration_steps in integration_steps_values:
        generation_start = time.perf_counter()
        gen_designs = generate_samples(
            model=model,
            design_shape=problem.design_space.shape,
            encoder_hidden_states=conditions_tensor,
            integration_steps=integration_steps,
            num_train_timesteps=num_train_timesteps,
            device=device,
            method=args.method,
            atol=args.atol,
            rtol=args.rtol
        )
        generation_runtime_sec = time.perf_counter() - generation_start
        gen_designs = gen_designs.squeeze(1)
        gen_designs_np = gen_designs.detach().cpu().numpy().reshape(args.n_samples, *problem.design_space.shape)
        gen_designs_np = np.clip(gen_designs_np, args.clip_min, args.clip_max)
        generation_samples_per_sec = args.n_samples / generation_runtime_sec if generation_runtime_sec > 0 else 0.0

        metrics_start = time.perf_counter()
        metrics_dict = metrics.metrics(
            problem,
            gen_designs_np,
            sampled_designs_np,
            sampled_conditions,
            sigma=args.sigma,
            gen_time=generation_runtime_sec,
            gen_speed=generation_samples_per_sec,
        )
        metrics_runtime_sec = time.perf_counter() - metrics_start
        evaluation_runtime_sec = generation_runtime_sec + metrics_runtime_sec
        generation_samples_per_sec = args.n_samples / generation_runtime_sec if generation_runtime_sec > 0 else float("nan")
        metrics_dict.update(
            {
                "seed": args.seed,
                "problem_id": args.problem_id,
                "model_id": "flow_matching_2d_cond",
                "n_samples": args.n_samples,
                "sigma": args.sigma,
                "integration_steps": integration_steps,
                "num_train_timesteps": num_train_timesteps,
                "layers_per_block": layers_per_block,
                "checkpoint_source": checkpoint_source,
                "condition_seed": condition_seed,
                "sampled_indices": ",".join(str(int(index)) for index in sampled_indices.tolist()),
                "generation_runtime_sec": generation_runtime_sec,
                "metrics_runtime_sec": metrics_runtime_sec,
                "evaluation_runtime_sec": evaluation_runtime_sec,
                "generation_samples_per_sec": generation_samples_per_sec,
                "method": args.method,
            }
        )
        all_rows.append(metrics_dict)

        out_path = output_dir / (
            f"metrics_flow_matching_2d_cond_{args.problem_id}_seed{args.seed}_steps{integration_steps}.csv"
        )
        write_metrics_csv([metrics_dict], str(out_path), append_output=False)
        print(
            f"step={integration_steps}: wrote {out_path} "
            f"(cog={metrics_dict['cog']:.6g}, fog={metrics_dict['fog']:.6g}, gen={generation_runtime_sec:.2f}s)"
        )

    if args.track and all_rows:
        run_name = args.run_name or (
            f"{args.problem_id}__flow_matching_2d_cond__ablation__seed{args.seed}__{int(time.time())}"
        )
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type=args.wandb_job_type,
            name=run_name,
            config={
                **vars(args),
                "model_id": "flow_matching_2d_cond",
                "checkpoint_source": checkpoint_source,
                "resolved_num_train_timesteps": num_train_timesteps,
                "resolved_layers_per_block": layers_per_block,
                "resolved_integration_steps_list": integration_steps_values,
            },
        )
        if run is None:
            raise RuntimeError("Failed to initialize Weights & Biases run")

        df = pd.DataFrame(all_rows).sort_values("integration_steps")
        run.log({"ablation/flow_matching_steps_table": wandb.Table(dataframe=df)})
        for row in df.to_dict(orient="records"):
            step = int(row["integration_steps"])
            run.log(
                {
                    f"ablation/cog/step_{step}": row["cog"],
                    f"ablation/fog/step_{step}": row["fog"],
                    f"ablation/mmd/step_{step}": row["mmd"],
                    f"ablation/dpp/step_{step}": row["dpp"],
                    f"ablation/viol/step_{step}": row["viol"],
                    f"ablation/runtime_sec/step_{step}": row["generation_runtime_sec"],
                    f"ablation/samples_per_sec/step_{step}": row["generation_samples_per_sec"],
                }
            )
        run.finish()

    print(f"Done in {time.perf_counter() - eval_start:.2f}s with {len(all_rows)} step settings.")
