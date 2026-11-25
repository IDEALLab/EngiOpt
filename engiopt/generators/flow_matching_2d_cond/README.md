# Conditional flow matching

Load a released EngOpt 2026 checkpoint and generate designs for the first two test conditions:

```python
from engibench.utils.all_problems import BUILTIN_PROBLEMS
import torch

from engiopt.utils.all_generators import BUILTIN_GENERATORS

problem_id = "beams2d"
problem = BUILTIN_PROBLEMS[problem_id]()
generator = BUILTIN_GENERATORS["flow_matching_2d_cond"].from_pretrained(
    problem,
    problem_id=problem_id,
    hf_repo_prefix="engiopt-engopt2026",
)
rows = problem.dataset["test"][:2]
conditions = torch.tensor([[rows[key][i] for key in generator.condition_keys] for i in range(2)])
designs = generator.sample(conditions, seed=2001)
```

To evaluate the same published weights and record their Hub path and revision in the score row:

```sh
python -m engiopt.evaluate --problem-id beams2d --generators flow_matching_2d_cond \
  --hf-repo-prefix engiopt-engopt2026 --seeds 1 --metrics mmd --on-error raise
```

The study prefix selects the Euler/16 package at the pinned release for that seed. For a different solver's selected checkpoint, download its directory from the [release](https://huggingface.co/IDEALLab/engiopt-engopt2026-flow-matching-2d-cond/tree/58ae97c4aa59784fbe22ab232b5d196ff088e4c7) and use `model_source="local", local_model_dir="..."`. Keep the same `problem_id` and training seed as the package.

Use `problem_id = "heatconduction2d"` for the thermal benchmark. The published weights expect the scalar conditions in the training order, as defined by EngiBench at `217394232bd628484a72b72ce624b1bdd955245a`. The package configuration selects the solver and its step count. Each Euler step uses one velocity evaluation, midpoint uses two, and RK4 uses four.

## Train a local checkpoint

For a CPU smoke test on Beams2D, run:

```sh
python -m engiopt.generators.flow_matching_2d_cond.flow_matching_2d_cond \
  --problem-id beams2d --device cpu --batch-size 2 --max-batches 1 --output ./cfm-smoke
```

Use a new output directory. The trainer reads the training split to determine its design range, even for a single update. To reload your own weights in Python, replace `hf_repo_prefix="engiopt-engopt2026"` in the example above with `model_source="local", local_model_dir="./cfm-smoke"`. A one-update model tests the pipeline, not design quality.

Without `--output`, the package goes into `local_checkpoints/flow_matching_2d_cond/{problem_id}/seed_{seed}`. Existing directories are never overwritten. The package records the number of optimizer steps, elapsed training seconds and device; dataset preparation and saving the checkpoint are not timed.

For a longer run, omit `--max-batches` and set `--n-epochs` and `--device`. The package contains the **final** weights. The study instead shortlisted checkpoints on validation MMD and chose between them on validation COG.

## Reproduce a published result

[Felder, Habibi, Massoudi and Fuge (2026)](https://arxiv.org/abs/2609.00863) report Beams2D cumulative optimality gap (COG) by solver and step count in Table 4. The following Python snippet reproduces the Euler/16 entry using the ten released per-seed test results. It downloads only metadata; it does not run inference or rerun the downstream optimizer.

```python
import json
from statistics import median, stdev
from urllib.request import urlopen

root = "https://huggingface.co/IDEALLab/engiopt-engopt2026-flow-matching-2d-cond/resolve"
revision = "58ae97c4aa59784fbe22ab232b5d196ff088e4c7"
cog = []
for seed in range(1, 11):
    package = f"beams2d/selected/v1/euler_16/seed_{seed}"
    with urlopen(f"{root}/{revision}/{package}/metadata.json", timeout=30) as response:
        metrics = json.load(response)["test_metrics"]
    assert (metrics["seed"], metrics["method"], metrics["integration_steps"]) == (seed, "euler", 16)
    cog.append(metrics["cog"])
print(f"{median(cog):.3f} +/- {stdev(cog):.3f}")  # 1.182 +/- 3.126
```

The first number is the **median**, not the mean; the second is the sample standard deviation across seeds. Each seed uses 50 test conditions. Test scores are read after checkpoint selection, never used to choose the epoch.

To regenerate the underlying scores rather than aggregate them, use the [study workflow and Slurm launchers](https://github.com/IDEALLab/EngiOpt/tree/8b9d30f4e350677a6234bc6dbae3a95c2b261937/reproducibility/conditional_flow_matching). The [campaign manifest](https://github.com/IDEALLab/EngiOpt/blob/8b9d30f4e350677a6234bc6dbae3a95c2b261937/reproducibility/conditional_flow_matching/manifest.json) fixes the run mapping; the [artifact index](https://github.com/IDEALLab/EngiOpt/blob/8b9d30f4e350677a6234bc6dbae3a95c2b261937/reproducibility/conditional_flow_matching/ARTIFACTS.md) lists the exact dataset and checkpoint revisions. Start with the dry run. The published COG is an optimizer-based metric; an MMD-only smoke test is not a substitute.
