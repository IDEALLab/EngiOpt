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

Use a new output directory. The trainer reads the training split to determine its design range, even for a single update. Reload the package using the example above with `local_model_dir="./cfm-smoke"` instead of downloading weights. A one-update model tests the pipeline, not design quality.

For a longer run, omit `--max-batches` and set `--n-epochs` and `--device`. The package contains the **final** weights. The study instead shortlisted checkpoints on validation MMD and chose between them on validation COG.
