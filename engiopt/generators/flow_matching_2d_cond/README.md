# Conditional flow matching

Load a released EngOpt 2026 checkpoint and generate designs for the first two test conditions:

```python
from pathlib import Path

from engibench.utils.all_problems import BUILTIN_PROBLEMS
from huggingface_hub import snapshot_download
import torch

from engiopt.utils.all_generators import BUILTIN_GENERATORS

problem_id = "beams2d"
package = f"{problem_id}/selected/v1/euler_16/seed_1"
root = snapshot_download(
    "IDEALLab/engiopt-engopt2026-flow-matching-2d-cond",
    revision="58ae97c4aa59784fbe22ab232b5d196ff088e4c7",
    allow_patterns=[f"{package}/{file}" for file in ("model.pth", "metadata.json", "run_config.json")],
)
problem = BUILTIN_PROBLEMS[problem_id]()
generator = BUILTIN_GENERATORS["flow_matching_2d_cond"].from_pretrained(
    problem, problem_id=problem_id, model_source="local", local_model_dir=str(Path(root) / package),
)
rows = problem.dataset["test"][:2]
conditions = torch.tensor([[rows[key][i] for key in generator.condition_keys] for i in range(2)])
designs = generator.sample(conditions, seed=2001)
```

Use `problem_id = "heatconduction2d"` for the thermal benchmark. The published weights expect the scalar conditions in the training order, as defined by EngiBench at `217394232bd628484a72b72ce624b1bdd955245a`. The package configuration selects the solver and its step count. Each Euler step uses one velocity evaluation, midpoint uses two, and RK4 uses four.
