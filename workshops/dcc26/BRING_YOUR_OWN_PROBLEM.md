# Bring Your Own EngiBench Problem

This guide is for participants who want to turn a real research problem, for example a PhD simulator, solver, or design study, into an EngiBench problem after the workshop.

Notebook 03 shows the core idea with an in-notebook toy cantilever. A production EngiBench contribution needs a little more: a package module, a hosted dataset, documentation, and the checks expected by the EngiBench repository.

## The Contract

An EngiBench problem is ready when another researcher can do this without knowing your simulator internals:

```python
from engibench.problems.my_problem import MyProblem

problem = MyProblem(seed=0)
design, idx = problem.random_design()
conditions = {"load": 1.0}
violations = problem.check_constraints(design, config=conditions)
objectives = problem.simulate(design, config=conditions)
fig, ax = problem.render(design)
problem.reset(seed=1)
```

The problem must define these pieces:

| Piece | What it answers | Where it lives |
|---|---|---|
| Design representation | What can a method output? | `design_space` |
| Operating conditions | What scenario is the design evaluated under? | `Conditions` and `conditions` |
| Objectives | What does better mean? | `objectives` |
| Simulator | How is a candidate scored? | `simulate_verbose`, optionally `simulate` |
| Constraints | What is invalid or outside scope? | `Conditions`, `Config`, `design_constraints`, `design_space` |
| Dataset | What examples train or compare methods? | `dataset_id`, hosted Hugging Face dataset |
| Baseline | What non-ML method should be beaten? | `optimize`, if available |
| Rendering | What is the canonical human view? | `render` |
| Reproducibility | How is simulator/random state reset? | `reset` |

## Before Writing Code

Write down these decisions first. If one is vague, the benchmark will be vague too.

1. **Design variables.** Define the exact shape, dtype, units, and bounds of one design. Decide whether a design is an array, image, graph, dict, geometry object, or another structured type.
2. **Conditions.** Separate what the method controls from what the world imposes. Loads, boundary conditions, material settings, operating points, target specs, and environment parameters usually belong in `Conditions`.
3. **Objectives.** Name each objective exactly as it will appear in the dataset. Use one scalar per objective and mark each as minimize or maximize.
4. **Constraints.** Separate physics impossibilities, benchmark scope limits, manufacturing rules, solver limits, and soft warnings. Do not hide validity inside a large penalty objective.
5. **Simulator.** Decide whether `simulate` can run locally in Python, needs a container, or calls an external binary.
6. **Baseline.** Pick the strongest practical classical method you can afford to run. If no optimizer is available, make that explicit by leaving `optimize` unimplemented or raising `NotImplementedError`.
7. **Dataset plan.** Decide how many condition points you will sample, how each reference design is generated, and which split policy gives train/test/val separation.
8. **Reproducibility.** Record units, seeds, solver versions, mesh settings, tolerances, hardware assumptions, and citations.

## Production File Layout

Create one package under `engibench/problems/`. Follow existing examples such as `beams2d`, `heatconduction2d`, or `photonics2d`.

```text
engibench/
  problems/
    my_problem/
      __init__.py
      v0.py
```

`__init__.py` should export exactly one `Problem` subclass. EngiBench discovery imports `engibench.problems.<problem_id>` and fails if the module exposes zero or multiple `Problem` subclasses.

```python
"""MyProblem problem module."""

from engibench.problems.my_problem.v0 import MyProblem

__all__ = ["MyProblem"]
```

Use `v0.py` for the first version. If a later change breaks dataset compatibility, create a new version module. If the change is only an implementation improvement with the same benchmark contract, keep the same version.

## Minimal Production Skeleton

This is the shape to copy into `engibench/problems/my_problem/v0.py`. Replace the placeholders with your real physics.

```python
from dataclasses import dataclass
from typing import Annotated, Any

from gymnasium import spaces
import numpy as np
import numpy.typing as npt

from engibench.constraint import THEORY
from engibench.constraint import bounded
from engibench.constraint import constraint
from engibench.core import ObjectiveDirection
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.core import SimulationResult


@constraint(categories=THEORY)
def design_is_physically_valid(design: npt.NDArray) -> None:
    """Example design-level constraint."""
    assert np.isfinite(design).all(), "design contains non-finite values"


class MyProblem(Problem[npt.NDArray]):
    """Short description of the engineering design problem."""

    version = 0
    objectives = (("objective_name", ObjectiveDirection.MINIMIZE),)

    @dataclass
    class Conditions:
        """Scenario variables that must appear as dataset columns."""

        load: Annotated[float, bounded(lower=0.0, upper=10.0).category(THEORY)] = 1.0
        """Load applied to the design, in the units used by the simulator."""

    @dataclass
    class Config(Conditions):
        """Conditions plus simulator or optimizer settings."""

        max_iter: Annotated[int, bounded(lower=1)] = 100

    conditions = Conditions()
    design_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float64)
    design_constraints = (design_is_physically_valid,)
    dataset_id = "IDEALLab/my_problem_v0"
    container_id = None

    def __init__(self, seed: int = 0, config: dict[str, Any] | None = None) -> None:
        super().__init__(seed=seed)
        self.config = self.Config(**(config or {}))
        self.conditions = self.Conditions(load=self.config.load)

    def simulate_verbose(self, design: npt.NDArray, config: dict[str, Any] | None = None) -> SimulationResult:
        cfg = self.Config(**{**self.config.__dict__, **(config or {})})
        value = float(np.sum(design) * cfg.load)
        return SimulationResult(np.array([value], dtype=np.float64))

    def random_design(self, dataset_split: str = "train", design_key: str = "optimal_design") -> tuple[npt.NDArray, int]:
        idx = int(self.np_random.integers(low=0, high=len(self.dataset[dataset_split])))
        return np.asarray(self.dataset[dataset_split][design_key][idx]), idx

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed=seed)
        # Reset any simulator-specific state here.

    def optimize(
        self, starting_point: npt.NDArray, config: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray, list[OptiStep]]:
        raise NotImplementedError

    def render(self, design: npt.NDArray, *, open_window: bool = False) -> Any:
        raise NotImplementedError
```

Important details:

- Subclass `Problem[...]` with the concrete design type. The repository test checks this.
- Implement `simulate_verbose`; the default `simulate` calls it and returns only `objective_values`.
- Define `dataset_id` and `container_id` as class attributes, even if `container_id = None`.
- Keep `Conditions` to benchmark scenario variables. Use `Config` for `Conditions` plus solver-only knobs such as `max_iter`, mesh resolution, tolerances, or filenames.
- Set `self.config` and `self.conditions` in `__init__` when constructor arguments can change defaults.
- Implement `reset`, even if it only calls `super().reset(seed=seed)`. The repository tests check that subclasses provide it directly.
- If `design_space` depends on `Config`, update both the class default and the instance value in `__init__`, as `beams2d` does.
- `random_design` should usually sample `optimal_design` from `problem.dataset["train"]`. For an early prototype without a dataset, sampling uniformly from `design_space` is fine, but that is not enough for a production EngiBench contribution.

## Dataset Requirements

Your hosted dataset should be a Hugging Face `DatasetDict` with these splits:

- `train`
- `test`
- `val`

The dataset columns should include:

- `optimal_design`, except for special cases that need a different documented design column;
- one column for every field in `Conditions`;
- one column for every objective name in `objectives`;
- optional `initial_design`, optimization histories, solver diagnostics, or metadata fields.

The objective column names must match exactly. If `objectives = (("mass_kg", ObjectiveDirection.MINIMIZE),)`, the dataset should have a `mass_kg` column.

For generative inverse design, each row should be one scenario and one reference design:

```text
optimal_design | load | length | mass_kg | optional_metadata...
```

Use deterministic scripts to generate the dataset. Keep the scripts, sampling policy, solver settings, and split logic in the problem folder or a clearly linked repository path.

## Constraint Rules

Use field constraints for scalar condition or config bounds:

```python
from typing import Annotated

from engibench.constraint import IMPL
from engibench.constraint import THEORY
from engibench.constraint import bounded

volfrac: Annotated[
    float,
    bounded(lower=0.0, upper=1.0).category(THEORY),
    bounded(lower=0.1, upper=0.9).warning().category(IMPL),
] = 0.35
```

Use `@constraint` functions for constraints involving the design, or involving multiple variables:

```python
import numpy.typing as npt

from engibench.constraint import constraint


@constraint
def volume_matches_condition(design: npt.NDArray, volfrac: float) -> None:
    assert abs(float(design.mean()) - volfrac) < 0.01, "design volume fraction does not match volfrac"
```

`check_constraints(design, config)` checks:

- constraints declared on `Config` fields;
- whether `design_space.contains(design)` is true;
- every function in `design_constraints`.

The keys needed by `design_constraints` must be present in the `config` dict passed to `check_constraints`. In the current EngiBench implementation, defaults from `Config` are used for field constraints, but they are not automatically merged into the argument dict used for `design_constraints`.

## Documentation Requirements

Add a problem page under `docs/problems/`, for example:

```text
docs/problems/my_problem.md
```

Use the EngiBench docs directives:

````md
# My Problem

``` {problem:table}
:lead: Your Name @github-handle
```

## Conditions

``` {problem:conditions}
```
````

Add a representative render image:

```text
docs/_static/img/problems/my_problem.png
```

Add the page to the Sphinx navigation. In the current EngiBench docs, direct problem pages are listed in `docs/index.md`. If your problem belongs to an existing family page, add it to that family page instead. Do not rely on `docs/problems/index.md`; that file is not present in the current repository layout.

## Validation Checklist

From the EngiBench repository root:

```sh
pip install -e ".[dev]"
```

For a narrower test install:

```sh
pip install -e ".[testing]"
```

If pytest reports that the `subtests` fixture is missing, install `pytest-subtests` as well. The full `.[dev]` install is the safer default when preparing a contribution.

Run discovery:

```sh
python - <<'PY'
from engibench.utils.all_problems import BUILTIN_PROBLEMS

assert "my_problem" in BUILTIN_PROBLEMS
print(BUILTIN_PROBLEMS["my_problem"])
PY
```

Run the implementation tests:

```sh
pytest tests/test_problem_implementations.py
```

If the simulation reference file for the new problem does not exist yet, create it intentionally:

```sh
CREATE_REF_FILES=missing pytest tests/test_problem_implementations.py
```

Then inspect and commit the generated reference JSON under `tests/reference/simulate/`.

Build the docs:

```sh
pip install -e ".[doc]"
cd docs
make dirhtml
```

For live preview:

```sh
cd docs
sphinx-autobuild -b dirhtml --watch ../engibench --re-ignore "pickle$" . _build
```

## Common Mistakes

- Copying Notebook 03 directly into `engibench/problems/` without adding `simulate_verbose`, `dataset_id`, `container_id`, and a hosted dataset.
- Putting solver knobs in `Conditions`. If it is not part of the design scenario that methods condition on, put it in `Config`.
- Returning a Python float from `simulate_verbose`. Return `SimulationResult(np.array([...]))`.
- Forgetting to implement `reset` in the subclass.
- Naming an objective in code but using a different dataset column name.
- Exporting multiple problem classes from a problem package `__init__.py`.
- Publishing only optimized designs without the conditions that produced them.
- Treating infeasible designs as high objective values instead of reporting independent constraint violations.
- Adding docs to `docs/problems/index.md`; update the current docs navigation instead.

## A Good First Milestone

For your own PhD problem, aim for this before trying to train a model:

1. One deterministic simulator call works on one hand-picked design.
2. `check_constraints` catches one invalid design and passes one valid design.
3. `render` produces a figure that another engineer can interpret.
4. `random_design` returns a design with the same type, shape, and dtype as `design_space`.
5. A tiny dataset with `train`, `test`, and `val` splits can be loaded through `problem.dataset`.
6. `pytest tests/test_problem_implementations.py` discovers the problem and fails only on known missing production artifacts.

Once those are true, the rest is scaling: more condition samples, stronger baselines, better documentation, and a cleaner dataset release.
