"""Registry of all generators in EngiOpt.

Mirrors `engibench.utils.all_problems`, so the two halves of the stack read the
same way::

    problem = BUILTIN_PROBLEMS["beams2d"]()
    generator = BUILTIN_GENERATORS["cgan_cnn_2d"].from_pretrained(problem, problem_id="beams2d", seed=1)
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, TYPE_CHECKING

from gymnasium import spaces

from engiopt.core import Generator
import engiopt.generators

if TYPE_CHECKING:
    from engibench.core import Problem

ADAPTER_MODULE = "adapter"
"""Each generator package implements the contract in this module."""


def list_generators(base_module: Any = engiopt.generators) -> dict[str, type[Generator]]:
    """Return a dict containing all available `Generator` classes defined in submodules of `base_module`."""
    module_path = next(iter(base_module.__path__))
    modules = pkgutil.iter_modules(path=[module_path], prefix="")
    return {
        m.name: extract_generator(importlib.import_module(f"{base_module.__package__}.{m.name}.{ADAPTER_MODULE}"))
        for m in modules
        if m.ispkg and not m.name.startswith("_")
    }


def extract_generator(module: Any) -> type[Generator]:
    """Get a `Generator` class defined in a module.

    Raises an exception if the module contains multiple `Generator` classes.
    """
    generator_types = [
        o
        for o in vars(module).values()
        if isinstance(o, type) and issubclass(o, Generator) and o is not Generator and not _is_abstract(o)
    ]
    try:
        (g,) = generator_types
    except ValueError:
        msg = f"Only one generator per adapter is allowed. Got {', '.join(t.__name__ for t in generator_types)}"
        raise ValueError(msg) from None
    return g


def _is_abstract(obj: type) -> bool:
    """Whether a class still has unimplemented abstract methods."""
    return bool(getattr(obj, "__abstractmethods__", frozenset()))


BUILTIN_GENERATORS = list_generators()


def design_kind_of(problem: Problem) -> str:
    """Classify a problem's design space as `1d`, `2d`, `3d`, or `dict`.

    Used to reject nonsensical pairings, such as a 2D CNN generator pointed at a
    3D problem.
    """
    if isinstance(problem.design_space, spaces.Dict):
        return "dict"
    ndim = len(problem.design_space.shape or ())
    return {1: "1d", 2: "2d", 3: "3d"}.get(ndim, f"{ndim}d")


def generators_for(problem: Problem) -> dict[str, type[Generator]]:
    """Return the generators whose `design_kinds` cover this problem's design space."""
    kind = design_kind_of(problem)
    return {name: g for name, g in BUILTIN_GENERATORS.items() if kind in g.design_kinds}


def problem_id_of(problem: Problem | type[Problem]) -> str:
    """Reverse-look-up a problem's registry key.

    EngiBench problems do not carry their own id, but checkpoint paths and
    leaderboard rows are keyed by it.

    Raises:
        ValueError: If the problem is not a built-in EngiBench problem.
    """
    from engibench.utils.all_problems import BUILTIN_PROBLEMS

    problem_cls = problem if isinstance(problem, type) else type(problem)
    for name, candidate in BUILTIN_PROBLEMS.items():
        if candidate is problem_cls:
            return name
    msg = f"{problem_cls.__name__} is not a built-in EngiBench problem; pass problem_id explicitly."
    raise ValueError(msg)
