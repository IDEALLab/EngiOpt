"""The CLI must not carry its own default spec version.

The library default moved to v2 while `evaluate.py` still built
`f"{problem_id}/v1"`, so a bare `python -m engiopt.evaluate` selected a spec
whose metric list no longer matched the registry (`KeyError: 'novelty'`).
The default lives in `EvalSpec.load` alone; the CLI passes `--spec` through.
"""

from __future__ import annotations

from typing import Any

import pytest

from engiopt import evaluate
from engiopt.evaluation.spec import EvalSpec


class _StopAtSpecError(Exception):
    """Raised by the fake to stop `main` before it loads anything real."""


def test_the_cli_without_a_spec_defers_to_the_library_default(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_for_problem(problem_id: str, *, spec: Any = None, **_: Any) -> Any:
        seen["spec"] = spec
        raise _StopAtSpecError

    monkeypatch.setattr(evaluate.Evaluator, "for_problem", fake_for_problem)
    with pytest.raises(_StopAtSpecError):
        evaluate.main(evaluate.Args())
    assert seen["spec"] is None, "no --spec means the library default, not a version the CLI invents"


def test_a_bare_problem_reference_loads_the_current_spec_version() -> None:
    """`EvalSpec.load("beams2d")` is what the CLI's no-flag path now resolves."""
    assert EvalSpec.load("beams2d").version == EvalSpec.version, "a bare reference follows the dataclass default"
