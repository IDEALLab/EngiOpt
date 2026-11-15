"""The metric declaration contract: one decorator, one direction, one sentence."""

from __future__ import annotations

import pytest

from engiopt.evaluation.registry import MetricRegistry
from engiopt.evaluation.registry import METRICS
from engiopt.evaluation.registry import register_metric


@pytest.fixture
def registry() -> MetricRegistry:
    """A registry of its own, so a test never publishes into the global one."""
    return MetricRegistry()


def test_a_custom_registry_is_not_bypassed_when_empty(registry: MetricRegistry) -> None:
    """An empty registry is falsy, which `registry or METRICS` silently ignored."""
    before = len(METRICS)

    @register_metric("solo", family="diversity", cost="cheap", higher_is_better=True, registry=registry)
    def solo(ctx: object) -> float:
        """A metric registered into a registry of its own."""
        return 0.0

    assert "solo" in registry
    assert len(METRICS) == before, "registration leaked into the global registry"


def test_the_docstring_is_the_question(registry: MetricRegistry) -> None:
    """No separate text to maintain: the first docstring line is what readers see."""

    @register_metric("share", family="feasibility", cost="cheap", higher_is_better=False, registry=registry)
    def share(ctx: object) -> float:
        """What fraction of the designs are infeasible?

        Longer explanation lives below the first line.
        """
        return 0.0

    assert registry["share"].description == "What fraction of the designs are infeasible?"
    assert registry.explain().loc["share", "direction"] == "lower is better"


def test_a_diagnostic_declares_no_direction(registry: MetricRegistry) -> None:
    @register_metric("diag", family="conditions", cost="cheap", higher_is_better=None, registry=registry)
    def diag(ctx: object) -> float:
        """A metric read beside the others, never ranked on."""
        return 0.0

    assert registry["diag"].direction == "diagnostic"


def test_two_metrics_cannot_claim_one_column(registry: MetricRegistry) -> None:
    register_metric(
        "first", family="diversity", cost="cheap", higher_is_better=True, outputs=("shared",), registry=registry
    )(lambda ctx: {"shared": 1.0})
    with pytest.raises(ValueError, match="already emitted"):
        register_metric(
            "second", family="diversity", cost="cheap", higher_is_better=True, outputs=("shared",), registry=registry
        )(lambda ctx: {"shared": 2.0})
