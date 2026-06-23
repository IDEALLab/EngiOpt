from __future__ import annotations

import pytest

from engiopt.timing import benchmark_callable
from engiopt.timing import summarize_durations


def test_summarize_durations_is_deterministic() -> None:
    summary = summarize_durations([0.3, 0.1, 0.2])

    assert summary.total_runtime_sec == pytest.approx(0.6)
    assert summary.mean_runtime_sec == pytest.approx(0.2)
    assert summary.median_runtime_sec == pytest.approx(0.2)
    assert summary.min_runtime_sec == pytest.approx(0.1)
    assert summary.max_runtime_sec == pytest.approx(0.3)
    assert summary.timed_repeats == 3


def test_benchmark_callable_runs_warmups_and_timed_repeats() -> None:
    calls = {"count": 0}

    def fn() -> None:
        calls["count"] += 1

    summary = benchmark_callable(fn, device="cpu", warmup_repeats=2, timed_repeats=3)

    assert calls["count"] == 5
    assert summary.timed_repeats == 3
    assert summary.total_runtime_sec >= 0.0


def test_benchmark_callable_rejects_invalid_repeat_counts() -> None:
    with pytest.raises(ValueError, match="warmup_repeats"):
        benchmark_callable(lambda: None, device="cpu", warmup_repeats=-1, timed_repeats=1)

    with pytest.raises(ValueError, match="timed_repeats"):
        benchmark_callable(lambda: None, device="cpu", warmup_repeats=0, timed_repeats=0)
