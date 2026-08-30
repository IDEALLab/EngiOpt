"""Tests for the public conditional flow-matching campaign manifest."""

from __future__ import annotations

import importlib.util
from pathlib import Path

EXPECTED_EXPERIMENTS = 220
RTX_4090_SEEDS = {1, 2, 3}
FIRST_RTX_3090_SEED = 4

MODULE_PATH = Path(__file__).parents[1] / "reproducibility" / "conditional_flow_matching" / "campaign.py"
SPEC = importlib.util.spec_from_file_location("conditional_flow_matching_campaign", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


def test_full_campaign_has_expected_unique_mapping() -> None:
    experiments = CAMPAIGN.full_experiments(RTX_4090_SEEDS)

    assert len(experiments) == EXPECTED_EXPERIMENTS
    assert len({experiment["key"] for experiment in experiments}) == EXPECTED_EXPERIMENTS
    assert len({experiment["train_task_id"] for experiment in experiments}) == EXPECTED_EXPERIMENTS
    assert len({experiment["eval_task_id"] for experiment in experiments}) == EXPECTED_EXPERIMENTS


def test_gpu_assignment_is_seed_based() -> None:
    experiments = CAMPAIGN.full_experiments(RTX_4090_SEEDS)

    assert {experiment["gpu"] for experiment in experiments if experiment["seed"] in RTX_4090_SEEDS} == {"rtx_4090"}
    assert {experiment["gpu"] for experiment in experiments if experiment["seed"] >= FIRST_RTX_3090_SEED} == {"rtx_3090"}


def test_smoke_exercises_all_models() -> None:
    experiments = CAMPAIGN.smoke_experiments()

    assert {experiment["model_id"] for experiment in experiments} == {
        "flow_matching_2d_cond",
        "diffusion_2d_cond",
        "cgan_cnn_2d",
    }
    assert {experiment["problem_id"] for experiment in experiments} == {"beams2d"}
    assert {experiment["seed"] for experiment in experiments} == {1}
