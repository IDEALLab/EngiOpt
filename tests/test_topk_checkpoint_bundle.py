from __future__ import annotations

import json

import pytest
import torch as th

from engiopt.topk_checkpoint_bundle import collect_topk_checkpoint_files
from engiopt.topk_checkpoint_bundle import _eval_only_checkpoint
from engiopt.topk_checkpoint_bundle import infer_checkpoint_package_label
from engiopt.topk_checkpoint_bundle import TopKBundleSpec


def _write_metrics(path, epochs):
    payload = {
        "top_k_epochs": [{"epoch": epoch, "metric_value": value} for epoch, value in epochs],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collects_diffusion_topk_files_in_metric_order(tmp_path):
    _write_metrics(tmp_path / "validation_metrics.json", [(9, 0.2), (19, 0.1), (29, 0.3)])
    th.save({"args": {"num_timesteps": 1000}}, tmp_path / "epoch_0010.pth")
    th.save({"args": {"num_timesteps": 1000}}, tmp_path / "epoch_0020.pth")
    th.save({"args": {"num_timesteps": 1000}}, tmp_path / "epoch_0030.pth")
    th.save({}, tmp_path / "final_model.pth")

    files, run_config = collect_topk_checkpoint_files(
        TopKBundleSpec(
            model_id="diffusion_2d_cond",
            problem_id="beams2d",
            seed=1,
            checkpoint_dir=tmp_path,
            top_k=2,
        )
    )

    assert set(files) == {"validation_metrics.json", "epoch_0020.pth", "epoch_0010.pth", "final_model.pth"}
    assert run_config["num_timesteps"] == 1000


def test_collects_cgan_generator_and_discriminator_checkpoints(tmp_path):
    _write_metrics(tmp_path / "validation_metrics.json", [(0, 0.2)])
    th.save({"args": {"generator_output_activation": "sigmoid"}}, tmp_path / "generator_epoch_0001.pth")
    th.save({}, tmp_path / "discriminator_epoch_0001.pth")

    files, run_config = collect_topk_checkpoint_files(
        TopKBundleSpec(
            model_id="cgan_cnn_2d",
            problem_id="heatconduction2d",
            seed=2,
            checkpoint_dir=tmp_path,
        )
    )

    assert "generator_epoch_0001.pth" in files
    assert "discriminator_epoch_0001.pth" in files
    assert run_config["generator_output_activation"] == "sigmoid"


def test_missing_referenced_checkpoint_fails_loudly(tmp_path):
    _write_metrics(tmp_path / "validation_metrics.json", [(4, 0.1)])

    with pytest.raises(FileNotFoundError, match="epoch_0005.pth"):
        collect_topk_checkpoint_files(
            TopKBundleSpec(
                model_id="flow_matching_2d_cond",
                problem_id="beams2d",
                seed=1,
                checkpoint_dir=tmp_path,
            )
        )


def test_infers_protocol_labels():
    assert infer_checkpoint_package_label("flow_matching_2d_cond", {"method": "euler", "integration_steps": 16}) == "euler_16"
    assert infer_checkpoint_package_label("diffusion_2d_cond", {"num_timesteps": 1000}) == "timesteps_1000"
    assert infer_checkpoint_package_label("cgan_cnn_2d", {"generator_output_activation": "sigmoid"}) == "activation_sigmoid"


def test_eval_only_checkpoint_strips_optimizer_state():
    checkpoint = {
        "args": {"layers_per_block": 2},
        "model": {"weight": th.tensor([1.0])},
        "model_config": {"num_timesteps": 1000},
        "design_min": th.tensor(0.0),
        "design_max": th.tensor(1.0),
        "optimizer": {"state": "large"},
        "optimizer_generator": {"state": "large"},
    }

    eval_checkpoint = _eval_only_checkpoint(checkpoint, "flow_matching_2d_cond")

    assert "model" in eval_checkpoint
    assert "args" in eval_checkpoint
    assert "model_config" in eval_checkpoint
    assert "design_min" in eval_checkpoint
    assert "design_max" in eval_checkpoint
    assert "optimizer" not in eval_checkpoint
    assert "optimizer_generator" not in eval_checkpoint
