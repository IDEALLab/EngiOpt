from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as th

import engiopt.selected_checkpoint_bundle as selected_bundle
from engiopt.selected_checkpoint_bundle import archive_selected_checkpoint_bundle
from engiopt.selected_checkpoint_bundle import SelectedCheckpointSpec
from engiopt.selected_checkpoint_bundle import stage_selected_checkpoint_bundle


def _selection_rows(checkpoint_name: str) -> list[dict[str, object]]:
    return [
        {
            "rank": 1,
            "epoch": 100,
            "checkpoint_path": checkpoint_name,
            "validation_mmd": 0.1,
            "validation_cog": 2.0,
            "validation_fog": 0.3,
            "validation_eval_mmd": 0.2,
            "validation_generation_seed": 1002,
        },
        {
            "rank": 2,
            "epoch": 120,
            "checkpoint_path": checkpoint_name,
            "validation_mmd": 0.2,
            "validation_cog": 1.0,
            "validation_fog": 0.2,
            "validation_eval_mmd": 0.3,
            "validation_generation_seed": 1003,
        },
    ]


def test_selected_cgan_bundle_contains_generator_and_selection_evidence(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "generator_epoch_0120.pth"
    th.save(
        {
            "args": {"generator_output_activation": "sigmoid", "latent_dim": 32},
            "generator": {"weight": th.tensor([1.0])},
            "optimizer_generator": {"large": "state"},
            "epoch": 119,
            "training_wandb": {"wandb_run_id": "train123"},
        },
        checkpoint_path,
    )
    metrics_path = tmp_path / "validation_metrics.json"
    metrics_path.write_text(json.dumps({"top_k_epochs": []}), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_save_checkpoint_package(**kwargs):
        captured.update(kwargs)
        staged = th.load(kwargs["checkpoint_files"]["generator.pth"], map_location="cpu")
        assert set(staged) == {"args", "epoch", "generator", "training_wandb"}
        assert "optimizer_generator" not in staged
        selection = json.loads(Path(kwargs["checkpoint_files"]["selection_results.json"]).read_text())
        assert selection["selected_rank"] == 2
        assert [row["selected"] for row in selection["candidates"]] == [False, True]
        return {
            "checkpoint_backend": "hf",
            "hf_repo_id": "IDEALLab/test",
            "hf_package_path": "beams2d/selected/release/activation_sigmoid/seed_1",
            "hf_model_ref": "hf://IDEALLab/test/beams2d/selected/release/activation_sigmoid/seed_1",
            "hf_revision": "abc123",
        }

    monkeypatch.setattr(selected_bundle, "save_checkpoint_package", fake_save_checkpoint_package)
    info = archive_selected_checkpoint_bundle(
        spec=SelectedCheckpointSpec(
            model_id="cgan_cnn_2d",
            problem_id="beams2d",
            seed=1,
            checkpoint_path=checkpoint_path,
            validation_metrics_path=metrics_path,
            selection_rows=_selection_rows(checkpoint_path.name),
            selected_rank=2,
            selected_epoch=120,
            top_k=2,
            selection_batch_size=50,
            selection_seed=124,
            test_seed=1,
            test_generation_seed=2001,
            release="release",
        ),
        checkpoint_backend="hf",
        hf_entity="IDEALLab",
        hf_repo_prefix="test",
        hf_private=False,
    )

    assert captured["extra_path_parts"] == ["selected", "release", "activation_sigmoid"]
    assert captured["upload_run_copy"] is False
    assert captured["primary_files"] == ["generator.pth"]
    assert set(captured["checkpoint_files"]) == {
        "generator.pth",
        "validation_metrics.json",
        "selection_results.json",
        "selection_results.csv",
    }
    assert len(info["selected_checkpoint_sha256"]) == 64


def test_selected_bundle_requires_exactly_one_selected_candidate(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "epoch_0100.pth"
    th.save({"args": {}, "model": {}, "epoch": 99}, checkpoint_path)
    metrics_path = tmp_path / "validation_metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(selected_bundle, "save_checkpoint_package", lambda **kwargs: kwargs)

    with pytest.raises(ValueError, match="exactly one"):
        archive_selected_checkpoint_bundle(
            spec=SelectedCheckpointSpec(
                model_id="flow_matching_2d_cond",
                problem_id="beams2d",
                seed=1,
                checkpoint_path=checkpoint_path,
                validation_metrics_path=metrics_path,
                selection_rows=_selection_rows(checkpoint_path.name),
                selected_rank=3,
                selected_epoch=100,
                top_k=2,
                selection_batch_size=50,
                selection_seed=124,
                test_seed=1,
                test_generation_seed=2001,
                release="release",
            ),
            checkpoint_backend="hf",
            hf_entity="IDEALLab",
            hf_repo_prefix="test",
            hf_private=False,
        )


def test_staged_bundle_is_upload_ready_and_protects_published_release(tmp_path):
    checkpoint_path = tmp_path / "epoch_0100.pth"
    th.save({"args": {"method": "euler", "integration_steps": 16}, "model": {}, "epoch": 99}, checkpoint_path)
    metrics_path = tmp_path / "validation_metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")
    spec = SelectedCheckpointSpec(
        model_id="flow_matching_2d_cond",
        problem_id="beams2d",
        seed=1,
        checkpoint_path=checkpoint_path,
        validation_metrics_path=metrics_path,
        selection_rows=_selection_rows(checkpoint_path.name),
        selected_rank=2,
        selected_epoch=100,
        top_k=2,
        selection_batch_size=50,
        selection_seed=124,
        test_seed=1,
        test_generation_seed=2001,
        release="release",
    )

    info = stage_selected_checkpoint_bundle(spec=spec, staging_root=tmp_path / "staging")
    bundle_dir = Path(info["selected_checkpoint_staging_dir"])
    metadata = json.loads((bundle_dir / "metadata.json").read_text())
    assert metadata["package_label"] == "euler_16"
    assert (bundle_dir / "model.pth").exists()
    assert (bundle_dir / "selection_results.csv").exists()

    (bundle_dir / "upload_receipt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="new release label"):
        stage_selected_checkpoint_bundle(spec=spec, staging_root=tmp_path / "staging")
