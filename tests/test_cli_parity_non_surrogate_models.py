"""CLI parity checks across non-surrogate generative model scripts.

These tests guard the harmonized flag surfaces introduced for:
- overwrite-safe metric writing in evaluators
- local checkpoint path support in evaluators
- standardized checkpoint interval/path controls in trainers
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

EVAL_FILES = [
    REPO_ROOT / "engiopt/cgan_1d/evaluate_cgan_1d.py",
    REPO_ROOT / "engiopt/cgan_2d/evaluate_cgan_2d.py",
    REPO_ROOT / "engiopt/cgan_cnn_2d/evaluate_cgan_cnn_2d.py",
    REPO_ROOT / "engiopt/cgan_cnn_3d/evaluate_cgan_cnn_3d.py",
    REPO_ROOT / "engiopt/cgan_vae/evaluate_cgan_vae.py",
    REPO_ROOT / "engiopt/diffusion_1d/evaluate_diffusion_1d.py",
    REPO_ROOT / "engiopt/diffusion_2d_cond/evaluate_diffusion_2d_cond.py",
    REPO_ROOT / "engiopt/flow_matching_2d_cond/evaluate_flow_matching_2d_cond.py",
    REPO_ROOT / "engiopt/gan_1d/evaluate_gan_1d.py",
    REPO_ROOT / "engiopt/gan_2d/evaluate_gan_2d.py",
    REPO_ROOT / "engiopt/gan_bezier/evaluate_gan_bezier.py",
    REPO_ROOT / "engiopt/gan_cnn_2d/evaluate_gan_cnn_2d.py",
    REPO_ROOT / "engiopt/pixel_cnn_pp_2d/evaluate_pixel_cnn_pp_2d.py",
    REPO_ROOT / "engiopt/vqgan/evaluate_vqgan.py",
]

TRAIN_FILES = [
    REPO_ROOT / "engiopt/cgan_1d/cgan_1d.py",
    REPO_ROOT / "engiopt/cgan_2d/cgan_2d.py",
    REPO_ROOT / "engiopt/cgan_bezier/cgan_bezier.py",
    REPO_ROOT / "engiopt/cgan_cnn_2d/cgan_cnn_2d.py",
    REPO_ROOT / "engiopt/cgan_cnn_3d/cgan_cnn_3d.py",
    REPO_ROOT / "engiopt/cgan_vae/cgan_vae.py",
    REPO_ROOT / "engiopt/diffusion_1d/diffusion_1d.py",
    REPO_ROOT / "engiopt/diffusion_2d_cond/diffusion_2d_cond.py",
    REPO_ROOT / "engiopt/flow_matching_2d_cond/flow_matching_2d_cond.py",
    REPO_ROOT / "engiopt/gan_1d/gan_1d.py",
    REPO_ROOT / "engiopt/gan_2d/gan_2d.py",
    REPO_ROOT / "engiopt/gan_bezier/gan_bezier.py",
    REPO_ROOT / "engiopt/gan_cnn_2d/gan_cnn_2d.py",
    REPO_ROOT / "engiopt/pixel_cnn_pp_2d/pixel_cnn_pp_2d.py",
    REPO_ROOT / "engiopt/vqgan/vqgan.py",
]


def _args_fields(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Args":
            fields: set[str] = set()
            for class_item in node.body:
                if isinstance(class_item, ast.AnnAssign) and isinstance(class_item.target, ast.Name):
                    fields.add(class_item.target.id)
            return fields
    raise AssertionError(f"Could not find Args dataclass in {py_file}")


def _args_defaults(py_file: Path) -> dict[str, object | None]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Args":
            defaults: dict[str, object | None] = {}
            for class_item in node.body:
                if (
                    isinstance(class_item, ast.AnnAssign)
                    and isinstance(class_item.target, ast.Name)
                    and isinstance(class_item.value, ast.Constant)
                ):
                    defaults[class_item.target.id] = class_item.value.value
            return defaults
    raise AssertionError(f"Could not find Args dataclass in {py_file}")


def _has_write_metrics_csv_call(py_file: Path) -> bool:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "write_metrics_csv":
            return True
    return False


def test_eval_cli_surface_non_surrogate_models() -> None:
    required_eval_fields = {
        "problem_id",
        "seed",
        "wandb_project",
        "wandb_entity",
        "n_samples",
        "sigma",
        "output_csv",
        "append_output",
        "checkpoint_path",
    }

    for py_file in EVAL_FILES:
        fields = _args_fields(py_file)
        missing = sorted(required_eval_fields - fields)
        assert not missing, f"{py_file} missing required eval fields: {missing}"

        defaults = _args_defaults(py_file)
        assert defaults.get("append_output") is True, f"{py_file} append_output should default to True"
        assert defaults.get("checkpoint_path") is None, f"{py_file} checkpoint_path should default to None"


def test_eval_uses_shared_metrics_writer() -> None:
    for py_file in EVAL_FILES:
        assert _has_write_metrics_csv_call(py_file), f"{py_file} must use write_metrics_csv"


def test_train_cli_surface_non_surrogate_models() -> None:
    shared_train_fields = {
        "problem_id",
        "algo",
        "track",
        "wandb_project",
        "wandb_entity",
        "seed",
        "save_model",
        "checkpoint_dir",
        "checkpoint_interval_epochs",
    }

    for py_file in TRAIN_FILES:
        fields = _args_fields(py_file)
        missing = sorted(shared_train_fields - fields)
        assert not missing, f"{py_file} missing shared train fields: {missing}"


def test_train_checkpoint_path_fields_present_by_family() -> None:
    expected_checkpoint_fields = {
        "engiopt/cgan_1d/cgan_1d.py": {"generator_checkpoint_path", "discriminator_checkpoint_path"},
        "engiopt/cgan_2d/cgan_2d.py": {"generator_checkpoint_path", "discriminator_checkpoint_path"},
        "engiopt/cgan_bezier/cgan_bezier.py": {"generator_checkpoint_path", "discriminator_checkpoint_path"},
        "engiopt/cgan_cnn_2d/cgan_cnn_2d.py": {"generator_checkpoint_path", "discriminator_checkpoint_path"},
        "engiopt/cgan_cnn_3d/cgan_cnn_3d.py": {"generator_checkpoint_path", "discriminator_checkpoint_path"},
        "engiopt/cgan_vae/cgan_vae.py": {"checkpoint_path"},
        "engiopt/diffusion_1d/diffusion_1d.py": {"checkpoint_path"},
        "engiopt/diffusion_2d_cond/diffusion_2d_cond.py": {"checkpoint_path"},
        "engiopt/flow_matching_2d_cond/flow_matching_2d_cond.py": {"checkpoint_path"},
        "engiopt/gan_1d/gan_1d.py": {"generator_checkpoint_path", "discriminator_checkpoint_path"},
        "engiopt/gan_2d/gan_2d.py": {"generator_checkpoint_path", "discriminator_checkpoint_path"},
        "engiopt/gan_bezier/gan_bezier.py": {"generator_checkpoint_path", "discriminator_checkpoint_path"},
        "engiopt/gan_cnn_2d/gan_cnn_2d.py": {"generator_checkpoint_path", "discriminator_checkpoint_path"},
        "engiopt/pixel_cnn_pp_2d/pixel_cnn_pp_2d.py": {"checkpoint_path"},
        "engiopt/vqgan/vqgan.py": {
            "cvqgan_checkpoint_path",
            "vqgan_checkpoint_path",
            "discriminator_checkpoint_path",
            "transformer_checkpoint_path",
        },
    }

    for rel_path, checkpoint_fields in expected_checkpoint_fields.items():
        py_file = REPO_ROOT / rel_path
        fields = _args_fields(py_file)
        missing = sorted(checkpoint_fields - fields)
        assert not missing, f"{py_file} missing checkpoint path fields: {missing}"
