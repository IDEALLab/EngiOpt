"""CLI parity checks for the three 2D baseline families.

This test keeps the shared train/eval interfaces aligned across:
- flow_matching_2d_cond
- diffusion_2d_cond
- cgan_cnn_2d
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _args_fields(py_file: Path) -> set[str]:
    """Return dataclass field names from class Args without importing modules."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Args":
            fields: set[str] = set()
            for class_item in node.body:
                if isinstance(class_item, ast.AnnAssign) and isinstance(class_item.target, ast.Name):
                    fields.add(class_item.target.id)
            return fields
    raise AssertionError(f"Could not find Args dataclass in {py_file}")


def test_eval_cli_parity_for_2d_models() -> None:
    """Ensure shared evaluation flags remain aligned across model families."""
    eval_files = [
        REPO_ROOT / "engiopt/flow_matching_2d_cond/evaluate_flow_matching_2d_cond.py",
        REPO_ROOT / "engiopt/diffusion_2d_cond/evaluate_diffusion_2d_cond.py",
        REPO_ROOT / "engiopt/cgan_cnn_2d/evaluate_cgan_cnn_2d.py",
    ]
    eval_field_sets = [_args_fields(path) for path in eval_files]

    required_shared_fields = {
        "problem_id",
        "seed",
        "wandb_project",
        "wandb_entity",
        "n_samples",
        "sigma",
        "output_csv",
        "append_output",
        "checkpoint_path",
        "checkpoint_source",
        "checkpoint_package_label",
        "hf_entity",
        "hf_repo_prefix",
    }

    for path, fields in zip(eval_files, eval_field_sets, strict=True):
        missing = sorted(required_shared_fields - fields)
        assert not missing, f"{path} missing shared eval fields: {missing}"


def test_train_cli_parity_for_2d_models() -> None:
    """Ensure shared training flags remain aligned across model families."""
    train_files = [
        REPO_ROOT / "engiopt/flow_matching_2d_cond/flow_matching_2d_cond.py",
        REPO_ROOT / "engiopt/diffusion_2d_cond/diffusion_2d_cond.py",
        REPO_ROOT / "engiopt/cgan_cnn_2d/cgan_cnn_2d.py",
    ]
    train_field_sets = [_args_fields(path) for path in train_files]

    required_shared_fields = {
        "problem_id",
        "algo",
        "track",
        "wandb_project",
        "wandb_entity",
        "seed",
        "save_model",
        "checkpoint_backend",
        "upload_top_k_checkpoints",
        "include_final_in_top_k_bundle",
        "hf_entity",
        "hf_repo_prefix",
        "hf_private",
        "checkpoint_package_label",
        "n_epochs",
        "batch_size",
        "n_cpu",
    }

    for path, fields in zip(train_files, train_field_sets, strict=True):
        missing = sorted(required_shared_fields - fields)
        assert not missing, f"{path} missing shared train fields: {missing}"
