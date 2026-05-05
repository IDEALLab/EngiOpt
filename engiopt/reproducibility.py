"""Utilities for reproducible training runs."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_training(seed: int) -> np.random.Generator:
    """Seed Python, NumPy, and PyTorch RNGs for training scripts."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return rng


def enable_strict_determinism(*, warn_only: bool = True) -> None:
    """Enable stricter deterministic settings without changing model logic."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(mode=True, warn_only=warn_only)


def make_dataloader_generator(seed: int) -> torch.Generator:
    """Return a seeded generator for deterministic DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
