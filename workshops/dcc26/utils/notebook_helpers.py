"""Helper utilities for DCC26 workshop notebooks."""

from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np
import torch as th


def set_global_seed(seed: int) -> None:
    """Set seeds for reproducibility across numpy, python, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def pick_device() -> th.device:
    """Pick an available torch device in priority order."""
    if th.backends.mps.is_available():
        return th.device("mps")
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def ensure_dir(path: str) -> str:
    """Ensure a directory exists and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: Any, path: str) -> None:
    """Save Python data as a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    """Load Python data from a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
