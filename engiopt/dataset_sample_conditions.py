"""Contains function for sampling conditions from the dataset.

Also formats for use in problem.optimize and problem.simulate
"""

from datasets import Dataset
from engibench.core import Problem
import numpy as np
import torch as th


def sample_conditions(
    problem: Problem,
    n_samples: int,
    device: th.device,
    seed: int,
    split: str = "test",
) -> tuple[th.Tensor, Dataset, np.ndarray, np.ndarray]:
    """Samples conditions and designs from the dataset and prepares tensors for the generator.

    Args:
    problem (Problem): The problem containing the dataset with conditions and designs.
    n_samples (int): Number of samples to draw.
    device (th.device): The device (e.g., 'cpu', 'mps', 'cuda') to place the tensors on.
    seed (int): Random seed for reproducibility.
    split (str): Dataset split to sample from. Must be one of "train", "val", or "test".

    Returns:
    conditions_tensor: A PyTorch tensor of sampled conditions, reshaped for the generator.
    sampled_conditions: A Hugging Face Dataset object of sampled conditions.
    sampled_designs_np: A NumPy array of sampled optimal designs.
    selected_indices: The indices of the sampled conditions and designs.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split '{split}'. Expected one of: train, val, test")

    # Set up held-out conditions.
    rng = np.random.default_rng(seed)

    # Extract the conditions
    dataset = problem.dataset[split]
    conditions_ds = dataset.select_columns(problem.conditions_keys)

    # Sample conditions and reference designs at random indices
    selected_indices = rng.choice(len(dataset), n_samples, replace=True)
    sampled_conditions = conditions_ds.select(selected_indices)
    sampled_designs_np = np.array(dataset["optimal_design"])[selected_indices]

    # Create tensor for conditions to be used in the generator
    conditions_tensor = th.tensor(list(sampled_conditions[:].values()), dtype=th.float32, device=device).T

    return conditions_tensor, sampled_conditions, sampled_designs_np, selected_indices
