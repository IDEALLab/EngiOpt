"""Contains function for sampling conditions from the dataset.

Also formats for use in problem.optimize and problem.simulate
"""

from datasets import Dataset
from engibench.core import Problem
import numpy as np
import torch as th

from engiopt.transforms import get_scalar_condition_keys


def sample_conditions(
    problem: Problem, n_samples: int, device: th.device, seed: int
) -> tuple[th.Tensor, Dataset, np.ndarray, np.ndarray]:
    """Samples conditions and designs from the dataset and prepares tensors for the generator.

    The two outputs serve different consumers and deliberately differ:

    - `sampled_conditions` keeps every condition column the dataset provides,
      including array-valued ones, because that is what the simulator and
      optimizer need to reproduce the scenario.
    - `conditions_tensor` keeps only the scalar conditions, because that is all a
      generator can take as a dense `(n_samples, n_conds)` input. Conditions that
      are solver settings rather than dataset columns (photonics2d) or that are
      images rather than scalars (thermoelastic2d) are excluded; see
      `engiopt.transforms.get_scalar_condition_keys`.

    Args:
    problem (Problem): The problem containing the dataset with conditions and designs.
    n_samples (int): Number of samples to draw.
    device (th.device): The device (e.g. 'cpu', 'mps', 'cuda') to place the tensors on.
    seed (int): Random seed for reproducibility.

    Returns:
    conditions_tensor: Scalar conditions as a (n_samples, n_scalar_conds) tensor.
    sampled_conditions: A Hugging Face Dataset of every available condition column.
    sampled_designs_np: A NumPy array of sampled optimal designs.
    selected_indices: The indices of the sampled conditions and designs.
    """
    ### Set up testing conditions ###
    rng = np.random.default_rng(seed)

    # Extract the conditions the dataset actually carries; the simulator wants all
    # of them, the generator only the scalar ones.
    dataset = problem.dataset["test"]
    available_keys = [key for key in problem.conditions_keys if key in dataset.column_names]
    scalar_keys = get_scalar_condition_keys(problem, dataset)
    conditions_ds = dataset.select_columns(available_keys)

    # Sample conditions and test_ds designs at random indices
    selected_indices = rng.choice(len(dataset), n_samples, replace=True)
    sampled_conditions = conditions_ds.select(selected_indices)
    sampled_designs_np = np.array(dataset["optimal_design"])[selected_indices]

    # Create tensor for conditions to be used in the generator
    conditions_tensor = th.tensor([sampled_conditions[key] for key in scalar_keys], dtype=th.float32, device=device).T

    return conditions_tensor, sampled_conditions, sampled_designs_np, selected_indices
