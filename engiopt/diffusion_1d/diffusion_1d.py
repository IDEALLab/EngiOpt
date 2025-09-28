"""Diffusion 1D algorithm for vector-based problems.

We are using the implementation from https://github.com/lucidrains/denoising-diffusion-pytorch.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import time
from typing import TYPE_CHECKING

from denoising_diffusion_pytorch import GaussianDiffusion1D
from denoising_diffusion_pytorch import Unet1D
from engibench.utils.all_problems import BUILTIN_PROBLEMS
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torchvision import transforms
import tqdm
import tyro

from engiopt.transforms import flatten_dict_factory
import wandb

if TYPE_CHECKING:
    from engibench.utils.problem import Problem


class Normalizer:
    """Normalizes or denormalizes the input tensor."""

    def __init__(self, min_val: th.Tensor, max_val: th.Tensor, eps: float = 1e-7):
        self.eps = eps
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, x: th.Tensor) -> th.Tensor:
        """Normalizes the input tensor."""
        return (x - self.min_val) / (self.max_val - self.min_val + self.eps)

    def denormalize(self, x: th.Tensor) -> th.Tensor:
        """Denormalizes the input tensor."""
        return x * (self.max_val - self.min_val + self.eps) + self.min_val


def prepare_data(problem: Problem, padding_size: int, device: th.device) -> tuple[th.utils.data.TensorDataset, Normalizer]:
    """Prepares the data for the generator and discriminator.

    Args:
        problem (Problem): The problem to prepare the data for.
        padding_size (int): The size of padding to add to the data.
        device (th.device): The device to prepare the data on.

    Returns:
        tuple[th.utils.data.TensorDataset, Normalizer]: The training dataset, and design normalizer.
    """
    training_ds = problem.dataset.with_format("torch", device=device)["train"]

    # Flatten the designs if they are a Dict
    if isinstance(problem.design_space, spaces.Box):
        transform = transforms.Lambda(lambda x: x.flatten(1))
    elif isinstance(problem.design_space, spaces.Dict):
        transform = flatten_dict_factory(problem, device)

    # Add padding to the transformed data
    transformed_data = transform(training_ds["optimal_design"][:])
    if padding_size > 0:
        padded_data = th.nn.functional.pad(transformed_data, (0, padding_size), mode="constant", value=0)
    else:
        padded_data = transformed_data
    training_ds = th.utils.data.TensorDataset(padded_data)

    # Create design normalizer
    design_tensors = training_ds.tensors[0].T
    design_min = design_tensors.amin(dim=tuple(range(1, design_tensors.ndim))).to(device)
    design_max = design_tensors.amax(dim=tuple(range(1, design_tensors.ndim))).to(device)
    design_normalizer = Normalizer(design_min, design_max)

    return training_ds, design_normalizer


@dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "airfoil"
    """Problem identifier."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this algorithm."""

    # Tracking
    track: bool = True
    """Track the experiment with wandb."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name."""
    seed: int = 1
    """Random seed."""
    save_model: bool = False
    """Saves the model to disk."""

    # Algorithm specific
    n_epochs: int = 200
    """number of epochs of training"""
    batch_size: int = 64
    """size of the batches"""
    lr: float = 3e-4
    """learning rate"""
    b1: float = 0.5
    """decay of first order momentum of gradient"""
    b2: float = 0.999
    """decay of first order momentum of gradient"""
    sample_interval: int = 400
    """interval between image samples"""
    auto_norm: bool = True
    """Automatically normalize the data when learning."""
    unet_dim: int = 32
    """Dimensions for the UNET1D"""
    n_channels: int = 1
    """number of input channels for the model"""


if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    if not isinstance(problem.design_space, (spaces.Box, spaces.Dict)):
        raise ValueError("This algorithm only works with Box or Dict spaces.")

    if isinstance(problem.design_space, spaces.Box):
        design_shape = problem.design_space.shape
    else:
        dummy_design, _ = problem.random_design()
        flattened = spaces.flatten(problem.design_space, dummy_design)
        design_shape = np.array(flattened).shape

    # Add padding for the UNet (1D requires the input to be divisible by 8)
    padding_size = (8 - design_shape[0] % 8) % 8  # Only pad if needed
    padded_size = design_shape[0] + padding_size
    if padding_size > 0:
        print(f"Padding design from {design_shape[0]} to {padded_size} dimensions")
    design_shape = (padded_size,)

    # Logging
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), save_code=True, name=run_name)

    # Seeding
    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True

    os.makedirs("images", exist_ok=True)

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    # ----------
    #  Models
    # ----------
    model = Unet1D(
        dim=args.unet_dim,  # Used for the sinusoidal positional embeddings
        channels=args.n_channels,  # Number of channels in the input
    ).to(device)

    diffusion = GaussianDiffusion1D(
        model,
        seq_length=np.prod(design_shape),
        auto_normalize=True,
    ).to(device)

    # Configure data loader
    training_ds, design_normalizer = prepare_data(problem, padding_size, device)

    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Optimizers
    optimizer = th.optim.Adam(diffusion.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # ----------
    #  Training
    #  Note that it could probably be faster using accelerate as in https://github.com/lucidrains/denoising-diffusion-pytorch
    # ----------
    for epoch in tqdm.trange(args.n_epochs):
        for i, data in enumerate(dataloader):
            designs = data[0]

            designs_flat = designs.view(designs.size(0), 1, -1)  # flattens designs to a batch of 1D tensors with 1 channel
            # Normalize the designs
            designs = design_normalizer.normalize(designs)

            # Learning
            optimizer.zero_grad()
            loss = diffusion(designs_flat)
            loss.backward()
            optimizer.step()

            # ----------
            #  Logging
            # ----------
            if args.track:
                batches_done = epoch * len(dataloader) + i
                wandb.log({"loss": loss.item(), "epoch": epoch, "batch": batches_done})
                print(f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] [loss: {loss.item()}]")

                # This saves a grid image of 25 generated designs every sample_interval
                if batches_done % args.sample_interval == 0:
                    # Extract 25 designs
                    designs = diffusion.sample(batch_size=25)

                    if designs.dim() == 3:  # noqa: PLR2004
                        designs = designs.squeeze(1)
                    # Denormalize the designs before rendering
                    designs = design_normalizer.denormalize(designs)

                    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

                    # Flatten axes for easy indexing
                    axes = axes.flatten()

                    # Plot each tensor as a scatter plot
                    for j, tensor in enumerate(designs):
                        # Remove padding if needed
                        if padding_size > 0:
                            tensor = tensor[:-padding_size]  # noqa: PLW2901

                        if isinstance(problem.design_space, spaces.Dict):
                            design = spaces.unflatten(problem.design_space, tensor.cpu().numpy())
                        else:
                            design = tensor.cpu().numpy()
                        fig, ax = problem.render(design)
                        ax.figure.canvas.draw()
                        img = np.array(fig.canvas.renderer.buffer_rgba())
                        axes[j].imshow(img)
                        axes[j].set_xticks([])  # Hide x ticks
                        axes[j].set_yticks([])  # Hide y ticks

                    plt.tight_layout()
                    img_fname = f"images/{batches_done}.png"
                    plt.savefig(img_fname)
                    plt.close()
                    wandb.log({"designs": wandb.Image(img_fname)})

                # --------------
                #  Save models
                # --------------
                if args.save_model and epoch == args.n_epochs - 1 and i == len(dataloader) - 1:
                    ckpt = {
                        "epoch": epoch,
                        "batches_done": batches_done,
                        "model": diffusion.state_dict(),
                        "loss": loss.item(),
                    }

                    th.save(ckpt, "model.pth")
                    if args.track:
                        artifact = wandb.Artifact(f"{args.problem_id}_{args.algo}_model", type="model")
                        artifact.add_file("model.pth")

                        wandb.log_artifact(artifact, aliases=[f"seed_{args.seed}"])

    wandb.finish()
