"""This code is largely based on the excellent PyTorch GAN repo: https://github.com/eriklindernoren/PyTorch-GAN.

We essentially refreshed the Python style, use wandb for logging, and made a few little improvements.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import time
from typing import TYPE_CHECKING

from engibench.utils.all_problems import BUILTIN_PROBLEMS
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn
from torchvision import transforms
import tqdm
import tyro

from engiopt.transforms import flatten_dict_factory
import wandb

if TYPE_CHECKING:
    from engibench.utils.problem import Problem


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
    n_epochs: int = 5000
    """number of epochs of training"""
    batch_size: int = 32
    """size of the batches"""
    lr_gen: float = 0.00005
    """learning rate for the generator"""
    lr_disc: float = 0.0002
    """learning rate for the discriminator"""
    b1: float = 0.5
    """decay of first order momentum of gradient"""
    b2: float = 0.999
    """decay of first order momentum of gradient"""
    n_cpu: int = 8
    """number of cpu threads to use during batch generation"""
    latent_dim: int = 4
    """dimensionality of the latent space"""
    sample_interval: int = 400
    """interval between image samples"""


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


class Generator(nn.Module):
    def __init__(self, latent_dim: int, design_shape: tuple[int, ...], design_normalizer: Normalizer):
        super().__init__()
        self.design_shape = design_shape  # Store design shape
        self.design_normalizer = design_normalizer

        def block(in_feat: int, out_feat: int, normalize: bool = True) -> list[nn.Module]:  # noqa: FBT001, FBT002
            layers: list[nn.Module] = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(design_shape))),
            nn.Tanh(),
        )

    def forward(self, z: th.Tensor) -> th.Tensor:
        design = self.model(z)
        return self.design_normalizer.denormalize(design.view(design.size(0), *self.design_shape))


class Discriminator(nn.Module):
    def __init__(self, design_shape: tuple[int, ...], design_normalizer: Normalizer):
        super().__init__()
        self.design_normalizer = design_normalizer

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(design_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, design: th.Tensor) -> th.Tensor:
        design_flat = design.view(design.size(0), -1)
        return self.model(self.design_normalizer.normalize(design_flat))


def prepare_data(problem: Problem, device: th.device) -> tuple[th.utils.data.TensorDataset, Normalizer]:
    """Prepares the data for the generator and discriminator.

    Args:
        problem (Problem): The problem to prepare the data for.
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

    training_ds = th.utils.data.TensorDataset(
        transform(training_ds["optimal_design"][:]),
        *[training_ds[key][:] for key in problem.conditions_keys],
    )

    # Create design normalizer
    design_tensors = training_ds.tensors[0].T
    design_min = design_tensors.amin(dim=tuple(range(1, design_tensors.ndim))).to(device)
    design_max = design_tensors.amax(dim=tuple(range(1, design_tensors.ndim))).to(device)
    design_normalizer = Normalizer(design_min, design_max)

    return training_ds, design_normalizer


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
        design_shape = spaces.flatten(problem.design_space, dummy_design).shape

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

    # Prepare data
    training_ds, design_normalizer = prepare_data(problem, device)

    # Initialize generator and discriminator
    generator = Generator(latent_dim=args.latent_dim, design_shape=design_shape, design_normalizer=design_normalizer)
    discriminator = Discriminator(design_shape=design_shape, design_normalizer=design_normalizer)
    # Loss function
    adversarial_loss = th.nn.BCELoss()

    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Optimizers
    optimizer_generator = th.optim.Adam(generator.parameters(), lr=args.lr_gen, betas=(args.b1, args.b2))
    optimizer_discriminator = th.optim.Adam(discriminator.parameters(), lr=args.lr_disc, betas=(args.b1, args.b2))

    # ----------
    #  Training
    # ----------
    for epoch in tqdm.trange(args.n_epochs):
        for i, data in enumerate(dataloader):
            designs = data[0]

            # Adversarial ground truths
            valid = th.ones((designs.size(0), 1), requires_grad=False, device=device)
            fake = th.zeros((designs.size(0), 1), requires_grad=False, device=device)

            # -----------------
            #  Train Generator
            # min log(1 - D(G(z))) <==> max log(D(G(z)))
            # -----------------
            optimizer_generator.zero_grad()

            # Sample noise as generator input
            z = th.randn((designs.size(0), args.latent_dim), device=device, dtype=th.float)

            # Generate a batch of images
            gen_designs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_designs), valid)

            g_loss.backward()
            optimizer_generator.step()

            # ---------------------
            #  Train Discriminator
            # max log(D(real)) + log(1 - D(G(z)))
            # ---------------------
            optimizer_discriminator.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(designs), valid)
            fake_loss = adversarial_loss(discriminator(gen_designs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_discriminator.step()

            # ----------
            #  Logging
            # ----------
            if args.track:
                batches_done = epoch * len(dataloader) + i
                wandb.log(
                    {
                        "d_loss": d_loss.item(),
                        "g_loss": g_loss.item(),
                        "epoch": epoch,
                        "batch": batches_done,
                    }
                )
                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
                )

                # This saves a grid image of 25 generated designs every sample_interval
                if batches_done % args.sample_interval == 0:
                    # Extract 25 designs
                    tensors = gen_designs.data[:25]
                    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

                    # Flatten axes for easy indexing
                    axes = axes.flatten()

                    # Plot each tensor as a scatter plot
                    for j, tensor in enumerate(tensors):
                        if isinstance(problem.design_space, spaces.Dict):
                            design = spaces.unflatten(problem.design_space, tensor.cpu().numpy())
                        else:
                            design = tensor.cpu().numpy()
                        # use problem's render method to get the image
                        fig, ax = problem.render(design)
                        ax.figure.canvas.draw()
                        img = np.array(fig.canvas.renderer.buffer_rgba())
                        axes[j].imshow(img)
                        axes[j].set_xticks([])  # Hide x ticks
                        axes[j].set_yticks([])  # Hide y ticks
                        plt.close(fig)  # Close the original figure to free memory

                    plt.tight_layout()
                    img_fname = f"images/{batches_done}.png"
                    plt.savefig(img_fname)
                    plt.close()
                    wandb.log({"designs": wandb.Image(img_fname)})

                # --------------
                #  Save models
                # --------------
                if args.save_model and epoch == args.n_epochs - 1 and i == len(dataloader) - 1:
                    ckpt_gen = {
                        "epoch": epoch,
                        "batches_done": batches_done,
                        "generator": generator.state_dict(),
                        "optimizer_generator": optimizer_generator.state_dict(),
                        "loss": g_loss.item(),
                    }
                    ckpt_disc = {
                        "epoch": epoch,
                        "batches_done": batches_done,
                        "discriminator": discriminator.state_dict(),
                        "optimizer_discriminator": optimizer_discriminator.state_dict(),
                        "loss": d_loss.item(),
                    }

                    th.save(ckpt_gen, "generator.pth")
                    th.save(ckpt_disc, "discriminator.pth")
                    if args.track:
                        artifact_gen = wandb.Artifact(f"{args.problem_id}_{args.algo}_generator", type="model")
                        artifact_gen.add_file("generator.pth")
                        artifact_disc = wandb.Artifact(f"{args.problem_id}_{args.algo}_discriminator", type="model")
                        artifact_disc.add_file("discriminator.pth")

                        wandb.log_artifact(artifact_gen, aliases=[f"seed_{args.seed}"])
                        wandb.log_artifact(artifact_disc, aliases=[f"seed_{args.seed}"])

    wandb.finish()
