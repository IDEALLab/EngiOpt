"""This code is largely based on the excellent PyTorch GAN repo: https://github.com/eriklindernoren/PyTorch-GAN.

We essentially refreshed the Python style, use wandb for logging, and made a few little improvements.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import time

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn
from torchvision import transforms
import tqdm
import tyro
import wandb


@dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "beams2d"
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
    n_epochs: int = 1000
    """number of epochs of training"""
    batch_size: int = 32
    """size of the batches"""
    lr: float = 3e-4
    """learning rate"""
    b1: float = 0.5
    """decay of first order momentum of gradient"""
    b2: float = 0.999
    """decay of first order momentum of gradient"""
    n_cpu: int = 8
    """number of cpu threads to use during batch generation"""
    latent_dim: int = 100
    """dimensionality of the latent space"""
    sample_interval: int = 400
    """interval between image samples"""


class Generator(nn.Module):
    """Unconditional GAN generator that outputs images matching ``design_shape``.

    Args:
        latent_dim (int): Dimensionality of the noise (latent) vector ``z``.
        design_shape (tuple[int, int]): Desired height x width for the final image.
        num_filters (list[int], optional): Number of feature map channels at every
            upsampling stage. Defaults to ``[256, 128, 64, 32]``.
        out_channels (int, optional): Number of channels in the output image (e.g.
            ``3`` for RGB). Defaults to ``1``.
    """

    def __init__(
        self,
        latent_dim: int,
        design_shape: tuple[int, int],
        num_filters: list[int] | None = None,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        num_filters = num_filters or [256, 128, 64, 32]
        self.design_shape = design_shape

        # Stem: project the latent vector to a 7x7 spatial map
        self.stem = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, num_filters[0], kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(inplace=True),
        )

        # 7x7 → 100x100 through four transposed conv blocks
        self.up_blocks = nn.Sequential(
            # 7→13
            nn.ConvTranspose2d(num_filters[0], num_filters[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True),
            # 13→25
            nn.ConvTranspose2d(num_filters[1], num_filters[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True),
            # 25→50
            nn.ConvTranspose2d(num_filters[2], num_filters[3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[3]),
            nn.ReLU(inplace=True),
            # 50→100
            nn.ConvTranspose2d(num_filters[3], out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: th.Tensor) -> th.Tensor:  # (B, latent_dim, 1, 1)
        x = self.stem(z)  # (B, num_filters[0], 7, 7)
        x = self.up_blocks(x)  # (B, out_channels, 100, 100)
        return transforms.Resize(self.design_shape)(x)  # match ``design_shape``


class Discriminator(nn.Module):
    """Unconditional GAN critic / discriminator for 100x100 images.

    Args:
        in_channels (int, optional): Number of channels in the input image. Defaults to ``1``.
        num_filters (list[int], optional): Channels in each downsampling stage. Defaults to ``[32, 64, 128, 256]``.
        out_channels (int, optional): Size of the final output map (usually ``1``). Defaults to ``1``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_filters: list[int] | None = None,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        num_filters = num_filters or [32, 64, 128, 256]

        # 100x100 → 7x7 feature map
        self.main = nn.Sequential(
            # 100→50
            nn.Conv2d(in_channels, num_filters[0], kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 50→25
            nn.Conv2d(num_filters[0], num_filters[1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[1]),
            nn.LeakyReLU(0.2, inplace=True),
            # 25→13
            nn.Conv2d(num_filters[1], num_filters[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[2]),
            nn.LeakyReLU(0.2, inplace=True),
            # 13→7
            nn.Conv2d(num_filters[2], num_filters[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[3]),
            nn.LeakyReLU(0.2, inplace=True),
            # 7→1
            nn.Conv2d(num_filters[3], out_channels, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = transforms.Resize((100, 100))(x)  # ensure input resolution
        return self.main(x)  # (B, out_channels, 1, 1)


if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    design_shape = problem.design_space.shape

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

    # Loss function
    adversarial_loss = th.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(args.latent_dim, design_shape)
    discriminator = Discriminator()

    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

    # Configure data loader
    training_ds = problem.dataset.with_format("torch", device=device)["train"]
    training_ds = th.utils.data.TensorDataset(training_ds["optimal_design"].flatten(1))
    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Optimizers
    optimizer_generator = th.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_discriminator = th.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    @th.no_grad()
    def sample_designs(n_designs: int) -> th.Tensor:
        """Samples n_designs from the generator."""
        # Sample noise
        z = th.randn((n_designs, args.latent_dim, 1, 1), device=device, dtype=th.float)

        return generator(z)

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
            z = th.randn((designs.size(0), args.latent_dim, 1, 1), device=device, dtype=th.float)

            # Generate a batch of images
            gen_designs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_designs)[:, 0, 0], valid)

            g_loss.backward()
            optimizer_generator.step()

            # ---------------------
            #  Train Discriminator
            # max log(D(real)) + log(1 - D(G(z)))
            # ---------------------
            optimizer_discriminator.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(
                discriminator(designs.reshape(-1, 1, design_shape[0], design_shape[1]))[:, 0, 0], valid
            )
            fake_loss = adversarial_loss(discriminator(gen_designs.detach())[:, 0, 0], fake)
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
                    designs = sample_designs(25)
                    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

                    # Flatten axes for easy indexing
                    axes = axes.flatten()

                    # Plot each tensor as a image plot
                    for j, tensor in enumerate(designs):
                        img = tensor.cpu().numpy().reshape(design_shape[0], design_shape[1])  # Extract x and y coordinates
                        axes[j].imshow(img)  # image plot
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
                    if args.save_model:
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
                        artifact_gen = wandb.Artifact(f"{args.problem_id}_{args.algo}_generator", type="model")
                        artifact_gen.add_file("generator.pth")
                        artifact_disc = wandb.Artifact(f"{args.problem_id}_{args.algo}_discriminator", type="model")
                        artifact_disc.add_file("discriminator.pth")

                        wandb.log_artifact(artifact_gen, aliases=[f"seed_{args.seed}"])
                        wandb.log_artifact(artifact_disc, aliases=[f"seed_{args.seed}"])
                        wandb.log_artifact(artifact_gen, aliases=[f"seed_{args.seed}"])
                        wandb.log_artifact(artifact_disc, aliases=[f"seed_{args.seed}"])

    wandb.finish()
