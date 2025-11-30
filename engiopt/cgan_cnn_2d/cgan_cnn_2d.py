"""Based on https://github.com/togheppi/cDCGAN/tree/master.

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
    n_epochs: int = 2
    """number of epochs of training"""
    batch_size: int = 32
    """size of the batches"""
    lr_gen: float = 0.0001
    """learning rate for the generator"""
    lr_disc: float = 0.0004
    """learning rate for the discriminator"""
    b1: float = 0.5
    """decay of first order momentum of gradient"""
    b2: float = 0.999
    """decay of first order momentum of gradient"""
    n_cpu: int = 8
    """number of cpu threads to use during batch generation"""
    latent_dim: int = 32
    """dimensionality of the latent space"""
    sample_interval: int = 400
    """interval between image samples"""


class Generator(nn.Module):
    """Conditional GAN generator that outputs 100 x 100 images.

    From noise + condition.

    Args:
        latent_dim (int): Dimensionality of the noise (latent) vector.
        n_conds (int): Number of conditional features (channels)
                             that will be given as (B, cond_features, 1, 1).
        num_filters (list of int): Number of filters in each upsampling stage.
                                   E.g., [256, 128, 64, 32].
        out_channels (int): Number of output channels in the final image (e.g. 3 for RGB).
    """

    def __init__(
        self,
        latent_dim: int,
        n_conds: int,
        design_shape: tuple[int, int],
        num_filters: list[int] = [256, 128, 64, 32],  # noqa: B006
        out_channels: int = 1,
    ):
        super().__init__()
        self.design_shape = design_shape  # Store design shape
        # Path for noise z
        self.z_path = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, num_filters[0] // 2, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters[0] // 2),
            nn.ReLU(inplace=True),
        )
        # Path for condition c
        self.c_path = nn.Sequential(
            nn.ConvTranspose2d(n_conds, num_filters[0] // 2, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters[0] // 2),
            nn.ReLU(inplace=True),
        )

        # After we concat these, total channels = num_filters[0].
        # We'll define 4 upsampling layers to go 7x7 -> 13x13 -> 25x25 -> 50x50 -> 100x100

        self.up_blocks = nn.Sequential(
            # 7x7 -> 13x13 (kernel=3, stride=2, pad=1)
            nn.ConvTranspose2d(num_filters[0], num_filters[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True),
            # 13x13 -> 25x25 (kernel=3, stride=2, pad=1)
            nn.ConvTranspose2d(num_filters[1], num_filters[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True),
            # 25x25 -> 50x50 (kernel=4, stride=2, pad=1)
            nn.ConvTranspose2d(num_filters[2], num_filters[3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[3]),
            nn.ReLU(inplace=True),
            # 50x50 -> 100x100 (kernel=4, stride=2, pad=1)
            nn.ConvTranspose2d(num_filters[3], out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: th.Tensor, c: th.Tensor) -> th.Tensor:
        """Forward pass for the Generator.

        Inputs:
            z: (B, z_dim, 1, 1).
            c: (B, cond_features, 1, 1).
        Output:
            out: (B, out_channels, 100, 100).
        """
        # Run noise & condition through separate "stem"
        z_feat = self.z_path(z)  # -> (B, num_filters[0]//2, 7, 7)
        c_feat = self.c_path(c)  # -> (B, num_filters[0]//2, 7, 7)

        # Concat along channel dim
        x = th.cat([z_feat, c_feat], dim=1)  # (B, num_filters[0], 7, 7)

        # Upsample through the main blocks
        out = self.up_blocks(x)  # -> (B, out_channels, 100, 100)

        # Resize Image
        return transforms.Resize((self.design_shape[0], self.design_shape[1]))(out)


class Discriminator(nn.Module):
    """Conditional GAN discriminator that takes 100 x 100 images.

    Plus a condition (as another 'image' of shape (B, cond_features, 100, 100))
    and outputs a real/fake score in [0, 1].

    Args:
        n_conds (int): Number of conditional channels to pass in parallel.
        in_channels (int): Number of channels in real images (e.g. 3 for RGB).
        num_filters (list of int): Number of filters in each downsampling stage.
                                   E.g., [32, 64, 128, 256].
        out_channels (int): Typically 1 for final real/fake score (sigmoid).
    """

    def __init__(
        self,
        n_conds: int,
        in_channels: int = 1,
        num_filters: list[int] = [32, 64, 128, 256],  # noqa: B006
        out_channels: int = 1,
    ):
        super().__init__()

        # Path for real image
        self.img_path = nn.Sequential(
            nn.Conv2d(in_channels, num_filters[0] // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Path for condition
        self.cond_path = nn.Sequential(
            nn.Conv2d(n_conds, num_filters[0] // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Combined path (50x50 -> 25x25 -> 13x13 -> 7x7)
        # We'll define 3 downsampling stages, then the final 7x7 -> 1x1
        self.down_blocks = nn.Sequential(
            # 50->25 (kernel=4, stride=2, pad=1)
            nn.Conv2d(num_filters[0], num_filters[1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[1]),
            nn.LeakyReLU(0.2, inplace=True),
            # 25->13 (kernel=3, stride=2, pad=1)
            nn.Conv2d(num_filters[1], num_filters[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[2]),
            nn.LeakyReLU(0.2, inplace=True),
            # 13->7 (kernel=3, stride=2, pad=1)
            nn.Conv2d(num_filters[2], num_filters[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[3]),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final 7x7 -> 1x1
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_filters[3], out_channels, kernel_size=7, stride=1, padding=0, bias=False), nn.Sigmoid()
        )

    def forward(self, x: th.Tensor, c: th.Tensor) -> th.Tensor:
        """Forward pass for the Discriminator.

        Inputs:
            x: (B, in_channels, 100, 100) - e.g. real or generated image
            c: (B, cond_features, 100, 100) - condition image
        Output:
            out: (B, out_channels, 1, 1).
        """
        # Resize Image
        x = transforms.Resize((100, 100))(x)

        # Separate stem for image and condition
        x_feat = self.img_path(x)  # (B, num_filters[0]//2, 50, 50)
        c = c.expand(-1, -1, 100, 100)
        c_feat = self.cond_path(c)  # (B, num_filters[0]//2, 50, 50)

        # Concat along channel dimension
        h = th.cat([x_feat, c_feat], dim=1)  # (B, num_filters[0], 50, 50)

        # Downsample
        h = self.down_blocks(h)  # -> (B, num_filters[3], 7, 7)

        # Final conv => (B, out_channels, 1, 1)
        return self.final_conv(h)


if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    design_shape = problem.design_space.shape
    conditions = problem.conditions
    n_conds = len(conditions)

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
    generator = Generator(latent_dim=args.latent_dim, n_conds=n_conds, design_shape=design_shape)
    discriminator = Discriminator(n_conds)

    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

    # Configure data loader
    training_ds = problem.dataset.with_format("torch", device=device)["train"]
    training_ds = th.utils.data.TensorDataset(
        training_ds["optimal_design"][:].flatten(1), *[training_ds[key][:] for key in problem.conditions_keys]
    )
    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Optimizers
    optimizer_generator = th.optim.Adam(generator.parameters(), lr=args.lr_gen, betas=(args.b1, args.b2))
    optimizer_discriminator = th.optim.Adam(discriminator.parameters(), lr=args.lr_disc, betas=(args.b1, args.b2))

    @th.no_grad()
    def sample_designs(n_designs: int) -> tuple[th.Tensor, th.Tensor]:
        """Samples n_designs from the generator."""
        # Sample noise
        z = th.randn((n_designs, args.latent_dim, 1, 1), device=device, dtype=th.float)

        linspaces = [
            th.linspace(conds[:, i].min(), conds[:, i].max(), n_designs, device=device) for i in range(conds.shape[1])
        ]

        desired_conds = th.stack(linspaces, dim=1)
        gen_imgs = generator(z, desired_conds.reshape(-1, conds.shape[1], 1, 1))
        return desired_conds, gen_imgs

    # ----------
    #  Training
    # ----------
    for epoch in tqdm.trange(args.n_epochs):
        for i, data in enumerate(dataloader):
            # THIS IS PROBLEM DEPENDENT
            designs = data[0]

            conds = th.stack((data[1:]), dim=1).reshape(-1, n_conds, 1, 1)

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
            gen_designs = generator(z, conds)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_designs, conds)[:, 0, 0], valid)

            g_loss.backward()
            optimizer_generator.step()

            # ---------------------
            #  Train Discriminator
            # max log(D(real)) + log(1 - D(G(z)))
            # ---------------------
            optimizer_discriminator.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(
                discriminator(designs.reshape(-1, 1, design_shape[0], design_shape[1]), conds)[:, 0, 0], valid
            )
            fake_loss = adversarial_loss(discriminator(gen_designs.detach(), conds)[:, 0, 0], fake)
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
                    desired_conds, designs = sample_designs(25)
                    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

                    # Flatten axes for easy indexing
                    axes = axes.flatten()

                    # Plot each tensor as a scatter plot
                    for j, tensor in enumerate(designs):
                        img = tensor.cpu().numpy().reshape(design_shape[0], design_shape[1])  # Extract x and y coordinates
                        dc = desired_conds[j].cpu()
                        axes[j].imshow(img)  # Scatter plot
                        title = [(conditions[i][0], f"{dc[i]:.2f}") for i in range(n_conds)]
                        title_string = "\n ".join(f"{condition}: {value}" for condition, value in title)
                        axes[j].title.set_text(title_string)  # Set title
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
