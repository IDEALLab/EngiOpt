"""Barebones Conditional GAN for 2D designs - simplest possible implementation.

This is a minimal, self-contained implementation of Conditional GAN (cGAN)
for 2D engineering designs using CNNs. Generates designs conditioned on
performance specifications.
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
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import tqdm
import tyro
import wandb


@dataclass
class Args:
    # Problem and tracking
    problem_id: str = "beams2d"
    """Problem ID to run. Must be one of the built-in problems in engibench."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """Algorithm name for tracking purposes."""
    track: bool = True
    """Whether to track with Weights & Biases."""
    wandb_project: str = "cgan_barebones"
    """WandB project name."""
    wandb_entity: str | None = None
    """WandB entity name. If None, uses the default entity."""
    seed: int = 1
    """Random seed for reproducibility."""
    save_model: bool = False
    """Whether to save the model after training."""
    sample_interval: int = 400
    """Interval for sampling designs during training."""

    # Training
    n_epochs: int = 200
    """Number of training epochs."""
    batch_size: int = 32
    """Batch size for training."""
    lr_gen: float = 1e-4
    """Learning rate for generator."""
    lr_disc: float = 4e-4
    """Learning rate for discriminator."""
    b1: float = 0.5
    """Adam beta1 parameter."""
    b2: float = 0.999
    """Adam beta2 parameter."""

    # Model architecture
    latent_dim: int = 32
    """Dimensionality of noise vector."""


class Generator(nn.Module):
    """Simple conditional generator: noise + conditions -> 100x100 design."""

    def __init__(self, latent_dim: int, n_conds: int, design_shape: tuple[int, int]):
        super().__init__()
        self.design_shape = design_shape

        # Separate paths for noise and conditions
        self.z_path = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.c_path = nn.Sequential(
            nn.ConvTranspose2d(n_conds, 128, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Upsampling blocks: 7x7 -> 13x13 -> 25x25 -> 50x50 -> 100x100
        self.up_blocks = nn.Sequential(
            # 7x7 -> 13x13
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 13x13 -> 25x25
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 25x25 -> 50x50
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 50x50 -> 100x100
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

        self.resize = transforms.Resize(design_shape)

    def forward(self, z: th.Tensor, c: th.Tensor) -> th.Tensor:
        """Generate design from noise and conditions.

        Args:
            z: Noise vector (B, latent_dim, 1, 1)
            c: Conditions (B, n_conds, 1, 1)

        Returns:
            Generated design (B, 1, H, W)
        """
        z_feat = self.z_path(z)  # (B, 128, 7, 7)
        c_feat = self.c_path(c)  # (B, 128, 7, 7)
        x = th.cat([z_feat, c_feat], dim=1)  # (B, 256, 7, 7)
        out = self.up_blocks(x)  # (B, 1, 100, 100)
        return self.resize(out)


class Discriminator(nn.Module):
    """Simple conditional discriminator: design + conditions -> real/fake."""

    def __init__(self, n_conds: int, design_shape: tuple[int, int]):
        super().__init__()
        self.resize = transforms.Resize((100, 100))

        # Image path: 100x100 -> features
        self.img_path = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 100->50
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 50->25
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Condition path: expand conditions to spatial dimensions
        self.cond_path = nn.Sequential(
            nn.ConvTranspose2d(n_conds, 64, kernel_size=25, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Combined path after concatenation
        self.combined = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),  # 25->13
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # 13->7
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=7, stride=1, padding=0, bias=False),  # 7->1
        )

    def forward(self, img: th.Tensor, c: th.Tensor) -> th.Tensor:
        """Classify design as real or fake given conditions.

        Args:
            img: Design (B, 1, H, W)
            c: Conditions (B, n_conds, 1, 1)

        Returns:
            Validity score (B, 1, 1, 1)
        """
        img = self.resize(img)  # (B, 1, 100, 100)
        img_feat = self.img_path(img)  # (B, 64, 25, 25)
        c_feat = self.cond_path(c)  # (B, 64, 25, 25)
        x = th.cat([img_feat, c_feat], dim=1)  # (B, 128, 25, 25)
        return self.combined(x)  # (B, 1, 1, 1)


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Load problem from EngiBench
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    design_shape = problem.design_space.shape
    n_conds = len(problem.conditions_keys)

    # Initialize W&B tracking
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            save_code=True,
            name=run_name,
        )

    # Set random seeds
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True

    os.makedirs("images", exist_ok=True)

    # Device selection
    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    print(f"Using device: {device}")

    # Build models
    generator = Generator(args.latent_dim, n_conds, design_shape).to(device)
    discriminator = Discriminator(n_conds, design_shape).to(device)

    # Optimizers
    optimizer_G = Adam(generator.parameters(), lr=args.lr_gen, betas=(args.b1, args.b2))
    optimizer_D = Adam(discriminator.parameters(), lr=args.lr_disc, betas=(args.b1, args.b2))

    # Loss function
    adversarial_loss = nn.MSELoss()

    # Load dataset
    hf = problem.dataset.with_format("torch")
    train_ds = hf["train"]

    x_train = train_ds["optimal_design"][:].unsqueeze(1).to(device)
    # Normalize to [-1, 1] for Tanh output
    x_train = x_train * 2.0 - 1.0

    # Get conditions
    conds_train = th.stack([train_ds[key][:] for key in problem.conditions_keys], dim=1).to(device)

    train_loader = DataLoader(
        TensorDataset(x_train, conds_train), batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # Training loop
    batches_done = 0
    for epoch in range(args.n_epochs):
        bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{args.n_epochs}")
        d_loss = th.tensor(0.0)
        g_loss = th.tensor(0.0)
        for i, (imgs, conds) in enumerate(bar):
            batch_size = imgs.size(0)

            # Adversarial ground truths
            valid = th.ones((batch_size, 1, 1, 1), device=device)
            fake = th.zeros((batch_size, 1, 1, 1), device=device)

            # ---------------------
            #  Train Generator
            # ---------------------
            optimizer_G.zero_grad()

            # Sample noise and generate fake images
            z = th.randn(batch_size, args.latent_dim, 1, 1, device=device)
            c = conds.unsqueeze(-1).unsqueeze(-1)  # (B, n_conds, 1, 1)

            gen_imgs = generator(z, c)

            # Generator loss: fool discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs, c), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real loss
            real_loss = adversarial_loss(discriminator(imgs, c), valid)

            # Fake loss
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), c), fake)

            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # Progress
            bar.set_postfix(
                {
                    "D_loss": f"{d_loss.item():.4f}",
                    "G_loss": f"{g_loss.item():.4f}",
                }
            )

            # Logging and sampling
            if args.track:
                wandb.log(
                    {
                        "d_loss": d_loss.item(),
                        "g_loss": g_loss.item(),
                        "epoch": epoch,
                        "batch": batches_done,
                    }
                )
                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
                )

                # This saves a grid image of 25 generated designs every sample_interval
                if batches_done % args.sample_interval == 0:
                    # Generate samples with linspace conditions
                    generator.eval()
                    with th.no_grad():
                        # Sample noise
                        z = th.randn((25, args.latent_dim, 1, 1), device=device, dtype=th.float)

                        # Create linspace for each condition
                        linspaces = [
                            th.linspace(conds_train[:, i].min(), conds_train[:, i].max(), 25, device=device)
                            for i in range(conds_train.shape[1])
                        ]

                        desired_conds = th.stack(linspaces, dim=1)
                        gen_imgs = generator(z, desired_conds.reshape(-1, conds_train.shape[1], 1, 1))

                        # Denormalize from [-1, 1] to [0, 1]
                        gen_imgs = (gen_imgs + 1.0) / 2.0

                        # Plot with condition annotations
                        fig, axes = plt.subplots(5, 5, figsize=(12, 12))
                        axes = axes.flatten()

                        for j in range(25):
                            img = gen_imgs[j].cpu().numpy().reshape(design_shape[0], design_shape[1])
                            dc = desired_conds[j].cpu()
                            axes[j].imshow(img)
                            title = [(problem.conditions_keys[i], f"{dc[i]:.2f}") for i in range(n_conds)]
                            title_string = "\n ".join(f"{condition}: {value}" for condition, value in title)
                            axes[j].title.set_text(title_string)
                            axes[j].set_xticks([])
                            axes[j].set_yticks([])

                        plt.tight_layout()
                        img_fname = f"images/{batches_done}.png"
                        plt.savefig(img_fname)
                        plt.close()
                        wandb.log({"designs": wandb.Image(img_fname)})

                    generator.train()

            batches_done += 1

        print(f"Epoch {epoch}/{args.n_epochs} - D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

    # Save models
    if args.save_model:
        th.save(generator.state_dict(), "generator_barebones.pth")
        th.save(discriminator.state_dict(), "discriminator_barebones.pth")

        if args.track:
            artifact = wandb.Artifact(f"{args.problem_id}_{args.algo}", type="model")
            artifact.add_file("generator_barebones.pth")
            artifact.add_file("discriminator_barebones.pth")
            wandb.log_artifact(artifact, aliases=[f"seed_{args.seed}"])

    if args.track:
        wandb.finish()

    print("Training complete!")
