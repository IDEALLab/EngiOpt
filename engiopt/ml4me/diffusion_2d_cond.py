"""Barebones Conditional Diffusion for 2D designs - simplest possible implementation.

This is a minimal, self-contained implementation of Conditional Denoising Diffusion
for 2D engineering designs. Uses a simple U-Net with condition embedding.
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
    wandb_project: str = "diffusion_barebones"
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
    lr: float = 1e-4
    """Learning rate."""

    # Diffusion parameters
    num_timesteps: int = 250
    """Number of diffusion timesteps."""
    beta_start: float = 1e-4
    """Starting beta value for noise schedule."""
    beta_end: float = 0.02
    """Ending beta value for noise schedule."""


class SimpleUNet(nn.Module):
    """Simple U-Net for denoising with time and condition embedding.

    Takes noisy image, timestep, and conditions as input.
    Outputs predicted noise.
    """

    def __init__(self, n_conds: int, time_emb_dim: int = 32):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Condition embedding
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_conds, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder (downsampling)
        self.enc1 = self._conv_block(1, 64)  # 100x100
        self.enc2 = self._conv_block(64, 128)  # 50x50
        self.enc3 = self._conv_block(128, 256)  # 25x25

        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)  # 12x12

        # Decoder (upsampling)
        self.dec3 = self._conv_block(512 + 256, 256)  # concat with enc3
        self.dec2 = self._conv_block(256 + 128, 128)  # concat with enc2
        self.dec1 = self._conv_block(128 + 64, 64)  # concat with enc1

        # Output
        self.out = nn.Conv2d(64, 1, kernel_size=1)

        # Embedding projection to match channel dimensions
        self.emb_proj = nn.Linear(time_emb_dim * 2, 512)  # time + cond

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Basic convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x: th.Tensor, t: th.Tensor, c: th.Tensor) -> th.Tensor:
        """Forward pass through U-Net.

        Args:
            x: Noisy image (B, 1, H, W)
            t: Timestep (B, 1)
            c: Conditions (B, n_conds)

        Returns:
            Predicted noise (B, 1, H, W)
        """
        # Embed time and conditions
        t_emb = self.time_mlp(t)  # (B, time_emb_dim)
        c_emb = self.cond_mlp(c)  # (B, time_emb_dim)
        emb = th.cat([t_emb, c_emb], dim=1)  # (B, time_emb_dim * 2)
        emb = self.emb_proj(emb)  # (B, 512)

        # Encoder
        e1 = self.enc1(x)  # (B, 64, 100, 100)
        e2 = self.enc2(self.pool(e1))  # (B, 128, 50, 50)
        e3 = self.enc3(self.pool(e2))  # (B, 256, 25, 25)

        # Bottleneck with embedding
        b = self.bottleneck(self.pool(e3))  # (B, 512, 12, 12)
        # Add embedding as bias across spatial dimensions
        b = b + emb.view(-1, 512, 1, 1)

        # Decoder with skip connections
        d3 = self.upsample(b)  # (B, 512, 24, 24)
        # Crop or pad to match e3 size (25x25)
        d3 = nn.functional.interpolate(d3, size=e3.shape[2:], mode="bilinear", align_corners=True)
        d3 = self.dec3(th.cat([d3, e3], dim=1))  # (B, 256, 25, 25)

        d2 = self.upsample(d3)  # (B, 256, 50, 50)
        d2 = self.dec2(th.cat([d2, e2], dim=1))  # (B, 128, 50, 50)

        d1 = self.upsample(d2)  # (B, 128, 100, 100)
        d1 = self.dec1(th.cat([d1, e1], dim=1))  # (B, 64, 100, 100)

        return self.out(d1)  # (B, 1, 100, 100)


class DiffusionModel:
    """Simple diffusion model with linear noise schedule."""

    def __init__(self, num_timesteps: int, beta_start: float, beta_end: float, device: th.device):
        self.num_timesteps = num_timesteps
        self.device = device

        # Linear noise schedule
        self.betas = th.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = th.cumprod(self.alphas, dim=0)

        # Precompute values for sampling
        self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = th.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = th.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - th.cat([th.tensor([1.0], device=device), self.alphas_cumprod[:-1]])) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_0: th.Tensor, t: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Forward diffusion: add noise to x_0 at timestep t.

        Returns:
            noisy_x, noise
        """
        noise = th.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        noisy_x = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_x, noise

    @th.no_grad()
    def p_sample(self, model: nn.Module, x_t: th.Tensor, t: int, c: th.Tensor) -> th.Tensor:
        """Reverse diffusion: denoise x_t at timestep t."""
        batch_size = x_t.shape[0]
        t_tensor = th.full((batch_size, 1), t, device=self.device, dtype=th.float32) / self.num_timesteps

        # Predict noise
        pred_noise = model(x_t, t_tensor, c)

        # Compute x_{t-1}
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        beta = self.betas[t]

        # Mean of p(x_{t-1} | x_t)
        x_t_minus_1_mean = (1.0 / th.sqrt(alpha)) * (
            x_t - (beta / th.sqrt(1.0 - alpha_cumprod)) * pred_noise
        )

        # Add noise if not final step
        if t > 0:
            noise = th.randn_like(x_t)
            variance = self.posterior_variance[t]
            x_t_minus_1 = x_t_minus_1_mean + th.sqrt(variance) * noise
        else:
            x_t_minus_1 = x_t_minus_1_mean

        return x_t_minus_1

    @th.no_grad()
    def sample(self, model: nn.Module, shape: tuple, c: th.Tensor) -> th.Tensor:
        """Sample from the model by running reverse diffusion."""
        batch_size = c.shape[0]
        # Start from pure noise
        x = th.randn(batch_size, *shape[1:], device=self.device)

        # Reverse diffusion
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(model, x, t, c)

        return x


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Load problem from EngiBench
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    design_shape = problem.design_space.shape
    assert design_shape is not None, "Design space shape must be defined"
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

    # Build model and diffusion
    model = SimpleUNet(n_conds).to(device)
    diffusion = DiffusionModel(args.num_timesteps, args.beta_start, args.beta_end, device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Loss function
    loss_fn = nn.MSELoss()

    # Load dataset
    hf = problem.dataset.with_format("torch")
    train_ds = hf["train"]

    x_train = train_ds["optimal_design"][:].unsqueeze(1).to(device)
    conds_train = th.stack([train_ds[key][:] for key in problem.conditions_keys], dim=1).to(device)

    train_loader = DataLoader(
        TensorDataset(x_train, conds_train), batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # Training loop
    batches_done = 0
    for epoch in range(args.n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{args.n_epochs}")
        for imgs, conds in bar:
            batch_size = imgs.shape[0]

            # Sample random timesteps
            t = th.randint(0, args.num_timesteps, (batch_size,), device=device)

            # Add noise to images
            noisy_imgs, noise = diffusion.q_sample(imgs, t)

            # Predict noise
            t_normalized = t.unsqueeze(1).float() / args.num_timesteps
            pred_noise = model(noisy_imgs, t_normalized, conds)

            # Compute loss
            loss = loss_fn(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Logging and sampling
            if args.track:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "epoch": epoch,
                        "batch": batches_done,
                    }
                )
                print(f"[Epoch {epoch}/{args.n_epochs}] [Batch {batches_done}/{len(train_loader)}] [Loss: {loss.item()}]")

                # This saves a grid image of 25 generated designs every sample_interval
                if batches_done % args.sample_interval == 0:
                    model.eval()
                    with th.no_grad():
                        # Create linspace for each condition
                        linspaces = [
                            th.linspace(conds_train[:, i].min(), conds_train[:, i].max(), 25, device=device)
                            for i in range(conds_train.shape[1])
                        ]

                        desired_conds = th.stack(linspaces, dim=1)
                        samples = diffusion.sample(model, (25, 1, *design_shape), desired_conds)
                        samples = th.clamp(samples, 0, 1)

                        # Plot with condition annotations
                        fig, axes = plt.subplots(5, 5, figsize=(12, 12))
                        axes = axes.flatten()

                        for j in range(25):
                            img = samples[j, 0].cpu().numpy()
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

                    model.train()

            batches_done += 1

        avg_loss = epoch_loss / n_batches
        print(f"Epoch {epoch}/{args.n_epochs} - Avg Loss: {avg_loss:.4f}")

    # Save model
    if args.save_model:
        th.save(model.state_dict(), "diffusion_barebones.pth")

        if args.track:
            artifact = wandb.Artifact(f"{args.problem_id}_{args.algo}", type="model")
            artifact.add_file("diffusion_barebones.pth")
            wandb.log_artifact(artifact, aliases=[f"seed_{args.seed}"])

    if args.track:
        wandb.finish()

    print("Training complete!")
