"""Barebones Normalizing Flow for 2D designs - simplest possible implementation.

This is a minimal, self-contained implementation of Normalizing Flows
for 2D engineering designs using simple coupling layers. Learns an invertible
mapping between design space and a simple distribution (Gaussian).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    wandb_project: str = "flow_barebones"
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

    # Model architecture
    n_flows: int = 12
    """Number of coupling layers."""
    hidden_dim: int = 512
    """Hidden dimension for coupling networks."""


class CouplingLayer(nn.Module):
    """Affine coupling layer.

    Splits input in half, uses first half to compute scale and translation
    for second half. This maintains invertibility.
    """

    def __init__(self, dim: int, hidden_dim: int, n_conds: int = 0):
        super().__init__()
        self.dim = dim
        self.split_dim = dim // 2

        # Network that computes scale and translation
        # Input: first half + conditions, Output: scale and translation for second half
        input_dim = self.split_dim + n_conds
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (dim - self.split_dim) * 2),  # scale and translation
        )

    def forward(self, x: th.Tensor, c: th.Tensor | None = None, reverse: bool = False) -> tuple[th.Tensor, th.Tensor]:
        """Forward or reverse pass through coupling layer.

        Args:
            x: Input (B, dim)
            c: Conditions (B, n_conds) or None
            reverse: If True, compute inverse transformation

        Returns:
            output, log_det_jacobian
        """
        # Split input
        x1, x2 = x[:, : self.split_dim], x[:, self.split_dim :]

        # Compute scale and translation from x1 (and conditions if provided)
        if c is not None:
            net_input = th.cat([x1, c], dim=1)
        else:
            net_input = x1

        net_out = self.net(net_input)
        log_s, t = net_out.chunk(2, dim=1)

        # Stabilize scale with tanh
        log_s = th.tanh(log_s)

        if not reverse:
            # Forward: x2' = x2 * exp(log_s) + t
            x2_out = x2 * th.exp(log_s) + t
            log_det = log_s.sum(dim=1)
        else:
            # Reverse: x2 = (x2' - t) / exp(log_s)
            x2_out = (x2 - t) * th.exp(-log_s)
            log_det = -log_s.sum(dim=1)

        return th.cat([x1, x2_out], dim=1), log_det


class NormalizingFlow(nn.Module):
    """Simple normalizing flow with coupling layers."""

    def __init__(self, dim: int, n_flows: int, hidden_dim: int, n_conds: int = 0):
        super().__init__()
        self.dim = dim
        self.n_conds = n_conds

        # Stack of coupling layers with alternating splits
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(CouplingLayer(dim, hidden_dim, n_conds))

        # Base distribution (standard Gaussian)
        self.register_buffer("base_loc", th.zeros(dim))
        self.register_buffer("base_scale", th.ones(dim))

    def forward(self, x: th.Tensor, c: th.Tensor | None = None) -> tuple[th.Tensor, th.Tensor]:
        """Forward pass: x -> z (design to latent).

        Returns:
            z, log_det_jacobian
        """
        log_det_total = th.zeros(x.shape[0], device=x.device)

        z = x
        for i, flow in enumerate(self.flows):
            z, log_det = flow(z, c, reverse=False)
            log_det_total += log_det

            # Permute for next layer (simple reversal)
            if i < len(self.flows) - 1:
                z = th.flip(z, dims=[1])

        return z, log_det_total

    def inverse(self, z: th.Tensor, c: th.Tensor | None = None) -> th.Tensor:
        """Inverse pass: z -> x (latent to design)."""
        x = z

        for i, flow in reversed(list(enumerate(self.flows))):
            # Undo permutation
            if i < len(self.flows) - 1:
                x = th.flip(x, dims=[1])

            x, _ = flow(x, c, reverse=True)

        return x

    def log_prob(self, x: th.Tensor, c: th.Tensor | None = None) -> th.Tensor:
        """Compute log probability of x under the model."""
        z, log_det = self.forward(x, c)

        # Log probability under base distribution (standard Gaussian)
        log_p_z = -0.5 * (z**2).sum(dim=1) - 0.5 * self.dim * math.log(2 * math.pi)

        # Add log determinant of Jacobian
        log_p_x = log_p_z + log_det

        return log_p_x

    def sample(self, n_samples: int, c: th.Tensor | None = None) -> th.Tensor:
        """Sample from the model."""
        # Sample from base distribution
        z = th.randn(n_samples, self.dim, device=self.base_loc.device)

        # Transform through inverse flow
        x = self.inverse(z, c)

        return x


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Load problem from EngiBench
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    design_shape = problem.design_space.shape
    design_dim = np.prod(design_shape)
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
    print(f"Original design shape: {design_shape}")
    print(f"Original design dimension: {design_dim}")
    print(f"Number of conditions: {n_conds}")

    # Resize to standardized (100, 100) for consistent performance across problems
    resize_to_standard = transforms.Resize((100, 100))
    resize_to_original = transforms.Resize(design_shape)
    standard_dim = 100 * 100  # 10000

    print(f"Resizing to standard shape: (100, 100)")
    print(f"Standard design dimension: {standard_dim}")

    # Build model with standard dimensions
    flow = NormalizingFlow(standard_dim, args.n_flows, args.hidden_dim, n_conds).to(device)

    # Optimizer
    optimizer = Adam(flow.parameters(), lr=args.lr)

    # Load dataset
    hf = problem.dataset.with_format("torch")
    train_ds = hf["train"]

    # Resize to (100, 100) and flatten
    x_train_original = train_ds["optimal_design"][:].unsqueeze(1)  # Add channel dim (N, 1, H, W)
    x_train = resize_to_standard(x_train_original).flatten(1).to(device)  # (N, 10000)
    conds_train: th.Tensor | None = None

    if n_conds > 0:
        conds_train = th.stack([train_ds[key][:] for key in problem.conditions_keys], dim=1).to(device)
        train_loader = DataLoader(
            TensorDataset(x_train, conds_train), batch_size=args.batch_size, shuffle=True, drop_last=True
        )
    else:
        train_loader = DataLoader(TensorDataset(x_train), batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Training loop
    batches_done = 0
    for epoch in range(args.n_epochs):
        flow.train()
        epoch_loss = 0.0
        n_batches = 0

        bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{args.n_epochs}")
        for batch in bar:
            if n_conds > 0:
                designs, conds = batch
            else:
                designs = batch[0]
                conds = None

            # Compute negative log likelihood
            log_prob = flow.log_prob(designs, conds)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            bar.set_postfix({"loss": f"{loss.item():.4f}", "log_prob": f"{log_prob.mean().item():.2f}"})

            # Logging and sampling
            if args.track:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "log_prob": log_prob.mean().item(),
                        "epoch": epoch,
                        "batch": batches_done,
                    }
                )
                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {batches_done}/{len(train_loader)}] [Loss: {loss.item()}] [Log prob: {log_prob.mean().item():.2f}]"
                )

                # This saves a grid image of 25 generated designs every sample_interval
                if batches_done % args.sample_interval == 0:
                    flow.eval()
                    with th.no_grad():
                        if n_conds > 0:
                            assert conds_train is not None, "conds_train must be defined when n_conds > 0"
                            # Create linspace for each condition
                            linspaces = [
                                th.linspace(conds_train[:, i].min(), conds_train[:, i].max(), 25, device=device)
                                for i in range(conds_train.shape[1])
                            ]

                            desired_conds = th.stack(linspaces, dim=1)
                            samples = flow.sample(25, desired_conds)
                        else:
                            # Unconditional sampling
                            samples = flow.sample(25, None)
                            desired_conds = None

                        samples = th.clamp(samples, 0, 1)
                        # Reshape to (100, 100) then resize to original shape
                        samples = samples.view(-1, 1, 100, 100)
                        samples = resize_to_original(samples).squeeze(1)  # Back to (N, H, W)

                        # Plot with condition annotations
                        fig, axes = plt.subplots(5, 5, figsize=(12, 12))
                        axes = axes.flatten()

                        for j in range(25):
                            img = samples[j].cpu().numpy()
                            axes[j].imshow(img)
                            if n_conds > 0 and desired_conds is not None:
                                dc = desired_conds[j].cpu()
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

                    flow.train()

            batches_done += 1

        avg_loss = epoch_loss / n_batches
        print(f"Epoch {epoch}/{args.n_epochs} - Avg Loss: {avg_loss:.4f}")

    # Save model
    if args.save_model:
        th.save(flow.state_dict(), "flow_barebones.pth")

        if args.track:
            artifact = wandb.Artifact(f"{args.problem_id}_{args.algo}", type="model")
            artifact.add_file("flow_barebones.pth")
            wandb.log_artifact(artifact, aliases=[f"seed_{args.seed}"])

    if args.track:
        wandb.finish()

    print("Training complete!")
