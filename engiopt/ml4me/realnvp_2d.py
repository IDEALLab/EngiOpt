"""Barebones RealNVP for 2D designs - simplest possible implementation.

This is a minimal, self-contained implementation of RealNVP (Real-valued Non-Volume
Preserving transformations) for 2D engineering designs. RealNVP is a specific type
of normalizing flow that uses affine coupling layers with careful design choices.
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
    wandb_project: str = "realnvp_barebones"
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
    n_flows: int = 8
    """Number of coupling layers."""
    hidden_dim: int = 512
    """Hidden dimension for coupling networks."""
    n_hidden_layers: int = 3
    """Number of hidden layers in coupling networks."""


class AffineCouplingLayer(nn.Module):
    """RealNVP affine coupling layer with proper scale parameterization."""

    def __init__(self, dim: int, hidden_dim: int, n_hidden_layers: int, mask: th.Tensor, n_conds: int = 0):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask)

        # Build coupling network: input -> scale and translation
        input_dim = int(mask.sum().item()) + n_conds
        output_dim = dim - int(mask.sum().item())

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

        # Separate outputs for scale and translation
        self.scale_net = nn.Linear(hidden_dim, output_dim)
        self.translation_net = nn.Linear(hidden_dim, output_dim)

        # Initialize scale to be close to identity
        nn.init.zeros_(self.scale_net.weight)
        nn.init.zeros_(self.scale_net.bias)

    def forward(self, x: th.Tensor, c: th.Tensor | None = None, reverse: bool = False) -> tuple[th.Tensor, th.Tensor]:
        """Forward or reverse pass through coupling layer.

        Args:
            x: Input (B, dim)
            c: Conditions (B, n_conds) or None
            reverse: If True, compute inverse transformation

        Returns:
            output, log_det_jacobian
        """
        # Apply mask to get the part we condition on
        x_masked = x * self.mask

        # Extract only the masked (non-zero) values for network input
        # The mask has 1s where we keep values, so we select those dimensions
        masked_indices = self.mask.bool()
        x_masked_values = x[:, masked_indices]

        # Compute scale and translation
        if c is not None:
            net_input = th.cat([x_masked_values, c], dim=1)
        else:
            net_input = x_masked_values

        hidden = self.net(net_input)
        log_s = self.scale_net(hidden)
        t = self.translation_net(hidden)

        # Stabilize scale with tanh (RealNVP uses this trick)
        log_s = th.tanh(log_s)

        # Apply transformation to unmasked part
        if not reverse:
            # Forward: y = x * exp(log_s) + t (on unmasked part)
            y = x_masked + (1 - self.mask) * (x * th.exp(log_s) + t)
            log_det = ((1 - self.mask) * log_s).sum(dim=1)
        else:
            # Reverse: x = (y - t) * exp(-log_s) (on unmasked part)
            y = x_masked + (1 - self.mask) * ((x - t) * th.exp(-log_s))
            log_det = -((1 - self.mask) * log_s).sum(dim=1)

        return y, log_det


class BatchNormLayer(nn.Module):
    """Batch normalization layer for flows (invertible)."""

    def __init__(self, dim: int, momentum: float = 0.1):
        super().__init__()
        self.dim = dim
        self.momentum = momentum

        # Running statistics
        self.register_buffer("running_mean", th.zeros(dim))
        self.register_buffer("running_var", th.ones(dim))

        # Learnable parameters
        self.log_gamma = nn.Parameter(th.zeros(dim))
        self.beta = nn.Parameter(th.zeros(dim))

    def forward(self, x: th.Tensor, reverse: bool = False) -> tuple[th.Tensor, th.Tensor]:
        """Forward or reverse pass through batch norm.

        Args:
            x: Input (B, dim)
            reverse: If True, compute inverse transformation

        Returns:
            output, log_det_jacobian
        """
        if not reverse:
            # Forward pass
            if self.training:
                # Use batch statistics
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False)

                # Update running statistics
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

                mean = batch_mean
                var = batch_var
            else:
                # Use running statistics
                mean = self.running_mean
                var = self.running_var

            # Normalize
            x_norm = (x - mean) / th.sqrt(var + 1e-5)

            # Scale and shift
            y = th.exp(self.log_gamma) * x_norm + self.beta

            # Log determinant
            log_det = self.log_gamma.sum() - 0.5 * th.log(var + 1e-5).sum()
            log_det = log_det.expand(x.shape[0])

            return y, log_det
        else:
            # Reverse pass
            mean = self.running_mean
            var = self.running_var

            # Undo scale and shift
            x_norm = (x - self.beta) / th.exp(self.log_gamma)

            # Undo normalization
            y = x_norm * th.sqrt(var + 1e-5) + mean

            # Log determinant (negative of forward)
            log_det = -(self.log_gamma.sum() - 0.5 * th.log(var + 1e-5).sum())
            log_det = log_det.expand(x.shape[0])

            return y, log_det


class RealNVP(nn.Module):
    """RealNVP: Real-valued Non-Volume Preserving transformation."""

    def __init__(self, dim: int, n_flows: int, hidden_dim: int, n_hidden_layers: int, n_conds: int = 0):
        super().__init__()
        self.dim = dim
        self.n_conds = n_conds

        # Create alternating masks (checkerboard pattern in 1D)
        self.flows = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(n_flows):
            # Alternate between masking first half and second half
            if i % 2 == 0:
                mask = th.zeros(dim)
                mask[: dim // 2] = 1
            else:
                mask = th.zeros(dim)
                mask[dim // 2 :] = 1

            self.flows.append(AffineCouplingLayer(dim, hidden_dim, n_hidden_layers, mask, n_conds))
            self.batch_norms.append(BatchNormLayer(dim))

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
        for flow, bn in zip(self.flows, self.batch_norms):
            z, log_det = flow(z, c, reverse=False)
            log_det_total += log_det

            z, log_det_bn = bn(z, reverse=False)
            log_det_total += log_det_bn

        return z, log_det_total

    def inverse(self, z: th.Tensor, c: th.Tensor | None = None) -> th.Tensor:
        """Inverse pass: z -> x (latent to design)."""
        x = z

        for flow, bn in reversed(list(zip(self.flows, self.batch_norms))):
            x, _ = bn(x, reverse=True)
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
        with th.no_grad():
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
    flow = RealNVP(standard_dim, args.n_flows, args.hidden_dim, args.n_hidden_layers, n_conds).to(device)

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
            th.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)  # Gradient clipping
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            bar.set_postfix({"loss": f"{loss.item():.4f}", "log_prob": f"{log_prob.mean().item():.2f}"})

            # Sampling and logging
            if args.track and batches_done % args.sample_interval == 0:
                wandb.log(
                    {
                        "epoch": epoch,
                        "batch": batches_done,
                        "loss": loss.item(),
                        "log_prob": log_prob.mean().item(),
                    }
                )

                # Generate samples
                flow.eval()
                with th.no_grad():
                    if n_conds > 0:
                        assert conds_train is not None, "conds_train must be defined when n_conds > 0"
                        # Sample with conditions from training set
                        sample_conds = conds_train[: min(25, len(conds_train))]
                        samples = flow.sample(len(sample_conds), sample_conds)
                    else:
                        # Unconditional sampling
                        samples = flow.sample(25, None)

                    samples = th.clamp(samples, 0, 1)
                    # Reshape to (100, 100) then resize to original shape
                    samples = samples.view(-1, 1, 100, 100)
                    samples = resize_to_original(samples).squeeze(1)  # Back to (N, H, W)

                    # Plot
                    n_samples = len(samples)
                    grid_size = int(np.ceil(np.sqrt(n_samples)))
                    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
                    axes = axes.flatten() if n_samples > 1 else [axes]

                    for idx in range(grid_size * grid_size):
                        if idx < n_samples:
                            axes[idx].imshow(samples[idx].cpu().numpy(), cmap="gray")
                            axes[idx].axis("off")
                        else:
                            axes[idx].axis("off")

                    plt.suptitle(f"Generated samples (epoch {epoch}, batch {batches_done})")
                    plt.tight_layout()
                    plt.savefig(f"images/realnvp_samples_{batches_done}.png")
                    plt.close()

                    wandb.log({"samples": wandb.Image(f"images/realnvp_samples_{batches_done}.png")})

                flow.train()

            batches_done += 1

        avg_loss = epoch_loss / n_batches
        print(f"Epoch {epoch}/{args.n_epochs} - Avg Loss: {avg_loss:.4f}")

    # Save model
    if args.save_model:
        th.save(flow.state_dict(), "realnvp_barebones.pth")

        if args.track:
            artifact = wandb.Artifact(f"{args.problem_id}_{args.algo}", type="model")
            artifact.add_file("realnvp_barebones.pth")
            wandb.log_artifact(artifact, aliases=[f"seed_{args.seed}"])

    if args.track:
        wandb.finish()

    print("Training complete!")
