"""Utility functions and classes for training neural network models."""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import wandb

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np

    from engiopt.surrogate_model.mlp_tabular_only import Args


# Dictionaries for activation & optimizer
ACTIVATIONS: dict[str, nn.Module] = {
    "relu": nn.ReLU(),
    "leakyrelu": nn.LeakyReLU(0.2),
    "prelu": nn.PReLU(),
    "rrelu": nn.RReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "gelu": nn.GELU(),
    "celu": nn.CELU(),
    "none": nn.Identity(),
}

OPTIMIZERS: dict[str, Callable] = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
    "adadelta": optim.Adadelta,
    "adamax": optim.Adamax,
    "asgd": optim.ASGD,
    "lbfgs": optim.LBFGS,
}


def get_device(device: str) -> torch.device:
    """Determine the best available device for PyTorch operations.

    Returns:
        torch.device: The device to use for PyTorch operations, prioritizing:
            1. MPS (Metal Performance Shaders) for Apple Silicon
            2. CUDA for NVIDIA GPUs
            3. CPU as fallback
    """
    if device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Invalid device: {device}")


def make_activation(activation: str) -> nn.Module:
    """Create a PyTorch activation function from a string name.

    Args:
        activation: Name of the activation function.

    Returns:
        The corresponding PyTorch activation module.

    Raises:
        ValueError: If the activation name is not supported.
    """
    if activation not in ACTIVATIONS:
        raise ValueError("Unsupported activation")
    return ACTIVATIONS[activation]


def make_optimizer(
    name: str, params: Iterable[torch.nn.Parameter], lr: float, weight_decay: float = 0.0
) -> optim.Optimizer:
    """Create a PyTorch optimizer from a string name.

    Args:
        name: Name of the optimizer.
        params: Parameters to optimize.
        lr: Learning rate.
        weight_decay: Weight decay (L2 penalty).

    Returns:
        The corresponding PyTorch optimizer.

    Raises:
        ValueError: If the optimizer name is not supported.
    """
    if name not in OPTIMIZERS:
        raise ValueError("Unsupported optimizer")
    return OPTIMIZERS[name](params, lr=lr, weight_decay=weight_decay)


class PlainTabularDataset(Dataset):
    """Dataset for plain tabular data (no shape columns)."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """Initialize the dataset.

        Args:
            x: Input features array.
            y: Target values array.
        """
        super().__init__()
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            A tuple of (input features, target value).
        """
        return self.x[idx], self.y[idx]


def _create_mlp(
    in_dim: int,
    hidden_size: int,
    hidden_layers: int,
    activation: str,
    device: torch.device,
    out_dim: int = 1,
) -> nn.Sequential:
    """Create a multi-layer perceptron model.

    Args:
        in_dim: Input dimension.
        hidden_size: Size of hidden layers.
        hidden_layers: Number of hidden layers.
        activation: Activation function name.
        out_dim: Output dimension. Default is 1.
        device: Device to create the model on.

    Returns:
        A sequential model with the specified architecture.
    """
    act_fn = make_activation(activation)
    layers: list[nn.Module] = []
    current_dim = in_dim

    for _ in range(hidden_layers):
        layers.append(nn.Linear(current_dim, hidden_size))
        layers.append(act_fn)
        current_dim = hidden_size

    layers.append(nn.Linear(current_dim, out_dim))
    return nn.Sequential(*layers).to(device)


def train_one_model(  # noqa: PLR0915, C901
    args: Args, train_loader: DataLoader, val_loader: DataLoader, device: torch.device
) -> tuple[Any, float]:
    """Train a single model (structured or unstructured).

    Args:
        args: Training arguments. This should include attributes like:
              structured, latent_dim, hidden_layers, hidden_size, activation,
              optimizer, learning_rate, lambda_lv, gamma, n_epochs, lr_decay,
              lr_decay_step, patience, and optionally device.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: The device to train the model on (CPU, CUDA, or MPS).

    Returns:
        A tuple of (model, best_val_loss).
    """
    in_dim = next(iter(train_loader))[0].shape[1]
    model = _create_mlp(
        in_dim=in_dim,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        device=device,
    )

    def train_step(batch_data: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_batch, y_batch = batch_data
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        preds = model(x_batch).squeeze(-1)
        return F.smooth_l1_loss(preds, y_batch)

    def valid_step(batch_data: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_batch, y_batch = batch_data
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        return F.smooth_l1_loss(model(x_batch).squeeze(-1), y_batch)

    opt = make_optimizer(args.optimizer, model.parameters(), lr=args.learning_rate, weight_decay=args.l2_lambda)
    sched = optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_weights = None

    for epoch in range(args.n_epochs):
        model.train()
        running_train_loss = 0.0
        n_train = len(train_loader.dataset)
        for batch_data in train_loader:
            opt.zero_grad()
            loss = train_step(batch_data)
            batch_size_ = len(batch_data[0])
            loss.backward()
            opt.step()
            running_train_loss += loss.item() * batch_size_
        epoch_train_loss = running_train_loss / n_train

        model.eval()
        running_val_loss = 0.0
        n_val = len(val_loader.dataset)
        with torch.no_grad():
            for batch_data in val_loader:
                loss_val = valid_step(batch_data)
                batch_size_ = len(batch_data[0])
                running_val_loss += loss_val.item() * batch_size_
        epoch_val_loss = running_val_loss / n_val

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_weights = model.state_dict()
        else:
            epochs_no_improve += 1

        if (epoch + 1) % args.lr_decay_step == 0:
            sched.step()

        print(f"[Epoch {epoch + 1}/{args.n_epochs}] Train: {epoch_train_loss:.4f}, Val: {epoch_val_loss:.4f}")
        if args.track:
            # Seed is tracked to distinguish between different runs of the same model in the ensemble
            wandb.log({"train_loss": epoch_train_loss, "val_loss": epoch_val_loss, "epoch": epoch, "seed": args.seed})
        if epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    print(f"Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model, best_val_loss
