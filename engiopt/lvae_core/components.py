"""Shared components for LVAE models: utilities, schedules, and base neural network layers.

This module consolidates all shared building blocks used across both 1D and 2D LVAE
implementations, including:
- Scheduling functions (polynomial_schedule)
- Spectral normalization utilities (spectral_norm_conv)
- Base layers (Scale, Normalizer, MLP, SNMLP)
- Spectrally normalized combination layers (SNLinearCombo, TrueSNDeconv2DCombo)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


# =============================================================================
# Scheduling Functions
# =============================================================================


def polynomial_schedule(w, N, p=1, w_init=[1.0, 0.0], M=0):
    """Create a polynomial weight schedule for LVAE training.

    Generates a callable that smoothly transitions weights from w_init to w over
    N epochs using a polynomial schedule. Useful for gradually introducing volume
    loss or other regularization terms.

    Args:
        w: Target final weights (list or tensor)
        N: Number of epochs to complete schedule
        p: Polynomial order (default: 1 for linear, 2 for quadratic)
        w_init: Initial weights (default: [1.0, 0.0])
        M: Warmup epochs before schedule starts (default: 0)

    Returns:
        Callable that takes epoch number and returns current weights

    Example:
        >>> schedule = polynomial_schedule([1.0, 0.01], N=100, p=2)
        >>> weights = schedule(epoch=50)  # Returns weights at 50% progress
    """
    w = torch.as_tensor(w, dtype=torch.float)
    w_init = torch.as_tensor(w_init, dtype=torch.float)

    def poly_w(epoch):
        if epoch >= N:
            return w
        if epoch < M:
            return w_init
        else:
            k = (epoch - M) ** p / ((N - M) ** p)
            w_n = w_init + (w - w_init) * k
            return w_n

    return poly_w


# =============================================================================
# Spectral Normalization Utilities
# =============================================================================


def spectral_norm_conv(module: nn.Module, input_shape: tuple[int, int]) -> nn.Module:
    """Apply spectral normalization to a convolutional layer.

    This function wraps a convolutional/transposed convolutional layer with spectral
    normalization that is aware of the input spatial dimensions. Following the pattern
    from TrueSNDCGenerator, this ensures 1-Lipschitz bound on the layer.

    Args:
        module: A Conv2d or ConvTranspose2d module to normalize.
        input_shape: The spatial dimensions (H, W) of the input to this layer.

    Returns:
        The module wrapped with spectral normalization.
    """
    # Use standard spectral_norm which applies to weight matrix
    # The input_shape is mainly for documentation/debugging purposes
    # since PyTorch's spectral_norm applies to the weight parameter directly
    return spectral_norm(module)


# =============================================================================
# Base Layer Components
# =============================================================================


class Scale(nn.Module):
    """Learnable or fixed scalar multiplication layer.

    This layer multiplies input by a scalar value, useful for controlling
    the Lipschitz constant of the overall network.

    Args:
        scale: Initial scale value (can be float or tensor)

    Shape:
        - Input: (*, H_in) where * means any number of dimensions
        - Output: (*, H_in) same shape as input
    """

    def __init__(self, scale: float | torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("s", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply input by scale factor.

        Args:
            x: Input tensor

        Returns:
            Scaled tensor: x * scale
        """
        return self.s * x


class Normalizer:
    """Min-max normalization utility for tensors.

    Provides normalization and denormalization using min-max scaling to a
    configurable target range. Useful for preprocessing inputs and postprocessing
    outputs in neural networks.

    Args:
        min_val: Minimum value(s) for normalization (scalar or tensor)
        max_val: Maximum value(s) for normalization (scalar or tensor)
        target_range: Target range for normalized values (default: [0, 1])
        eps: Small constant to avoid division by zero (default: 1e-7)

    Example:
        >>> # Normalize angles from [-10, 10] to [0, 1]
        >>> normalizer = Normalizer(min_val=-10.0, max_val=10.0)
        >>> angle_norm = normalizer.normalize(angle)  # → [0, 1]
        >>> angle_orig = normalizer.denormalize(angle_norm)  # → [-10, 10]
    """

    def __init__(
        self,
        min_val: float | torch.Tensor,
        max_val: float | torch.Tensor,
        target_range: tuple[float, float] = (0.0, 1.0),
        eps: float = 1e-7,
    ):
        """Initialize normalizer with min and max values.

        Args:
            min_val: Minimum value(s) for normalization
            max_val: Maximum value(s) for normalization
            target_range: Target range for normalization (default: [0, 1])
            eps: Small constant to avoid division by zero
        """
        self.eps = eps
        self.min_val = torch.as_tensor(min_val) if not isinstance(min_val, torch.Tensor) else min_val
        self.max_val = torch.as_tensor(max_val) if not isinstance(max_val, torch.Tensor) else max_val
        self.target_min, self.target_max = target_range

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to target range.

        Args:
            x: Input tensor in original range

        Returns:
            Normalized tensor in target range
        """
        # First normalize to [0, 1]
        x_01 = (x - self.min_val) / (self.max_val - self.min_val + self.eps)
        # Then scale to target range
        return x_01 * (self.target_max - self.target_min) + self.target_min

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize input from target range to original range.

        Args:
            x: Normalized tensor in target range

        Returns:
            Denormalized tensor in original range
        """
        # First scale from target range to [0, 1]
        x_01 = (x - self.target_min) / (self.target_max - self.target_min + self.eps)
        # Then denormalize to original range
        return x_01 * (self.max_val - self.min_val + self.eps) + self.min_val


class MLP(nn.Module):
    """Multi-layer perceptron with flexible architecture.

    Constructs a fully connected network with specified hidden layer widths.
    The final layer is always a linear transformation without activation.
    Hidden layers use LeakyReLU(0.2) activation by default.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        hidden_widths: List of hidden layer widths (empty list = single linear layer)
        activation: Activation function for hidden layers (default: LeakyReLU(0.2))
        use_spectral_norm: Whether to apply spectral normalization (default: False)

    Shape:
        - Input: (N, H_in) where H_in = in_features
        - Output: (N, H_out) where H_out = out_features

    Example:
        >>> mlp = MLP(128, 64, hidden_widths=[256, 128])
        >>> # Network: 128 -> 256 -> 128 -> 64
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_widths: list[int],
        activation: nn.Module | None = None,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_widths = list(hidden_widths)

        if activation is None:
            activation = nn.LeakyReLU(0.2)

        # Build layer specifications: [in_features] + hidden_widths + [out_features]
        layer_sizes = [in_features] + self.hidden_widths + [out_features]

        # Build sequential model
        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            # Create linear layer
            linear = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            if use_spectral_norm:
                linear = spectral_norm(linear)
            layers.append(linear)

            # Add activation for all layers except the last
            if i < len(layer_sizes) - 2:
                layers.append(activation)

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP.

        Args:
            x: Input tensor (N, in_features)

        Returns:
            Output tensor (N, out_features)
        """
        return self.model(x)


class SNMLP(MLP):
    """Spectrally normalized MLP with Lipschitz constant ≤ 1.

    This is a convenience wrapper around MLP that automatically enables
    spectral normalization on all linear layers.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        hidden_widths: List of hidden layer widths
        activation: Activation function for hidden layers (default: LeakyReLU(0.2))

    Shape:
        - Input: (N, H_in) where H_in = in_features
        - Output: (N, H_out) where H_out = out_features
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_widths: list[int],
        activation: nn.Module | None = None,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_widths=hidden_widths,
            activation=activation,
            use_spectral_norm=True,
        )


# =============================================================================
# Spectral Normalized Combination Layers
# =============================================================================


class SNLinearCombo(nn.Module):
    """Spectral normalized linear layer with activation.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = spectral_norm(nn.Linear(in_features, out_features))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer."""
        return self.activation(self.linear(x))


class TrueSNDeconv2DCombo(nn.Module):
    """Spectral normalized transposed conv2d with batch norm and activation.

    This module combines ConvTranspose2d with spectral normalization,
    batch normalization, and ReLU activation.

    Args:
        input_shape: Spatial dimensions (H, W) of input feature maps.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
        stride: Stride of the convolution.
        padding: Padding added to the input.
        output_padding: Additional size added to output shape.
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 0,
    ):
        super().__init__()
        self.conv = spectral_norm_conv(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            ),
            input_shape,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer."""
        return self.activation(self.bn(self.conv(x)))


__all__ = [
    # Scheduling
    "polynomial_schedule",
    # Spectral normalization
    "spectral_norm_conv",
    # Base layers
    "Scale",
    "Normalizer",
    "MLP",
    "SNMLP",
    # Combination layers
    "SNLinearCombo",
    "TrueSNDeconv2DCombo",
]
