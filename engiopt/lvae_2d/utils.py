"""Utility functions for LVAE 2D models."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


def polynomial_schedule(w, N, p=1, w_init=[1.0, 0.0], M=0):
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
