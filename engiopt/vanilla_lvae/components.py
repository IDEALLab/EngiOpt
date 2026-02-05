"""Standalone components for vanilla LVAE models.

This module provides encoder, decoder, and MLP architectures for
Lipschitz-constrained autoencoders. These components are self-contained
to allow the vanilla_lvae module to be used independently.

Components:
    - Encoder2D: Convolutional encoder for 2D designs
    - TrueSNDecoder2D: Spectrally normalized 2D decoder with tunable Lipschitz bound
    - SNMLPPredictor: Spectrally normalized MLP for performance prediction
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from torchvision import transforms


def spectral_norm_conv(module: nn.Module, input_shape: tuple[int, int]) -> nn.Module:  # noqa: ARG001
    """Apply spectral normalization to a convolutional layer.

    Args:
        module: A Conv2d or ConvTranspose2d module to normalize.
        input_shape: The spatial dimensions (H, W) of the input to this layer.

    Returns:
        The module wrapped with spectral normalization.
    """
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


class Encoder2D(nn.Module):
    """Convolutional encoder for 2D designs.

    Architecture: Input -> Conv layers -> Latent vector
    - Input   [100x100]
    - Conv1   [50x50]   (k=4, s=2, p=1)
    - Conv2   [25x25]   (k=4, s=2, p=1)
    - Conv3   [13x13]   (k=3, s=2, p=1)
    - Conv4   [7x7]     (k=3, s=2, p=1)
    - Conv5   [1x1]     (k=7, s=1, p=0)

    Args:
        latent_dim: Dimension of the latent space.
        design_shape: Original design shape (H, W) for reference.
        resize_dimensions: Dimensions to resize input to before encoding.
    """

    def __init__(
        self,
        latent_dim: int,
        design_shape: tuple[int, int],
        resize_dimensions: tuple[int, int] = (100, 100),
    ) -> None:
        super().__init__()
        self.resize_in = transforms.Resize(resize_dimensions)
        self.design_shape = design_shape

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 100->50
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 50->25
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # 25->13
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),  # 13->7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final 7x7 conv produces (B, latent_dim, 1, 1) -> flatten to (B, latent_dim)
        self.to_latent = nn.Conv2d(512, latent_dim, kernel_size=7, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder.

        Args:
            x: Input designs (B, 1, H, W)

        Returns:
            Latent codes (B, latent_dim)
        """
        x = self.resize_in(x)  # (B, 1, 100, 100)
        h = self.features(x)  # (B, 512, 7, 7)
        return self.to_latent(h).flatten(1)  # (B, latent_dim)


class TrueSNDeconv2DCombo(nn.Module):
    """Spectral normalized transposed conv2d with batch norm and activation.

    Args:
        input_shape: Spatial dimensions (H, W) of input feature maps.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
        stride: Stride of the convolution.
        padding: Padding added to the input.
        output_padding: Additional size added to output shape.
    """

    def __init__(  # noqa: PLR0913
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


class TrueSNDecoder2D(nn.Module):
    """2D decoder with spectral normalization for Lipschitz-constrained decoding.

    Same architecture as a standard decoder but with spectral normalization applied
    to all linear and convolutional layers. The lipschitz_scale parameter controls
    the sharpness of the decoder's output by scaling activations before the final
    sigmoid, allowing more expressive decoders while keeping output in [0, 1].

    This is critical for:
    - Preventing isotropic shrinkage during volume minimization
    - Stable gradient flow in constrained optimization
    - Bounded perturbations in design space

    Architecture: Latent vector -> Deconv layers -> Output
    • Latent   [latent_dim]
    • Linear   [512x7x7]
    • Reshape  [512x7x7]
    • Deconv1  [256x13x13]  (k=3, s=2, p=1)
    • Deconv2  [128x25x25]  (k=3, s=2, p=1)
    • Deconv3  [64x50x50]   (k=4, s=2, p=1)
    • Deconv4  [1x100x100]  (k=4, s=2, p=1)
    • Scale by lipschitz_scale, then Sigmoid -> [0, 1]

    Args:
        latent_dim: Dimension of the latent space
        design_shape: Original design shape (H, W) for resizing output
        lipschitz_scale: Scales pre-sigmoid activations. Higher values allow sharper
            decoder Jacobians (more expressive) while keeping output in [0, 1].
            Default: 1.0 (strict 1-Lipschitz before sigmoid).
    """

    def __init__(
        self,
        latent_dim: int,
        design_shape: tuple[int, int],
        lipschitz_scale: float = 1.0,
    ):
        super().__init__()
        self.design_shape = design_shape
        self.resize_out = transforms.Resize(self.design_shape)
        self.lipschitz_scale = lipschitz_scale

        # Spectral normalized linear projection
        self.proj = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 512 * 7 * 7)),
            nn.ReLU(inplace=True),
        )

        # Build deconvolutional layers with spectral normalization (no final sigmoid)
        self.deconv = nn.Sequential(
            # 7->13 (input shape: 7x7)
            TrueSNDeconv2DCombo(
                input_shape=(7, 7),
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            # 13->25 (input shape: 13x13)
            TrueSNDeconv2DCombo(
                input_shape=(13, 13),
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            # 25->50 (input shape: 25x25)
            TrueSNDeconv2DCombo(
                input_shape=(25, 25),
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            # 50->100 (input shape: 50x50) - final conv, no activation yet
            spectral_norm_conv(
                nn.ConvTranspose2d(
                    64,
                    1,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False,
                ),
                (50, 50),
            ),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to 2D design with Lipschitz constraint.

        The lipschitz_scale is applied before sigmoid, allowing sharper decoder
        Jacobians while keeping output in [0, 1].

        Args:
            z: Latent codes (B, latent_dim)

        Returns:
            Reconstructed designs (B, 1, H, W) in range [0, 1]
        """
        x = self.proj(z).view(z.size(0), 512, 7, 7)  # (B, 512, 7, 7)
        x = self.deconv(x)  # (B, 1, 100, 100) - pre-sigmoid activations
        x = self.resize_out(x)  # (B, 1, H_orig, W_orig)
        return torch.sigmoid(x * self.lipschitz_scale)  # Scale then squash to [0, 1]


class SNMLPPredictor(nn.Module):
    """Spectral normalized MLP for performance prediction from latent codes.

    Enforces c-Lipschitz continuity to ensure small steps in latent space correspond
    to bounded steps in performance space. The lipschitz_scale parameter scales
    activations before the final linear layer, allowing sharper internal gradients
    while maintaining bounded Lipschitz continuity.

    This is critical for:
    - Ensuring latent space respects performance information
    - Smooth optimization in latent space
    - Interpretable performance gradients

    Args:
        input_dim: Input dimension (latent_dim + n_conditions for conditional)
        output_dim: Output dimension (number of performance metrics)
        hidden_dims: Tuple of hidden layer widths (default: (256, 128))
        lipschitz_scale: Scales hidden layer activations before final layer. Higher
            values allow sharper gradients (more expressive). Default: 1.0.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        lipschitz_scale: float = 1.0,
    ):
        super().__init__()
        self.lipschitz_scale = lipschitz_scale

        # Hidden layers with spectral normalization
        hidden_layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            hidden_layers.append(SNLinearCombo(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.hidden = nn.Sequential(*hidden_layers)

        # Final layer: spectral normalized Linear (no activation)
        self.output = spectral_norm(nn.Linear(prev_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict performance from latent codes (and optionally conditions).

        The lipschitz_scale is applied to hidden layer outputs before the final
        linear layer, allowing sharper gradients while maintaining bounded Lipschitz.

        Args:
            x: Input tensor (B, input_dim) containing [latent_codes] or [latent_codes, conditions]

        Returns:
            Predicted performance (B, output_dim)
        """
        h = self.hidden(x)  # Hidden layer outputs
        return self.output(h * self.lipschitz_scale)  # Scale then project


__all__ = [
    "Encoder2D",
    "SNLinearCombo",
    "SNMLPPredictor",
    "TrueSNDecoder2D",
    "TrueSNDeconv2DCombo",
    "spectral_norm_conv",
]
