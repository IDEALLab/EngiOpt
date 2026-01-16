"""2D-specific encoder and decoder architectures for LVAE models.

This module provides standard convolutional encoder/decoder architectures used
across 2D LVAE implementations. All architectures assume 100×100 input resolution
and use the following spatial progression:
- Encoder: 100→50→25→13→7→1
- Decoder: 7→13→25→50→100
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import transforms

from engiopt.lvae_core import SNLinearCombo, TrueSNDeconv2DCombo, spectral_norm_conv

__all__ = [
    "Encoder2D",
    "Decoder2D",
    "TrueSNDecoder2D",
    "SNMLPPredictor",
]


class Encoder2D(nn.Module):
    """2D convolutional encoder: Input → Latent vector.

    Standard encoder architecture for 100×100 images using strided convolutions:
    • Input   [100×100]
    • Conv1   [50×50]   (k=4, s=2, p=1)
    • Conv2   [25×25]   (k=4, s=2, p=1)
    • Conv3   [13×13]   (k=3, s=2, p=1)
    • Conv4   [7×7]     (k=3, s=2, p=1)
    • Conv5   [1×1]     (k=7, s=1, p=0)

    Args:
        latent_dim: Dimension of the latent space
        design_shape: Original design shape (H, W) for resizing output
        resize_dimensions: Target dimensions for input resizing (default: (100, 100))
    """

    def __init__(
        self,
        latent_dim: int,
        design_shape: tuple[int, int],
        resize_dimensions: tuple[int, int] = (100, 100),
    ):
        super().__init__()
        self.resize_in = transforms.Resize(resize_dimensions)
        self.design_shape = design_shape

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 100→50
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 50→25
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # 25→13
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),  # 13→7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final 7×7 conv produces (B, latent_dim, 1, 1) → flatten to (B, latent_dim)
        self.to_latent = nn.Conv2d(512, latent_dim, kernel_size=7, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode 2D design to latent vector.

        Args:
            x: Input design (B, 1, H, W)

        Returns:
            Latent codes (B, latent_dim)
        """
        x = self.resize_in(x)  # (B, 1, 100, 100)
        h = self.features(x)  # (B, 512, 7, 7)
        return self.to_latent(h).flatten(1)  # (B, latent_dim)


class Decoder2D(nn.Module):
    """2D convolutional decoder: Latent vector → Image.

    Standard decoder architecture using transposed convolutions:
    • Latent   [latent_dim]
    • Linear   [512×7×7]
    • Reshape  [512×7×7]
    • Deconv1  [256×13×13]  (k=3, s=2, p=1)
    • Deconv2  [128×25×25]  (k=3, s=2, p=1)
    • Deconv3  [64×50×50]   (k=4, s=2, p=1)
    • Deconv4  [1×100×100]  (k=4, s=2, p=1)

    Args:
        latent_dim: Dimension of the latent space
        design_shape: Original design shape (H, W) for resizing output
    """

    def __init__(self, latent_dim: int, design_shape: tuple[int, int]):
        super().__init__()
        self.design_shape = design_shape
        self.resize_out = transforms.Resize(self.design_shape)

        self.proj = nn.Sequential(
            nn.Linear(latent_dim, 512 * 7 * 7),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            # 7→13
            nn.ConvTranspose2d(
                512,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 13→25
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 25→50
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 50→100
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to 2D design.

        Args:
            z: Latent codes (B, latent_dim)

        Returns:
            Reconstructed designs (B, 1, H, W)
        """
        x = self.proj(z).view(z.size(0), 512, 7, 7)  # (B, 512, 7, 7)
        x = self.deconv(x)  # (B, 1, 100, 100)
        return self.resize_out(x)  # (B, 1, H_orig, W_orig)


class TrueSNDecoder2D(nn.Module):
    """2D decoder with spectral normalization for Lipschitz-constrained decoding.

    Same architecture as Decoder2D but with spectral normalization applied to all
    linear and convolutional layers, ensuring a bounded Lipschitz constant. This is
    critical for:
    - Stable gradient flow in constrained optimization
    - Bounded perturbations in design space
    - Theoretical guarantees for volume minimization

    The lipschitz_scale parameter relaxes the strict 1-Lipschitz constraint:
    - lipschitz_scale = 1.0: Strict 1-Lipschitz (default)
    - lipschitz_scale > 1.0: Relaxed c-Lipschitz for higher expressiveness

    Args:
        latent_dim: Dimension of the latent space
        design_shape: Original design shape (H, W) for resizing output
        lipschitz_scale: Lipschitz constant multiplier (default: 1.0)
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
            nn.utils.parametrizations.spectral_norm(nn.Linear(latent_dim, 512 * 7 * 7)),
            nn.ReLU(inplace=True),
        )

        # Build deconvolutional layers with spectral normalization
        self.deconv = nn.Sequential(
            # 7→13 (input shape: 7×7)
            TrueSNDeconv2DCombo(
                input_shape=(7, 7),
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            # 13→25 (input shape: 13×13)
            TrueSNDeconv2DCombo(
                input_shape=(13, 13),
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            # 25→50 (input shape: 25×25)
            TrueSNDeconv2DCombo(
                input_shape=(25, 25),
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            # 50→100 (input shape: 50×50) - final layer with sigmoid
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
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to 2D design with Lipschitz constraint.

        Args:
            z: Latent codes (B, latent_dim)

        Returns:
            Reconstructed designs (B, 1, H, W) scaled by lipschitz_scale
        """
        x = self.proj(z).view(z.size(0), 512, 7, 7)  # (B, 512, 7, 7)
        x = self.deconv(x)  # (B, 1, 100, 100)
        x = self.resize_out(x)  # (B, 1, H_orig, W_orig)
        return x * self.lipschitz_scale


class SNMLPPredictor(nn.Module):
    """Spectral normalized MLP for performance prediction from latent codes.

    Enforces c-Lipschitz continuity to ensure small steps in latent space correspond
    to bounded steps in performance space. This is critical for:
    - Smooth optimization in latent space
    - Interpretable performance gradients
    - Stable surrogate-based design search

    Uses SNLinearCombo (spectral normalized Linear + ReLU) for hidden layers and
    spectral normalized Linear for the output layer.

    The lipschitz_scale parameter relaxes the strict 1-Lipschitz constraint:
    - lipschitz_scale = 1.0: Strict 1-Lipschitz (default)
    - lipschitz_scale > 1.0: Relaxed c-Lipschitz for high-dynamic-range objectives

    Args:
        input_dim: Input dimension (latent_dim + n_conditions for conditional)
        output_dim: Output dimension (number of performance metrics)
        hidden_dims: Tuple of hidden layer widths (default: (256, 128))
        lipschitz_scale: Lipschitz constant multiplier (default: 1.0)
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

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(SNLinearCombo(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # Final layer: spectral normalized Linear (no activation)
        layers.append(nn.utils.parametrizations.spectral_norm(nn.Linear(prev_dim, output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict performance from latent codes (and optionally conditions).

        Args:
            x: Input tensor (B, input_dim) containing [latent_codes] or [latent_codes, conditions]

        Returns:
            Predicted performance (B, output_dim) scaled by lipschitz_scale
        """
        return self.net(x) * self.lipschitz_scale
