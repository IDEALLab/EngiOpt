"""Network components shared by the LVAE generators.

Encoder, decoder, and MLP architectures for Lipschitz-constrained autoencoders.
The Lipschitz bound is what makes latent distances meaningful: without it, a
decoder is free to stretch some latent directions arbitrarily, and any metric
computed in that space measures the decoder's parameterization rather than the
designs.

Components:
    - `Encoder2D`: convolutional encoder for 2D designs
    - `TrueSNDecoder2D`: spectrally normalized 2D decoder with tunable Lipschitz bound
    - `SNMLPPredictor`: spectrally normalized MLP for performance prediction
    - `ConditionEncoder2D`: encodes scalar and image conditions for conditional decoding
    - `LatentWhitening`: PCA-rotation whitening, so pruning sees decorrelated axes

Used by `engiopt.generators.lvae_2d`, `plvae_2d`, and `constrained_plvae_2d`,
and by `engiopt.lvae.checkpoints` when rebuilding an encoder as a metric
instrument.
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


class LatentWhitening(nn.Module):
    """PCA-rotation whitening for latent codes.

    Decorrelates latent dimensions by rotating into the principal component basis
    derived from a running covariance estimate (EMA). Unlike ZCA whitening, this
    preserves per-dimension variance — only the rotation is applied, not the
    scaling to unit variance — so volume regularization and pruning still see
    meaningful per-dimension scales.

    The rotation matrix is updated with ``torch.no_grad`` and treated as a
    constant during backprop. Gradients flow through the matrix multiply but
    the eigendecomposition itself is not differentiated, avoiding numerical
    instability. Sign-correction keeps eigenvectors aligned across updates.

    Args:
        dim: Number of latent dimensions.
        momentum: EMA momentum for running statistics (like BatchNorm). Default: 0.1.
    """

    running_mean: torch.Tensor
    running_cov: torch.Tensor
    _rotation: torch.Tensor
    num_batches_tracked: torch.Tensor

    def __init__(self, dim: int, momentum: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_cov", torch.eye(dim))
        self.register_buffer("_rotation", torch.eye(dim))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def _update_stats(self, z: torch.Tensor) -> None:
        """Update running mean, covariance, and PCA rotation from a batch."""
        mean = z.mean(0)
        centered = z - mean
        cov = (centered.T @ centered) / max(z.shape[0] - 1, 1)

        self.num_batches_tracked += 1
        self.running_mean.lerp_(mean, self.momentum)
        self.running_cov.lerp_(cov, self.momentum)

        # Eigendecomposition of running covariance for PCA rotation
        _, eigvecs = torch.linalg.eigh(self.running_cov)
        # Sign correction: align each eigenvector with its predecessor
        signs = torch.sign((eigvecs * self._rotation).sum(0))
        signs[signs == 0] = 1
        self._rotation.copy_(eigvecs * signs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Center and rotate latent codes into decorrelated PCA basis.

        Args:
            z: Raw latent codes (B, dim).

        Returns:
            Decorrelated latent codes (B, dim) with diagonal covariance.
        """
        if self.training:
            self._update_stats(z)
        return (z - self.running_mean) @ self._rotation


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
        whitening: If True, apply PCA-rotation whitening after the final
            convolutional layer to decorrelate latent dimensions by construction.
    """

    def __init__(
        self,
        latent_dim: int,
        design_shape: tuple[int, int],
        resize_dimensions: tuple[int, int] = (100, 100),
        *,
        whitening: bool = False,
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
        self.whiten = LatentWhitening(latent_dim) if whitening else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder.

        Args:
            x: Input designs (B, 1, H, W)

        Returns:
            Latent codes (B, latent_dim)
        """
        x = self.resize_in(x)  # (B, 1, 100, 100)
        h = self.features(x)  # (B, 512, 7, 7)
        z = self.to_latent(h).flatten(1)  # (B, latent_dim)
        if self.whiten is not None:
            z = self.whiten(z)
        return z


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


class TrueSNDecoder2D(nn.Module):
    """2D decoder with spectral normalization for Lipschitz-constrained decoding.

    Same architecture as a standard decoder but with spectral normalization applied
    to all linear and convolutional layers. The lipschitz_scale parameter sets the
    effective Lipschitz bound of the decoder: each SN layer is 1-Lipschitz, and the
    final output is scaled by lipschitz_scale, giving an overall bound of exactly
    lipschitz_scale.

    Output is unbounded (no sigmoid) so that the Lipschitz bound is honest everywhere
    — no saturation artifacts. Clamp to [0, 1] at inference for valid designs.

    This is critical for:
    - Preventing isotropic shrinkage during volume minimization
    - Stable gradient flow in constrained optimization
    - Honest Lipschitz bound that matches lipschitz_scale exactly

    Architecture: Latent vector -> Deconv layers -> Output
    • Latent   [latent_dim]
    • Linear   [512x7x7]
    • Reshape  [512x7x7]
    • Deconv1  [256x13x13]  (k=3, s=2, p=1)
    • Deconv2  [128x25x25]  (k=3, s=2, p=1)
    • Deconv3  [64x50x50]   (k=4, s=2, p=1)
    • Deconv4  [1x100x100]  (k=4, s=2, p=1)
    • Scale by lipschitz_scale (raw output, no activation)

    Args:
        latent_dim: Dimension of the latent space
        design_shape: Original design shape (H, W) for resizing output
        lipschitz_scale: Effective Lipschitz bound of the decoder. The overall
            Lipschitz constant equals exactly this value. Default: 1.0.
        cond_dim: Dimension of condition embedding to concatenate with z before
            projection. When 0 (default), decoder is unconditional.
    """

    def __init__(
        self,
        latent_dim: int,
        design_shape: tuple[int, int],
        lipschitz_scale: float = 1.0,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.design_shape = design_shape
        self.resize_out = transforms.Resize(self.design_shape)
        self.lipschitz_scale = lipschitz_scale
        self.cond_dim = cond_dim

        # Spectral normalized linear projection (input includes condition embedding when cond_dim > 0)
        self.proj = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim + cond_dim, 512 * 7 * 7)),
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

    def forward(self, z: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Decode latent vector to 2D design with Lipschitz constraint.

        Output is raw (unbounded) so the Lipschitz bound equals lipschitz_scale
        exactly. Clamp to [0, 1] at inference for valid designs.

        Args:
            z: Latent codes (B, latent_dim)
            cond: Condition embedding (B, cond_dim) or None. When provided,
                concatenated with z before projection.

        Returns:
            Reconstructed designs (B, 1, H, W), unbounded (clamp at inference)
        """
        if cond is not None:
            z = torch.cat([z, cond], dim=-1)
        x = self.proj(z).view(z.size(0), 512, 7, 7)  # (B, 512, 7, 7)
        x = self.deconv(x)  # (B, 1, 100, 100)
        x = self.resize_out(x)  # (B, 1, H_orig, W_orig)
        return x * self.lipschitz_scale


class SNMLPPredictor(nn.Module):
    """Spectral normalized MLP for performance prediction from latent codes.

    Enforces c-Lipschitz continuity to ensure small steps in latent space correspond
    to bounded steps in performance space. Each SN layer is 1-Lipschitz, and the
    lipschitz_scale multiplies hidden outputs before the final SN layer, giving an
    overall Lipschitz bound of exactly lipschitz_scale.

    This is critical for:
    - Ensuring latent space respects performance information
    - Smooth optimization in latent space
    - Interpretable performance gradients

    Args:
        input_dim: Input dimension (latent_dim + n_conditions for conditional)
        output_dim: Output dimension (number of performance metrics)
        hidden_dims: Tuple of hidden layer widths (default: (256, 128))
        lipschitz_scale: Effective Lipschitz bound of the predictor. The overall
            Lipschitz constant equals exactly this value. Default: 1.0.
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


class ConditionEncoder2D(nn.Module):
    """Convolutional encoder for image-sized conditions (e.g., boundary matrices).

    Encodes image conditions into a compact embedding vector that can be shared
    between decoder and predictor paths. Uses BatchNorm (not spectral norm) because
    the condition encoder processes fixed input data — the Lipschitz property only
    matters along the z-direction, which is guaranteed by the SN decoder.

    Architecture mirrors Encoder2D but with n_img_conds input channels:
    - Input   [100x100]
    - Conv1   [50x50]   (k=4, s=2, p=1)
    - Conv2   [25x25]   (k=4, s=2, p=1)
    - Conv3   [13x13]   (k=3, s=2, p=1)
    - Conv4   [7x7]     (k=3, s=2, p=1)
    - Conv5   [1x1]     (k=7, s=1, p=0) -> cond_embed_dim

    Args:
        n_img_conds: Number of image condition channels (e.g., 1 for a single boundary matrix).
        cond_embed_dim: Dimension of the output embedding vector.
        resize_dimensions: Dimensions to resize input conditions to before encoding.
    """

    def __init__(
        self,
        n_img_conds: int,
        cond_embed_dim: int = 64,
        resize_dimensions: tuple[int, int] = (100, 100),
    ) -> None:
        super().__init__()
        self.resize_in = transforms.Resize(resize_dimensions)

        self.features = nn.Sequential(
            nn.Conv2d(n_img_conds, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 100->50
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

        self.to_embedding = nn.Conv2d(512, cond_embed_dim, kernel_size=7, stride=1, padding=0, bias=True)

    def forward(self, img_conds: torch.Tensor) -> torch.Tensor:
        """Encode image conditions to embedding vector.

        Args:
            img_conds: Image conditions (B, n_img_conds, H, W).

        Returns:
            Condition embedding (B, cond_embed_dim).
        """
        x = self.resize_in(img_conds)  # (B, n_img_conds, 100, 100)
        h = self.features(x)  # (B, 512, 7, 7)
        return self.to_embedding(h).flatten(1)  # (B, cond_embed_dim)


__all__ = [
    "ConditionEncoder2D",
    "Encoder2D",
    "LatentWhitening",
    "SNLinearCombo",
    "SNMLPPredictor",
    "TrueSNDecoder2D",
    "TrueSNDeconv2DCombo",
    "spectral_norm_conv",
]
