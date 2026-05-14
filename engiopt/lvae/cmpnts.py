"""
Components for the Latent Autoencoder (LAE).
    Contains:
        - EncoderBlock: convolutional down-sampling block (no time conditioning)
        - DecoderBlock: convolutional up-sampling block (no time conditioning)
        - MiddleBlock: bottleneck block
        - LinearCombo / MLP: fully-connected helpers (shared with DDM)
        - AirfoilConditional / AirfoilConditionalPerf: condition embedding
"""

import torch
from torch import nn
from torch.nn.utils import spectral_norm as sn


# ---------------------------------------------------------------------------
# Convolutional building blocks
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """Conv1d down-sampling block — mirrors Down_Block without time conditioning."""

    def __init__(self, in_channel, out_channel, kernel_size=3,
                 padding=1, padding_mode='circular', activation=nn.GELU(), norm=True,
                 use_sn: bool = False):
        super().__init__()
        wrap = sn if use_sn else (lambda x: x)
        self.conv1 = nn.Sequential(
            wrap(nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size,
                           padding=padding, padding_mode=padding_mode)),
            activation,
        )
        self.conv2 = nn.Sequential(
            wrap(nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size,
                           padding=padding, padding_mode=padding_mode)),
            activation,
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.norm = norm
        self.norm1 = nn.BatchNorm1d(out_channel)
        self.norm2 = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        x = self.conv1(x)
        if self.norm:
            x = self.norm1(x)
        x = self.conv2(x)
        if self.norm:
            x = self.norm2(x)
        # returns (pooled, skip)
        return self.pool(x), x


class DecoderBlock(nn.Module):
    """ConvTranspose1d up-sampling block — mirrors Up_Block without time conditioning."""

    def __init__(self, in_channel, out_channel, kernel_size=3,
                 padding=1, padding_mode='circular', activation=nn.GELU(), norm=True, final=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(2 * in_channel, in_channel, kernel_size=kernel_size,
                      padding=padding, padding_mode=padding_mode),
            activation,
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, kernel_size=kernel_size,
                      padding=padding, padding_mode=padding_mode),
            activation,
        )
        self.upsample = nn.ConvTranspose1d(in_channel, out_channel, kernel_size=2, stride=2)
        self.norm = norm
        self.norm1 = nn.BatchNorm1d(in_channel)
        self.norm2 = nn.BatchNorm1d(in_channel)
        self.final = final

    def forward(self, x):
        x = self.conv1(x)
        if self.norm:
            x = self.norm1(x)
        x = self.conv2(x)
        if self.norm:
            x = self.norm2(x)
        if not self.final:
            x = self.upsample(x)
        return x


class MiddleBlock(nn.Module):
    """Bottleneck block — mirrors Middle_Block without time conditioning."""

    def __init__(self, in_channel, middle_channel, out_channel, kernel_size=3,
                 padding=1, padding_mode='circular', activation=nn.GELU(), norm=True,
                 use_sn: bool = False):
        super().__init__()
        wrap = sn if use_sn else (lambda x: x)
        self.conv1 = nn.Sequential(
            wrap(nn.Conv1d(in_channel, middle_channel, kernel_size=kernel_size,
                           padding=padding, padding_mode=padding_mode)),
            activation,
        )
        self.conv2 = nn.Sequential(
            wrap(nn.Conv1d(middle_channel, middle_channel, kernel_size=kernel_size,
                           padding=padding, padding_mode=padding_mode)),
            activation,
        )
        self.upsample = wrap(nn.ConvTranspose1d(middle_channel, out_channel, kernel_size=2, stride=2))
        self.norm = norm
        self.norm1 = nn.BatchNorm1d(middle_channel)
        self.norm2 = nn.BatchNorm1d(middle_channel)

    def forward(self, x):
        x = self.conv1(x)
        if self.norm:
            x = self.norm1(x)
        x = self.conv2(x)
        if self.norm:
            x = self.norm2(x)
        return self.upsample(x)


# ---------------------------------------------------------------------------
# Fully-connected helpers (identical to DDM cmpnts)
# ---------------------------------------------------------------------------

class _Combo(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None

    def forward(self, input):
        return self.model(input)


class LinearCombo(_Combo):
    """Regular fully connected layer combo."""
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(alpha),
        )


class MLP(nn.Module):
    """Fully connected network.

    Shape:
        - Input:  ``(N, in_features)``
        - Output: ``(N, out_features)``
    """
    def __init__(self, in_features: int, out_features: int, layer_width: list,
                 combo=LinearCombo):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.model = self._build_model(layer_width, combo)

    def forward(self, input):
        return self.model(input)

    def _build_model(self, layer_width, combo):
        model = nn.Sequential()
        for idx, (in_ftr, out_ftr) in enumerate(zip(
            [self.in_features] + layer_width,
            layer_width + [self.out_features],
        )):
            model.add_module(str(idx), combo(in_ftr, out_ftr))
        return model


class AirfoilConditional(nn.Module):
    """Embeds Bezier control points/weights to a fixed-size conditioning vector."""
    def __init__(self, in_channels: int, in_dim: int, out_features: int = 10,
                 mlp_layers: list = [96, 64, 32]):
        super().__init__()
        self.mlp = MLP(in_dim * in_channels, out_features, layer_width=mlp_layers)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, wcp: torch.Tensor):
        return self.mlp(self.flatten(wcp))


class AirfoilConditionalPerf(nn.Module):
    """Embeds control points + performance parameters to a conditioning vector."""
    def __init__(self, in_channels: int, in_dim: int, inp_paras_dim: int,
                 out_features: int = 10, mlp_layers: list = [96, 64, 32]):
        super().__init__()
        self.mlp = MLP(in_dim * in_channels + inp_paras_dim + 1, out_features,
                       layer_width=mlp_layers)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, wcp: torch.Tensor, inp_paras: torch.Tensor, alpha: torch.Tensor):
        x = self.flatten(wcp)
        x = torch.cat((x, inp_paras, alpha), dim=1)
        return self.mlp(x)


