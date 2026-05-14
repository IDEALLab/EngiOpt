"""
Encoder and decoder networks for the LAE.
    Contains:
        - LAEEncoder: maps BAE latents + conditions → deterministic z
        - LAEDecoder: maps z + conditions → BAE latents, AoA, eta_y, pressure, performance
          All Conv1d, ConvTranspose1d, and Linear layers in both encoder and decoder use
          spectral normalisation for Lipschitz control.
"""

import torch
from torch import nn
from torch.nn.utils import spectral_norm as sn

from .cmpnts import (
    EncoderBlock, MiddleBlock,
    MLP, LinearCombo, AirfoilConditional,
)
from .utils import convert_str_to_activ


class LAEEncoder(nn.Module):
    """Convolutional encoder: (z_opt, c) → z  (deterministic, Lipschitz+LV).

    Args:
        w_dim:            Number of spanwise slices (default 9).
        latent_channels:  BAE latent channels per slice (default 3).
        latent_length:    BAE latent sequence length (default 30).
        lae_latent_dim:   Dimension of the LAE latent z.
        c_dim:            Dimension of the scalar condition vector.
        c_dim_latent:     Projected condition dimension.
        down_channels:    Feature map widths for encoder blocks.
        middle_channel:   Bottleneck feature map width.
        c_net_hidden_layers: Hidden layer sizes for the condition MLP.
        c_net_activation: Activation string for the condition MLP.
        block_activation: Activation string for conv blocks.
        padding_mode:     Padding mode for Conv1d ('circular' or 'zeros').
        block_norms:      Whether to use batch norm in each block.
    """

    def __init__(
        self,
        w_dim: int = 9,
        latent_channels: int = 3,
        latent_length: int = 30,
        lae_latent_dim: int = 64,
        c_dim: int = 4,
        c_dim_latent: int = 16,
        down_channels: list = [64, 128, 256, 512],
        middle_channel: int = 256,
        c_net_hidden_layers: list = [32, 32],
        c_net_activation: str = 'GELU',
        block_activation: str = 'GELU',
        padding_mode: str = 'circular',
        block_norms: list = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        act = convert_str_to_activ(block_activation)
        c_act = convert_str_to_activ(c_net_activation)

        self.w_dim = w_dim
        self.latent_channels = latent_channels
        self.latent_length = latent_length
        self.lae_latent_dim = lae_latent_dim
        self.c_dim = c_dim
        self.c_dim_latent = c_dim_latent

        # Condition embedding: scalar c → c_dim_latent vector (SN on every Linear)
        c_layers = [c_dim] + c_net_hidden_layers + [c_dim_latent]
        self.c_net = nn.Sequential(*[
            nn.Sequential(sn(nn.Linear(c_layers[i], c_layers[i + 1])), c_act)
            for i in range(len(c_layers) - 1)
        ])

        # Input channels: w_dim slices of latent_channels each, plus c_dim_latent broadcast
        in_ch = w_dim * latent_channels + c_dim_latent

        if block_norms is None:
            block_norms = [True] * len(down_channels)

        # Encoder down-sampling blocks (SN on all Conv layers)
        self.down_blocks = nn.ModuleList()
        ch = in_ch
        for out_ch, norm in zip(down_channels, block_norms):
            self.down_blocks.append(
                EncoderBlock(ch, out_ch, padding_mode=padding_mode,
                             activation=act, norm=norm, use_sn=True)
            )
            ch = out_ch

        # Bottleneck (SN on all Conv layers)
        self.middle = MiddleBlock(
            ch, middle_channel, down_channels[-1],
            padding_mode=padding_mode, activation=act, use_sn=True,
        )

        # Global average pooling → deterministic latent (SN on Linear)
        pool_out = down_channels[-1]
        self.dropout = nn.Dropout(p=dropout)
        self.mu_head = sn(nn.Linear(pool_out, lae_latent_dim))

        # Learnable span-position embeddings: each of the w_dim slices gets a
        # per-channel additive bias so the encoder knows which slice is root/tip.
        self.span_embed = nn.Embedding(w_dim, latent_channels)

    def forward(
        self,
        z_opt: torch.Tensor,   # [B, w_dim, latent_channels, latent_length]
        c: torch.Tensor,       # [B, c_dim]
    ):
        B = z_opt.shape[0]

        # Add span-position bias to each slice so the encoder can distinguish
        # root from tip even when the input slices are identical.
        slice_idx = torch.arange(self.w_dim, device=z_opt.device)
        span_emb = self.span_embed(slice_idx)          # [w_dim, latent_channels]
        z_opt = z_opt + span_emb.unsqueeze(0).unsqueeze(-1)  # broadcast over B and ll

        # Flatten slice dimension: [B, w_dim * latent_channels, latent_length]
        x_in = z_opt.reshape(B, self.w_dim * self.latent_channels, self.latent_length)

        # Broadcast condition along sequence dimension
        c_emb = self.c_net(c)                                         # [B, c_dim_latent]
        c_emb = c_emb.unsqueeze(-1).expand(-1, -1, x_in.shape[-1])   # [B, c_dim_latent, L]
        x_in = torch.cat([x_in, c_emb], dim=1)                       # [B, in_ch, L]

        # Encode
        for block in self.down_blocks:
            x_in, _ = block(x_in)

        x_in = self.middle(x_in)

        # Global average pool → [B, pool_out]
        h = x_in.mean(dim=-1)

        mu = self.mu_head(self.dropout(h))  # [B, lae_latent_dim]
        return mu


class LAEDecoder(nn.Module):
    """Convolutional decoder with spectral-normalised layers:
    (z, c) → (z_opt_pred, alpha_pred, eta_y_pred, pressure_pred, perf_pred).

    Global-head architecture: all slice outputs are predicted jointly from a
    single flattened feature vector h_flat, then reshaped to [B, w_dim, ...].

    Spectral normalisation is applied to every Conv1d, ConvTranspose1d, and
    Linear layer in the decoder for Lipschitz control.

    Args:
        w_dim:            Number of spanwise slices (default 9).
        latent_channels:  BAE latent channels per slice (default 3).
        latent_length:    BAE latent sequence length (default 30).
        pressure_length:  Number of pressure sample points per slice (default 192).
        perf_dim:         Dimension of the performance target (default 2, i.e. cd, cl).
        lae_latent_dim:   Dimension of the LAE latent z.
        c_dim:            Dimension of the scalar condition vector.
        c_dim_latent:     Projected condition dimension.
        up_channels:      Feature map widths for decoder blocks.
        c_net_hidden_layers: Hidden layer sizes for the condition MLP.
        c_net_activation: Activation string for the condition MLP.
        block_activation: Activation string for conv blocks.
        padding_mode:     Padding mode for Conv1d.
        block_norms:      Whether to use batch norm in each block.
        base_length:      Sequence length at the start of decoding (before up-sampling).
    """

    def __init__(
        self,
        w_dim: int = 9,
        latent_channels: int = 3,
        latent_length: int = 30,
        pressure_length: int = 192,
        perf_dim: int = 2,
        lae_latent_dim: int = 64,
        c_dim: int = 4,
        c_dim_latent: int = 16,
        up_channels: list = [512, 256, 128, 64],
        c_net_hidden_layers: list = [32, 32],
        c_net_activation: str = 'GELU',
        block_activation: str = 'GELU',
        padding_mode: str = 'circular',
        block_norms: list = None,
        base_length: int = 2,
        use_pressure: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        act = convert_str_to_activ(block_activation)
        c_act = convert_str_to_activ(c_net_activation)

        self.w_dim = w_dim
        self.latent_channels = latent_channels
        self.latent_length = latent_length
        self.pressure_length = pressure_length
        self.perf_dim = perf_dim
        self.lae_latent_dim = lae_latent_dim
        self.c_dim = c_dim
        self.c_dim_latent = c_dim_latent
        self.base_length = base_length

        # Condition embedding (all Linear layers get SN)
        c_layers = [c_dim] + c_net_hidden_layers + [c_dim_latent]
        self.c_net = nn.Sequential(*[
            nn.Sequential(sn(nn.Linear(c_layers[i], c_layers[i + 1])), c_act)
            for i in range(len(c_layers) - 1)
        ])

        # Project z + c_emb to a spatial feature map [B, up_channels[0], base_length]
        self.z_proj = sn(nn.Linear(lae_latent_dim + c_dim_latent, up_channels[0] * base_length))

        if block_norms is None:
            block_norms = [True] * len(up_channels)

        # Decoder up-sampling blocks — every Conv1d / ConvTranspose1d gets SN
        self.up_blocks = nn.ModuleList()
        for i, (in_ch, norm) in enumerate(zip(up_channels, block_norms)):
            out_ch = up_channels[i + 1] if i + 1 < len(up_channels) else up_channels[-1]
            final = (i == len(up_channels) - 1)
            block = nn.Sequential(
                sn(nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1,
                             padding_mode=padding_mode)),
                act,
                nn.BatchNorm1d(out_ch) if norm else nn.Identity(),
                sn(nn.ConvTranspose1d(out_ch, out_ch, kernel_size=2, stride=2))
                if not final else nn.Identity(),
            )
            self.up_blocks.append(block)

        final_ch = up_channels[-1]

        # Output sequence length: base_length * 2^(n_blocks - 1) doublings
        out_seq_len = base_length * (2 ** (len(up_channels) - 1))
        flat_size = final_ch * out_seq_len

        self.dropout = nn.Dropout(p=dropout)

        # Global heads — all slice outputs predicted jointly from h_flat then reshaped.
        self.z_opt_head    = sn(nn.Linear(flat_size, w_dim * latent_channels * latent_length))
        self.eta_y_head    = sn(nn.Linear(flat_size, w_dim))
        self.alpha_head    = sn(nn.Linear(flat_size, 1))
        self.perf_head     = sn(nn.Linear(flat_size, perf_dim))
        self.use_pressure  = use_pressure
        if use_pressure:
            self.pressure_head = sn(nn.Linear(flat_size, w_dim * pressure_length))

    def forward(
        self,
        z: torch.Tensor,   # [B, lae_latent_dim]
        c: torch.Tensor,   # [B, c_dim]
    ):
        B = z.shape[0]

        c_emb = self.c_net(c)                       # [B, c_dim_latent]
        zc = torch.cat([z, c_emb], dim=-1)          # [B, lae_latent_dim + c_dim_latent]

        h = self.z_proj(zc)                         # [B, up_channels[0] * base_length]
        h = h.reshape(B, -1, self.base_length)      # [B, up_channels[0], base_length]

        for block in self.up_blocks:
            h = block(h)

        # Single global feature vector
        h_flat = self.dropout(h.reshape(B, -1))     # [B, flat_size]

        alpha_pred = self.alpha_head(h_flat)                                                          # [B, 1]
        perf_pred  = self.perf_head(h_flat)                                                           # [B, perf_dim]
        z_opt_pred = self.z_opt_head(h_flat).reshape(B, self.w_dim, self.latent_channels, self.latent_length)
        eta_y_pred = self.eta_y_head(h_flat).reshape(B, self.w_dim, 1)
        if self.use_pressure:
            pressure_pred = self.pressure_head(h_flat).reshape(B, self.w_dim, self.pressure_length)
        else:
            pressure_pred = torch.zeros(B, self.w_dim, self.pressure_length, device=z.device)

        return z_opt_pred, alpha_pred, eta_y_pred, pressure_pred, perf_pred


# ---------------------------------------------------------------------------
# Backward-compat aliases (checkpoints created before the rename still load)
# ---------------------------------------------------------------------------
LVAEEncoder = LAEEncoder
LVAEDecoder = LAEDecoder
