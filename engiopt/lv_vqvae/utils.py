"""Architectural blocks and utilities for LV-VQVAE.

This module contains:
- An "Online Clustered Codebook" vector quantizer (used by VQVAE/CVQVAE).
- Standard encoder/decoder building blocks (residual, attention, up/downsample, etc.).
- LV + dynamic pruning utilities for latent channel dimension reduction (n_z).
- A minimal GPT-2 style Transformer (nanoGPT-style) used for Stage 2 token modeling.

Notes for this version:
- Discriminator/adversarial training is removed.
- Perceptual loss (LPIPS/VGG) is removed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as f
from torch.nn.utils.parametrizations import spectral_norm


def _entropy_from_probs(p: th.Tensor) -> th.Tensor:
    """Shannon entropy (nats) of a probability vector."""
    p = p.clamp_min(1e-12)
    return -(p * p.log()).sum()


def set_data_variance(x: th.Tensor) -> None:
    """Set the data variance from training data for NMSE computation.

    Should be called once before training with the full training dataset
    or a representative sample.

    Args:
        x: Training data tensor of shape (N, ...).
    """
    var = x.var().item()
    if var < 1e-10:  # noqa: PLR2004
        var = 1.0  # Fallback for constant data
    return th.tensor(var, device=x.device)  # also not included from orig. implementation: self._data_var_set = True


def token_stats_from_indices(indices: th.Tensor, *, vocab_size: int) -> dict[str, float]:
    """Token usage stats from flat index tensor (no pruning mask assumed yet).

    Returns:
        token_perplexity: exp(H) (effective number of codes used)
        token_perplexity_frac: token_perplexity / vocab_size (0..1)
        token_usage_frac: unique_used / vocab_size (0..1)
    """
    idx = indices.view(-1).to(dtype=th.long)
    counts = th.bincount(idx, minlength=vocab_size).float()
    total = counts.sum().clamp_min(1.0)
    probs = counts / total
    entropy = _entropy_from_probs(probs)
    ppl = float(entropy.exp().item())
    unique = int((counts > 0).sum().item())
    usage_frac = float(unique / max(1, vocab_size))
    ppl_frac = float(ppl / max(1, vocab_size))
    return {
        "token_perplexity": ppl,
        "token_perplexity_frac": ppl_frac,
        "token_usage_frac": usage_frac,
    }


def loss_vol(z: th.Tensor, *, active_mask: th.Tensor, frozen_std: th.Tensor, eta: float) -> th.Tensor:
    """Volume loss = exp(mean(log(std_i + eta))) over latent dims.

    Assumes z is (B, C, H, W): std is over (B, H, W) per channel C.
    """
    # Pruned dims use frozen std (captured at prune time),
    # active dims use current std. This makes pruning volume-neutral.
    s = frozen_std.clone()
    if active_mask.any():
        s[active_mask] = z[:, active_mask, :, :].std(dim=(0, 2, 3))

    return th.exp(th.log(s + float(eta)).mean())


def make_sorted_std_plot(
    *,
    zstd: th.Tensor,
    title: str = "Sorted EMA STD. per Dimension",
):
    """Creates a figure of sorted dimension stds, including zeroed out (pruned) ones.

    zstd: shape [latent_dim] (EMA or batch std)
    active_mask: shape [latent_dim] bool
    Produces a matplotlib figure showing sorted stds for all dims,
    with active vs pruned indicated.
    """
    zstd_cpu = zstd.detach().float().cpu().numpy()

    order = np.argsort(zstd_cpu)[::-1]  # DESCENDING
    sorted_std = zstd_cpu[order]

    x = np.arange(len(sorted_std))

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=300)

    # Log scale can't display zeros/negatives; clamp for plotting only
    eps = 1e-12
    sorted_std_plot = np.clip(sorted_std, eps, None)

    # Bar plot for all dims
    ax.bar(x, sorted_std_plot)

    ax.set_title(title)
    ax.set_xlabel("dimension index (sorted by std, desc)")
    ax.set_ylabel("std")
    ax.grid(visible=True, linewidth=0.3, alpha=0.5)

    # log y-axis with fixed limits 1e-1 to 1e1
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e1)

    plt.tight_layout()
    return fig


class Codebook(nn.Module):
    """Improved version over vector quantizer, with the dynamic initialization for the unoptimized "dead" vectors.

    Parameters:
        num_codebook_vectors (int): number of codebook entries
        latent_dim (int): dimensionality of codebook entries
        beta (float): weight for the commitment loss
        decay (float): decay for the moving average of code usage
        distance (str): distance type for looking up the closest code
        anchor (str): anchor sampling methods
        first_batch (bool): if true, the offline version of the model
        contras_loss (bool): if true, use the contras_loss to further improve the performance
        init (bool): if true, the codebook has been initialized
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        num_codebook_vectors: int,
        latent_dim: int,
        beta: float = 0.01,  # Default: 0.25. Set to a lower value for the LV-VQVAE warm-up.
        decay: float = 0.99,
        distance: str = "cos",
        anchor: str = "probrandom",
        first_batch: bool = False,
        contras_loss: bool = False,
        init: bool = False,
    ):
        super().__init__()

        self.num_embed = num_codebook_vectors
        self.embed_dim = latent_dim
        self.beta = beta
        self.decay = decay
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.init = init

        self.pool = FeaturePool(self.num_embed, self.embed_dim)
        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.register_buffer("embed_prob", th.zeros(self.num_embed))

    def forward(self, z: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = z.view(-1, self.embed_dim)

        # Normalize inputs and weights to unit sphere immediately.
        # This prevents collapse by forcing the model to learn angular features.
        z_flattened = f.normalize(z_flattened, dim=1)
        z = z_flattened.view(z.shape)  # Update reference for loss calc
        self.embedding.weight.data = f.normalize(self.embedding.weight.data, dim=1)

        # clculate the distance
        if self.distance == "l2":
            # negative squared distance (so argmax == nearest)
            d = (
                -th.sum(z_flattened.detach() ** 2, dim=1, keepdim=True)
                - th.sum(self.embedding.weight**2, dim=1)
                + 2 * th.einsum("bd,dn->bn", z_flattened.detach(), rearrange(self.embedding.weight, "n d-> d n"))
            )
        elif self.distance == "cos":
            # cosine similarity (argmax == nearest)
            normed_z_flattened = f.normalize(z_flattened, dim=1).detach()
            normed_codebook = f.normalize(self.embedding.weight, dim=1)
            d = th.einsum("bd,dn->bn", normed_z_flattened, rearrange(normed_codebook, "n d -> d n"))

        #  TODO: Prevent selecting pruned/inactive codebook entries.
        #  This is required for consistency with downstream transformer masking of inactive tokens.

        # encoding
        sort_distance, indices = d.sort(dim=1)
        # look up the closest point for the indices
        encoding_indices = indices[:, -1]
        encodings = th.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # quantize and unflatten
        z_q = th.matmul(encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        loss = self.beta * th.mean((z_q.detach() - z) ** 2) + th.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()
        # count
        avg_probs = th.mean(encodings, dim=0)
        perplexity = th.exp(-th.sum(avg_probs * th.log(avg_probs + 1e-10)))
        min_encodings = encodings

        # online clustered reinitialization for unoptimized points
        if self.training:
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha=1 - self.decay)
            # running average updates
            if self.anchor in ["closest", "random", "probrandom"] and (not self.init):
                # closest sampling
                if self.anchor == "closest":
                    sort_distance, indices = d.sort(dim=0)
                    random_feat = z_flattened.detach()[indices[-1, :]]
                # feature pool based random sampling
                elif self.anchor == "random":
                    random_feat = self.pool.query(z_flattened.detach())
                # probabilitical based random sampling
                elif self.anchor == "probrandom":
                    norm_distance = f.softmax(d.t(), dim=1)
                    prob = th.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                # decay parameter based on the average usage
                decay = (
                    th.exp(-(self.embed_prob * self.num_embed * 10) / (1 - self.decay) - 1e-3)
                    .unsqueeze(1)
                    .repeat(1, self.embed_dim)
                )
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True
            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = d.sort(dim=0)
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0) / self.num_embed)) :, :].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[: int(sort_distance.size(0) * 1 / 2), :]
                dis = th.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = f.cross_entropy(dis, th.zeros((dis.size(0),), dtype=th.long, device=dis.device))
                loss += contra_loss

        return z_q, encoding_indices, loss, min_encodings, perplexity


class FeaturePool:
    """Implements a feature buffer that stores previously encoded features.

    This buffer enables us to initialize the codebook using a history of generated features rather than the ones produced by the latest encoders.

    Parameters:
        pool_size (int): the size of feature buffer
        dim (int): the dimension of each feature
    """

    def __init__(self, pool_size: int, dim: int = 64):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (th.rand((pool_size, dim)) * 2 - 1) / pool_size

    def query(self, features: th.Tensor) -> th.Tensor:
        """Return features from the pool."""
        self.features = self.features.to(features.device)
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size:  # if the batch size is large enough, directly update the whole codebook
                random_feat_id = th.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features : num] = features
                self.nums_features = num
        elif features.size(0) > int(self.pool_size):
            random_feat_id = th.randint(0, features.size(0), (int(self.pool_size),))
            self.features = features[random_feat_id]
        else:
            random_id = th.randperm(self.pool_size)
            self.features[random_id[: features.size(0)]] = features

        return self.features


class GroupNorm(nn.Module):
    """Group Normalization block to be used in VQVAE Encoder.

    Parameters:
        channels (int): number of channels in the input feature map
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.gn(x)


class Swish(nn.Module):
    """Swish activation function to be used in VQVAE Encoder."""

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x * th.sigmoid(x)


class ResidualBlock(nn.Module):
    """Residual block to be used in VQVAE Encoder.

    Parameters:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        return x + self.block(x)


class DownSampleBlock(nn.Module):
    """Down-sampling block to be used in VQVAE Encoder.

    Parameters:
        channels (int): number of channels in the input feature map
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: th.Tensor) -> th.Tensor:
        pad = (0, 1, 0, 1)
        x = f.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class NonLocalBlock(nn.Module):
    """Non-local attention block to be used in VQVAE Encoder.

    Parameters:
        channels (int): number of channels in the input feature map
    """

    def __init__(self, channels: int):
        super().__init__()
        self.in_channels = channels

        self.gn = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: th.Tensor) -> th.Tensor:
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        attn = th.bmm(q, k)
        attn = attn * (int(c) ** (-0.5))
        attn = f.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        a = th.bmm(v, attn)
        a = a.reshape(b, c, h, w)

        return x + a


class LinearCombo(nn.Module):
    """Regular fully connected layer combo for the CVQVAE if enabled.

    Parameters:
        in_features (int): number of input features
        out_features (int): number of output features
        alpha (float): negative slope for LeakyReLU
    """

    def __init__(self, in_features: int, out_features: int, alpha: float = 0.2):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(in_features, out_features), nn.LeakyReLU(alpha))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)


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
        self.conv = spectral_norm(
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

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass through the layer."""
        return self.activation(self.bn(self.conv(x)))


###########################################
########## GPT-2 BASE CODE BELOW ##########
###########################################
class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch lacks bias=False support)."""

    def __init__(self, ndim: int, *, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(th.ones(ndim))
        self.bias = nn.Parameter(th.zeros(ndim)) if bias else None

    def forward(self, x: th.Tensor) -> th.Tensor:
        return f.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Causal self-attention with FlashAttention fallback when unavailable."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(f, "scaled_dot_product_attention")
        if not self.flash:
            warnings.warn(
                "Falling back to non-flash attention; PyTorch >= 2.0 enables FlashAttention.",
                stacklevel=2,
            )
            self.register_buffer(
                "bias",
                th.tril(th.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, t, c = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)

        if self.flash:
            y = f.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :t, :t] == 0, float("-inf"))
            att = f.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Feed-forward block used inside Transformer blocks."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Transformer block: LayerNorm -> Self-Attn -> residual; LayerNorm -> MLP -> residual."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 uses 50257; padded to multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # GPT-2 uses biases in Linear/LayerNorm


class GPT(nn.Module):
    """Minimal GPT-2 style Transformer."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                th.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            th.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                th.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            th.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: th.Tensor,
        targets: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor | None]:
        """Forward pass returning logits and optional cross-entropy loss."""
        device = idx.device
        _, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}; block size is {self.config.block_size}"
        pos = th.arange(0, t, dtype=th.long, device=device)

        tok_emb = self.transformer["wte"](idx)
        pos_emb = self.transformer["wpe"](pos)
        x = self.transformer["drop"](tok_emb + pos_emb)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)

        logits = self.lm_head(x)
        loss: th.Tensor | None
        if targets is not None:
            loss = f.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None
        return logits, loss


###########################################
########## GPT-2 BASE CODE ABOVE ##########
###########################################




###########################################
########## LV-VQVAE BLOCKS BELOW ##########
###########################################

class GroupSort(nn.Module):
    """Provably 1-Lipschitz activation function.

    Splits channels into groups and sorts them to preserve gradient norm.
    Works for both 4D (Conv) and 2D (Linear) inputs.
    """
    def __init__(self, group_size=2):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: th.Tensor) -> th.Tensor:
        # Handle 4D Input (Conv2d): (B, C, H, W)
        if x.dim() == 4:  # noqa: PLR2004
            b, c, h, w = x.shape
            if c % self.group_size != 0:
                raise ValueError(f"Channels ({c}) must be divisible by group_size ({self.group_size})")
            # Reshape -> Sort -> Flatten back
            x = x.reshape(b, c // self.group_size, self.group_size, h, w)
            x, _ = x.sort(dim=2)
            return x.reshape(b, c, h, w)

        # Handle 2D Input (Linear/MLP): (B, Features)
        if x.dim() == 2:  # noqa: PLR2004
            b, c = x.shape
            if c % self.group_size != 0:
                raise ValueError(f"Features ({c}) must be divisible by group_size ({self.group_size})")
            # Reshape -> Sort -> Flatten back
            x = x.reshape(b, c // self.group_size, self.group_size)
            x, _ = x.sort(dim=2)
            return x.reshape(b, c)

        raise ValueError(f"GroupSort expects 2D or 4D input, got {x.dim()}D")


class ScaledTanh(nn.Module):
    """Scaled Tanh activation.

    Ensures outputs are bounded [-scale, scale] and strictly Lipschitz < 1.
    """
    def __init__(self, scale=0.999):
        super().__init__()
        self.scale = scale

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.scale * th.tanh(x)


class TrueSNUpsample(nn.Module):
    """1-Lipschitz Upsampling using SN-TransposedConv."""
    def __init__(self, channels: int, n_power_iterations: int = 1):
        super().__init__()
        self.up_conv = spectral_norm(
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
            n_power_iterations=n_power_iterations
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.up_conv(x)


class TrueSNResidualBlock(nn.Module):
    """1-Lipschitz Residual Block.

    Uses GroupSort and scales residual branch by 1/sqrt(2).
    """
    def __init__(self, in_channels: int, out_channels: int, group_size: int = 2, n_power_iterations: int = 1, dilation: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        padding = dilation

        # Pre-activation style: GroupSort -> SNConv -> GroupSort -> SNConv
        self.block = nn.Sequential(
            GroupSort(group_size=group_size),
            spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, padding_mode="reflect"),
                n_power_iterations=n_power_iterations
            ),
            GroupSort(group_size=group_size),
            spectral_norm(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, padding_mode="reflect"),
                n_power_iterations=n_power_iterations
            )
        )

        if in_channels != out_channels:
            self.shortcut = spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                n_power_iterations=n_power_iterations
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: th.Tensor) -> th.Tensor:
        # Scale to maintain unit variance / Lipschitz bound
        return (self.shortcut(x) + self.block(x)) / (2**0.5)

###########################################
########## LV-VQVAE BLOCKS ABOVE ##########
###########################################
