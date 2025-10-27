"""Architectural blocks and other utils for the Vector Quantized Generative Adversarial Network (VQGAN)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import warnings

from einops import rearrange
import requests
import torch as th
from torch import nn
from torch.nn import functional as f
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights
import tqdm

# URL and checkpoint for LPIPS model
URL_MAP = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}

CKPT_MAP = {"vgg_lpips": "vgg.pth"}


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
        beta: float = 0.25,
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

        # clculate the distance
        if self.distance == "l2":
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = (
                -th.sum(z_flattened.detach() ** 2, dim=1, keepdim=True)
                - th.sum(self.embedding.weight**2, dim=1)
                + 2 * th.einsum("bd, dn-> bn", z_flattened.detach(), rearrange(self.embedding.weight, "n d-> d n"))
            )
        elif self.distance == "cos":
            # cosine distances from z to embeddings e_j
            normed_z_flattened = f.normalize(z_flattened, dim=1).detach()
            normed_codebook = f.normalize(self.embedding.weight, dim=1)
            d = th.einsum("bd,dn->bn", normed_z_flattened, rearrange(normed_codebook, "n d -> d n"))

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


class Discriminator(nn.Module):
    """PatchGAN-style discriminator.

    Adapted from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538
    This assumes we never use a discriminator for the CVQGAN, since it is generally a much simpler model.

    Parameters:
        num_filters_last: Number of filters in the last conv layer.
        n_layers: Number of convolutional layers.
        image_channels: Number of channels in the input image.
    """

    def __init__(self, num_filters_last: int = 64, n_layers: int = 3, image_channels: int = 1):
        super().__init__()

        # Convolutional backbone (PatchGAN)
        layers: list[nn.Module] = [
            nn.Conv2d(image_channels, num_filters_last, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2**i, 8)
            layers += [
                nn.Conv2d(
                    num_filters_last * num_filters_mult_last,
                    num_filters_last * num_filters_mult,
                    kernel_size=4,
                    stride=2 if i < n_layers else 1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m: nn.Module) -> None:
        """Custom weight initialization (DCGAN-style)."""
        classname = m.__class__.__name__
        if "Conv" in classname:
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        elif "BatchNorm" in classname:
            nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass with optional CVQGAN adapter."""
        return self.model(x)


class GroupNorm(nn.Module):
    """Group Normalization block to be used in VQGAN Encoder and Decoder.

    Parameters:
        channels (int): number of channels in the input feature map
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.gn(x)


class Swish(nn.Module):
    """Swish activation function to be used in VQGAN Encoder and Decoder."""

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x * th.sigmoid(x)


class ResidualBlock(nn.Module):
    """Residual block to be used in VQGAN Encoder and Decoder.

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


class UpSampleBlock(nn.Module):
    """Up-sampling block to be used in VQGAN Decoder.

    Parameters:
        channels (int): number of channels in the input feature map
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = f.interpolate(x, scale_factor=2.0)
        return self.conv(x)


class DownSampleBlock(nn.Module):
    """Down-sampling block to be used in VQGAN Encoder.

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
    """Non-local attention block to be used in VQGAN Encoder and Decoder.

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
    """Regular fully connected layer combo for the CVQGAN if enabled.

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


class ScalingLayer(nn.Module):
    """Channel-wise affine normalization used by LPIPS."""

    def __init__(self):
        super().__init__()
        self.register_buffer("shift", th.tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", th.tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, x: th.Tensor) -> th.Tensor:
        return (x - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """1x1 conv with dropout (per-layer LPIPS linear head)."""

    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)


class VGG16(nn.Module):
    """Torchvision VGG16 feature extractor sliced at LPIPS tap points."""

    def __init__(self):
        super().__init__()
        vgg_feats = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        blocks = [vgg_feats[i] for i in range(30)]
        self.slice1 = nn.Sequential(*blocks[0:4])  # relu1_2
        self.slice2 = nn.Sequential(*blocks[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*blocks[9:16])  # relu3_3
        self.slice4 = nn.Sequential(*blocks[16:23])  # relu4_3
        self.slice5 = nn.Sequential(*blocks[23:30])  # relu5_3
        self.requires_grad_(requires_grad=False)

    def forward(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return (h1, h2, h3, h4, h5)


class GreyscaleLPIPS(nn.Module):
    """LPIPS for greyscale/topological data with optional 'raw' aggregation.

    ``use_raw=True`` is often preferable for non-natural images since learned
    linear heads are tuned on natural RGB photos.

    Parameters:
        use_raw: If True, average raw per-layer squared diffs (no linear heads).
        clamp_output: Clamp the final loss to ``>= 0``.
        robust_clamp: Clamp inputs to [0, 1] before feature extraction.
        warn_on_clamp: If True, log warnings when inputs fall outside [0, 1].
        freeze: If True, disables grads on all params.
        ckpt_name: Key in URL_MAP/CKPT_MAP for loading LPIPS heads.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        use_raw: bool = True,
        clamp_output: bool = False,
        robust_clamp: bool = True,
        warn_on_clamp: bool = False,
        freeze: bool = True,
        ckpt_name: str = "vgg_lpips",
    ):
        super().__init__()
        self.use_raw = use_raw
        self.clamp_output = clamp_output
        self.robust_clamp = robust_clamp
        self.warn_on_clamp = warn_on_clamp

        self.scaling_layer = ScalingLayer()
        self.channels = (64, 128, 256, 512, 512)
        self.vgg = VGG16()
        self.linears = nn.ModuleList([NetLinLayer(c) for c in self.channels])

        self._load_from_pretrained(name=ckpt_name)
        if freeze:
            self.requires_grad_(requires_grad=False)

    def forward(self, real_x: th.Tensor, fake_x: th.Tensor) -> th.Tensor:
        """Compute greyscale-aware LPIPS distance between two batches."""
        if self.robust_clamp:
            real_x = th.clamp(real_x, 0.0, 1.0)
            fake_x = th.clamp(fake_x, 0.0, 1.0)

        # Promote greyscale -> RGB for VGG features
        if real_x.shape[1] == 1:
            real_x = real_x.repeat(1, 3, 1, 1)
        if fake_x.shape[1] == 1:
            fake_x = fake_x.repeat(1, 3, 1, 1)

        fr = self.vgg(self.scaling_layer(real_x))
        ff = self.vgg(self.scaling_layer(fake_x))
        diffs = [(self._norm_tensor(a) - self._norm_tensor(b)) ** 2 for a, b in zip(fr, ff)]

        if self.use_raw:
            parts = [self._spatial_average(d).mean(dim=1, keepdim=True) for d in diffs]
        else:
            parts = [self._spatial_average(self.linears[i](d)) for i, d in enumerate(diffs)]

        loss = th.stack(parts, dim=0).sum()
        if self.clamp_output:
            loss = th.clamp(loss, min=0.0)
        return loss

    # Helpers
    @staticmethod
    def _norm_tensor(x: th.Tensor) -> th.Tensor:
        """L2-normalize channels per spatial location: BxCxHxW -> BxCxHxW."""
        norm = th.sqrt(th.sum(x**2, dim=1, keepdim=True))
        return x / (norm + 1e-10)

    @staticmethod
    def _spatial_average(x: th.Tensor) -> th.Tensor:
        """Average over spatial dimensions with dims kept: BxCxHxW -> BxCx1x1."""
        return x.mean(dim=(2, 3), keepdim=True)

    def _load_from_pretrained(self, *, name: str) -> None:
        """Load LPIPS linear heads (and any required buffers) from a checkpoint."""
        ckpt = self._get_ckpt_path(name, "vgg_lpips")
        state_dict = th.load(ckpt, map_location=th.device("cpu"), weights_only=True)
        self.load_state_dict(state_dict, strict=False)

    @staticmethod
    def _download(url: str, local_path: str, *, chunk_size: int = 1024) -> None:
        """Stream a file to disk with a progress bar."""
        os.makedirs(os.path.split(local_path)[0], exist_ok=True)
        with requests.get(url, stream=True, timeout=10) as r:
            total_size = int(r.headers.get("content-length", 0))
            with tqdm.tqdm(total=total_size, unit="B", unit_scale=True) as pbar, open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(len(data))

    def _get_ckpt_path(self, name: str, root: str) -> str:
        """Return local path to a pretrained LPIPS checkpoint; download if missing."""
        assert name in URL_MAP, f"Unknown LPIPS checkpoint name: {name!r}"
        path = os.path.join(root, CKPT_MAP[name])
        if not os.path.exists(path):
            self._download(URL_MAP[name], path)
        return path


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
