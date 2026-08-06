"""Least-volume autoencoders, shared by the LVAE generators and latent metrics.

A least-volume autoencoder collapses unused latent dimensions rather than
letting them carry noise, leaving an *active* subspace whose width estimates the
design manifold's intrinsic dimension. That makes the same trained model useful
in two roles:

- as a **generator**, via `engiopt.generators.lvae_2d`, `plvae_2d`, and
  `constrained_plvae_2d`;
- as the **measuring instrument** for latent-space metrics, via
  `engiopt.evaluation.metrics.latent`.

Both roles read the same HuggingFace checkpoint package. Nothing here depends on
Weights & Biases.

Because a latent metric is only meaningful relative to the instrument that
produced it, an evaluation spec must pin the instrument's configuration; see
`load_lvae_encoder`.
"""

from engiopt.lvae.checkpoints import build_encoder
from engiopt.lvae.checkpoints import load_lvae_encoder
from engiopt.lvae.config import LVAEConfig
from engiopt.lvae.encode import decode_designs
from engiopt.lvae.encode import encode_active
from engiopt.lvae.encode import encode_designs
from engiopt.lvae.encode import get_active_mask
from engiopt.lvae.encode import latent_dim_of
from engiopt.lvae.encode import PrunedEncoder

__all__ = [
    "LVAEConfig",
    "PrunedEncoder",
    "build_encoder",
    "decode_designs",
    "encode_active",
    "encode_designs",
    "get_active_mask",
    "latent_dim_of",
    "load_lvae_encoder",
]
