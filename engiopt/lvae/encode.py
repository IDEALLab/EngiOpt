"""Encoding designs into the pruned latent space.

Latent-space metrics are only meaningful on the *active* subspace. Dynamic
pruning freezes collapsed dimensions at a constant, so those columns carry no
information about a design -- including them would dilute every distance with
axes that are identical for every sample.

`get_active_mask` is therefore the gate between raw latent codes and anything
that measures them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch as th
from torch import nn

if TYPE_CHECKING:
    import numpy.typing as npt

DEFAULT_BATCH_SIZE = 256


class PrunedEncoder(nn.Module):
    """An encoder with the LVAE pruning mask applied.

    Pruned dimensions are clamped to the frozen values they held when pruning
    fired, matching `LeastVolumeAE_DynamicPruning.encode`.

    Args:
        encoder: The raw encoder module.
        pruning_mask: Boolean tensor, `True` where a dimension was pruned.
        frozen_z: Latent values pruned dimensions are held at.
    """

    pruning_mask: th.Tensor
    frozen_z: th.Tensor

    def __init__(self, encoder: nn.Module, pruning_mask: th.Tensor, frozen_z: th.Tensor) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("pruning_mask", pruning_mask.bool())
        self.register_buffer("frozen_z", frozen_z)

    @property
    def latent_dim(self) -> int:
        """Total latent width, including pruned dimensions."""
        return int(self.pruning_mask.numel())

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Encode, then clamp pruned dimensions to their frozen values.

        Args:
            x: Input designs `(B, 1, H, W)`.

        Returns:
            Latent codes `(B, latent_dim)`.
        """
        z = self.encoder(x)
        # Out-of-place: an in-place write here would break autograd for any
        # caller that encodes under grad, and silently alias the caller's tensor.
        frozen = self.frozen_z.to(z.device, z.dtype).expand_as(z)
        return th.where(self.pruning_mask.to(z.device), frozen, z)


def latent_dim_of(encoder: nn.Module) -> int:
    """Infer an encoder's latent width.

    Args:
        encoder: A `PrunedEncoder` or a raw `Encoder2D`.

    Returns:
        The number of latent dimensions.

    Raises:
        AttributeError: If the width cannot be determined from the module.
    """
    if isinstance(encoder, PrunedEncoder):
        return encoder.latent_dim
    to_latent = getattr(encoder, "to_latent", None)
    if to_latent is not None and hasattr(to_latent, "out_channels"):
        return int(to_latent.out_channels)
    raise AttributeError(f"cannot infer latent_dim from {type(encoder).__name__}")


def get_active_mask(encoder: nn.Module) -> npt.NDArray[np.bool_]:
    """Return a boolean mask of active (unpruned) latent dimensions.

    Args:
        encoder: A `PrunedEncoder`, or a raw encoder that was never pruned.

    Returns:
        Boolean array of shape `(latent_dim,)`; `True` marks an active dimension.
        A raw encoder has no pruning information, so every dimension is active.
    """
    if isinstance(encoder, PrunedEncoder):
        return ~encoder.pruning_mask.detach().cpu().numpy().astype(bool)
    # A full-width mask, not a length-1 array: callers index `z[:, mask]`, and a
    # short mask would silently keep only the leading dimension.
    return np.ones(latent_dim_of(encoder), dtype=bool)


def encode_designs(
    encoder: nn.Module,
    designs: npt.NDArray,
    device: th.device | str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> npt.NDArray:
    """Encode designs into latent codes.

    Args:
        encoder: A trained LVAE encoder.
        designs: Designs of shape `(N, H, W)` or `(N, 1, H, W)`.
        device: Device to run encoding on.
        batch_size: Designs per forward pass.

    Returns:
        Latent codes of shape `(N, latent_dim)`.
    """
    encoder.eval()
    device = th.device(device) if isinstance(device, str) else device

    designs_t = th.as_tensor(np.asarray(designs)).float()
    if designs_t.ndim == 3:  # noqa: PLR2004
        designs_t = designs_t.unsqueeze(1)

    codes: list[npt.NDArray] = []
    with th.no_grad():
        for start in range(0, len(designs_t), batch_size):
            batch = designs_t[start : start + batch_size].to(device)
            codes.append(encoder(batch).cpu().numpy())
    return np.concatenate(codes, axis=0)


def decode_designs(
    decoder: nn.Module,
    latent_codes: npt.NDArray,
    device: th.device | str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> npt.NDArray:
    """Decode latent codes back into designs.

    Args:
        decoder: A trained LVAE decoder.
        latent_codes: Latent codes of shape `(N, latent_dim)`.
        device: Device to run decoding on.
        batch_size: Codes per forward pass.

    Returns:
        Designs of shape `(N, H, W)`.
    """
    decoder.eval()
    device = th.device(device) if isinstance(device, str) else device

    codes_t = th.as_tensor(np.asarray(latent_codes)).float()

    designs: list[npt.NDArray] = []
    with th.no_grad():
        for start in range(0, len(codes_t), batch_size):
            batch = codes_t[start : start + batch_size].to(device)
            designs.append(decoder(batch).squeeze(1).cpu().numpy())
    return np.concatenate(designs, axis=0)


def encode_active(
    encoder: nn.Module,
    designs: npt.NDArray,
    device: th.device | str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> npt.NDArray:
    """Encode designs and keep only the active latent dimensions.

    This is the form latent-space metrics consume: pruned columns are constant
    across every design and would only dilute distances.

    Args:
        encoder: A trained LVAE encoder.
        designs: Designs of shape `(N, H, W)` or `(N, 1, H, W)`.
        device: Device to run encoding on.
        batch_size: Designs per forward pass.

    Returns:
        Latent codes of shape `(N, n_active)`.
    """
    codes = encode_designs(encoder, designs, device, batch_size)
    return codes[:, get_active_mask(encoder)]
