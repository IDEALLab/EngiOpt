"""Lightweight stand-ins for expensive objects, shared across test modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch as th

from engiopt.lvae.checkpoints import LoadedLVAE
from engiopt.lvae.components import Encoder2D
from engiopt.lvae.components import TrueSNDecoder2D
from engiopt.lvae.config import LVAEConfig
from engiopt.lvae.encode import PrunedEncoder

if TYPE_CHECKING:
    import numpy.typing as npt

STUB_LATENT_DIM = 4


def stub_lvae(
    design_shape: tuple[int, ...],
    *,
    latent_dim: int = STUB_LATENT_DIM,
    pruned_dims: list[int] | None = None,
) -> LoadedLVAE:
    """An untrained LVAE with both halves, for tests about plumbing not weights.

    Args:
        design_shape: The problem's design shape.
        latent_dim: Latent width.
        pruned_dims: Dimensions to mark pruned; `None` leaves the encoder raw.

    Returns:
        A `LoadedLVAE` whose `resolved` is `None` -- nothing under test reads it.
    """
    shape: tuple[int, int] = (int(design_shape[0]), int(design_shape[1]))
    encoder: th.nn.Module = Encoder2D(latent_dim=latent_dim, design_shape=shape)

    if pruned_dims:
        mask = th.zeros(latent_dim, dtype=th.bool)
        mask[pruned_dims] = True
        encoder = PrunedEncoder(encoder, mask, th.zeros(latent_dim))

    lvae = LoadedLVAE(
        encoder=encoder.eval(),
        config=LVAEConfig(latent_dim=latent_dim, perf_dim=latent_dim, resize_dimensions=(100, 100), design_shape=shape),
        resolved=None,  # type: ignore[arg-type]
    )
    # `LoadedLVAE.decoder` is a cached_property, so assigning to it fills the
    # cache and nothing ever tries to rebuild one from published weights.
    lvae.decoder = TrueSNDecoder2D(latent_dim=latent_dim, design_shape=shape).eval()
    return lvae


def stub_designs(n: int, design_shape: tuple[int, ...], seed: int = 0) -> npt.NDArray:
    """Random designs of the right shape and dtype."""
    import numpy as np

    return np.random.default_rng(seed).random((n, *design_shape)).astype(np.float32)


class StubConditions:
    """A minimal stand-in for a HuggingFace conditions split."""

    def __init__(self, n: int, keys: tuple[str, ...] = ("volfrac", "rmin"), seed: int = 0) -> None:
        import numpy as np

        rng = np.random.default_rng(seed)
        self.column_names = list(keys)
        self._columns = {key: rng.random(n) for key in keys}

    def __getitem__(self, key: object) -> object:
        if isinstance(key, str):
            return self._columns[key]
        return {name: column[key] for name, column in self._columns.items()}
