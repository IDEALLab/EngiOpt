"""Shared `Generator` behaviour for the LVAE family.

All three LVAE variants sample the same way, and the reasoning is worth stating
once. An autoencoder has no prior to draw from, so "generate a design" needs a
distribution over latent codes supplied from somewhere. Volume regularization
leaves a compact active subspace, and the empirical distribution of *training*
codes inside it is the only region the decoder was ever fit on.

So: fit a Gaussian to the training set's active-dimension codes at load time,
sample from that, and hold pruned dimensions at the frozen values the encoder
itself would emit. Drawing from a unit Gaussian over all `latent_dim`
dimensions instead would place most samples far outside the trained region and
put mass on axes that carry no information at all.

Concrete adapters supply only their weights filename and the class that
rebuilds their decoder.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import torch as th

from engiopt.lvae.checkpoints import build_encoder
from engiopt.lvae.components import TrueSNDecoder2D
from engiopt.lvae.config import LVAEConfig
from engiopt.lvae.encode import encode_designs
from engiopt.lvae.encode import get_active_mask

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint

LATENT_FIT_SAMPLES = 512
"""Training designs encoded to estimate the latent distribution."""

COVARIANCE_JITTER = 1e-6
"""Added to the covariance diagonal; near-pruned dimensions can be singular."""


def fit_latent_gaussian(
    encoder: th.nn.Module,
    problem: Problem,
    device: th.device,
    n_samples: int = LATENT_FIT_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the training-set latent distribution over active dimensions.

    Args:
        encoder: The trained encoder, pruning wrapper included.
        problem: Problem whose training split supplies the designs.
        device: Device to encode on.
        n_samples: Training designs to encode.

    Returns:
        `(mean, covariance, active_mask)`, where mean and covariance describe
        only the active subspace.
    """
    active_mask = get_active_mask(encoder)
    designs = np.asarray(problem.dataset["train"]["optimal_design"][:n_samples], dtype=np.float32)
    codes = encode_designs(encoder, designs, device)[:, active_mask]

    mean = codes.mean(axis=0)
    covariance = np.cov(codes, rowvar=False) + COVARIANCE_JITTER * np.eye(codes.shape[1])
    return mean, covariance, active_mask


class LVAESamplingMixin:
    """Latent sampling shared by every LVAE adapter.

    Concrete adapters set `weights_filename` and call `prepare` inside `build`.
    """

    weights_filename: str

    def __init__(
        self,
        decoder: th.nn.Module,
        latent_mean: np.ndarray,
        latent_cov: np.ndarray,
        active_mask: np.ndarray,
        frozen_z: np.ndarray,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.decoder = decoder
        self.latent_mean = latent_mean
        self.latent_cov = latent_cov
        self.active_mask = active_mask
        self.frozen_z = frozen_z

    @classmethod
    def prepare(
        cls,
        resolved: ResolvedCheckpoint,
        problem: Problem,
        device: th.device,
    ) -> dict[str, Any]:
        """Rebuild the decoder and fit the latent distribution.

        Args:
            resolved: The downloaded checkpoint package.
            problem: Problem supplying the design shape and training split.
            device: Device to place the decoder on.

        Returns:
            Keyword arguments for the adapter's constructor.
        """
        design_shape = tuple(problem.design_space.shape)
        config = LVAEConfig.from_run_config(resolved.run_config, design_shape)  # type: ignore[arg-type]
        checkpoint = th.load(resolved.files[cls.weights_filename], map_location=device, weights_only=False)

        decoder = TrueSNDecoder2D(
            latent_dim=config.latent_dim,
            design_shape=config.design_shape,
            lipschitz_scale=config.decoder_lipschitz_scale,
        )
        decoder.load_state_dict(checkpoint["decoder"])
        decoder.to(device).eval()

        encoder = build_encoder(checkpoint, config, device)
        latent_mean, latent_cov, active_mask = fit_latent_gaussian(encoder, problem, device)

        frozen = checkpoint.get("pruning_frozen_z")
        frozen_z = frozen.detach().cpu().numpy() if frozen is not None else np.zeros(config.latent_dim, dtype=np.float32)

        return {
            "decoder": decoder,
            "latent_mean": latent_mean,
            "latent_cov": latent_cov,
            "active_mask": active_mask,
            "frozen_z": frozen_z,
        }

    def sample_latent_and_decode(self, n: int) -> th.Tensor:
        """Draw `n` latent codes from the fitted distribution and decode them.

        Args:
            n: Number of designs to generate.

        Returns:
            Generated designs, shape `(n, 1, H, W)`.
        """
        rng = np.random.default_rng(int(th.randint(0, 2**31 - 1, (1,)).item()))
        active = rng.multivariate_normal(self.latent_mean, self.latent_cov, size=n)

        z = np.tile(self.frozen_z.astype(np.float64), (n, 1))
        z[:, self.active_mask] = active

        device = next(self.decoder.parameters()).device
        return self.decoder(th.from_numpy(z).float().to(device))
