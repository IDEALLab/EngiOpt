"""`Generator` adapter for the constrained performance-predicting LVAE.

An LVAE is an autoencoder, not a latent-variable model with a prior, so there is
no distribution to sample from a priori. Volume regularization does, however,
leave a compact active subspace, and the empirical distribution of training
codes inside it is what "plausible latent code" means for this model. This
adapter therefore fits a Gaussian to the training set's active-dimension codes
at load time and samples from that, holding pruned dimensions at the frozen
values the encoder itself would produce.

Sampling any other way -- a unit Gaussian over all `latent_dim` dimensions, say
-- would place most draws far outside the region the decoder was ever trained
on, and would put mass on pruned axes that carry no information at all.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import torch as th

from engiopt.core import ConditionBatch
from engiopt.core import Generator
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
"""Added to the covariance diagonal; pruned-adjacent dimensions can be singular."""


class ConstrainedPerfLVAE2D(Generator):
    """Constrained performance-predicting least-volume autoencoder."""

    algo_id = "constrained_plvae_2d"

    conditional = False
    """The decoder is unconditional unless trained with `--conditional-decoder`."""

    design_kinds = ("2d",)

    checkpoint_files = ("constrained_plvae.pth",)

    primary_state_key = "decoder"

    output_clip: tuple[Any, Any] | None = (1e-3, 1.0)
    """The decoder emits unbounded values so its Lipschitz bound holds exactly."""

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
    def build(
        cls,
        resolved: ResolvedCheckpoint,
        problem: Problem,
        device: th.device,
        **base: Any,
    ) -> ConstrainedPerfLVAE2D:
        """Rebuild the decoder and estimate the latent distribution to sample from.

        Args:
            resolved: The downloaded checkpoint package.
            problem: The problem being evaluated, used for the design shape and
                the training split the latent fit is drawn from.
            device: Device to place the decoder on.
            **base: Base-class arguments supplied by `from_pretrained`.

        Returns:
            A ready-to-sample generator.

        Raises:
            NotImplementedError: If the run used a conditional decoder, which
                needs the condition encoder and rasterized image conditions
                that evaluation does not currently reconstruct.
        """
        design_shape = tuple(problem.design_space.shape)
        config = LVAEConfig.from_run_config(resolved.run_config, design_shape)  # type: ignore[arg-type]

        if resolved.run_config.get("conditional_decoder", False):
            raise NotImplementedError(
                f"{cls.algo_id} checkpoints trained with --conditional-decoder are not yet loadable for "
                "evaluation: sampling would need the condition encoder and rasterized image conditions."
            )

        checkpoint = th.load(resolved.files["constrained_plvae.pth"], map_location=device, weights_only=False)

        decoder = TrueSNDecoder2D(
            latent_dim=config.latent_dim,
            design_shape=config.design_shape,
            lipschitz_scale=config.decoder_lipschitz_scale,
        )
        decoder.load_state_dict(checkpoint["decoder"])
        decoder.to(device).eval()

        encoder = build_encoder(checkpoint, config, device)
        active_mask = get_active_mask(encoder)

        designs = np.asarray(problem.dataset["train"]["optimal_design"][:LATENT_FIT_SAMPLES], dtype=np.float32)
        codes = encode_designs(encoder, designs, device)[:, active_mask]

        latent_mean = codes.mean(axis=0)
        latent_cov = np.cov(codes, rowvar=False) + COVARIANCE_JITTER * np.eye(codes.shape[1])

        frozen = checkpoint.get("pruning_frozen_z")
        frozen_z = frozen.detach().cpu().numpy() if frozen is not None else np.zeros(config.latent_dim, dtype=np.float32)

        return cls(
            decoder=decoder,
            latent_mean=latent_mean,
            latent_cov=latent_cov,
            active_mask=active_mask,
            frozen_z=frozen_z,
            **base,
        )

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        """Draw latent codes from the fitted distribution and decode them.

        Args:
            conditions: Unused; this decoder is unconditional.
            n: Number of designs to generate.

        Returns:
            Generated designs, shape `(n, 1, H, W)`.
        """
        del conditions

        rng = np.random.default_rng(th.randint(0, 2**31 - 1, (1,)).item())
        active = rng.multivariate_normal(self.latent_mean, self.latent_cov, size=n)

        z = np.tile(self.frozen_z.astype(np.float64), (n, 1))
        z[:, self.active_mask] = active

        device = next(self.decoder.parameters()).device
        return self.decoder(th.from_numpy(z).float().to(device))
