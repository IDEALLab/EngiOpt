"""`Generator` contract for the VQGAN + transformer model."""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

import torch as th

from engiopt.core import ConditionBatch
from engiopt.core import Generator
from engiopt.generators.vqgan.vqgan import VQGAN
from engiopt.generators.vqgan.vqgan import VQGANTransformer
from engiopt.transforms import resize_to

if TYPE_CHECKING:
    from engibench.core import Problem

    from engiopt.checkpoint_store import ResolvedCheckpoint


class VQGANGenerator(Generator):
    """Discrete latent VQGAN whose codes are sampled autoregressively by a transformer.

    Conditions are themselves quantized (by a second, condition-side VQGAN) into
    the transformer's start-of-sequence tokens. Training may drop constant
    condition columns and rescale the rest; both are recorded in the checkpoint,
    so `Generator.sample` hands `_sample` the conditions already in that form and
    this adapter does no preprocessing of its own.
    """

    algo_id = "vqgan"
    conditional = True
    design_kinds = ("2d",)
    checkpoint_files = ("vqgan.pth", "transformer.pth")
    """`cvqgan.pth` is loaded by `build` too, but only conditional runs write one,
    so it cannot be required here. The package content hash covers it regardless."""
    primary_state_key = "transformer"
    output_clip = (1e-3, 1.0)

    def __init__(self, net: VQGANTransformer, latent_size: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.net = net
        self.latent_size = latent_size

    @classmethod
    def build(cls, resolved: ResolvedCheckpoint, problem: Problem, device: th.device, **base: Any) -> VQGANGenerator:
        """Rebuild the design VQGAN, the condition VQGAN, and the transformer."""
        config = resolved.run_config
        vqgan = VQGAN(
            device=device,
            is_c=False,
            encoder_channels=config["encoder_channels"],
            encoder_start_resolution=config["image_size"],
            encoder_attn_resolutions=config["encoder_attn_resolutions"],
            encoder_num_res_blocks=config["encoder_num_res_blocks"],
            decoder_channels=config["decoder_channels"],
            decoder_start_resolution=config["latent_size"],
            decoder_attn_resolutions=config["decoder_attn_resolutions"],
            decoder_num_res_blocks=config["decoder_num_res_blocks"],
            image_channels=config["image_channels"],
            latent_dim=config["latent_dim"],
            num_codebook_vectors=config["num_codebook_vectors"],
        )
        vqgan.load_state_dict(th.load(resolved.files["vqgan.pth"], map_location=device, weights_only=True)["vqgan"])
        vqgan.eval().to(device)
        cvqgan = VQGAN(
            device=device,
            is_c=True,
            cond_feature_map_dim=config["cond_feature_map_dim"],
            cond_dim=config["cond_dim"],
            cond_hidden_dim=config["cond_hidden_dim"],
            cond_latent_dim=config["cond_latent_dim"],
            cond_codebook_vectors=config["cond_codebook_vectors"],
        )
        cvqgan_state = cls._load_cvqgan(resolved.root_dir, config=config, device=device)
        if cvqgan_state is not None:
            cvqgan.load_state_dict(cvqgan_state)
        cvqgan.eval().to(device)
        net = VQGANTransformer(
            conditional=config["conditional"],
            vqgan=vqgan,
            cvqgan=cvqgan,
            image_size=config["image_size"],
            decoder_channels=config["decoder_channels"],
            cond_feature_map_dim=config["cond_feature_map_dim"],
            num_codebook_vectors=config["num_codebook_vectors"],
            n_layer=config["n_layer"],
            n_head=config["n_head"],
            n_embd=config["n_embd"],
            dropout=config["dropout"],
        )
        net.load_state_dict(
            th.load(resolved.files["transformer.pth"], map_location=device, weights_only=True)[cls.primary_state_key]
        )
        net.eval().to(device)
        return cls(net=net, latent_size=config["latent_size"], problem=problem, device=device, **base)

    @classmethod
    def _load_cvqgan(cls, package_root: str, *, config: dict[str, Any], device: th.device) -> dict[str, Any] | None:
        """Load the condition-side VQGAN, which ships inside the same package.

        Raises:
            FileNotFoundError: If the model is conditional but the package has
                no `cvqgan.pth`, which would make its conditions unusable.
        """
        path = os.path.join(package_root, "cvqgan.pth")
        if not os.path.exists(path):
            if config["conditional"]:
                raise FileNotFoundError(f"Conditional VQGAN needs cvqgan.pth, missing from {package_root}")
            return None
        return th.load(path, map_location=device, weights_only=True)["cvqgan"]

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        """Sample a full grid of latent codes autoregressively, then decode it."""
        if self.run_config["conditional"]:
            start_tokens = self.net.encode_to_z(x=conditions.require_tensor(self.algo_id), is_c=True)[1]
        else:
            start_tokens = th.ones(n, 1, dtype=th.int64, device=self.device) * self.net.sos_token
        codes = self.net.sample(
            x=th.empty(n, 0, dtype=th.int64, device=self.device),
            c=start_tokens,
            steps=self.latent_size**2,
        )
        return resize_to(data=self.net.z_to_image(codes), h=self.design_shape[0], w=self.design_shape[1])
