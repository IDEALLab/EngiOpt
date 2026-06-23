"""Archive a local top-k checkpoint directory to durable checkpoint storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from engiopt.topk_checkpoint_bundle import archive_topk_checkpoint_bundle
from engiopt.topk_checkpoint_bundle import CheckpointArchiveMode
from engiopt.topk_checkpoint_bundle import CheckpointBackend
from engiopt.topk_checkpoint_bundle import TopKBundleSpec

ModelId = Literal["flow_matching_2d_cond", "diffusion_2d_cond", "cgan_cnn_2d"]


@dataclass
class Args:
    """Arguments for archiving an existing top-k checkpoint directory."""

    checkpoint_dir: str
    """Directory containing validation_metrics.json and top-k checkpoints."""
    model_id: ModelId
    """Model family identifier."""
    problem_id: str
    """EngiBench problem identifier."""
    seed: int
    """Training seed."""
    top_k: int = 5
    """Number of top validation-MMD checkpoints to archive."""
    checkpoint_backend: CheckpointBackend = "hf"
    """Durable backend. Use 'none' for a dry no-upload run."""
    checkpoint_archive_mode: CheckpointArchiveMode = "eval"
    """Archive mode: 'eval' strips optimizer state; 'full' preserves training checkpoints."""
    hf_entity: str = ""
    """HF org/user where checkpoint packages are stored. Empty infers the token username."""
    hf_repo_prefix: str = "engiopt"
    """HF repo prefix used for model-family repositories."""
    hf_private: bool = False
    """Create/use private HF model repositories."""
    checkpoint_package_label: str | None = None
    """Optional package label, e.g. euler_16 or corrected_sigmoid."""
    include_discriminator: bool = True
    """Include cGAN discriminator checkpoints when present."""
    no_final: bool = False
    """Do not include final/best checkpoint files in the package."""


def main() -> None:
    args = tyro.cli(Args)
    info = archive_topk_checkpoint_bundle(
        spec=TopKBundleSpec(
            model_id=args.model_id,
            problem_id=args.problem_id,
            seed=args.seed,
            checkpoint_dir=Path(args.checkpoint_dir),
            top_k=args.top_k,
            package_label=args.checkpoint_package_label,
            include_final=not args.no_final,
            include_discriminator=args.include_discriminator,
            archive_mode=args.checkpoint_archive_mode,
        ),
        checkpoint_backend=args.checkpoint_backend,
        hf_entity=args.hf_entity,
        hf_repo_prefix=args.hf_repo_prefix,
        hf_private=args.hf_private,
    )
    print("Archived top-k checkpoint bundle:")
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
