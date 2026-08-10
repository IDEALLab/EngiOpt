"""Repackage W&B-era artifact directories as checkpoint packages.

The 2D checkpoints sitting in `artifacts/` predate the `Generator` contract:
they hold `generator.pth` and nothing else, while the loader also wants the
`run_config.json` that says how to rebuild the network. Everything it needs is
recoverable from the weights themselves, so this script reads the shapes and
writes the package.

This exists so the IDETC challenge has a *legitimate* field to sit alongside the
constructed adversaries before the Euler sweep publishes to HuggingFace. Once
those checkpoints are on the Hub the bank resolves them there and this script is
dead code -- deliberately, and it should be deleted then.

    python workshops/idetc26/tools/stage_local_checkpoints.py
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import torch as th

REPO_ROOT = Path(__file__).resolve().parents[3]
DESTINATION = REPO_ROOT / "workshops" / "idetc26" / "build" / "checkpoints"

SOURCES = {
    "cgan_cnn_2d": REPO_ROOT / "artifacts" / "beams2d_cgan_cnn_2d_generator:v20" / "generator.pth",
    "gan_cnn_2d": REPO_ROOT / "artifacts" / "beams2d_gan_cnn_2d_generator:v21" / "generator.pth",
}
"""Artifact directories to repackage, keyed by the `algo_id` that loads them."""

FIRST_LAYER = {"cgan_cnn_2d": "z_path.0.weight", "gan_cnn_2d": "stem.0.weight"}
"""The transposed convolution whose input channel count is the latent dimension."""

OUTPUT_ACTIVATION = "tanh"
"""What these runs trained with.

Confirmed empirically rather than assumed: reloading the cGAN under `sigmoid`
gives designs whose mean density is 0.61 against a requested 0.26, and an MMD of
0.67 against 0.037. The weights only make sense one way.
"""


def stage(algo: str, source: Path, destination: Path) -> dict[str, object]:
    """Write one checkpoint package, deriving its run config from the weights.

    Args:
        algo: The `algo_id` that will load this package.
        source: Path to the legacy `generator.pth`.
        destination: Package directory to write.

    Returns:
        The run config that was written.

    Raises:
        FileNotFoundError: If the source checkpoint is missing.
        KeyError: If the weights do not carry the expected first layer.
    """
    if not source.exists():
        raise FileNotFoundError(f"No checkpoint at {source}.")

    checkpoint = th.load(source, map_location="cpu", weights_only=False)
    state = checkpoint.get("generator", checkpoint)
    latent_dim = int(state[FIRST_LAYER[algo]].shape[0])

    run_config: dict[str, object] = {"latent_dim": latent_dim, "generator_output_activation": OUTPUT_ACTIVATION}

    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination / "generator.pth")
    (destination / "run_config.json").write_text(json.dumps(run_config, indent=2))
    (destination / "metadata.json").write_text(
        json.dumps(
            {"staged_from": str(source.relative_to(REPO_ROOT)), "note": "Legacy W&B artifact, repackaged."}, indent=2
        )
    )
    return run_config


def main() -> None:
    """Stage every known legacy checkpoint."""
    for algo, source in SOURCES.items():
        config = stage(algo, source, DESTINATION / algo)
        print(f"{algo}: latent_dim={config['latent_dim']} -> {(DESTINATION / algo).relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
