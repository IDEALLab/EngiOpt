"""List the checkpoint packages published for a model family.

A sweep publishes one package per configuration and seed, addressed by
`{problem_id}/cfg_{fingerprint}/seed_{seed}`. The fingerprint is a hash of the
run config, so it identifies a configuration unambiguously but says nothing
about what that configuration *was*. This reads each package's `run_config.json`
and prints the two together, which is what you need to choose one -- picking a
latent-metric instrument out of a sweep, or finding which configuration earned a
leaderboard row.

Usage::

    python -m engiopt.list_checkpoints --algo constrained_plvae_2d
    python -m engiopt.list_checkpoints --algo cgan_cnn_2d --problem-id beams2d
    python -m engiopt.list_checkpoints --algo constrained_plvae_2d --show nmse-threshold-rec
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi
import tyro

from engiopt.checkpoint_store import build_hf_repo_id

PACKAGE_PATTERN = re.compile(r"^(?P<problem>[^/]+)/(?:cfg_(?P<fingerprint>[^/]+)/)?seed_(?P<seed>\d+)/")
"""Matches the package layout written by `save_checkpoint_package`."""

RUN_CONFIG_FILE = "run_config.json"

ALREADY_A_COLUMN = {"problem_id", "seed", "algo"}
"""Run-config keys that duplicate a column the table already prints."""


@dataclass(frozen=True)
class CheckpointPackage:
    """One published checkpoint, and the configuration behind it."""

    problem_id: str
    seed: int
    fingerprint: str | None
    """None for the canonical `{problem}/seed_N` path a default-config run also writes."""
    run_config: dict[str, object]

    @property
    def is_canonical(self) -> bool:
        """Whether this is the default-config path `from_pretrained` reads without a fingerprint."""
        return self.fingerprint is None


def discover(repo_id: str, *, problem_id: str | None = None) -> list[CheckpointPackage]:
    """Find every checkpoint package in a model-family repo.

    Args:
        repo_id: HuggingFace model repo, e.g. `IDEALLab/engiopt-cgan-cnn-2d`.
        problem_id: Restrict to one problem.

    Returns:
        Packages sorted by problem, then fingerprint, then seed.
    """
    api = HfApi()
    seen: dict[tuple[str, str | None, int], None] = {}

    for path in api.list_repo_files(repo_id):
        match = PACKAGE_PATTERN.match(path)
        if match is None:
            continue
        key = (match["problem"], match["fingerprint"], int(match["seed"]))
        if problem_id is None or key[0] == problem_id:
            seen[key] = None

    packages: list[CheckpointPackage] = []
    for problem, fingerprint, seed in seen:
        prefix = f"{problem}/cfg_{fingerprint}/seed_{seed}" if fingerprint else f"{problem}/seed_{seed}"
        try:
            local = hf_hub_download(repo_id=repo_id, filename=f"{prefix}/{RUN_CONFIG_FILE}")
            config = json.loads(open(local).read())  # noqa: SIM115
        except Exception:  # noqa: BLE001 - a package without a readable config is still worth listing
            config = {}
        packages.append(CheckpointPackage(problem_id=problem, seed=seed, fingerprint=fingerprint, run_config=config))

    return sorted(packages, key=lambda p: (p.problem_id, p.fingerprint or "", p.seed))


def main(
    algo: str,
    problem_id: str | None = None,
    show: tuple[str, ...] = (),
    hf_entity: str = "IDEALLab",
    hf_repo_prefix: str = "engiopt",
) -> None:
    """Print the published checkpoints for one model family.

    Args:
        algo: Model family, e.g. `constrained_plvae_2d`.
        problem_id: Restrict to one problem.
        show: Run-config keys to display. Defaults to whichever keys actually
            differ across the packages found, since those are the ones that
            distinguish one configuration from another.
        hf_entity: HF org/user holding the checkpoint repos.
        hf_repo_prefix: Prefix of the per-model-family repo.
    """
    repo_id = build_hf_repo_id(hf_entity, hf_repo_prefix, algo)
    packages = discover(repo_id, problem_id=problem_id)

    if not packages:
        print(f"No checkpoint packages found in {repo_id}.")
        return

    keys = list(show)
    if not keys:
        # Show only what varies -- shared values say nothing about which to pick
        # -- and drop anything already shown as its own column.
        all_keys = {key for pkg in packages for key in pkg.run_config} - ALREADY_A_COLUMN
        keys = sorted(
            key
            for key in all_keys
            if len({json.dumps(pkg.run_config.get(key), sort_keys=True, default=str) for pkg in packages}) > 1
        )

    print(f"{repo_id}  ({len(packages)} packages)\n")
    header = f"{'problem':<18} {'fingerprint':<12} {'seed':>4}  " + "  ".join(f"{k:<22}" for k in keys)
    print(header)
    print("-" * len(header))
    for pkg in packages:
        fingerprint = pkg.fingerprint or "(canonical)"
        values = "  ".join(f"{pkg.run_config.get(key, '-')!s:<22}" for key in keys)
        print(f"{pkg.problem_id:<18} {fingerprint:<12} {pkg.seed:>4}  {values}")

    print("\nTo pin one as the latent-metric instrument, put this in the problem's eval spec:")
    example = packages[0]
    print(
        '  "latent_instrument": {'
        f'"algo": "{algo}", "seed": {example.seed}, '
        f'"config_fingerprint": "{example.fingerprint or ""}"' + "}"
    )


if __name__ == "__main__":
    tyro.cli(main)
