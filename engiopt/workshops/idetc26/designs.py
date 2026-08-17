"""Sampled designs, kept on disk so a room of people never pays for them twice.

Sampling is the workshop's only unavoidable wait. Drawing the spec's conditions
from every model in the bank means running a diffusion model on a free-tier CPU
while forty people watch a cell with no output, and a kernel restart makes them
watch it again. None of that wait teaches anything: the designs are a *fixture*
of the session, identical for every team, and the interesting costs -- what a
metric costs, what the simulator costs -- are elsewhere.

So designs are written to disk once and read back everywhere. A store is a
search path: an explicit directory first, then the copy that ships inside the
package, so a Colab runtime with no checkout still opens instantly. Anything not
found is sampled live and written back, which means the first person to ask for
a fresh seed pays for it and nobody else does.

What is stored beside the array matters as much as the array. `gen_seconds` is a
ranked column, so a replayed cost has to say *which machine measured it* rather
than silently pass off a workstation's timing as this laptop's -- and the spec
version and condition digest are stored so a cache built against different
conditions is refused rather than scored.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np

PACKAGE_CACHE = Path(__file__).resolve().parent / "cache"
"""The cache that ships inside the installed package.

Inside `engiopt/`, not beside the notebooks, because Colab installs the package
and never sees the repository. Anything outside the package is not there on the
day.
"""

REPO_CACHE = Path(__file__).resolve().parents[3] / "workshops" / "idetc26" / "cache"
"""The working copy in a source checkout, which `build_design_cache` writes."""


@dataclass(frozen=True)
class CachedDesigns:
    """One model's designs at one seed, plus what is needed to score them honestly.

    Attributes:
        designs: The generated designs, in the order the spec asks its
            conditions in.
        sample_seconds: What generating them cost, on `machine`.
        model_params: Parameter count of the model that produced them.
        machine: Where the timing was measured. `gen_seconds` is only comparable
            within one machine, so replaying a cost without saying whose it is
            would be a fabrication.
        built_at: When the cache entry was written.
        spec_version: Spec the conditions came from.
        condition_digest: Digest of those conditions, so a cache built against a
            different draw is refused instead of scored.
    """

    designs: np.ndarray
    sample_seconds: float | None = None
    model_params: int | None = None
    machine: str = "unknown"
    built_at: str = ""
    spec_version: str = ""
    condition_digest: str | None = None

    @property
    def replayed_cost_note(self) -> str:
        """How to describe this entry's `gen_seconds` in a results table."""
        if self.sample_seconds is None:
            return "not measured"
        return f"{self.sample_seconds:.3g}s, measured on {self.machine}"


class DesignStore:
    """Reads and writes cached designs for one problem under one spec.

    Attributes:
        problem_id: Which problem the designs are for.
        spec_version: Which spec's conditions they answer.
        condition_digest: Digest of that spec's resolved conditions.
        roots: Directories searched for an entry, in order.
        write_root: Where a freshly sampled entry is written.
    """

    def __init__(
        self,
        problem_id: str,
        *,
        spec_version: str,
        condition_digest: str | None = None,
        roots: tuple[Path, ...] | None = None,
        write_root: Path | None = None,
    ) -> None:
        self.problem_id = problem_id
        self.spec_version = spec_version
        self.condition_digest = condition_digest
        self.roots = roots if roots is not None else (REPO_CACHE, PACKAGE_CACHE)
        self.write_root = write_root or self.roots[0]

    def path_in(self, root: Path, key: str, seed: int) -> Path:
        """Where one entry lives under `root`.

        The spec version is a directory rather than a filename suffix so that a
        stale spec's cache can be deleted in one move.
        """
        return root / self.problem_id / self.spec_version / f"{_slug(key)}__seed{seed}.npz"

    def load(self, key: str, seed: int) -> CachedDesigns | None:
        """Read one entry, or None if no root holds a usable one.

        An entry drawn against different conditions is skipped with a printed
        reason rather than returned: scoring it would put two different question
        sets in the same column.
        """
        for root in self.roots:
            path = self.path_in(root, key, seed)
            if not path.exists():
                continue
            try:
                entry = _read(path)
            except Exception as exc:  # noqa: BLE001 - a corrupt cache must not sink the session
                print(f"  [cache] ignoring {path.name}: {type(exc).__name__}: {exc}")
                continue
            if self.condition_digest and entry.condition_digest and entry.condition_digest != self.condition_digest:
                print(
                    f"  [cache] ignoring {path.name}: it was drawn against conditions "
                    f"{entry.condition_digest}, and this spec resolves to {self.condition_digest}."
                )
                continue
            return entry
        return None

    def store(
        self,
        key: str,
        seed: int,
        designs: np.ndarray,
        *,
        sample_seconds: float | None = None,
        model_params: int | None = None,
    ) -> Path:
        """Write one entry under `write_root` and return where it landed."""
        path = self.path_in(self.write_root, key, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "problem_id": self.problem_id,
            "spec_version": self.spec_version,
            "condition_digest": self.condition_digest,
            "key": key,
            "seed": seed,
            "sample_seconds": sample_seconds,
            "model_params": model_params,
            "machine": _machine(),
            "built_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(timespec="seconds"),
        }
        np.savez_compressed(path, designs=np.asarray(designs, dtype=np.float32), meta=json.dumps(meta))
        return path

    def holds(self, key: str, seed: int) -> bool:
        """Whether a usable entry exists, without unpacking the array."""
        return any(self.path_in(root, key, seed).exists() for root in self.roots)


def _read(path: Path) -> CachedDesigns:
    """Load one `.npz` entry written by `DesignStore.store`."""
    with np.load(path, allow_pickle=False) as payload:
        designs = np.asarray(payload["designs"])
        meta: dict[str, Any] = json.loads(str(payload["meta"]))
    return CachedDesigns(
        designs=designs,
        sample_seconds=meta.get("sample_seconds"),
        model_params=meta.get("model_params"),
        machine=meta.get("machine", "unknown"),
        built_at=meta.get("built_at", ""),
        spec_version=meta.get("spec_version", ""),
        condition_digest=meta.get("condition_digest"),
    )


def _slug(key: str) -> str:
    """Filesystem-safe version of a bank key like `cgan_cnn_2d#3`."""
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in key)


def _machine() -> str:
    """Short description of the machine a timing was measured on."""
    return f"{platform.system()} {platform.machine()}"
