"""Tests for identifying which EngiBench produced a spec and which one scored a row.

A version string does not pin a problem definition: 0.2.0 from PyPI and 0.2.0
from `main` point `thermoelastic2d` at different datasets. The spec's condition
and dataset checks catch that class of difference. What they cannot catch is a
change to `simulate` or `optimize`, which moves IOG/COG/FOG without touching any
condition name -- so the revision is recorded on both sides.
"""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

from engiopt.evaluation import spec as spec_mod
from engiopt.evaluation.evaluator import PROVENANCE_COLUMNS


def test_a_vcs_install_reports_its_recorded_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """`pip install "engibench @ git+..."` leaves no `.git`, but records the commit.

    This is the case CI hits, and the one that has to work: the spec files carry
    `0.2.0+0a028c03d02b`, and `freeze()` must be able to reproduce that string.
    """
    payload = json.dumps({"url": "https://github.com/IDEALLab/EngiBench", "vcs_info": {"commit_id": "0a028c03d02bfeed"}})

    class _Distribution:
        @staticmethod
        def read_text(name: str) -> str | None:
            return payload if name == "direct_url.json" else None

    import importlib.metadata as importlib_metadata

    monkeypatch.setattr(importlib_metadata, "distribution", lambda _name: _Distribution())

    assert spec_mod._installed_vcs_commit() == "0a028c03d02b"


def test_a_wheel_without_vcs_info_reports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plain PyPI wheel has a `direct_url.json` only sometimes, and never a commit."""

    class _Distribution:
        @staticmethod
        def read_text(_name: str) -> str | None:
            return json.dumps({"url": "file:///tmp/engibench.whl", "archive_info": {}})

    import importlib.metadata as importlib_metadata

    monkeypatch.setattr(importlib_metadata, "distribution", lambda _name: _Distribution())

    assert spec_mod._installed_vcs_commit() is None


def test_an_installed_package_is_never_read_as_a_checkout(tmp_path: Path) -> None:
    """The bug this guards: `git -C` searches *parent* directories.

    A wheel unpacked into a virtualenv inside the EngiOpt repository would make
    `git -C .../site-packages/..` answer with EngiOpt's commit and record it as
    EngiBench's -- a wrong sha, which is worse than no sha.
    """
    site_packages = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)

    assert spec_mod._source_checkout_commit(site_packages) is None


def test_a_directory_inside_an_unrelated_repo_is_not_that_repo(tmp_path: Path) -> None:
    """Only the repository *root* counts as the package's own checkout."""
    import subprocess

    repo = tmp_path / "some_repo"
    (repo / "nested" / "engibench").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)

    # `nested` sits inside a git repo but is not its root, so it is not a checkout
    # of the package -- exactly the shape a vendored install takes.
    assert spec_mod._source_checkout_commit(repo / "nested") is None


def test_freeze_and_the_row_report_the_same_kind_of_string() -> None:
    """The spec's "frozen against" and the row's "evaluated on" must be comparable."""
    from engiopt.evaluation.evaluator import engibench_version as row_version

    assert row_version() == spec_mod.engibench_version()


def test_the_runtime_engibench_is_recorded_on_every_row() -> None:
    """Recorded, not enforced.

    Enforcing an exact match would refuse to evaluate for any contributor whose
    EngiBench differs by a commit; partitioning ranks by it would mean two models
    could never be compared unless scored on identical builds. Recording keeps
    the difference visible at neither cost.
    """
    assert "engibench_version" in PROVENANCE_COLUMNS


def test_the_version_carries_a_sha_when_one_is_identifiable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The composed value is `<version>+<sha>`, which is what the specs record."""
    monkeypatch.setattr(spec_mod, "_installed_vcs_commit", lambda: "abcdef123456")

    assert spec_mod.engibench_version().endswith("+abcdef123456")


def test_the_version_degrades_to_the_bare_release(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing identifying the source, the release version is all there is."""
    monkeypatch.setattr(spec_mod, "_installed_vcs_commit", lambda: None)
    monkeypatch.setattr(spec_mod, "_source_checkout_commit", lambda _root: None)

    assert "+" not in spec_mod.engibench_version()


def test_the_committed_specs_record_a_resolvable_engibench(tmp_path: Path) -> None:
    """Every committed spec should say which EngiBench it was frozen against."""
    spec_files = sorted(spec_mod.SPEC_ROOT.glob("*/*.json"))
    assert spec_files, "no committed specs found"

    missing: list[str] = []
    for path in spec_files:
        payload: dict[str, Any] = json.loads(path.read_text())
        if not payload.get("engibench_version"):
            missing.append(f"{path.parent.name}/{path.stem}")

    assert not missing, f"specs with no recorded EngiBench: {missing}"
