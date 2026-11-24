"""Tests for generator discovery and the `Generator` contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import torch as th

from engiopt.core import ConditionBatch
from engiopt.core import Generator
from engiopt.utils.all_generators import BUILTIN_GENERATORS
from engiopt.utils.all_generators import extract_generator


def test_every_generator_directory_has_an_adapter() -> None:
    """Discovery must find an adapter for every model package."""
    packages = {p.name for p in Path("engiopt/generators").iterdir() if p.is_dir() and not p.name.startswith("_")}
    assert set(BUILTIN_GENERATORS) == packages


def test_algo_ids_match_directory_names() -> None:
    """`algo_id` keys the leaderboard and the HF repo, so it must match the package."""
    for name, generator in BUILTIN_GENERATORS.items():
        assert generator.algo_id == name


def test_template_is_not_registered() -> None:
    """The copy-me template must never appear as a real model."""
    assert "_template" not in BUILTIN_GENERATORS


def test_the_template_loader_rebuilds_what_the_template_script_saves(fake_problem: Any, tmp_path: Path) -> None:
    """The two template files are a pair, and a copy of them must work before anything is changed.

    The training script writes its network's weights under one filename and one
    key. The loader reads that filename and that key and rebuilds the network
    from the saved settings. Nothing checks at import time that the two agree,
    so this does: save the way the script saves, load the way the loader loads.
    """
    from engiopt.core import CONDITION_KEYS_METADATA_FIELD
    from engiopt.generators._template.adapter import MyModel
    from engiopt.generators._template.my_model import Generator as TemplateNet

    keys = ("volfrac", "rmin")
    latent_dim = 8
    trained = TemplateNet(latent_dim=latent_dim, n_conds=len(keys), design_shape=fake_problem.design_space.shape)
    th.save({"generator": trained.state_dict()}, tmp_path / "generator.pth")

    class _Package:
        run_config: ClassVar[dict[str, Any]] = {"latent_dim": latent_dim}
        files: ClassVar[dict[str, str]] = {"generator.pth": str(tmp_path / "generator.pth")}
        metadata: ClassVar[dict[str, Any]] = {CONDITION_KEYS_METADATA_FIELD: list(keys)}

    loaded = MyModel.build(
        _Package,
        problem=fake_problem,
        device=th.device("cpu"),
        problem_id="fake",
        seed=1,
        condition_keys=keys,
    )

    for name, weights in trained.state_dict().items():
        assert th.equal(loaded.net.state_dict()[name], weights), name
    designs = loaded.sample(th.full((4, len(keys)), 0.5), n=4, seed=0)
    assert designs.shape == (4, *fake_problem.design_space.shape)


def test_every_generator_declares_the_files_it_needs() -> None:
    """`checkpoint_files` drives resolution, so it must be non-empty and concrete."""
    for name, generator in BUILTIN_GENERATORS.items():
        assert generator.checkpoint_files, f"{name} declares no checkpoint files"
        assert all(f.endswith(".pth") for f in generator.checkpoint_files), name


def test_extract_generator_rejects_ambiguous_modules() -> None:
    """Two generators in one adapter module would make the registry ambiguous."""

    class _Module:
        First = type("First", (_StubGenerator,), {"algo_id": "a"})
        Second = type("Second", (_StubGenerator,), {"algo_id": "b"})
        __name__ = "fake.adapter"

    with pytest.raises(ValueError, match="Only one generator per adapter"):
        extract_generator(_Module)


class _StubGenerator(Generator):
    """Minimal in-memory generator, standing in for a trained model."""

    algo_id = "stub"
    conditional = True
    design_kinds = ("2d",)

    @classmethod
    def build(cls, resolved: Any, problem: Any, device: Any, **base: Any) -> _StubGenerator:
        raise NotImplementedError

    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor:
        # Deliberately the wrong shape and out of range, to prove `sample` fixes both.
        return th.rand(n, int(np.prod(self.design_shape))) * 4.0 - 2.0


def _stub(problem: Any) -> _StubGenerator:
    return _StubGenerator(problem, problem_id="fake", seed=1, device=th.device("cpu"))


def test_sample_normalizes_shape_and_records_cost(fake_problem: Any) -> None:
    """`sample` reshapes to the design shape and times the call."""
    generator = _stub(fake_problem)
    designs = generator.sample(th.zeros(5, 2), n=5)
    assert designs.shape == (5, *fake_problem.design_space.shape)
    assert generator.last_sample_seconds is not None
    assert generator.last_sample_seconds >= 0


def test_sample_applies_output_clip(fake_problem: Any) -> None:
    """`output_clip` bounds designs that a simulator would otherwise reject."""
    generator = _stub(fake_problem)
    generator.output_clip = (0.1, 0.9)
    designs = generator.sample(th.zeros(4, 2), n=4)
    assert designs.min() >= 0.1
    assert designs.max() <= 0.9


def test_sample_infers_n_from_conditions(fake_problem: Any) -> None:
    """Omitting `n` uses the number of supplied conditions."""
    designs = _stub(fake_problem).sample(th.zeros(7, 2))
    assert designs.shape[0] == 7


def test_sample_is_reproducible_for_a_fixed_seed(fake_problem: Any) -> None:
    """Same seed, same designs -- the basis of every reproducibility claim."""
    generator = _stub(fake_problem)
    first = generator.sample(th.zeros(3, 2), n=3, seed=123)
    second = generator.sample(th.zeros(3, 2), n=3, seed=123)
    np.testing.assert_array_equal(first, second)


def test_condition_batch_reports_a_clear_error_when_empty() -> None:
    """A conditional model given nothing should say so by name."""
    with pytest.raises(ValueError, match="my_algo is conditional"):
        ConditionBatch(tensor=None).require_tensor("my_algo")


def test_registry_is_a_plain_mapping_like_builtin_problems() -> None:
    """`BUILTIN_GENERATORS` mirrors `BUILTIN_PROBLEMS`: a dict of name -> class."""
    assert isinstance(BUILTIN_GENERATORS, dict)
    assert all(isinstance(g, type) and issubclass(g, Generator) for g in BUILTIN_GENERATORS.values())


def test_every_checkpoint_load_uses_the_restricted_unpickler() -> None:
    """A `.pth` file is pickled Python, so loading one runs whatever it contains.

    `weights_only=True` swaps in an unpickler that can rebuild tensors and
    nothing else. Every published checkpoint loads under it, so the unrestricted
    form buys nothing and costs arbitrary code execution the moment a checkpoint
    comes from someone else's HuggingFace account.
    """
    adapters = [Path("engiopt/generators", name, "adapter.py") for name in BUILTIN_GENERATORS]
    sources = [Path("engiopt/core.py"), Path("engiopt/generators/_template/adapter.py"), *adapters]

    offenders = []
    for path in sources:
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if "th.load(" in line and "weights_only=True" not in line:
                offenders.append(f"{path}:{number}")
    assert not offenders, "checkpoint loads without weights_only=True: " + ", ".join(offenders)
