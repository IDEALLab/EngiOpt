"""Fixed-step flow matching integrates a field at the checkpoint's time scale."""

from types import SimpleNamespace

import pytest
import torch

from engiopt.checkpoint_store import ResolvedCheckpoint
from engiopt.generators.flow_matching_2d_cond.adapter import FlowMatching2DCond
from engiopt.generators.flow_matching_2d_cond.model import integrate


@pytest.mark.parametrize(("method", "expected", "calls"), [("euler", 0.0, 1), ("midpoint", 0.5, 2), ("rk4", 0.5, 4)])
def test_linear_time_field(method, expected, calls):
    times = []

    def field(x, time, conditions):
        times.append(time.item())
        return SimpleNamespace(sample=torch.ones_like(x) * time * conditions)

    result = integrate(field, torch.zeros(1), torch.tensor([0.1]), 1, 11, method)
    assert len(times) == calls
    torch.testing.assert_close(result, torch.tensor([expected]))


@pytest.mark.parametrize(("method", "step_factor"), [("euler", 1.5), ("midpoint", 1.625), ("rk4", 1.6484375)])
def test_state_dependent_field(method, step_factor):
    # For dx/dt = x, each solver takes a different Taylor approximation to exp(dt).
    def field(x, time, conditions):
        return SimpleNamespace(sample=x)

    result = integrate(field, torch.ones(1), torch.zeros(1), 2, 1000, method)
    torch.testing.assert_close(result, torch.tensor([step_factor**2]))


@pytest.mark.parametrize(("steps", "method"), [(0, "euler"), (1, "unsupported")])
def test_invalid_solver(steps, method):
    with pytest.raises(ValueError, match="Invalid flow integration"):
        integrate(None, torch.ones(1), torch.zeros(1), steps, 1000, method)


@pytest.mark.parametrize(
    ("problem_id", "seed", "message"),
    [("beams2d", 2, "Checkpoint seed"), ("heatconduction2d", 1, "Checkpoint problem_id"), ("beams2d", 1, "conditions")],
)
def test_refuse_mismatched_study_package(fake_problem, problem_id, seed, message):
    config = {"seed": 1, "problem_id": "beams2d"}
    package = ResolvedCheckpoint("local", "", {}, config, {"bundle_type": "selected_checkpoint"})
    # Even four names are not interchangeable: this order swaps the first two.
    keys = ("rmin", "volfrac", "forcedist", "overhang_constraint")
    with pytest.raises(ValueError, match=message):
        FlowMatching2DCond.build(
            package, fake_problem, torch.device("cpu"), problem_id=problem_id, seed=seed, condition_keys=keys
        )


def test_study_prefix_resolves_pinned_weights(monkeypatch):
    from engiopt.generators.flow_matching_2d_cond import adapter

    monkeypatch.setattr(adapter, "resolve_checkpoint_reference", lambda **kwargs: kwargs)
    resolved = FlowMatching2DCond.resolve_checkpoint(
        problem_id="heatconduction2d", seed=3, hf_repo_prefix="engiopt-engopt2026"
    )
    assert resolved == {
        "model_source": "hf",
        "model_ref": "hf://IDEALLab/engiopt-engopt2026-flow-matching-2d-cond/heatconduction2d/selected/v1/euler_16/seed_3",
        "required_files": ["model.pth"],
        "revision": "58ae97c4aa59784fbe22ab232b5d196ff088e4c7",
    }


@pytest.mark.parametrize("name", ["volume", "volfrac"])
def test_selected_thermal_weights_accept_both_dataset_versions(name, tmp_path, fake_problem, monkeypatch):
    from engiopt.generators.flow_matching_2d_cond import adapter

    net = torch.nn.Linear(2, 1)
    weights = tmp_path / "model.pth"
    torch.save({"model": net.state_dict(), "design_shape": fake_problem.design_space.shape, "encoder_hid_dim": 2}, weights)
    config = {"problem_id": "heatconduction2d", "seed": 1, "layers_per_block": 2, "clip_min": 0.001, "clip_max": 1.0}
    package = ResolvedCheckpoint(
        "local", str(tmp_path), {"model.pth": str(weights)}, config, {"bundle_type": "selected_checkpoint"}
    )
    monkeypatch.setattr(adapter, "build_model", lambda *_: torch.nn.Linear(2, 1))
    args = dict(problem_id="heatconduction2d", seed=1)
    loaded = FlowMatching2DCond.build(package, fake_problem, torch.device("cpu"), condition_keys=(name, "length"), **args)
    assert loaded.condition_keys == (name, "length")
    with pytest.raises(ValueError, match="conditions"):
        FlowMatching2DCond.build(package, fake_problem, torch.device("cpu"), condition_keys=("length", name), **args)


def test_study_metrics_are_read_only(monkeypatch):
    import pandas as pd

    from engiopt import evaluate

    args = evaluate.Args(attach_metrics=True, hf_repo_prefix="engiopt-engopt2026", push_to="example/board")
    board = pd.DataFrame([{"algo_id": "flow_matching_2d_cond"}])
    monkeypatch.setattr(evaluate, "push_to_hub", lambda *_: pytest.fail("unexpected board upload"))
    monkeypatch.setattr(evaluate, "publish_checkpoint_metrics", lambda **_: pytest.fail("unexpected checkpoint upload"))
    with pytest.raises(ValueError, match="Omit --attach-metrics"):
        evaluate._publish(args, board)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"hf_entity": "another-team", "hf_repo_prefix": "engiopt-engopt2026"},
        {"hf_repo_prefix": "engiopt-engopt2026", "config_fingerprint": "special"},
        {"hf_repo_prefix": "engiopt-engopt2026", "model_source": "local", "local_model_dir": "pkg"},
    ],
)
def test_other_sources_use_standard_layout(monkeypatch, options):
    from engiopt import core

    monkeypatch.setattr(core, "resolve_named_checkpoint", lambda **kwargs: kwargs)
    resolved = FlowMatching2DCond.resolve_checkpoint(problem_id="beams2d", seed=7, **options)
    assert resolved["algo"] == "flow_matching_2d_cond"
    assert resolved["seed"] == 7
    assert resolved["local_model_dir"] == options.get("local_model_dir")
