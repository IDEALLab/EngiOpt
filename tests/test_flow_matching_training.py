"""Exercise training, package writing and inference without downloading a dataset."""

import json
from types import SimpleNamespace

from datasets import Dataset
import numpy as np
import torch

from engiopt.generators.flow_matching_2d_cond import adapter
from engiopt.generators.flow_matching_2d_cond import flow_matching_2d_cond as training


class _Field(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x, time, cond):
        return SimpleNamespace(sample=self.scale * x)


def test_train_and_reload(tmp_path, monkeypatch, fake_problem):
    split = Dataset.from_dict(dict(fake_problem.dataset["train"][:]))
    monkeypatch.setattr(type(fake_problem), "dataset", property(lambda self: {"train": split}))
    monkeypatch.setitem(training.BUILTIN_PROBLEMS, "beams2d", lambda: fake_problem)
    for module in (training, adapter):
        monkeypatch.setattr(module, "build_model", lambda *_: _Field())

    package = tmp_path / "checkpoint"
    training.train(training.Args(output=package, device="cpu", batch_size=2, max_batches=1))
    assert json.loads((package / "metadata.json").read_text())["training_updates"] == 1
    generator = adapter.FlowMatching2DCond.from_pretrained(
        fake_problem, problem_id="beams2d", device=torch.device("cpu"), model_source="local", local_model_dir=str(package)
    )
    assert generator.net.scale.item() != 0
    designs = generator.sample(torch.zeros(2, 2), seed=2001)
    assert designs.shape == (2, *fake_problem.design_space.shape)
    assert np.isfinite(designs).all()
