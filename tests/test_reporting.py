from __future__ import annotations

import pandas as pd

from engiopt.reporting import load_metric_shards
from engiopt.reporting import summarize_metrics
from engiopt.reporting import write_metrics_csv


def test_write_metrics_csv_overwrites_by_default(tmp_path):
    output_path = tmp_path / "metrics_seed1.csv"

    write_metrics_csv([{"model_id": "foo", "problem_id": "beams2d", "seed": 1, "cog": 10.0}], str(output_path))
    write_metrics_csv([{"model_id": "foo", "problem_id": "beams2d", "seed": 1, "cog": 20.0}], str(output_path))

    frame = pd.read_csv(output_path)
    assert len(frame) == 1
    assert frame.iloc[0]["cog"] == 20.0


def test_load_metric_shards_uses_latest_row(tmp_path):
    shard_path = tmp_path / "metrics_flow_matching_2d_cond_beams2d_seed1.csv"
    write_metrics_csv(
        [
            {"model_id": "flow_matching_2d_cond", "problem_id": "beams2d", "seed": 1, "cog": 10.0},
            {"model_id": "flow_matching_2d_cond", "problem_id": "beams2d", "seed": 1, "cog": 5.0},
        ],
        str(shard_path),
        append_output=False,
    )

    loaded = load_metric_shards(str(tmp_path), problem_id="beams2d", latest_only=True)
    assert len(loaded) == 1
    assert loaded.iloc[0]["cog"] == 5.0


def test_summarize_metrics_builds_mean_and_std():
    raw = pd.DataFrame(
        [
            {"problem_id": "beams2d", "model_id": "fm", "seed": 1, "source_file": "a.csv", "cog": 0.0, "mmd": 1.0},
            {"problem_id": "beams2d", "model_id": "fm", "seed": 2, "source_file": "b.csv", "cog": 2.0, "mmd": 3.0},
        ]
    )

    summary_long, summary_wide = summarize_metrics(raw, metrics=("cog", "mmd"))

    cog_row = summary_long[(summary_long["model_id"] == "fm") & (summary_long["metric"] == "cog")].iloc[0]
    assert cog_row["mean"] == 1.0
    assert summary_wide.iloc[0]["cog_mean"] == 1.0
