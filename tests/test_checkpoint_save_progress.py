"""Completed windows remain observable while an atomic checkpoint is saving."""
import json
import sys

import pytest

from tidegraph.experiment import checkpoint, container_job, jobs
from tidegraph.experiment.engine import run


def request():
    return dict(schema="tide-experiment-v1", model=dict(recipe=dict(topology="fc", depth=2,
        region_width=1, width=4)), task=dict(samples=2, positions=2, input_pattern="staggered-v1"),
        execution=dict(device="cpu"), measurement=dict(steps=1, warmup=1))


@pytest.mark.parametrize("failure_window", [None, 0, 1])
def test_save_progress_preserves_last_durable_boundary(tmp_path, monkeypatch, failure_window):
    job = tmp_path/"job"
    container_job.prepare(job, command=[sys.executable, "-V"], cwd=tmp_path, timeout=10)
    output = job/"run"
    original = checkpoint.save
    observed = []

    def save(owner, directory, resolved, identity, next_window):
        saved = json.loads((output/"result.json").read_text())
        summary = jobs.status(job)["experiment"]
        index = next_window-1
        assert summary["phase"] == "checkpoint-save" and summary["window"] == index
        assert summary["execution_status"] == "running"
        row = saved["warmup" if index == 0 else "metrics"][-1]
        assert row["window"] == index and row["seconds"] >= 0
        assert row["checkpoint_scheduled"]
        assert (output/row["output_artifact"]).is_file()
        assert "checkpoint_seconds" not in row
        if index == 0:
            assert summary["checkpoint"] == {"state": "not_checked"}
        else:
            assert summary["checkpoint"]["next_window"] == index
            assert summary["checkpoint"]["state"] == "saved"
            assert checkpoint.locate(output)[1]["next_window"] == index
        observed.append(index)
        if index == failure_window:
            raise RuntimeError("injected save failure")
        return original(owner, directory, resolved, identity, next_window)

    monkeypatch.setattr(checkpoint, "save", save)
    if failure_window is None:
        result = run(request(), output)
        assert result["state"] == "passed" and result["phase"] == "complete"
        assert result["checkpoint"]["next_window"] == 2
        assert result["checkpoint"]["exact_resume"] is True
        assert observed == [0, 1]
    else:
        with pytest.raises(RuntimeError, match="injected save failure"):
            run(request(), output)
        result = json.loads((output/"result.json").read_text())
        assert result["state"] == "failed" and result["execution_status"] == "failed"
        assert result["phase"] == "checkpoint-save" and result["window"] == failure_window
        assert result["validation"]["complete"] == "not_checked"
        if failure_window == 0:
            assert result["checkpoint"] == {"state": "not_checked"}
        else:
            assert result["checkpoint"]["next_window"] == 1
            assert result["checkpoint"]["exact_resume"] == "not_checked"
            assert checkpoint.locate(output)[1]["next_window"] == 1
