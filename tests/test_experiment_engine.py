"""Installed execution separates full qualification, minimal return and resume."""
from copy import deepcopy
import json
import pytest
import torch
from tidegraph.experiment.config import resolve
from tidegraph.experiment.engine import run, check
from tidegraph.experiment.checkpoint import locate
from tidegraph.experiment.validation import outcome


def config(mode="inference", **execution):
    return dict(schema="tide-experiment-v1", model=dict(recipe=dict(topology="fc", depth=2,
        region_width=1, width=4, history=4)), task=dict(samples=2, positions=2, input_pattern="staggered-v1"),
        execution=dict(device="cpu", **execution), measurement=dict(mode=mode, steps=1, warmup=0))


@pytest.mark.parametrize("mode", ["inference", "training"])
def test_complete_independent_and_serial_exact_checkpoint(tmp_path, mode):
    result = run(config(mode), tmp_path/"run")
    assert result["state"] == "passed"
    assert result["validation"]["strict_passed"]
    assert result["validation"]["complete"] == "checked"
    assert result["checkpoint"]["exact_resume"] is True
    assert result["execution_status"] == "completed"
    assert len(result["metrics"]) == 1 and result["warmup"] == []
    path, metadata = locate(tmp_path/"run")
    assert metadata["next_window"] == 1 and (path/"state.pt").is_file()


def test_external_resume_matches_uninterrupted_with_optimizer_and_task(tmp_path):
    one = config("training")
    run(one, tmp_path/"first")
    resumed = run(one, tmp_path/"second", resume=tmp_path/"first")
    whole = deepcopy(one)
    whole["measurement"]["steps"] = 2
    run(whole, tmp_path/"whole")
    assert resumed["start_window"] == 1 and resumed["state"] == "passed"
    left, _ = locate(tmp_path/"second")
    right, _ = locate(tmp_path/"whole")
    from tidegraph.compare import equivalent
    for name in ("state.pt", "task.pt"):
        equivalent(torch.load(left/name, weights_only=True), torch.load(right/name, weights_only=True), atol=0, rtol=0)


def test_minimal_return_is_unchecked_and_preserves_complete_recovery(tmp_path):
    c = config(return_mode="outputs")
    c["measurement"]["validation"] = "none"
    result = run(c, tmp_path/"minimal")
    assert result["state"] == "completed" and not result["strict_passed"]
    assert result["validation"]["numeric"] == "not_checked"
    assert result["checkpoint"]["exact_resume"] is True
    assert result["timing_contract"]["return_mode"] == "outputs"


def test_strict_full_check_includes_independent_roots_and_training(tmp_path):
    report = check(config(), tmp_path/"base", "full")
    assert report["state"] == "passed"
    roots = next(x for x in report["checks"] if x["name"] == "independent-roots")
    assert "connected-zero" in roots["roots"]
    assert "private-raw-state-roots-after-nonposition-cut" in roots["roots"]


def test_only_numeric_exploration_is_accepted():
    report = dict(strict_passed=False, numeric="failed", discrete="passed")
    assert outcome(report, "strict") == "failed"
    assert outcome(report, "explore") == "exploratory_accepted"
    report["discrete"] = "failed"
    assert outcome(report, "explore") == "failed"


def test_resume_rejects_changed_identity_before_creating_output(tmp_path):
    c = config()
    run(c, tmp_path/"first")
    c["task"]["seed"] = 12
    with pytest.raises(ValueError, match="workload_id"):
        run(c, tmp_path/"bad", resume=tmp_path/"first")
    assert not (tmp_path/"bad").exists()


def test_capacity_refusal_is_recorded_before_model_allocation(tmp_path):
    c = config()
    c["resources"] = dict(max_materialize_bytes=1)
    with pytest.raises(ValueError, match="capacity preflight"):
        run(c, tmp_path/"refused")
    record = json.loads((tmp_path/"refused"/"result.json").read_text())
    assert record["execution_status"] == "failed"
    assert "construction_seconds" not in record


def test_separate_packing_observation_preserves_records_but_is_not_throughput(tmp_path):
    c = config(packing="packed")
    c["measurement"]["observation"] = "packing"
    report = run(c, tmp_path/"observed")
    assert report["validation"]["strict_passed"] and report["checkpoint"]["exact_resume"]
    assert not report["throughput_sample"] and not report["throughput_qualified"]
    assert report["observation"]["packing"]["metrics"]["packing/schema_version"] == 1
    assert len(report["metrics"][0]["boundary_activity"]["samples"]) == 2
    from tidegraph.experiment.report import summarize
    assert summarize(report, tmp_path/"observed"/"result.json")["samples_per_second"] is None


def test_host_snapshots_and_atomic_checkpoint_generations_are_budgeted():
    c = config()
    c["measurement"].update(warmup=1, steps=3)
    r = resolve(c)
    assert r["resources"]["live_results"] == 1
    assert r["resources"]["host_results"] == 6
    assert r["capacity_budget"]["disk"]["checkpoint_generations"] == 2


def test_unchecked_run_rejects_nonfinite_task_state_without_module(tmp_path, monkeypatch):
    from tidegraph.experiment.tasks import WeightedOutputTask
    from tidegraph.comparison import NonFiniteMismatch
    monkeypatch.setattr(WeightedOutputTask, "state_dict", lambda self: {"progress": torch.tensor(float("nan"))})
    c = config(return_mode="metrics")
    c["measurement"].update(validation="none", checkpoint_check=False)
    with pytest.raises(NonFiniteMismatch):
        run(c, tmp_path/"nonfinite-task")
    record = json.loads((tmp_path/"nonfinite-task"/"result.json").read_text())
    assert record["execution_status"] == "failed"
    assert record["checkpoint"]["state"] == "not_checked"


def test_resident_task_state_is_checked_even_without_returned_tensors():
    from types import SimpleNamespace
    from tidegraph.experiment.resident import Resident
    from tidegraph.comparison import NonFiniteMismatch
    owner = Resident.__new__(Resident)
    owner.spec = {"return_mode": "metrics"}
    owner.training, owner.inputs, owner.span, owner.devices = False, [], 1, ()
    owner.task = SimpleNamespace(inputs=lambda *a, **kw: [],
        state_dict=lambda: {"progress": torch.tensor(float("inf"))})
    owner.owner = SimpleNamespace(advance=lambda *a, **kw: {"stats": {}})
    with pytest.raises(NonFiniteMismatch):
        owner.step(0)
