"""Cadence changes durability, never inference/optimizer window semantics."""
from copy import deepcopy
import json
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.experiment.config import resolve
from tidegraph.experiment.engine import run
from tidegraph.experiment.checkpoint import locate
from tidegraph.experiment.checkpoint_policy import publication_plan, due


def request(mode="training", interval=1, steps=3):
    return dict(schema="tide-experiment-v1", model=dict(recipe=dict(topology="fc", depth=2,
        region_width=1, width=4)), task=dict(samples=2, positions=2, input_pattern="staggered-v1"),
        execution=dict(device="cpu"), measurement=dict(mode=mode, steps=steps, warmup=0,
                                                     checkpoint_interval=interval))


@pytest.mark.parametrize("mode", ["inference", "training"])
def test_final_only_and_every_window_have_exact_states_slots_and_next_window(tmp_path, mode):
    every = run(request(mode), tmp_path/"every")
    final = run(request(mode, interval=3), tmp_path/"final")
    assert every["state"] == final["state"] == "passed"
    assert every["checkpoint"]["exact_resume"] and final["checkpoint"]["exact_resume"]
    assert [r["checkpoint_scheduled"] for r in final["metrics"]] == [False, False, True]
    left, a = locate(tmp_path/"every")
    right, b = locate(tmp_path/"final")
    assert a["next_window"] == b["next_window"] == 3
    for name in ("state.pt", "task.pt"):
        equivalent(torch.load(left/name, weights_only=True), torch.load(right/name, weights_only=True), atol=0, rtol=0)
    assert final["admission"]["disk"]["checkpoint_generations"] == 1


def test_failure_keeps_only_the_last_published_boundary(tmp_path, monkeypatch):
    from tidegraph.experiment.eager import Eager
    original = Eager.step
    def step(self, index, **kwargs):
        if index == 2:
            raise RuntimeError("injected failure in unsaved window")
        return original(self, index, **kwargs)
    monkeypatch.setattr(Eager, "step", step)
    with pytest.raises(RuntimeError, match="unsaved window"):
        run(request(interval=2, steps=4), tmp_path/"failed")
    _, metadata = locate(tmp_path/"failed")
    assert metadata["next_window"] == 2
    result = json.loads((tmp_path/"failed/result.json").read_text())
    assert result["execution_status"] == "failed" and result["checkpoint"]["exact_resume"] == "not_checked"


def test_resume_alignment_changes_generation_budget_and_final_is_always_saved():
    m = dict(warmup=0, steps=3, checkpoint_interval=3)
    initial = publication_plan(m)
    resumed = publication_plan(m, 1)
    assert initial["simultaneous_generations"] == 1 and resumed["simultaneous_generations"] == 2
    assert [i for i in range(2, 5) if due(resumed, i)] == [3, 4]
    for start in range(5):
        plan = publication_plan(dict(m, steps=1), start)
        assert due(plan, start+1) and plan["publications"] == 1


def test_tmpfs_reduction_and_default_identity_remain_explicit():
    c = request(interval=3)
    c["resources"] = dict(storage_medium="tmpfs")
    final = resolve(c)["capacity_budget"]
    c["measurement"]["checkpoint_interval"] = 1
    every = resolve(c)["capacity_budget"]
    assert every["host"]["future_tmpfs_file_bytes"] == 2*final["host"]["future_tmpfs_file_bytes"]
    assert every["disk"]["estimated_new_bytes"]-final["disk"]["estimated_new_bytes"] == final["checkpoint_policy"]["estimated_file_tensor_bytes"]
    with pytest.raises(ValueError, match="checkpoint_interval"):
        bad = deepcopy(c);bad["measurement"]["checkpoint_interval"] = 0;resolve(bad)
    with pytest.raises(ValueError, match="standalone currently requires 1"):
        c["measurement"]["checkpoint_interval"] = 3;c["execution"]["client"] = "libtorch";resolve(c)


def test_exact_cpu_recovery_keeps_original_integer_float_and_finite_contracts():
    from tidegraph.compare import DiscreteMismatch, NumericMismatch, NonFiniteMismatch, collect_comparison
    clock = torch.tensor([2**60+1, 2**60+2], dtype=torch.int64)
    value = torch.tensor([[0., -0.], [2., -3.]]).t()
    equivalent(dict(clock=clock, value=value), dict(clock=clock.clone(), value=value.clone()), atol=0, rtol=0)
    with pytest.raises(DiscreteMismatch):
        equivalent(clock, clock+1, atol=0, rtol=0)
    with pytest.raises(NumericMismatch):
        equivalent(value, value+1e-7, atol=0, rtol=0)
    with pytest.raises(NonFiniteMismatch):
        equivalent(torch.tensor([float("nan")]), torch.tensor([float("nan")]), atol=0, rtol=0)
    report = collect_comparison(value, value.clone(), atol=0, rtol=0)
    assert report["compared_leaves"]["numeric"] == 1 and report["strict_passed"]
