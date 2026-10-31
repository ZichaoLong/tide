"""Root failures remain failures, with durable evidence of the failed boundary."""
import json
import pytest
from tidegraph.experiment import checks


def configuration():
    return dict(schema="tide-experiment-v1", model=dict(recipe=dict(topology="fc", depth=2,
        region_width=1, width=4, history=4)), task=dict(samples=2, positions=2,
        input_pattern="staggered-v1"), execution=dict(device="cpu"),
        measurement=dict(mode="inference", steps=1, warmup=0))


@pytest.mark.parametrize("failed_call,stage,completed", [
    (1, "reference-vjp", []), (2, "candidate-vjp", []),
    (4, "candidate-vjp", ["all"]),
])
def test_failed_root_is_identified_without_marking_untested_training_passed(
        tmp_path, monkeypatch, failed_call, stage, completed):
    original, count = checks._vjp, 0
    def fail(loss, leaves):
        nonlocal count
        count += 1
        if count == failed_call:
            live = json.loads((tmp_path/"check/result.json").read_text())
            assert live["state"] == "running"
            assert live["phase"] == "independent-roots"
            assert live["root_progress"]["stage"] == stage
            raise RuntimeError("injected event failure")
        return original(loss, leaves)
    monkeypatch.setattr(checks, "_vjp", fail)
    with pytest.raises(RuntimeError, match="injected event failure"):
        checks.check(configuration(), tmp_path/"check", "full")
    record = json.loads((tmp_path/"check/result.json").read_text())
    assert record["state"] == "failed" and record["error"]["type"] == "RuntimeError"
    assert record["root_progress"] == dict(stage=stage,
        root="all" if failed_call < 3 else "first-output", completed_roots=completed)
    assert [c["name"] for c in record["checks"]] == ["complete-independent-inference-and-exact-resume"]
    assert not (tmp_path/"check/training").exists()


def test_completed_check_keeps_all_roots_and_training(tmp_path):
    record = checks.check(configuration(), tmp_path/"check", "full")
    assert record["state"] == "passed" and record["phase"] == "complete"
    roots = next(c for c in record["checks"] if c["name"] == "independent-roots")["roots"]
    assert record["root_progress"]["completed_roots"] == roots
    assert "connected-zero" in roots and "private-raw-state-roots-after-nonposition-cut" in roots
    assert record["checks"][-1]["name"] == "complete-independent-training-and-exact-resume"
    assert record["checks"][-1]["state"] == "passed"
