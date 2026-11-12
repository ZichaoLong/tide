"""Finite LH checker: independent small graphs and classification witnesses."""
import copy
import os
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from lh_tensor_check import evaluate, compare, capture, raw_columns
from test_lh_tensor_baseline import fixture


@pytest.mark.parametrize("client", ["python", "native"])
@pytest.mark.parametrize("level", ["public", "default"])
def test_small_lh_all_raw_control_support_ports_and_shared_task_vjp(client, level):
    anchor = fixture(torch.float64, profile="textbook")
    x = torch.linspace(-.2, .4, 24, dtype=torch.float64).reshape(2, 3, 4)
    support = torch.tensor([[[True, False, True], [True, True, False]],
                            [[True, True, False], [False, True, False]]])
    plan = dict(positions=3)
    options = dict(client=client, library=os.environ.get("TIDE_BUILD_DIR"), capture_bytes=2**24)
    reference, direct = evaluate(anchor.binding, anchor.spec, plan, x, support, level="direct", **options)
    actual, candidate = evaluate(anchor.binding, anchor.spec, plan, x, support, level=level, **options)
    report = compare(actual, reference)
    assert report["strict_numerical_projection_passed"], report
    for name in ("candidate_events", "selected_events", "visited_edges"):
        assert direct["actual_work"][name] == candidate["actual_work"][name]
    assert len(actual["controls"]) == len(anchor.binding.graph.nodes)
    assert all(t.dtype == torch.int64 for name, t in actual["raw"].items()
               if name.endswith(("times", "last_time", "observations")))
    assert len(actual["outputs"]) == len(anchor.binding.graph.outputs)
    assert candidate["root_count"] == direct["root_count"]
    broken = copy.deepcopy(actual)
    last = next(name for name in reversed(broken["raw"]) if name.endswith("times") and broken["raw"][name].numel())
    broken["raw"][last][-1] += 1
    assert compare(broken, reference)["hard_failure"]


def test_classification_distinguishes_rounding_nonfinite_support_and_connections():
    expected = {"raw": {"x": torch.ones(2)}, "active": {"n": torch.tensor([True])},
                "gradients": {"unused": None}}
    actual = copy.deepcopy(expected)
    actual["raw"]["x"][-1] += .001
    report = compare(actual, expected)
    assert report["numerical_failure"] and not report["hard_failure"]
    actual["active"]["n"][0] = False
    assert compare(actual, expected)["hard_failure"]
    actual = copy.deepcopy(expected); actual["raw"]["x"][-1] = float("nan")
    report = compare(actual, expected)
    assert report["hard_failure"] and report["sections"]["raw"]["failure_kind"] == "nonfinite"
    actual = copy.deepcopy(expected); actual["gradients"]["unused"] = torch.zeros(3)
    report = compare(actual, expected)
    assert report["strict_numerical_projection_passed"]
    assert not report["sections"]["gradients"]["connection_equivalent"]
    actual["gradients"]["unused"][2] = 1.
    report = compare(actual, expected)
    assert report["numerical_failure"] and not report["hard_failure"]


def test_candidate_raw_decode_only_reads_actual_columns_and_capture_is_bounded():
    from types import SimpleNamespace
    from tidegraph.weighted.column_layout import SequenceLayout
    from tidegraph.weighted.kv_columns_state import KVStateColumns
    keys = torch.tensor([[2., 3.]], requires_grad=True)
    state = KVStateColumns(SequenceLayout(torch.tensor([0, 1]), torch.tensor([2**54+7])),
        keys, keys*2, keys*3, torch.tensor([.5]), torch.zeros(1, 3),
        torch.tensor([2**54+7]), torch.tensor([4]))
    result = raw_columns(SimpleNamespace(states={3: state}))
    assert result["node/3/sample/0/times"].item() == 2**54+7
    assert torch.equal(result["node/3/sample/0/keys"], keys)
    gradient, = torch.autograd.grad(result["node/3/sample/0/keys"].sum(), keys)
    assert torch.equal(gradient, torch.ones_like(keys))
    with pytest.raises(MemoryError, match="capture"):
        capture({"raw": result}, 1)
