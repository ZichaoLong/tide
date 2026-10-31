"""Independent local anchors, earliest strict events and explicit FP64 policy."""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from core5_inference_difference import delta, first_kv_witness, first_strict_trace, literal_kv, literal_source
from weighted_numeric_differences import first_trace_difference
from tidegraph.planning import ExecutionSpec
from tidegraph.planning.precision import validate


def test_literal_kv_matches_hand_calculated_low_rank_write():
    memory = SimpleNamespace(config=SimpleNamespace(epsilon=3.5),
        base=torch.tensor([[1., 0.], [0., 1.], [1., 1.], [2., -1.]]),
        adapter_v=torch.tensor([[2., 0.]]), adapter_u=torch.tensor([[1.], [-1.], [2.], [0.]]))
    value = literal_kv(memory, torch.tensor([3., 4.]))
    torch.testing.assert_close(value["normal"], torch.tensor([.75, 1.], dtype=torch.float64), atol=0, rtol=0)
    torch.testing.assert_close(value["write"], torch.tensor([2.25, -.5, 4.75, .5], dtype=torch.float64), atol=0, rtol=0)
    assert memory.base.dtype == torch.float32


def test_literal_source_preserves_every_physical_atom_and_positive_mass():
    from tidegraph.records import Atom
    from tidegraph.source_class import SourceClasses
    from tidegraph.weighted.programs import WeightedAggregate
    program = WeightedAggregate("test", SourceClasses((), (0, 0, 0)), {}, 0., .1, 1)
    event = dict(fiber=[Atom(0, 0, 1, 0, p, 0, torch.tensor(value)) for p, value in
                       enumerate(([2., 1., 4.], [1., 4., 1.], [-1., 900., 900.]))])
    torch.testing.assert_close(literal_source(program, event), torch.tensor([3., 2., 3.], dtype=torch.float64), atol=0, rtol=0)


def test_literal_source_handles_private_low_rank_transform_before_weighting():
    from tidegraph.records import Atom
    from tidegraph.source_class import SourceClasses
    from tidegraph.weighted.programs import WeightedAggregate
    from tidegraph.weighted.source_binding import SourceTransform
    # A=1+1*3=4, B=1+1*1=2; norm(3)=3/4, so the hidden scalar is 3.
    param = lambda value: torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))
    transform = SourceTransform(param([[1.]]), param([[1.]]), param([0.]),
                                tuple(param([[v]]) for v in (1., 3., 1., 1.)))
    program = WeightedAggregate("test", SourceClasses((), (0,)), {0: transform}, .5, 7., 1)
    event = dict(fiber=[Atom(0, 0, 1, 0, 0, 0, torch.tensor([2., 3.]))])
    torch.testing.assert_close(literal_source(program, event),
        torch.tensor([2., 3.+3./(1.+math.exp(-3.))], dtype=torch.float64), atol=1e-14, rtol=0)


def test_actual_kv_trace_witness_diagnoses_a_changed_write_without_replaying():
    from tidegraph import Continuation
    from tidegraph.planning.materialize import materialize
    from tidegraph.planning.recipes import prepare
    from tidegraph.reference import run
    from weighted_benchmark_runners import snapshot
    plan, _ = prepare(dict(topology="fc", width=8, region_width=2, depth=2),
                      samples=1, positions=3, input_pattern="staggered-v1")
    bound, inputs, span = materialize(plan, max_bytes=1000000, dtype=torch.float64)
    with torch.no_grad():
        expected = snapshot(run(bound.graph, bound.model, Continuation(bound.graph.identity, 1),
                                inputs, span, sealed_until=span))
    actual = deepcopy(expected)
    first = min(actual.trace, key=lambda e: (e["time"], e["batch"], e["node"]))
    first["proposal_slots"]["keys"][-1, 0] += .01
    witness = first_kv_witness(bound, actual, expected)
    assert witness["state"] == "observed" and witness["fiber_exact_error"] is None
    assert witness["event"] == [first["time"], first["batch"], first["node"]]
    assert witness["stored_write"]["failing_coordinates"] == 1
    assert witness["candidate_write_vs_same_content_fp64"]["failing_coordinates"] == 1
    assert witness["reference_write_vs_same_content_fp64"]["failing_coordinates"] == 0
    assert witness["coherent_fiber_difference"]["exact"]


def test_first_exact_and_strict_event_are_distinct_and_chronological():
    expected = SimpleNamespace(trace=[dict(time=t, batch=0, node=0, content=torch.zeros(1)) for t in (3, 1)])
    actual = deepcopy(expected)
    actual.trace[0]["content"] += 2e-5
    actual.trace[1]["content"] += 1e-7
    assert first_trace_difference(actual, expected)["left_event"] == [1, 0, 0]
    assert first_strict_trace(actual, expected)["event"] == [3, 0, 0]
    actual.trace[1]["clock"] = torch.tensor(2**54+1, dtype=torch.int64)
    expected.trace[1]["clock"] = torch.tensor(2**54, dtype=torch.int64)
    assert first_strict_trace(actual, expected)["event"] == [1, 0, 0]
    actual.trace.pop()
    assert "identities differ" in first_strict_trace(actual, expected)["reason"]


def test_anchor_statistics_use_original_payload_tolerance_and_refuse_nonfinite():
    row = delta(torch.tensor([1.5e-6, 1e-7]), torch.zeros(2, dtype=torch.float64), torch.float32)
    assert row["failing_coordinates"] == 1 and row["atol"] == 1e-6 and row["rtol"] == 1e-5
    assert row["flat_index"] == 0 and not row["exact"]
    assert delta(torch.tensor([float("nan")]), torch.zeros(1), torch.float32)["reason"] == "nonfinite operand"


@pytest.mark.parametrize("client", ["pytorch", "python-native", "libtorch"])
def test_cuda_eager_fp64_has_explicit_independent_execution_identity(client):
    narrow = ExecutionSpec(client=client, device="cuda:0", placement="mixed-c").record()
    wide = ExecutionSpec(client=client, device="cuda:0", placement="mixed-c",
                         dtype="float64", scoring_dtype="float64").record()
    assert wide["execution_id"] != narrow["execution_id"]
    assert wide["requested"]["dtype"] == wide["effective"]["scoring_dtype"] == "float64"
    validate("float64", torch.device("cuda:0"), "python" if client == "pytorch" else "native")


@pytest.mark.parametrize("settings", [
    dict(device="cuda:0", placement="resident", client="python-native", scoring_dtype="float64"),
    dict(device="cuda:0", peer_devices=("cuda:0", "cuda:1"), scoring_dtype="float64"),
    dict(device="cuda:0", scoring_dtype="float32"),
    dict(device="npu:0", placement="mixed-a", scoring_dtype="float64")])
def test_unqualified_fp64_domains_stay_explicitly_rejected(settings):
    with pytest.raises(ValueError, match="FP64"):
        ExecutionSpec(dtype="float64", **settings)
    with pytest.raises(ValueError, match="FP64"):
        validate("float64", torch.device("cuda:0"), "resident")


def test_cpu_diagnostic_runs_actual_schedules_without_throughput_claim(tmp_path):
    from tidegraph.planning.recipes import prepare
    from durable_records import write_json
    plan, _ = prepare(dict(topology="fc", width=8, region_width=2, depth=2),
                      samples=2, positions=3, input_pattern="staggered-v1")
    write_json(tmp_path/"workload.json", plan)
    result = subprocess.run([sys.executable, str(ROOT/"scripts/diagnose_core5_inference.py"),
        "--workload", str(tmp_path/"workload.json"), "--device", "cpu", "--placement", "native",
        "--implementation", "python", "--dtype", "float64", "--windows", "1", "--allow-dirty",
        "--output-dir", str(tmp_path/"diagnostic")], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path/"diagnostic/result.json").read_text())
    assert report["state"] == "observed_match" and report["workload_id"] == plan["sha256"]
    assert report["windows"][0]["complete_error"] is None
    assert "metrics" not in report and "no throughput" in report["scope"]
