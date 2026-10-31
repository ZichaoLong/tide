"""Complete observations keep exact storage values, private fields and resume."""
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from weighted_benchmark_return import _copy_group, snapshot


def exact(a, b):
    if isinstance(a, torch.Tensor):
        assert a.shape == b.shape and a.dtype == b.dtype and b.device.type == "cpu"
        # Byte comparison also checks signed zero, NaN payload and int64 clocks.
        assert torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                           b.contiguous().reshape(-1).view(torch.uint8))
        assert not b.requires_grad and b.grad_fn is None
    elif is_dataclass(a):
        assert type(a) is type(b)
        for field in fields(a):
            exact(getattr(a, field.name), getattr(b, field.name))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


@pytest.fixture
def device():
    # An explicit accelerator request must resolve or fail; never skip it.
    from tidegraph.runtime import resolve_device
    return resolve_device(os.environ.get("TIDE_RETURN_TEST_DEVICE", "cpu"))[0]


@pytest.mark.parametrize("storage_dtype", [torch.float16, torch.float32, torch.float64, torch.int64, torch.bool],
                         ids=["fp16", "fp32", "fp64", "int64", "bool"])
def test_literal_packed_copy_is_byte_exact_and_each_occurrence_is_private(device, storage_dtype):
    values = [2**54+7, -1, 0, 2**54+8] if storage_dtype == torch.int64 else (
        [True, False, False, True] if storage_dtype == torch.bool else [0., -0., float("nan"), float("inf")])
    a = torch.tensor(values, dtype=storage_dtype, device=device)
    b = torch.ones((2, 2), dtype=storage_dtype, device=device)
    outputs = [None]*3
    _copy_group([(a, [0, 2]), (b, [1])], outputs)
    exact(a, outputs[0]); exact(b, outputs[1]); exact(a, outputs[2])
    assert len({x.untyped_storage().data_ptr() for x in outputs}) == 3
    original = outputs[2].clone()
    outputs[0][0] = 1
    exact(a, original); exact(original, outputs[2])


def test_full_snapshot_preserves_none_layout_and_original_autograd_roots(device, monkeypatch):
    import weighted_benchmark_return as returns
    monkeypatch.setattr(returns, "PACK_BYTES", 32)
    groups, copy_group = [], returns._copy_group
    def bounded(rows, outputs):
        size = sum(value.numel()*value.element_size() for value, _ in rows)
        assert size <= returns.PACK_BYTES
        groups.append(size)
        copy_group(rows, outputs)
    monkeypatch.setattr(returns, "_copy_group", bounded)
    a = torch.arange(4., device=device, requires_grad=True)
    b = torch.ones((), device=device, requires_grad=True)
    disconnected = torch.ones((), device=device, requires_grad=True)
    large = torch.arange(12., device=device)
    noncontiguous = large.reshape(3, 4).t()
    value = dict(first=a, again=a, view=a.view(2, 2), control=b, absent=None,
                 clock=torch.tensor([2**54+7, -1], device=device), mask=torch.tensor([True, False], device=device),
                 empty=torch.empty((0, 3), device=device), transposed=noncontiguous, large=large)
    before = snapshot(value)
    versions = [x._version for x in (a, b, disconnected, large)]
    packed = snapshot(value, packing="packed")
    if device.type != "cpu":
        assert groups  # The explicit device request exercised the packed path.
    exact(before, packed)
    assert packed["transposed"].stride() == before["transposed"].stride()
    assert packed["large"].stride() == before["large"].stride()
    packed["first"].fill_(99)
    exact(before["again"], packed["again"]); exact(before["view"], packed["view"])
    exact(before, snapshot(value))
    assert versions == [x._version for x in (a, b, disconnected, large)]
    ga, gb, gc = torch.autograd.grad(3*a.sum()+0*b, (a, b, disconnected), allow_unused=True)
    assert torch.equal(ga, torch.full_like(a, 3)) and gb.item() == 0 and gc is None
    with pytest.raises(ValueError, match="return packing"):
        snapshot(value, packing="auto")


@pytest.mark.parametrize("mode", ["inference", "training"])
def test_real_complete_records_two_windows_and_exact_new_owner_resume(tmp_path, mode, device):
    from tidegraph.compare import equivalent
    from weighted_benchmark_cases import Workload, fixture
    from weighted_benchmark_precision import promote
    from weighted_benchmark_runners import Eager
    from weighted_benchmark_resume import check
    config = Workload("pdg", "kv", width=4, samples=2, positions=2)
    def create():
        binding, inputs, span = fixture(config)
        binding, inputs = promote(binding, inputs, torch.float64)
        return binding, inputs, span
    args = SimpleNamespace(mode=mode, implementation="python", placement="native", schedule="prefill",
        dtype="float64", scoring_dtype="float64", profile=False, steps=1, packing="packed",
        return_packing="packed", workers=1, native_library=None, peer_devices=None)
    candidate = Eager(*create(), config, args, device)
    reference = Eager(*create(), config, args, torch.device("cpu"), reference=True)
    assert candidate.return_packing == "packed" and reference.return_packing == "scalar"
    try:
        for index in range(2):
            actual, expected = candidate.step(index), reference.step(index)
            equivalent(actual[0], expected[0])
            torch.testing.assert_close(torch.tensor(actual[1], dtype=torch.float64),
                                       torch.tensor(expected[1], dtype=torch.float64), atol=1e-10, rtol=1e-8)
            result = actual[0]
            assert result.trace and result.messages and result.continuation.pending and result.continuation.history
            exact(result, snapshot(result, packing="packed"))
            # Returned observations never become the next live continuation.
            result.continuation.pending[0].value.fill_(99)
        equivalent(candidate.parameters(), reference.parameters())
        equivalent(candidate.training_state(), reference.training_state())
        assert check(candidate, create, config, args, device, tmp_path)["exact"]
    finally:
        candidate.close(); reference.close()


def test_identity_launcher_and_evidence_do_not_confuse_operator_and_return_packing():
    from tidegraph.planning import ExecutionSpec
    from tidegraph.planning.study import case, command
    from core5_study_records import accept_result
    for client in ("pytorch", "python-native", "libtorch"):
        scalar = ExecutionSpec(client=client).record()
        packed = ExecutionSpec(client=client, return_packing="packed").record()
        assert scalar["execution_id"] != packed["execution_id"]
        assert scalar["effective"]["packing"] == packed["effective"]["packing"]
    with pytest.raises(ValueError, match="eager"):
        ExecutionSpec(client="python-native", device="cuda:0", placement="resident", return_packing="packed")
    with pytest.raises(ValueError, match="return packing"):
        ExecutionSpec(return_packing="auto")
    study = json.loads((ROOT/"benchmarks/core5-study-v1.json").read_text())
    for name in ("python-cpu", "cpp-cpu"):
        study["executions"][name]["return_packing"] = "packed"
        record = case(study, "check-fc-kv", name)
        argv = command(record, python="PYTHON", root=ROOT, output="OUT", workload="PLAN",
                       standalone_binary="CPP", fixture="MATERIAL")
        assert argv[argv.index("--return-packing")+1] == "packed"
        result = dict(state="passed", post_measurement_complete_records=True,
                      workload_id=record["plan"]["sha256"], requested_dtype="float32")
        with pytest.raises(ValueError, match="return packing"):
            accept_result(record, result)


def test_standalone_packed_return_checks_actual_forward_and_disk_resume(tmp_path, device):
    binary = os.environ.get("TIDE_WEIGHTED_BENCHMARK")
    if not binary:
        pytest.skip("standalone qualification supplies TIDE_WEIGHTED_BENCHMARK")
    fixture = tmp_path/"fixture.json"
    made = subprocess.run([sys.executable, str(ROOT/"scripts/export_weighted_benchmark.py"),
        "--native-library", os.environ["TIDE_BUILD_DIR"], "--topology", "pdg", "--output", str(fixture)],
        capture_output=True, text=True, timeout=30)
    assert made.returncode == 0, made.stderr
    out = tmp_path/"run"
    result = subprocess.run([binary, "--fixture", str(fixture), "--device", str(device), "--dtype", "float64",
        "--mode", "inference", "--steps", "1", "--return-packing", "packed", "--checkpoint-check", "true",
        "--output-dir", str(out)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    record = json.loads((out/"result.json").read_text())
    assert record["return_packing"] == "packed" and record["post_measurement_complete_records"]
    assert record["exact_resume"]["exact"] and record["metrics"][0]["pending"]
    assert any(row["phase"] == "exact_resume" for row in record["memory"]["phases"])
    checked = subprocess.run([str(Path(binary).with_name("weighted-return-check")), "--device", str(device), "--dtype", "float64"],
                             capture_output=True, text=True, timeout=30)
    assert checked.returncode == 0, checked.stderr
    refused = subprocess.run([binary, "--fixture", str(fixture), "--device", "cuda:0",
        "--implementation", "resident", "--return-packing", "packed", "--output-dir", str(tmp_path/"bad")],
        capture_output=True, text=True, timeout=10)
    assert refused.returncode and "only to eager" in refused.stderr and not (tmp_path/"bad").exists()
