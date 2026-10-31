"""Precision/verification controls must not hide a change of material or slots."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from weighted_benchmark_cases import Workload, fixture
from weighted_benchmark_precision import promote, validate
from weighted_plan_materialize import materialize
from weighted_topology import topology
from weighted_workload_plan import make_plan


def test_explicit_resident_capacities_do_not_accept_algorithm_or_lossy_integer_options(tmp_path):
    from weighted_benchmark_limits import load
    p = tmp_path/"limits.json"
    p.write_text('{"queue": 4096, "stages": 32, "cache": 4}')
    limits, sha = load(p)
    assert limits == dict(queue=4096, stages=32, cache=4) and len(sha) == 64
    for values in ({"prefill": True}, {"cache": 0}, {"cache": 1.5}, {"cache": True}, {"stages": 2**63}):
        p.write_text(json.dumps(values))
        with pytest.raises(ValueError):load(p)


def test_explicit_promotion_preserves_material_and_parameter_input_aliases():
    b, xs, _ = fixture(Workload())
    original = {n:p.detach().double() for n,p in b.model.named_parameters()}
    aliases = {}
    for n,p in b.model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(p), set()).add(n)
    # Add an equal-valued independent input. It must not join the real broadcast.
    from dataclasses import replace
    xs += [replace(xs[0], value=xs[0].value.clone())]
    ids = [id(x.value) for x in xs]
    b, ys = promote(b, xs, torch.float64)
    for i in range(len(xs)):
        assert torch.equal(ys[i].value, xs[i].value.double())
        for j in range(len(xs)):
            assert (ys[i].value is ys[j].value) == (ids[i] == ids[j])
    after = {}
    for n,p in b.model.named_parameters(remove_duplicate=False):
        after.setdefault(id(p), set()).add(n)
    assert {frozenset(x) for x in aliases.values()} == {frozenset(x) for x in after.values()}
    for n,p in b.model.named_parameters():
        assert p.dtype == torch.float64 and torch.equal(p, original[n])


def test_fp64_materialization_charges_requested_storage_before_allocation():
    p, _ = make_plan(topology("fc", depth=2, region_width=2), width=4, samples=1, positions=2)
    floor = p["memory_floor"]
    required = floor["logical_parameter_bytes"]+floor["legacy_inert_buffer_bytes"]
    with pytest.raises(ValueError, match="ceiling"):
        materialize(p, max_bytes=2*required-1, dtype=torch.float64)
    b, xs, _ = materialize(p, max_bytes=2*required, dtype=torch.float64)
    assert all(x.dtype == torch.float64 for x in b.model.parameters())
    assert all(x.value.dtype == torch.float64 for x in xs)


@pytest.mark.parametrize("device,implementation", [("npu", "python"), ("cuda", "resident"), ("cpu", "resident")])
def test_fp64_benchmark_refuses_an_undeclared_path(device, implementation):
    with pytest.raises(ValueError, match="FP64 benchmark requires CPU or single-card CUDA eager"):
        validate("float64", SimpleNamespace(type=device), implementation)


@pytest.mark.parametrize("field", ["optimizer", "gradients"])
def test_python_benchmark_rejects_bad_training_state_with_unchanged_parameters(tmp_path, monkeypatch, field):
    import benchmark_weighted
    from weighted_benchmark_runners import Eager
    original = Eager.training_state
    count = 0
    def corrupt(candidate):
        nonlocal count
        value = original(candidate)
        if count == 0:
            if field == "optimizer":
                next(iter(value["optimizer"]["state"].values()))["momentum_buffer"].add_(1)
            else:
                name = next(n for n,v in value["gradients"].items() if v is not None)
                value["gradients"][name] = None
        count += 1
        return value
    monkeypatch.setattr(Eager, "training_state", corrupt)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    out = tmp_path/"run"
    monkeypatch.setattr(sys, "argv", ["benchmark_weighted.py", "--device", "cpu", "--implementation", "python",
        "--dtype", "float64", "--steps", "1", "--allow-dirty", "--output-dir", str(out)])
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    with pytest.raises(AssertionError):
        benchmark_weighted.main()
    result = json.loads((out/"result.json").read_text())
    assert result["state"] == "failed" and len(result["metrics"]) == 1
    assert result["precision"]["effective_dtype"] == "float64"
    assert not result.get("optimizer_values_checked", False)


def test_eager_benchmark_checks_actual_new_owner_next_window(tmp_path, monkeypatch):
    import benchmark_weighted
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    out = tmp_path/"resume"
    monkeypatch.setattr(sys, "argv", ["benchmark_weighted.py", "--device", "cpu", "--implementation", "python",
        "--dtype", "float64", "--steps", "1", "--checkpoint-check", "--allow-dirty", "--output-dir", str(out)])
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    benchmark_weighted.main()
    record = json.loads((out/"result.json").read_text())
    assert record["state"] == "passed" and record["exact_resume"]["exact"]
    assert record["exact_resume"]["next_window"] == 2 and record["optimizer_values_checked"]


def test_half_benchmark_records_master_precision_threads_and_exact_restore(tmp_path, monkeypatch):
    import benchmark_weighted
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["benchmark_weighted.py", "--device", "cpu", "--implementation", "python",
        "--dtype", "float16", "--scoring-dtype", "float32", "--steps", "1", "--intraop-threads", "2",
        "--checkpoint-check", "--allow-dirty", "--output-dir", str(tmp_path/"half")])
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    try:
        benchmark_weighted.main()
    finally:
        torch.set_num_threads(1)
    record = json.loads((tmp_path/"half/result.json").read_text())
    assert record["state"] == "passed" and record["exact_resume"]["exact"]
    assert record["precision"]["master_dtype"] == "float32" and record["precision"]["loss_scale"] == 1
    assert record["effective_threads"]["intraop"] == 2
    assert record["execution"]["requested"]["dtype"] == "float16"
    assert record["final_parameter_gradients_checked"] and record["optimizer_values_checked"]
