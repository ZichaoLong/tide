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


@pytest.mark.parametrize("device,implementation", [("npu", "python"), ("cuda", "native"), ("cpu", "resident")])
def test_fp64_benchmark_refuses_an_undeclared_path(device, implementation):
    with pytest.raises(ValueError, match="CPU eager"):
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
