"""Column execution placement vs the independently placed scalar schedule."""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, ExecutionPlacement
from tidegraph.placement import place_model
from tidegraph.reference import run as reference
from tidegraph.comparison import equivalent
from tidegraph.weighted import KVConfig, RecurrentConfig, SettleColumnPlan, eager_runner
from test_settle_columns import general, inputs, external, observations, objective
from isolated_cases import vjp


CASES = [("kv", mode) for mode in ("0", "-", "+")]+[(f, "+") for f in ("sum", "affine", "delta", "gated_delta", "ssd")]


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("family,mode", CASES)
def test_temporary_scoring_columns_keep_raw_owners_and_shared_vjp(implementation, family, mode):
    dtype = torch.float32
    if family in {"kv", "sum"}:
        memory = KVConfig(memory_read="sum" if family == "sum" else "qry", epsilon=.3)
    else:
        args = dict(memory_family=family, memory_read="state", epsilon=.3)
        if family in {"gated_delta", "ssd"}:
            args.update(heads=2, conv_size=3)
        memory = RecurrentConfig(**args)
    binding, spec = general(dtype, memory=memory, read_mode=mode)
    domain, ports = inputs(dtype)
    placement = ExecutionPlacement("cpu", scoring_dtype="float64")
    plan = SettleColumnPlan(binding, spec, implementation=implementation, placement=placement)
    actual = plan.run_ports(domain, ports)
    expected = reference(binding.graph, place_model(binding.graph, binding.model, placement),
        Continuation(binding.graph.identity, 2), external(spec, domain, ports), 50, sealed_until=50)
    a, e = observations(actual, expected, spec)
    equivalent(a, e, atol=1e-6, rtol=1e-5)
    assert all(t.dtype == torch.float64 for n in actual.nodes.values() for t in n.descriptors.values())
    assert all(n.controls.dtype == dtype and n.window.content.dtype == dtype for n in actual.nodes.values())
    roots = {n: p for n, p in binding.model.named_parameters() if p.requires_grad}
    roots.update({f"{p}.{n}": getattr(m, n) for p, m in ports.items() for n in ("strength", "content")})
    equivalent(vjp(objective(a), roots), vjp(objective(e), roots), atol=1e-6, rtol=1e-5)
    equivalent(vjp(objective(a), roots, zero=True), vjp(objective(e), roots, zero=True), atol=0, rtol=0)
    public = eager_runner(binding, implementation, placement, trace=False).run_ports(spec, domain, ports, stop_position=10)
    assert public.execution['schedule'] == 'settle-columns-v1'
    assert public.execution['placement']['scoring_dtype'] == 'float64'


def test_read_copy_excludes_unused_full_and_update_matrices(monkeypatch):
    from tidegraph.weighted import column_placement
    binding, spec = general(torch.float32, memory=KVConfig(memory_read="sum"))
    domain, ports = inputs(torch.float32)
    seen = []
    original = column_placement.mapper
    def mapped(values, device, dtype):
        seen.extend(id(x) for x in values)
        return original(values, device, dtype)
    monkeypatch.setattr(column_placement, 'mapper', mapped)
    SettleColumnPlan(binding, spec, placement=ExecutionPlacement('cpu', scoring_dtype='float64')).run_ports(domain, ports)
    for n in binding.model.nodes:
        assert id(n.kernel.base) not in seen
        assert id(n.full_program.content.up) not in seen
        assert id(n.kernel.summary) in seen


def test_control_cast_cannot_remove_positive_support():
    from tidegraph.weighted.column_placement import controls
    with pytest.raises(ValueError, match='positive support'):
        controls(torch.tensor([1e-60], dtype=torch.float64), torch.ones(1, dtype=torch.float32))


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("mode", ["regional_total", "regional_total_zero", "pooled_attention"])
def test_placed_regional_heads_and_distinct_sending_maps(mode, implementation):
    kwargs = dict(local_score="match", region_score="att", region_key="content", region_evidence="grouped")
    if mode == "pooled_attention":
        kwargs.pop("region_evidence")
    if mode == "regional_total":
        kwargs.update(regularizer="ent", ent_alpha=1.5)
    binding, spec = general(torch.float32, mode, memory=KVConfig(epsilon=.3), **kwargs)
    domain, ports = inputs(torch.float32)
    p = ExecutionPlacement("cpu", scoring_dtype="float64")
    actual = SettleColumnPlan(binding, spec, placement=p, implementation=implementation).run_ports(domain, ports)
    expected = reference(binding.graph, place_model(binding.graph, binding.model, p), Continuation(binding.graph.identity, 2),
                         external(spec, domain, ports), 50, sealed_until=50)
    a, e = observations(actual, expected, spec)
    equivalent(a, e, atol=1e-6, rtol=1e-5)
    roots = {n: t for n, t in binding.model.named_parameters() if t.requires_grad}
    equivalent(vjp(objective(a), roots), vjp(objective(e), roots), atol=1e-6, rtol=1e-5)
