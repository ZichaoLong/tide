"""Default node-group dispatch and all finite Describe heads, independent oracle."""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation
from tidegraph.compare import equivalent
from tidegraph.content import Content
from tidegraph.reference import run as reference
from tidegraph.weighted import bind, fully_connected, KVConfig, SettleColumnPlan, layered_inputs
from tidegraph.experiment.work_observation import observe_work
from isolated_cases import vjp
from test_settle_columns import observations, objective
from test_weighted_describe_columns import CASES, regional, independent_head
from test_column_query_groups import make_queries, scalar
from tidegraph.weighted.programs import Describe
from tidegraph.weighted.kv_column_groups import read_kv_columns_many, full_kv_columns_many
from tidegraph.weighted.column_query_ops import contract


def independent_fields(cases):
    """Scalar memory schedule, literal heads and independent mass/content roots.

    Content's event adapter cats these two roots and consequently connects w to
    content and x to mass through zero derivatives. Ordinary column roots do not
    have that adapter; the literal head must consume their actual separate roots.
    """
    results = []
    for node, old, mass, content, layout, _ in cases:
        fields = []
        for b, (a, z) in enumerate(layout.spans):
            state = old[b]
            for j in range(a, z):
                time = int(layout.times[j])
                state = node.kernel.step(node, state, Content(torch.cat((mass[j:j+1], content[j]))), time)
                memory = node.kernel.mem(state, time, content[j])
                fields.append(independent_head(node.read_program, node.kernel, mass[j], content[j], memory))
        results.append({name: torch.stack([row[name] for row in fields]) for name in fields[0]})
    return results


@pytest.mark.parametrize("case", range(len(CASES)))
def test_grouped_all_heads_preserve_each_independent_field(case):
    cases, reads, fulls, roots = make_queries("qry")
    roots = {k: v for k, v in roots.items() if not any(part in k for part in ("read_program", "kernel", "full_program", "aggregate_program"))}
    for i, (node, *_rest) in enumerate(cases):
        node.read_program = Describe("node-group-check", node.kernel.config, regional(case), torch.Generator().manual_seed(101+i), torch.float64)
        node.read_program.score.bias.requires_grad_(False)
        for name, p in node.named_parameters():
            if p.requires_grad and all(p is not t for t in roots.values()):
                roots[f"{i}.{name}"] = p
    actual = read_kv_columns_many(reads, max_workspace_bytes=2**28)
    scalar_values, _ = scalar(cases, fulls, "+")
    equivalent(actual, scalar_values)
    expected = independent_fields(cases)
    equivalent(actual, expected)
    for field, value in actual[0].items():
        assert value.requires_grad == expected[0][field].requires_grad
        if not value.requires_grad:
            continue
        bar = torch.linspace(.2, .8, value.numel(), dtype=value.dtype).reshape_as(value)
        for zero in (False, True):
            a = vjp((value*bar).sum(), roots, zero=zero)
            e = vjp((expected[0][field]*bar).sum(), roots, zero=zero)
            equivalent(a, e)
            assert a["2.content"] is None


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("case", [0, 1, 4, 7])
def test_default_schedule_calls_real_groups_and_matches_scalar(case, implementation):
    spec = fully_connected(2, 3)
    binding = bind(spec.graph, memories=KVConfig(width=2, summary_width=2, heads=1, memory_read="qry"),
                   regions=regional(case), dtype=torch.float64)
    spec = replace(spec, graph=binding.graph)
    x = torch.linspace(.2, .7, 12, dtype=torch.float64).reshape(2, 3, 2).requires_grad_()
    plan = SettleColumnPlan(binding, spec, implementation=implementation)
    with observe_work(implementation) as work:
        actual = plan.run(x)
    metrics = work["metrics"]
    assert metrics["packing/bucket/kv_read_nodes/rows"] == 6
    assert metrics["packing/bucket/kv_read_nodes/calls"] == 2
    assert metrics["packing/projection/private_bmm/calls"] > 0
    expected = reference(binding.graph, binding.model, Continuation(binding.graph.identity, 2),
                         layered_inputs(spec, x), 3*spec.stride, sealed_until=3*spec.stride)
    a, e = observations(actual, expected, spec)
    equivalent(a, e)
    roots = {name: p for name, p in binding.model.named_parameters() if p.requires_grad} | {"input": x}
    equivalent(vjp(objective(a), roots), vjp(objective(e), roots))


def test_empty_banks_keep_query_parameters_unused_and_full_out_connected():
    cases, reads, fulls, roots = make_queries("qry")
    from tidegraph.weighted.kv_columns_state import KVStateColumns, prepare_kv_columns
    from tidegraph.weighted.kv_columns import read_kv_columns, full_kv_columns
    for i, (node, old, _mass, content, layout, _supplied) in enumerate(cases):
        left = KVStateColumns.from_states([node.initial() for _ in old])
        mass = torch.zeros(len(layout.times), dtype=content.dtype, requires_grad=True)
        window = prepare_kv_columns(node.kernel, layout, mass, content, left)
        reads[i] = node, window, "+", left
        fulls[i] = node, window, fulls[i][2]
    actual = read_kv_columns_many(reads, max_workspace_bytes=2**28)
    full = full_kv_columns_many(fulls, max_workspace_bytes=2**28)
    expected = [read_kv_columns(n, w, read_mode=m, initial=s) for n, w, m, s in reads]
    e_full = [full_kv_columns(*row) for row in fulls]
    equivalent(actual, expected);equivalent([v.content for v in full], [v.content for v in e_full])
    a = vjp(full[0].content.sum(), roots);e = vjp(e_full[0].content.sum(), roots)
    equivalent(a, e)
    assert a["0.full_program.content.query"] is None
    assert a["0.full_program.content.out"] is not None
    assert torch.count_nonzero(a["0.full_program.content.out"]) == 0


def test_contraction_frozen_output_and_higher_order_rejection():
    x = torch.ones(2, 3, 4, dtype=torch.float64, requires_grad=True)
    w = torch.ones(2, 4, 2, dtype=torch.float64, requires_grad=True)
    a, b = contract("matmul", [(x, w), (x.detach().clone(), w.detach().clone())])
    assert a.requires_grad and not b.requires_grad
    with pytest.raises(ValueError, match="first-order"):
        torch.autograd.grad(a.sum(), x, create_graph=True)
