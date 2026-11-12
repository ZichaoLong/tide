"""Candidate-count packing against literal per-frame independent region calls."""
from dataclasses import replace
import os
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.history import History
from tidegraph.region import RegionInput, RegionLayout
from tidegraph.weighted.regional_groups import region_groups
from tidegraph.weighted.regional_packing import RegionalPacking
from test_weighted_region_batch import fixture


def device():
    from tidegraph.runtime import resolve_device
    return resolve_device(os.environ.get("TIDE_COLUMN_DEVICE", "cpu"))[0]


def material(dtype, case, target="cpu"):
    p, requests = fixture(dtype, case)
    p.to(target)
    domains = {0: [0, 2], 1: [0, 2], 2: [1, 3], 3: [1, 3], 4: [2, 3]}
    fields = {}
    for v, rows in domains.items():
        fields[v] = {name: torch.stack([requests[row].candidates[v % 2][1][name] for row in rows])
                     .to(target).detach().requires_grad_() for name in requests[0].candidates[0][1]}
    request = replace(requests[0], payload_device=torch.device(target))
    return p, RegionalPacking({v: torch.tensor(rows) for v, rows in domains.items()}), fields, request


def literal(p, packing, fields, request):
    outputs = {v: [] for v in fields}
    layout = RegionLayout.create(request.layout.spec, tuple(fields))
    for row in sorted({r for rows in packing.domains.values() for r in rows}):
        candidates = tuple((v, {name: t[rows.index(row)] for name, t in fields[v].items()})
                           for v, rows in packing.domains.items() if row in rows)
        r = replace(request, candidates=candidates, layout=layout, time=2**54+row)
        for v, value in p.step(r).controls.items():
            outputs[v].append(value)
    return {v: torch.stack(values) for v, values in outputs.items()}


def compare_roots(a, b, roots):
    ga = torch.autograd.grad(a, roots, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(b, roots, retain_graph=True, allow_unused=True)
    equivalent(ga, gb)
    return ga


@pytest.mark.parametrize("case", range(12))
def test_all_sending_and_scoring_with_disjoint_nodes(dtype, case):
    p, packing, fields, request = material(dtype, case, device())
    actual = region_groups(p, packing, fields, control=device(), selection=device())
    expected = literal(p, packing, fields, request)
    assert [(g.candidates, len(g.rows)) for g in packing.groups] == [(2, 2), (3, 2)]
    equivalent(actual, expected)
    roots = [*p.parameters(), *(t for f in fields.values() for t in f.values())]
    for v in (0, 2, 4):
        for factor in (1., 0.):
            compare_roots(factor*actual[v].square().sum(), factor*expected[v].square().sum(), roots)
    compare_roots(sum((v+1)*x.square().sum() for v, x in actual.items()),
                  sum((v+1)*x.square().sum() for v, x in expected.items()), roots)
    # Node 2 never shares a candidate row with output node 0. Its independent
    # descriptor roots stay None, even though both enter one physical kernel.
    grads = torch.autograd.grad(actual[0].sum(), list(fields[2].values()), retain_graph=True, allow_unused=True)
    assert all(t is None for t in grads)
    with torch.inference_mode():
        inference = region_groups(p, packing, fields, control=device(), selection=device())
    equivalent(inference, actual)
    assert all(not x.requires_grad for x in inference.values())


def test_caps_and_normal_branches_preserve_field_dependencies():
    from tidegraph import Graph, Node, Region
    from tidegraph.weighted import bind, KVConfig, RegionalConfig
    graph = Graph(tuple(Node(0) for _ in range(4)), (), (Region(1),), (0,), (0,))
    config = RegionalConfig(selection_mode="regional_total", region_control_mode="fixed", total_fraction=1., k=1)
    b = bind(graph, memories=KVConfig(width=2, summary_width=2), regions=config, dtype=torch.float64)
    p = b.model.regions[0]
    packing = RegionalPacking({v: torch.tensor([v//2]) for v in range(4)})
    fields = {v: {"w": torch.tensor([.1], dtype=torch.float64, requires_grad=True),
                  "mu": torch.tensor([.3 if v < 2 else .8], dtype=torch.float64, requires_grad=True),
                  "score": torch.tensor([.1*(v+1)], dtype=torch.float64, requires_grad=True),
                  "evidence": torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)} for v in range(4)}
    request = RegionInput(History(), 0, (), b.graph.region_layouts[0], torch.float64, torch.device("cpu"))
    actual = region_groups(p, packing, fields, control=torch.device("cpu"), selection=torch.device("cpu"))
    expected = literal(p, packing, fields, request)
    roots = [x for f in fields.values() for x in f.values()]
    for v in range(4):
        compare_roots(actual[v].sum(), expected[v].sum(), roots)
        compare_roots(actual[v].sum()*0, expected[v].sum()*0, roots)


def test_aliases_original_versions_and_first_order():
    p, packing, fields, request = material(torch.float64, 3)
    p.query.out_weight = p.query.in_weight
    fields[2]["score"] = fields[0]["score"]
    actual = region_groups(p, packing, fields, control=torch.device("cpu"), selection=torch.device("cpu"))
    expected = literal(p, packing, fields, request)
    roots = [*p.parameters(), fields[0]["score"]]
    compare_roots(sum(x.square().sum() for x in actual.values()), sum(x.square().sum() for x in expected.values()), roots)
    with pytest.raises(ValueError, match="first-order"):
        torch.autograd.grad(actual[0].sum(), roots, create_graph=True, allow_unused=True)
    with torch.no_grad():
        fields[2]["mu"].add_(.1)  # Even a descriptor unrelated to this output.
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        torch.autograd.grad(actual[0].sum(), roots, allow_unused=True)
