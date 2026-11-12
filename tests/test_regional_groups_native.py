"""LibTorch node-group roots checked against independent scalar formulas."""
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.weighted.native import region_config
from test_regional_groups import material, literal, compare_roots, device


def execute(program, packing, fields):
    import _tide_native as core
    weights = core.RegionWeights()
    weights.extra = dict(program.named_parameters(remove_duplicate=False))
    core.configure_weighted_region(weights, region_config(core, program.config))
    # Match the literal formula's device. The standalone native default keeps
    # selection on CPU; callers, including the Settle plan, bind placement.
    placement = core.ExecutionPlacement()
    placement.control = "payload"
    placement.selection = "payload"
    return core.weighted_region_groups(weights, packing.domains, fields, placement)


@pytest.mark.parametrize("case", range(12))
def test_native_all_sending_scoring_and_disjoint_roots(dtype, case):
    program, packing, fields, request = material(dtype, case, device())
    actual = execute(program, packing, fields)
    expected = literal(program, packing, fields, request)
    equivalent(actual, expected)
    roots = [*program.parameters(), *(value for row in fields.values() for value in row.values())]
    for node in (0, 2, 4):
        for factor in (1., 0.):
            compare_roots(factor*actual[node].square().sum(), factor*expected[node].square().sum(), roots)
    disconnected = torch.autograd.grad(actual[0].sum(), list(fields[2].values()), retain_graph=True, allow_unused=True)
    assert all(value is None for value in disconnected)
    with torch.inference_mode():
        equivalent(execute(program, packing, fields), actual)


def test_native_aliases_versions_and_declared_first_order():
    program, packing, fields, request = material(torch.float64, 3)
    program.query.out_weight = program.query.in_weight
    fields[2]["score"] = fields[0]["score"]
    actual = execute(program, packing, fields)
    expected = literal(program, packing, fields, request)
    roots = [*program.parameters(), fields[0]["score"]]
    compare_roots(sum(x.square().sum() for x in actual.values()), sum(x.square().sum() for x in expected.values()), roots)
    with pytest.raises((ValueError, RuntimeError), match="first-order"):
        torch.autograd.grad(actual[0].sum(), roots, create_graph=True, allow_unused=True)
    with torch.no_grad():
        fields[2]["mu"].add_(.1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        torch.autograd.grad(actual[0].sum(), roots, allow_unused=True)


def test_native_caps_row_does_not_connect_scores_or_disjoint_nodes():
    from tidegraph import Graph, Node, Region
    from tidegraph.history import History
    from tidegraph.region import RegionInput
    from tidegraph.weighted import bind, KVConfig, RegionalConfig
    from tidegraph.weighted.regional_packing import RegionalPacking
    graph = Graph(tuple(Node(0) for _ in range(4)), (), (Region(1),), (0,), (0,))
    config = RegionalConfig(selection_mode="regional_total", region_control_mode="fixed", total_fraction=1., k=1)
    bound = bind(graph, memories=KVConfig(width=2, summary_width=2), regions=config, dtype=torch.float64)
    program = bound.model.regions[0]
    packing = RegionalPacking({v: torch.tensor([v//2]) for v in range(4)})
    fields = {v: dict(w=torch.tensor([.1], dtype=torch.float64, requires_grad=True),
                      mu=torch.tensor([.3 if v < 2 else .8], dtype=torch.float64, requires_grad=True),
                      score=torch.tensor([.1*(v+1)], dtype=torch.float64, requires_grad=True),
                      evidence=torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)) for v in range(4)}
    request = RegionInput(History(), 0, (), bound.graph.region_layouts[0], torch.float64, torch.device("cpu"))
    actual = execute(program, packing, fields)
    expected = literal(program, packing, fields, request)
    roots = [value for row in fields.values() for value in row.values()]
    for node in range(4):
        compare_roots(actual[node].sum(), expected[node].sum(), roots)
        compare_roots(actual[node].sum()*0, expected[node].sum()*0, roots)


def test_native_literal_ids_and_linear_publication(dtype):
    import _tide_native as core
    from test_regional_packing import fixture, BIG
    domains, source = fixture(dtype)
    rows, nodes, fields, values = core.weighted_regional_packing({v: r.tolist() for v, r in domains.items()}, source)
    assert rows == [[11], [2, 5, 8]]
    assert nodes[1].tolist() == [[3, 90], [3, BIG], [90, BIG]]
    assert nodes[1].dtype == torch.int64
    for node in domains:
        equivalent(values[node], source[node]['w'])
    # The publication transpose returns None for an entirely unused group.
    bars = torch.autograd.grad(values[99].sum()*0, [p['w'] for p in fields], allow_unused=True)
    assert bars[1] is None
    equivalent(bars[0], torch.zeros_like(fields[0]['w']))
