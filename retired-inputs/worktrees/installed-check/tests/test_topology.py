"""Independent structural witnesses for the pinned current instance graphs."""
from dataclasses import replace
from itertools import product

import pytest

from tide.topology import TopologyConfig, build_topology


def address(node):
    return node.layer, node.position, node.role, node.local


def endpoints(graph, source_class=None):
    return {(address(graph.nodes[e.src]), address(graph.nodes[e.dst]), e.source_class)
            for e in graph.edges if source_class is None or e.source_class == source_class}


def assert_records(graph):
    assert tuple(n.id for n in graph.nodes) == tuple(range(len(graph.nodes)))
    assert tuple(e.id for e in graph.edges) == tuple(range(len(graph.edges)))
    assert tuple(address(n) for n in graph.nodes) == tuple(sorted(address(n) for n in graph.nodes))
    assert sorted(i for group in graph.regions for i in group) == list(range(len(graph.nodes)))
    assert {i for edges in graph.incoming for i in edges} == set(range(len(graph.edges)))
    assert {i for edges in graph.outgoing for i in edges} == set(range(len(graph.edges)))
    for region, members in enumerate(graph.regions):
        assert all(graph.nodes[i].region == region and graph.ranks[i] == graph.region_ranks[region]
                   for i in members)
    for edge in graph.edges:
        assert edge.id in graph.incoming[edge.dst] and edge.id in graph.outgoing[edge.src]
        assert graph.ranks[edge.src] < graph.ranks[edge.dst]
        assert graph.nodes[edge.src].region != graph.nodes[edge.dst].region
        assert edge.delay > 0
    assert graph.input_nodes == tuple(n.id for n in graph.nodes if n.layer == 0)
    assert graph.output_nodes == tuple(n.id for n in graph.nodes
                                       if n.role == "D" and n.layer == len(graph.widths) - 1)
    assert not any(n.role == "H" and n.layer == len(graph.widths) - 1 for n in graph.nodes)


@pytest.mark.parametrize("layers,n", [(1, 1), (1, 8), (4, 3), (8, 8)])
def test_fc_closed_form(layers, n):
    graph = build_topology(TopologyConfig(layers=layers, n=n))
    assert len(graph.nodes) == layers * n
    assert len(graph.regions) == layers
    assert len(graph.edges) == (layers - 1) * n * n
    assert graph.period == layers + 2
    assert graph.input_time(7) == (layers + 2) * 7 + 1
    assert all(e.source_class == "DD" and e.delay == 1 for e in graph.edges)
    if layers == 1:
        assert graph.input_nodes == graph.output_nodes
    assert_records(graph)


@pytest.mark.parametrize("depth,n", [(1, 1), (1, 3), (3, 8)])
def test_lattice_independent_word_addresses(depth, n):
    graph = build_topology(TopologyConfig(kind="lattice", depth=depth, n=n))
    expected = set()
    for layer in range(2 * depth):
        words = product((0, 1), repeat=min(layer, 2 * depth - layer))
        for word in words:
            descendants = [word + (bit,) for bit in (0, 1)] if layer < depth else [word[:-1]]
            position = int("".join(map(str, word)) or "0", 2)
            for target_word in descendants:
                target = int("".join(map(str, target_word)) or "0", 2)
                for i in range(n):
                    for offset in {0, 1 % n}:
                        expected.add(((layer, position, "D", i),
                                      (layer + 1, target, "D", (i + offset) % n), "DDL"))
    assert endpoints(graph) == expected
    assert len(graph.regions) == 3 * 2 ** depth - 2
    assert graph.period == 2 * depth + 3
    assert_records(graph)


@pytest.mark.parametrize("depth,plain_nodes,plain_edges,hub_nodes,hub_edges", [
    (2, 288, 1152, 356, 1912), (3, 736, 3200, 916, 5368), (4, 1760, 7808, 2196, 13144),
])
def test_exchange_published_counts(depth, plain_nodes, plain_edges, hub_nodes, hub_edges):
    for hubs, nodes, edges in [(0, plain_nodes, plain_edges), (4, hub_nodes, hub_edges)]:
        graph = build_topology(TopologyConfig(kind="exchange", depth=depth, n=16, hubs=hubs))
        assert (len(graph.nodes), len(graph.edges)) == (nodes, edges)
        assert len(graph.widths) == 3 * depth + 1
        assert graph.period == len(graph.widths) + 1
        assert_records(graph)


@pytest.mark.parametrize("depth,rounds,n", [(1, 5, 8), (2, 5, 8), (3, 2, 16)])
def test_exchange_each_plateau_target_exact_bit_cube(depth, rounds, n):
    graph = build_topology(TopologyConfig(kind="exchange", depth=depth, rounds=rounds, n=n))
    assert sum(graph.widths) == (rounds + 3) * 2 ** depth - 2
    origins = [set() for _ in graph.nodes]
    for node in graph.nodes:
        if node.layer == depth:
            origins[node.id].add((node.position, node.local))
    region_masks, local_masks = {0}, {0}
    for layer in range(depth, depth + rounds):
        region_bit, local_bit = 1 << (layer % depth), 1 << (layer % (n.bit_length() - 1))
        region_masks |= {mask ^ region_bit for mask in region_masks}
        local_masks |= {mask ^ local_bit for mask in local_masks}
        for edge in graph.edges:
            if graph.nodes[edge.src].layer == layer:
                origins[edge.dst].update(origins[edge.src])
        for node in graph.nodes:
            if node.layer == layer + 1:
                assert origins[node.id] == {(node.position ^ a, node.local ^ b)
                                            for a in region_masks for b in local_masks}


def test_multiscale_published_exact_endpoints():
    graph = build_topology(TopologyConfig(kind="multiscale", profile="9", n=16, hubs=4))
    def targets(source, family):
        return {dst for src, dst, kind in endpoints(graph, family) if src == source}
    assert targets((2, 1, "D", 7), "DD1") == {(3, q, "D", i) for q in (2, 3, 4, 5) for i in (7, 8)}
    assert targets((2, 1, "D", 7), "DD2") == {(4, 6, "D", 2), (4, 7, "D", 5)}
    assert targets((2, 1, "H", 1), "HD2") == {(4, 6, "D", 4), (4, 7, "D", 12)}


@pytest.mark.parametrize("profile,positions,with_hubs", [
    ("compact-5-dd4", 6, 212), ("compact-7", 10, 356), ("compact-9", 16, 572),
    ("compact-9-two-inputs", 20, 712), ("compact-15", 22, 788),
])
def test_compact_counts(profile, positions, with_hubs):
    for hubs in (0, 4):
        graph = build_topology(TopologyConfig(kind="multiscale", profile=profile, n=32, hubs=hubs))
        assert sum(graph.widths) == positions
        assert len(graph.nodes) == (with_hubs if hubs else 32 * positions)
        assert_records(graph)


@pytest.mark.parametrize("profile,positions", [
    ("5", 24), ("5-dd4", 24), ("7", 40), ("9", 48), ("13", 104),
    ("15", 120), ("17", 152), ("25", 280), ("33", 408),
    ("compact-5-dd4", 6), ("compact-7", 10), ("compact-9", 16),
    ("compact-9-two-inputs", 20), ("compact-15", 22),
])
def test_multiscale_detail_path_delay_formula(profile, positions):
    graph = build_topology(TopologyConfig(kind="multiscale", profile=profile, n=8, timing="timed"))
    assert sum(graph.widths) == positions
    delays = [set() for _ in graph.nodes]
    for node in graph.nodes:
        if node.layer == 0:
            delays[node.id].add(0)
        for eid in graph.outgoing[node.id]:
            edge = graph.edges[eid]
            delays[edge.dst].update(value + edge.delay for value in delays[node.id])
    expected = set(range(1, 5)) if profile.endswith("dd4") else set(range(len(graph.widths) // 2, len(graph.widths)))
    assert all(delays[i] == expected for i in graph.output_nodes)


@pytest.mark.parametrize("profile,n,hubs", [
    ("compact-9", 8, 0), ("compact-9", 32, 4),
    ("compact-9-two-inputs", 8, 0), ("compact-9-two-inputs", 32, 4),
    ("compact-15", 8, 0), ("compact-15", 16, 0), ("compact-15", 32, 0),
])
def test_compact_each_output_phase_covers_every_first_layer_detail(profile, n, hubs):
    graph = build_topology(TopologyConfig(kind="multiscale", profile=profile, n=n,
                                        hubs=hubs, timing="timed"))
    roots = [i for i in graph.input_nodes if graph.nodes[i].role == "D"]
    origins = [[0] * 4 for _ in graph.nodes]
    for bit, node in enumerate(roots):
        origins[node][0] = 1 << bit
    for node in graph.nodes:
        for eid in graph.outgoing[node.id]:
            edge = graph.edges[eid]
            for phase, sources in enumerate(origins[node.id]):
                origins[edge.dst][(phase + 1) % 4] |= sources
    expected = (1 << len(roots)) - 1
    assert all(sources == expected for i in graph.output_nodes for sources in origins[i])


@pytest.mark.parametrize("n,probes,node_count,edge_count,hub_degrees", [
    (16, (1, 1, 1), 3024, 23840, (32, 24)),
    (32, (1, 1, 1), 5456, 41760, (48, 24)),
    (64, (2, 2, 1), 10320, 78912, (80, 28)),
    (128, (4, 4, 2), 20048, 153920, (144, 40)),
])
def test_multiscale_published_probe_scaling(n, probes, node_count, edge_count, hub_degrees):
    graph = build_topology(TopologyConfig(kind="multiscale", profile="17", n=n, hubs=4))
    assert graph.config.probes == probes
    assert (len(graph.nodes), len(graph.edges)) == (node_count, edge_count)
    for role, expected in (("D", (12, 14)), ("H", hub_degrees)):
        assert (max(len(graph.incoming[node.id]) for node in graph.nodes if node.role == role),
                max(len(graph.outgoing[node.id]) for node in graph.nodes if node.role == role)) == expected


def test_timing_changes_only_delays_and_period():
    config = TopologyConfig(kind="multiscale", profile="compact-9", n=8, hubs=4, timing="timed")
    timed, settle = build_topology(config), build_topology(replace(config, timing="settle"))
    common = build_topology(replace(config, period=settle.period))
    assert endpoints(timed) == endpoints(settle) == endpoints(common)
    assert timed.nodes == settle.nodes
    assert timed.period == 4 and settle.period == 10 and common.period == 10
    assert all(e.delay == 1 for e in timed.edges)
    assert any(e.delay > 1 for e in settle.edges)
    for edge in settle.edges:
        assert edge.delay == settle.ranks[edge.dst] - settle.ranks[edge.src]
    assert_records(settle)


@pytest.mark.parametrize("partition", ["group", "singleton", "mixed"])
def test_partitions_and_hub_induced_deletion(partition):
    config = TopologyConfig(kind="multiscale", profile="compact-7", n=8, hubs=4, partition=partition)
    full = build_topology(config)
    custom = tuple(tuple(l == 1 and p == 0 for p in range(w)) for l, w in enumerate(full.widths))
    for mask in ("all", "none", "checkerboard", custom):
        graph = build_topology(replace(config, hub_mask=mask))
        kept = {address(n) for n in graph.nodes}
        assert endpoints(graph) == {e for e in endpoints(full) if e[0] in kept and e[1] in kept}
        assert endpoints(graph, "DD1") == endpoints(full, "DD1")
        assert endpoints(graph, "DD2") == endpoints(full, "DD2")
        for layer, width in enumerate(graph.widths):
            for position in range(width):
                ds = [n for n in graph.nodes if (n.layer, n.position, n.role) == (layer, position, "D")]
                hs = [n for n in graph.nodes if (n.layer, n.position, n.role) == (layer, position, "H")]
                assert len({n.region for n in ds}) == 1
                if hs:
                    assert len({n.region for n in hs}) == (4 if partition == "singleton" else 1)
                    assert (hs[0].region == ds[0].region) == (partition == "mixed")
                if mask == "checkerboard":
                    assert bool(hs) == (layer < len(graph.widths) - 1 and (layer + position) % 2 == 0)
        assert_records(graph)


@pytest.mark.parametrize("config", [
    TopologyConfig(n=0), TopologyConfig(n=True), TopologyConfig(layers=0),
    TopologyConfig(kind="pdg"), TopologyConfig(timing="timed"), TopologyConfig(hubs=4),
    TopologyConfig(kind="lattice", layers=4), TopologyConfig(kind="lattice", depth=2, layers=7),
    TopologyConfig(kind="exchange", n=3), TopologyConfig(kind="exchange", rounds=0),
    TopologyConfig(kind="exchange", depth=2, rounds=3, layers=9),
    TopologyConfig(kind="exchange", hubs=3), TopologyConfig(kind="exchange", partition="mixed"),
    TopologyConfig(kind="multiscale", profile="unknown"),
    TopologyConfig(kind="multiscale", profile="compact-7", layers=9),
    TopologyConfig(kind="multiscale", period=9), TopologyConfig(period=7),
    TopologyConfig(kind="multiscale", hub_mask="unknown"),
    TopologyConfig(kind="multiscale", hubs=4, probes=(3, 1, 1)),
    TopologyConfig(kind="multiscale", probes=(1, 1, 1)),
    TopologyConfig(kind="multiscale", hubs=4, hub_mask=((True,),)),
])
def test_invalid_bindings_fail(config):
    with pytest.raises(ValueError):
        build_topology(config)


def test_explicit_mask_rejects_terminal_hubs():
    graph = build_topology(TopologyConfig(kind="multiscale", profile="compact-7", hubs=4))
    with pytest.raises(ValueError, match="terminal hubs"):
        build_topology(replace(graph.config, hub_mask=tuple((True,) * w for w in graph.widths)))


def test_resolved_configuration_and_deterministic_records():
    for kind, count in [("fc", 4), ("lattice", 7), ("exchange", 10), ("multiscale", 9)]:
        graph = build_topology(TopologyConfig(kind=kind))
        assert graph.config.layers == count and graph.config.period == graph.period
        assert build_topology(graph.config) == graph
