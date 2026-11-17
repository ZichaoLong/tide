"""Actual parameter inventory versus the immutable upstream capacity formulas."""
from collections import defaultdict
import importlib
import importlib.util
from pathlib import Path
import sys

import pytest

from tide import ModelConfig, TideModel, TopologyConfig


def fixed_capacity():
    root = Path(__file__).resolve().parents[1] / "math/reference/backbones"
    name = "_tide_fixed_capacity_backbones"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / "__init__.py",
                                                     submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(name + ".capacity")


CAPACITY = fixed_capacity()


def reference_nodes(profile, n, hubs):
    spec = CAPACITY.PROFILES[profile]
    widths = spec["widths"]
    if spec["family"] == "fc":
        nodes = [(layer, 0, "d", i) for layer in range(len(widths)) for i in range(n)]
        sources = [{"ext" if layer == 0 else "DD"} for layer, _, _, _ in nodes]
        return nodes, sources
    if spec["family"] == "multiscale":
        graph = CAPACITY.ms.construct(widths, n=n, h=4, probes=CAPACITY.ms.probes_for(n),
                                      placement="all" if hubs else "none", add_dd4=spec["add_dd4"])
    else:
        graph = CAPACITY.st.construct(widths, n=n, h=hubs, exchange=spec["family"] == "exchange")
    nodes, _, _, incoming, _, _ = graph
    sources = [{kind for _, kind in parents} | ({"ext"} if node[0] == 0 else set())
               for node, parents in zip(nodes, incoming)]
    return nodes, sources


def independent_inventory(profile, config):
    """Count mathematical ownership, without production module/group helpers.

    The reference count() covers group partitions. Its per-matrix inventory is
    also used to apply the written mixed KV exception and singleton omission.
    """
    topology = config.topology
    nodes, sources = reference_nodes(profile, topology.n, topology.hubs)
    region_nodes = defaultdict(list)
    regions = []
    for i, (layer, position, role, local) in enumerate(nodes):
        region = ((layer, position) if topology.partition == "mixed" else
                  (layer, position, role, local) if topology.partition == "singleton" and role == "h" else
                  (layer, position, role))
        regions.append(region)
        region_nodes[region].append(i)
    region_types = {region: tuple(sorted({nodes[i][2] for i in members}))
                    for region, members in region_nodes.items()}
    bases, private, embedding_slots = {}, 0, set()
    agg = CAPACITY.aggregation_config(config.d, config.aggregation)
    source_width = agg["agg_hidden_width"] + (config.d if agg["agg_attention_width"] else 0)
    for i, (layer, position, role, local) in enumerate(nodes):
        kind = "S" if config.memory == "S" or config.memory == "X" and role == "h" else "K"
        scored = not (topology.partition == "singleton" and role == "h")
        group = (i,) if config.sharing == "independent" else (layer, role, kind)
        if config.sharing == "region_shared":
            group = (regions[i], *group)
        matrices, vectors = CAPACITY.inventory(config.d, kind, config.aggregation)
        for name, rows, columns in matrices:
            if not scored and name == "encode":
                continue
            owner = group
            if name == "KV" and config.sharing != "independent":
                owner = (regions[i],) if config.sharing == "region_shared" else (layer, region_types[regions[i]])
            bases["matrix", owner, name] = rows * columns
            if config.sharing != "independent":
                private += min(config.adapter_rank, rows, columns) * (rows + columns)
        for name, width in vectors:
            if scored or not name.startswith("encode."):
                bases["vector", group, name] = width
        private += int(scored)  # Final local score bias remains node-private.
        embedding_slots.update((group, source) for source in sources[i])
    region_groups = set()
    for region, members in region_nodes.items():
        if topology.partition == "singleton" and region_types[region] == ("h",):
            continue
        owner = (nodes[members[0]][0], region_types[region]) if config.sharing == "layer_shared" else region
        region_groups.add(owner)
    boundary_roles = {role for layer, _, role, _ in nodes if layer == 0}
    boundary = len(boundary_roles) * (config.input_dim or config.d) * config.d
    return (sum(bases.values()) + private + len(embedding_slots)*source_width
            + len(region_groups)*CAPACITY.regional_parameter_size(config.d) + boundary)


# Each main memory, reception and sharing choice is represented; this is not
# the Cartesian product. Registered n=8 makes capacity.count directly usable.
CASES = [
    ("fc-4", TopologyConfig(n=8, layers=4), "K", "source_residual", "independent", 8, 64),
    ("fc-4", TopologyConfig(n=8, layers=4), "S", "incoming_light_64", "region_shared", 8, 64),
    ("fc-4", TopologyConfig(n=8, layers=4), "X", "incoming_light_128", "layer_shared", 4, 64),
    ("fc-4", TopologyConfig(n=8, layers=4), "K", "incoming_wide", "region_shared", 16, 128),
    ("lattice-3", TopologyConfig(kind="lattice", n=8, depth=1), "S", "incoming_wide", "independent", 8, 64),
    ("lattice-3", TopologyConfig(kind="lattice", n=8, depth=1), "X", "source_residual", "region_shared", 16, 128),
    ("lattice-3", TopologyConfig(kind="lattice", n=8, depth=1), "K", "incoming_light_64", "layer_shared", 8, 64),
    ("exchange-4", TopologyConfig(kind="exchange", n=8, depth=1, hubs=4), "K", "incoming_light_128", "independent", 8, 64),
    ("exchange-4", TopologyConfig(kind="exchange", n=8, depth=1, hubs=4), "S", "source_residual", "region_shared", 4, 64),
    ("exchange-4", TopologyConfig(kind="exchange", n=8, depth=1, hubs=4), "X", "incoming_wide", "layer_shared", 16, 64),
    ("multiscale-compact-5-dd4", TopologyConfig(kind="multiscale", n=8, profile="compact-5-dd4", hubs=4), "K", "incoming_light_64", "region_shared", 8, 64),
    ("multiscale-compact-5-dd4", TopologyConfig(kind="multiscale", n=8, profile="compact-5-dd4", hubs=4), "S", "incoming_light_128", "layer_shared", 4, 64),
    ("multiscale-compact-5-dd4", TopologyConfig(kind="multiscale", n=8, profile="compact-5-dd4", hubs=4), "X", "source_residual", "independent", 8, 64),
]


@pytest.mark.parametrize("profile,topology,memory,aggregation,sharing,rank,d", CASES)
def test_registered_parameter_count_matches_fixed_capacity(profile, topology, memory, aggregation, sharing, rank, d):
    config = ModelConfig(topology=topology, d=d, memory=memory, aggregation=aggregation,
                         sharing=sharing, adapter_rank=rank, input_dim=17)
    model = TideModel(config)
    expected = CAPACITY.count(profile, topology.n, topology.hubs, d, memory, sharing, rank, aggregation)["total"]
    expected += d * 17 * (2 if topology.hubs else 1)
    assert independent_inventory(profile, config) == expected
    assert sum(parameter.numel() for parameter in model.parameters()) == expected


@pytest.mark.parametrize("partition,memory,sharing", [
    ("mixed", "K", "independent"), ("mixed", "K", "region_shared"),
    ("mixed", "K", "layer_shared"), ("mixed", "S", "region_shared"),
    ("mixed", "X", "layer_shared"), ("singleton", "S", "independent"),
    ("singleton", "X", "region_shared"), ("singleton", "K", "layer_shared"),
])
def test_mixed_kv_base_exception_and_singleton_omission(partition, memory, sharing):
    topology = TopologyConfig(kind="multiscale", n=8, profile="compact-5-dd4", hubs=4, partition=partition)
    config = ModelConfig(topology=topology, d=64, memory=memory, sharing=sharing, aggregation="incoming_light_64")
    model = TideModel(config)
    expected = independent_inventory("multiscale-compact-5-dd4", config)
    assert sum(parameter.numel() for parameter in model.parameters()) == expected
    if partition == "mixed" and memory == "K" and sharing != "independent":
        kv = model.memories["K"].kv
        role_pairs = [(members[0], next(v for v in members if model.graph.nodes[v].role == "H"))
                      for members in model.graph.regions if any(model.graph.nodes[v].role == "H" for v in members)]
        assert all(kv.groups[a] == kv.groups[b] for a, b in role_pairs)


