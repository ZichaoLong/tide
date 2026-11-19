"""Construction-only parameter ownership; safe for meta-device capacity plans.

Each row of a packed Parameter belongs to one or more mathematical regions.
The shared input projection is the sole global owner. No device tensor is read
back to discover this metadata, and no floating payload is materialized.
"""
from dataclasses import dataclass

from .parameters import NodeLinear, NodeVector
from .reception import SourceEmbedding


@dataclass(frozen=True)
class ParameterRows:
    name: str
    parameter: object
    regions: tuple[tuple[int, ...], ...]
    category: str

    @property
    def row_elements(self):
        return self.parameter.numel() // len(self.regions)


def parameter_ownership(model):
    domains = {
        "receive": tuple(range(len(model.graph.nodes))),
        "describe": tuple(node for node, _ in sorted(model.description_ids.items(), key=lambda item: item[1])),
    }
    domains.update({"memories."+kind: tuple(node for node, _ in sorted(owners.items(), key=lambda item: item[1]))
                    for kind, owners in model.memory_ids.items()})
    source_classes = [set(model.graph.edges[edge].source_class for edge in model.graph.incoming[node.id]) |
                      ({"ext"} if node.layer == 0 else set()) for node in model.graph.nodes]
    records, handled = [], set()

    def add(name, parameter, regions, category):
        if id(parameter) in handled:
            raise ValueError("unexpected Parameter alias across ownership modules: "+name)
        handled.add(id(parameter))
        regions = tuple(tuple(sorted(set(row))) for row in regions)
        if category not in {"regional_network", "input_projection"} and parameter.shape[0] != len(regions):
            raise ValueError("static row count differs from Parameter shape: "+name)
        if not regions or parameter.numel() % len(regions):
            raise ValueError("invalid static Parameter row ownership: "+name)
        records.append(ParameterRows(name, parameter, regions, category))

    for path, module in model.named_modules():
        if isinstance(module, (NodeLinear, NodeVector)):
            prefix = next(name for name in domains if path.startswith(name+"."))
            nodes = domains[prefix]
            private = [(model.graph.nodes[node].region,) for node in nodes]
            grouped = {}
            for group, regions in zip(module._host_groups, private):
                grouped.setdefault(group, set()).update(regions)
            bases = [grouped[group] for group in range(len(grouped))]
            for leaf, parameter in module.named_parameters(recurse=False):
                adapter = leaf in {"left", "right"}
                add(path+"."+leaf, parameter, private if adapter else bases,
                    "private_adapter" if adapter else "owner_parameter")
        elif isinstance(module, SourceEmbedding):
            # Match SourceEmbedding's insertion order without reading lookup,
            # which is allowed to be a meta tensor in a planning construction.
            slots = {}
            for node, names in enumerate(source_classes):
                for name in sorted(names):
                    key = model.receive.a._host_groups[node], name
                    slots.setdefault(key, set()).add(model.graph.nodes[node].region)
            add(path+".weight", module.weight, slots.values(), "source_embedding")
    regional = {}
    for region, group in enumerate(model.region_group):
        if group >= 0:
            regional.setdefault(group, set()).add(region)
    for name, parameter in model.named_parameters():
        if id(parameter) in handled:
            continue
        if name.startswith("inputs."):
            add(name, parameter, [()], "input_projection")
        elif name.startswith("regional."):
            add(name, parameter, [regional[int(name.split(".")[1])]], "regional_network")
        else:
            raise ValueError("unclassified placed Parameter: "+name)
    if sum(row.parameter.numel() for row in records) != sum(p.numel() for p in model.parameters()):
        raise AssertionError("static ownership differs from unique Parameters")
    return tuple(records)


def row_partitions(record, region_partitions, input_partition=0):
    owners = []
    for regions in record.regions:
        partitions = {region_partitions[region] for region in regions} if regions else {input_partition}
        if len(partitions) != 1:
            raise ValueError(f"shared parameter {record.name} spans region partitions; regions={regions}")
        owners.append(next(iter(partitions)))
    return tuple(owners)


def region_parameter_units(model, records=None):
    """Exact unique row groups, with sharing explicitly represented once."""
    units = {}
    for record in parameter_ownership(model) if records is None else records:
        size = record.row_elements
        for regions in record.regions:
            unit = units.setdefault(regions, dict(regions=list(regions), unique_parameters=0,
                                                  parameter_bytes=0, parameter_components={}))
            unit["unique_parameters"] += size
            unit["parameter_bytes"] += size*record.parameter.element_size()
            components = unit["parameter_components"]
            components[record.name] = components.get(record.name, 0)+size
    return [units[key] for key in sorted(units)]


def balanced_region_partitions(model, count, *, input_partition=0, records=None):
    """Greedy exact-parameter balance over indivisible shared-region components.

    This is a deterministic storage plan, not a communication/performance model.
    It never duplicates a base or splits a mathematical region.
    """
    if type(count) is not int or count < 1 or not 0 <= input_partition < count:
        raise ValueError("positive partition count and valid input partition required")
    records = parameter_ownership(model) if records is None else records
    parent = list(range(len(model.graph.regions)))

    def find(region):
        while parent[region] != region:
            parent[region] = parent[parent[region]]
            region = parent[region]
        return region

    for record in records:
        for regions in record.regions:
            if regions:
                first = find(regions[0])
                for region in regions[1:]:
                    parent[find(region)] = first
    groups = {}
    for region in range(len(parent)):
        groups.setdefault(find(region), []).append(region)
    if count > len(groups):
        raise ValueError("more partitions than indivisible shared-region components")
    weights = dict.fromkeys(groups, 0)
    loads = [0]*count
    for record in records:
        for regions in record.regions:
            size = record.row_elements*record.parameter.element_size()
            if regions:
                weights[find(regions[0])] += size
            else:
                loads[input_partition] += size
    layout = [None]*len(parent)
    used = set()
    for index, root in enumerate(sorted(groups, key=lambda root: (-weights[root], min(groups[root])))):
        candidates = [part for part in range(count) if part not in used] if index < count else range(count)
        part = min(candidates, key=lambda part: (loads[part], part))
        used.add(part)
        loads[part] += weights[root]
        for region in groups[root]:
            layout[region] = part
    return tuple(layout)
