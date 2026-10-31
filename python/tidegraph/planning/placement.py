"""Allocation-free payload placement, matching ownership.place_payloads.

This plans one mathematical model, not replicas. Shared node/region parameters
are indivisible; port uses may copy their single canonical leaf. No Torch import
or numeric trajectory is needed, and a plan is never device qualification.
"""
from collections.abc import Mapping
import re
from .model import fingerprint
from .workload import validate_plan


def devices_list(devices):
    if isinstance(devices, str):
        raise ValueError("devices must be a nonempty sequence of explicit device names")
    result = tuple(devices)
    if not result or any(not isinstance(d, str) or not re.fullmatch(r"cpu|(?:cuda|npu):[0-9]+", d)
                         for d in result):
        raise ValueError("devices require cpu or explicit cuda:N/npu:N indices")
    if len(set(result)) != len(result) or len({d.split(":")[0] for d in result}) != 1:
        raise ValueError("devices must be distinct and use one backend")
    return result


def _unit(role, nodes):
    parts = role.split(".")
    if parts[0] == "nodes":
        return int(parts[1])
    if parts[0] == "regions":
        return nodes+int(parts[1])
    return None


def _port_node(role, graph):
    field, index = role.split(".")
    index = int(index)
    if field in {"input_scale", "output_scale"}:
        return graph["inputs" if field == "input_scale" else "outputs"][index]
    return graph["edges"][index][1 if field == "agg_scale" else 0]


def _components(plan, inv, scalar):
    graph = plan["graph"]
    n, nr = graph["nodes"], graph["regions"]
    parent = list(range(n+nr))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)

    owner_units = {}
    witnesses = []
    for name, row in inv["owners"].items():
        units = sorted({u for alias in row["aliases"] if (u := _unit(alias, n)) is not None})
        owner_units[name] = units
        for u in units[1:]:
            union(units[0], u)
        if len(units) > 1:
            witnesses.append(dict(parameter=name, aliases=row["aliases"], units=units))
    members = [[] for _ in range(nr)]
    for v, r in enumerate(graph["node_regions"]):
        members[r].append(v)
    # A region with no already-owned parameter follows its first member. A
    # shared region component instead follows the earliest region/node owner.
    fallback = []
    for r in range(nr):
        owner = root(n+r)
        if owner >= n:
            v = members[r][0]
            union(n+r, v)
            fallback.append(dict(region=r, node=v, reason="first member owns unbound region parameters"))
    groups = {}
    for v in range(n):
        groups.setdefault(root(v), dict(nodes=[], regions=[], parameters=[], parameter_bytes=0,
                                       inert_buffer_bytes=0, shared_parameters=[]))["nodes"].append(v)
    for r in range(nr):
        groups[root(n+r)]["regions"].append(r)
    owner_groups = {}
    for name, row in inv["owners"].items():
        units = owner_units[name]
        # Inventory order is node, region, input/output/aggregate/edge scales,
        # the same first-use precedence as the runtime payload owner.
        anchor = units[0] if units else _port_node(row["aliases"][0], graph)
        owner_groups[name] = key = root(anchor)
        groups[key]["parameters"].append(name)
        groups[key]["parameter_bytes"] += row["elements"]*scalar
    d = plan["width"]+1
    for key, group in groups.items():
        group["id"] = min(group["nodes"])
        group["inert_buffer_bytes"] = len(group["nodes"])*(d*d+3*d)*scalar
        group["static_bank_bytes"] = group["parameter_bytes"]+group["inert_buffer_bytes"]
    for witness in witnesses:
        groups[root(witness.pop("units")[0])]["shared_parameters"].append(witness)
    return groups, owner_groups, fallback


def _explicit_map(node_devices, nodes, devices):
    if isinstance(node_devices, Mapping):
        normalized = {}
        for key, value in node_devices.items():
            index = int(key) if isinstance(key, str) and key.isascii() and key.isdecimal() else key
            if type(index) is not int or str(index) != str(key) or index in normalized:
                raise ValueError("node_devices keys must be canonical nonnegative integer node IDs")
            normalized[index] = value
        if set(normalized) != set(range(nodes)):
            raise ValueError("node_devices must cover every node exactly once")
        result = tuple(normalized[v] for v in range(nodes))
    elif isinstance(node_devices, (tuple, list)):
        result = tuple(node_devices)
    else:
        raise ValueError("explicit strategy requires node_devices sequence or node-ID mapping")
    if len(result) != nodes or any(d not in devices for d in result):
        raise ValueError("node_devices must cover every node using declared devices")
    if result[0] != devices[0]:
        raise ValueError("node_devices[0] must equal the first device (eager primary owner)")
    return result


def plan_payload_placement(plan, devices, *, strategy="balanced-shared-groups",
                           node_devices=None, dtype="float32", device_limits=None):
    """Place canonical parameter components without constructing the model.

    ``device_limits`` maps devices to available bytes *after* caller reserves.
    Greedy weights are exact canonical parameters plus legacy inert buffers,
    not total runtime peaks. Node zero stays on the first/primary device. Other
    groups are ordered by descending bytes, then smallest node ID, and choose
    the least-loaded fitting device, breaking ties in requested device order.
    """
    inv = validate_plan(plan)
    devices = devices_list(devices)
    if dtype not in {"float16", "float32", "float64"}:
        raise ValueError("dtype must be float16, float32 or float64")
    if strategy not in {"explicit", "region-round-robin", "balanced-shared-groups"}:
        raise ValueError("unknown payload placement strategy")
    if strategy != "explicit" and node_devices is not None:
        raise ValueError("node_devices applies to the explicit placement strategy")
    limits = dict(device_limits or {})
    if set(limits)-set(devices) or any(type(x) is not int or x < 0 for x in limits.values()):
        raise ValueError("device_limits requires declared device keys and nonnegative integer bytes")
    scalar = {"float16": 2, "float32": 4, "float64": 8}[dtype]
    groups, owner_groups, fallback = _components(plan, inv, scalar)
    n = plan["graph"]["nodes"]
    node_map = [None]*n
    if strategy == "explicit":
        node_map = list(_explicit_map(node_devices, n, devices))
    elif strategy == "region-round-robin":
        node_map = [devices[r % len(devices)] for r in plan["graph"]["node_regions"]]
    loads = dict.fromkeys(devices, 0)
    ordered = sorted(groups.values(), key=lambda row: (-row["static_bank_bytes"], row["id"]))
    if strategy == "balanced-shared-groups":
        ordered.sort(key=lambda row: 0 if 0 in row["nodes"] else 1)
    for group in ordered:
        size = group["static_bank_bytes"]
        if strategy == "balanced-shared-groups":
            choices = (devices[0],) if 0 in group["nodes"] else devices
            fitting = [d for d in choices if d not in limits or loads[d]+size <= limits[d]]
            if not fitting:
                raise ValueError(f"placement group nodes={group['nodes']} needs {size} indivisible bytes; "
                                 "no declared device has room after existing groups and reserves")
            device = min(fitting, key=lambda d: (loads[d], devices.index(d)))
            for v in group["nodes"]:
                node_map[v] = device
        else:
            mapped = {node_map[v] for v in group["nodes"]}
            if len(mapped) != 1:
                name = group["shared_parameters"][0]["parameter"] if group["shared_parameters"] else "region owner"
                raise ValueError(f"node_devices splits shared parameter {name}; co-locate nodes {group['nodes']}")
            device = mapped.pop()
            if device in limits and loads[device]+size > limits[device]:
                raise ValueError(f"placement on {device} exceeds device_limits after indivisible group {group['nodes']}")
        loads[device] += size
        group["device"] = device
    parameter_devices = {name: groups[key]["device"] for name, key in owner_groups.items()}
    region_map = [None]*plan["graph"]["regions"]
    per_device = {d: dict(nodes=[], regions=[], parameter_elements=0, trainable_elements=0,
                         frozen_elements=0, parameter_bytes=0, inert_buffer_bytes=0, static_bank_bytes=0)
                  for d in devices}
    for group in groups.values():
        row = per_device[group["device"]]
        for key in ("nodes", "regions"):
            row[key].extend(group[key])
        for r in group["regions"]:
            region_map[r] = group["device"]
        for key in ("parameter_bytes", "inert_buffer_bytes", "static_bank_bytes"):
            row[key] += group[key]
    for name, row in inv["owners"].items():
        target = per_device[parameter_devices[name]]
        target["parameter_elements"] += row["elements"]
        target["trainable_elements" if row["trainable"] else "frozen_elements"] += row["elements"]
    crossing = [i for i, (a, b, _) in enumerate(plan["graph"]["edges"]) if node_map[a] != node_map[b]]
    body = dict(schema="tide-payload-placement-v1", workload_id=plan["sha256"],
        model_id=plan.get("model_id"), inventory_sha256=inv["sha256"], dtype=dtype,
        requested=dict(devices=list(devices), strategy=strategy,
                       node_devices=list(node_map) if strategy == "explicit" else None, device_limits=limits),
        devices=list(devices), node_devices=node_map, region_devices=region_map,
        parameter_devices=parameter_devices, per_device=per_device,
        co_location_groups=sorted(groups.values(), key=lambda row: row["id"]),
        largest_indivisible_group=max(groups.values(), key=lambda row: (row["static_bank_bytes"], -row["id"])),
        region_fallbacks=fallback, cross_device_edge_ids=crossing,
        weight_basis="canonical payload parameters plus node inert buffers; excludes dynamic runtime storage",
        automatic_admission=False, qualification="static request only; runtime ownership and device capability checks required")
    return dict(body, placement_id=fingerprint(body))
