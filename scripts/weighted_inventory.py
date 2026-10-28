"""Allocation-free canonical parameter inventory for the weighted bind factory.

Fixed math: 0770593 architecture §2, KV §1–4, recurrent §1–3, allocation §3.
Shapes, source aliases and frozen owners are separate from physical residency.
No Torch import, model execution, numeric prepass or learned tensor allocation.
"""
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from weighted_shapes import dim, memory_shapes, full_shapes, describe_shapes, region_shapes


def mapping(value):
    return asdict(value) if is_dataclass(value) else dict(value)


def repeated(value, size):
    return [mapping(value) for _ in range(size)] if isinstance(value, dict) or is_dataclass(value) else [mapping(x) for x in value]


def class_widths(widths):
    """Decode canonical JSON object keys without truncating float/bool labels."""
    result = {}
    for key, value in widths.items():
        label = int(key) if type(key) is str and key.isascii() and key.isdecimal() else key
        if (type(label) is not int or not 0 <= label < 2**63
                or type(key) is str and str(label) != key or label in result):
            raise ValueError("source width labels must be exact nonnegative int64")
        result[label] = value
    return result


def inventory(graph, memories, regions, *, classes=None, source_binding=None,
              source_hidden_width=4, aliases=None, frozen=()):
    """Return exact logical shapes/aliases for bind, with optional explicit ties.

    graph: node_regions, edges=[source,target,delay], inputs, outputs. Arbitrary
    source labels are int64, not a uniqueness constraint on physical atoms.
    aliases maps full role names to role names; frozen lists full role names.
    Runtime validation still owns numeric configuration and graph legality.
    """
    nodes = graph["node_regions"]
    n = len(nodes)
    if not 1 <= n <= 200000 or any(type(r) is not int or not 0 <= r < n for r in nodes):
        raise ValueError("invalid node regions")
    nr = max(nodes)+1
    if set(nodes) != set(range(nr)):
        raise ValueError("regions must be nonempty and contiguous")
    memories, regions = repeated(memories, n), repeated(regions, nr)
    if len(memories) != n or len(regions) != nr:
        raise ValueError("node/region configuration count mismatch")
    widths = {dim(c, "width", 4) for c in memories}
    if len(widths) != 1:
        raise ValueError("weighted payload width must agree")
    d = widths.pop()
    sources = mapping(source_binding or {"mode": "region"})
    mode, rank = sources.get("mode", "region"), sources.get("rank")
    if mode not in {"region", "node", "source", "adapted"}:
        raise ValueError("invalid source binding")
    if mode == "adapted":
        dim({"rank": rank}, "rank")
    elif rank is not None:
        raise ValueError("source rank is only applicable to adapted")
    classes = mapping(classes) if classes is not None else dict(edges=[0]*len(graph["edges"]), inputs=[0]*len(graph["inputs"]))
    if len(classes["edges"]) != len(graph["edges"]) or len(classes["inputs"]) != len(graph["inputs"]):
        raise ValueError("source class domain mismatch")
    if any(type(x) is not int or not 0 <= x < 2**63 for x in (*classes["edges"], *classes["inputs"])):
        raise ValueError("source labels must be nonnegative int64")
    labels = [set() for _ in nodes]
    for edge, label in zip(graph["edges"], classes["edges"]):
        if len(edge) != 3 or any(type(x) is not int for x in edge) or not (0 <= edge[0] < n and 0 <= edge[1] < n and 0 < edge[2] < 2**63):
            raise ValueError("invalid physical edge")
        labels[edge[1]].add(label)
    for target, label in zip(graph["inputs"], classes["inputs"]):
        if type(target) is not int or not 0 <= target < n:
            raise ValueError("invalid input owner")
        labels[target].add(label)
    all_labels = set().union(*labels)
    if isinstance(source_hidden_width, dict):
        hidden = class_widths(source_hidden_width)
        if set(hidden) != all_labels:
            raise ValueError("source width class domain mismatch")
        for v in hidden.values():
            dim({"width": v}, "width")
    else:
        hidden = dict.fromkeys(all_labels, dim({"width": source_hidden_width}, "width"))
    roles, shared = {}, {}

    def add(name, shape, category, key=None, trainable=True):
        shape = tuple(shape)
        if name in roles or any(type(x) is not int or not 0 < x < 2**63 for x in shape):
            raise ValueError("duplicate/invalid parameter role")
        count = math.prod(shape)
        if count >= 2**63:
            raise ValueError("parameter element count exceeds int64")
        owner = shared.setdefault(key, name) if key is not None else name
        if owner in roles and roles[owner]["shape"] != list(shape):
            raise ValueError("shared parameter shapes disagree")
        roles[name] = dict(shape=list(shape), owner=owner, category=category, trainable=trainable, elements=count)

    for v, c in enumerate(memories):
        j, prefix = nodes[v], f"nodes.{v}."
        for name, shape in memory_shapes(c).items():
            add(prefix+"kernel."+name, shape, "memory", ("kv_base", j) if name == "base" else None)
        for category, sub, shapes in (("full", "full_program.content.", full_shapes(c)),
                                      ("describe", "read_program.", describe_shapes(c, regions[j]))):
            for name, shape in shapes.items():
                add(prefix+sub+name, shape, category)
        for label in sorted(labels[v]):
            h = hidden[label]
            key = v if mode == "node" else (j, label) if mode == "source" else j
            embed_key = (v if mode in {"node", "adapted"} else j, label)
            base = prefix+f"aggregate_program.transforms.{label}."
            add(base+"a", (h, d), "aggregate", ("source_a", mode, key))
            add(base+"b", (d, h), "aggregate", ("source_b", mode, key))
            add(base+"embedding", (h,), "aggregate", ("source_e", mode, embed_key))
            if mode == "adapted":
                for name, shape in zip(("a_left", "a_right", "b_left", "b_right"), ((h, rank), (rank, d), (d, rank), (rank, h))):
                    add(base+name, shape, "aggregate")
    for j, config in enumerate(regions):
        for name, shape in region_shapes(config).items():
            add(f"regions.{j}."+name, shape, "region")
    for category, size in (("input_scale", len(graph["inputs"])), ("output_scale", len(graph["outputs"])),
                           ("agg_scale", len(graph["edges"])), ("edge_scale", len(graph["edges"]))):
        for index in range(size):
            add(f"{category}.{index}", (), "transport", trainable=False)
    # Ties join whole existing alias classes; callers cannot split an existing
    # shared owner by providing a second name or freezing just one alias.
    parents = {row["owner"]: row["owner"] for row in roles.values()}

    def owner(key):
        while parents[key] != key:
            key = parents[key]
        return key

    for a, b in (aliases or {}).items():
        if a not in roles or b not in roles or roles[a]["shape"] != roles[b]["shape"] or roles[a]["trainable"] != roles[b]["trainable"]:
            raise ValueError("invalid explicit parameter alias")
        parents[owner(roles[a]["owner"])] = owner(roles[b]["owner"])
    if any(name not in roles for name in frozen):
        raise ValueError("unknown frozen parameter")
    frozen_owners = {owner(roles[name]["owner"]) for name in frozen}
    owners = {}
    for name, row in roles.items():
        key = owner(row["owner"])
        row["owner"] = key
        if key in frozen_owners:
            row["trainable"] = False
        if key not in owners:
            owners[key] = dict(shape=row["shape"], elements=row["elements"], trainable=row["trainable"], aliases=[], categories=[])
        owners[key]["aliases"].append(name)
        if row["category"] not in owners[key]["categories"]:
            owners[key]["categories"].append(row["category"])
    totals = {"trainable": 0, "frozen": 0, "role_elements": sum(x["elements"] for x in roles.values())}
    for row in owners.values():
        totals["trainable" if row["trainable"] else "frozen"] += row["elements"]
    totals["unique"] = totals["trainable"]+totals["frozen"]
    result = dict(schema="tide-weighted-inventory-v1", upstream="0770593838a37c790a568226930567ed60f99b83",
                  nodes=n, regions=nr, edges=len(graph["edges"]), roles=roles, owners=owners, totals=totals)
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result
