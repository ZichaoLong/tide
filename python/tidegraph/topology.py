"""Small deterministic topology factories; explicit Graph records remain general."""
from dataclasses import asdict
from .graph import Edge, Graph, Node, Region


def topology(kind, *, size=4, layers=None, delay=1, budget=None,
             module=None, node_overrides=None, region_overrides=None, dormant_layers=0):
    """Return (Graph, region ranks) without allocating model tensors.

    Layered graphs connect every node to every node of the following active
    layer. Dormant layers are explicit and are never counted as active work.
    Parallel edges and other wiring can be supplied through GraphConfig.graph.
    """
    if type(delay) is not int or delay < 1:
        raise ValueError("delay must be a positive integer")
    if type(size) is not int or size < 1:
        raise ValueError("size must be a positive integer")
    if type(dormant_layers) is not int or dormant_layers < 0:
        raise ValueError("dormant_layers must be a nonnegative integer")
    if kind in {"chain", "ring", "self_loop"}:
        if layers is not None or dormant_layers:
            raise ValueError("layers/dormant_layers require a layered topology")
        if kind == "self_loop" and size != 1:
            raise ValueError("self_loop requires size=1")
        if kind == "ring" and size < 2:
            raise ValueError("ring requires at least two nodes")
        owners = list(range(size))
        edges = [Edge(v, v + 1, delay) for v in range(size - 1)]
        if kind in {"ring", "self_loop"}:
            edges.append(Edge(size - 1, 0, delay))
        inputs, outputs, widths = (0,), (size - 1,), [1] * size
    elif kind == "diamond":
        if size != 4 or layers is not None or dormant_layers:
            raise ValueError("diamond requires size=4 and no layers/dormant_layers")
        owners, widths = [0, 1, 1, 2], [1, 2, 1]
        edges = [Edge(a, b, delay) for a, b in ((0, 1), (0, 2), (1, 3), (2, 3))]
        inputs, outputs = (0,), (3,)
    elif kind == "layered":
        if not isinstance(layers, (list, tuple)) or not layers or any(type(x) is not int or x < 1 for x in layers):
            raise ValueError("layered topology requires positive layer sizes")
        active = len(layers)
        widths = list(layers) + [layers[-1]] * dormant_layers
        offsets = [0]
        for width in widths:
            offsets.append(offsets[-1] + width)
        owners = [r for r, width in enumerate(widths) for _ in range(width)]
        edges = [Edge(a, b, delay) for r in range(active - 1)
                 for a in range(offsets[r], offsets[r + 1])
                 for b in range(offsets[r + 1], offsets[r + 2])]
        inputs = tuple(range(offsets[1]))
        outputs = tuple(range(offsets[active - 1], offsets[active]))
    else:
        raise ValueError(f"unknown topology kind: {kind}")
    from .config import _record
    nodes = [_record(Node, dict(region=region, **(module or {}))) for region in owners]
    regions = [Region(width if budget is None else budget) for width in widths]
    for records, overrides in ((nodes, node_overrides), (regions, region_overrides)):
        for key, values in (overrides or {}).items():
            if isinstance(key, str) and key.isdigit():
                key = int(key)
            if type(key) is not int or not 0 <= key < len(records):
                raise ValueError("topology override owner is out of range")
            if not isinstance(values, dict):
                raise ValueError("topology override must be an object")
            records[key] = _record(type(records[key]), asdict(records[key]) | values)
    graph = Graph(tuple(nodes), tuple(edges), tuple(regions), inputs, outputs)
    return graph, tuple(1 + delay * r for r in range(len(regions)))
