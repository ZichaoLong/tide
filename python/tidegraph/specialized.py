"""Independent fixed ring/chain/diamond and layered schedules."""
from collections import defaultdict
import math
from .blocks import canonicalize
from .records import Atom, Result
from .validation import validate_window
from .specialized_step import frame


def validate_topology(graph, topology):
    n = len(graph.nodes)
    if graph.inputs != (0,) or graph.outputs != (n - 1,):
        raise ValueError("specialization requires one endpoint input/output")
    if topology not in {"self_loop", "ring", "chain", "diamond"}:
        raise ValueError("invalid specialization")
    if topology == "diamond":
        regions = tuple(node.region for node in graph.nodes)
        if n != 4 or regions not in {(0, 1, 2, 3), (0, 1, 1, 2)}:
            raise ValueError("diamond requires ordered singleton or paired middle regions")
        expected = [(0, 1), (0, 2), (1, 3), (2, 3)]
        layers = [(0,), (1, 2), (3,)] if regions[1] == regions[2] else [(v,) for v in range(n)]
    else:
        if len(graph.regions) != n or any(node.region != i for i, node in enumerate(graph.nodes)):
            raise ValueError("specialization requires singleton ordered regions")
        if (topology == "self_loop" and n != 1) or (topology == "ring" and n < 2):
            raise ValueError("invalid specialization node count")
        expected = ([(i, (i + 1) % n) for i in range(n)] if topology in {"self_loop", "ring"}
                    else [(i, i + 1) for i in range(n - 1)])
        layers = [(v,) for v in range(n)]
    if [(e.source, e.target) for e in graph.edges] != expected:
        raise ValueError("specialization topology mismatch")
    return layers


def _execute(graph, model, initial, external, stop, seal, layers, cyclic, mode, zeta):
    if mode not in {"hard", "hst", "softp"} or not math.isfinite(zeta):
        raise ValueError("invalid specialization options")
    atoms, ledger = validate_window(graph, model, initial, external, stop, seal)
    q = initial.fork(); q.ledger = ledger
    inbox = defaultdict(list)
    for a in list(q.pending) + atoms:
        inbox[a.node, a.batch, a.time].append(a)
    events, messages, outputs = [], [], []
    def apply(nodes, batch, time):
        block = frame(graph, model, q, nodes, batch, time, inbox, mode, zeta)
        for e in block:
            if not e["active"]:
                continue
            node = e["node"]
            ports = graph.port_indexes[1]
            for slot, payload in e["emitted"].items():
                kind, source = ports.bindings[ports.offsets[node] + slot]
                if kind == 0:
                    outputs.append((batch, time, source, payload * model.output_scale[source]))
                    continue
                edge = graph.edges[source]
                arrival = time + edge.delay
                if arrival >= 2**63:
                    raise ValueError("logical time overflow")
                a = Atom(batch, edge.target, arrival, 1, source, time, payload * model.edge_scale[source])
                inbox[edge.target, batch, arrival].append(a); messages.append(a)
        events.extend(block)
    if cyclic:
        # Strict positive delay makes nodes independent at a fixed logical tick.
        for time in range(q.cut, stop):
            for layer in layers:
                for batch in range(q.batch_size):
                    apply(layer, batch, time)
    else:
        # Fixed dependency layers, not the generic planner or streaming queue.
        for layer in layers:
            coordinates = sorted({(t, b) for (v, b, t), xs in inbox.items()
                                  if v in layer and xs and q.cut <= t < stop})
            for time, batch in coordinates:
                apply(layer, batch, time)
    q.pending = [a for bucket in inbox.values() for a in bucket if a.kind == 1 and a.time >= stop]
    q.cut = stop
    return canonicalize(graph, Result(q, events, outputs, messages, {"candidate_events": len(events)}))


def run(graph, model, initial, external, stop, *, sealed_until, topology, mode="hard", zeta=1.0):
    layers = validate_topology(graph, topology)
    return _execute(graph, model, initial, external, stop, sealed_until, layers,
                    topology in {"self_loop", "ring"}, mode, zeta)


def settle_chain(spec, model, initial, values, *, mode="hard", zeta=1.0):
    """Fixed-chain SettleGraph anchor; independent token propagation schedule."""
    validate_topology(spec.graph, "chain")
    if initial.cut % spec.stride or initial.pending:
        raise ValueError("SettleGraph needs a complete position cut")
    result = run(spec.graph, model, initial, spec.external(values, initial.cut // spec.stride),
                 initial.cut + values.shape[1] * spec.stride,
                 sealed_until=initial.cut + values.shape[1] * spec.stride,
                 topology="chain", mode=mode, zeta=zeta)
    result.outputs = [(b, t - spec.rank(spec.graph.outputs[0]) + spec.output_rank, p, v)
                      for b, t, p, v in result.outputs]
    return result


def settle_layered(spec, model, initial, values, *, mode="hard", zeta=1.0):
    """Fully connected adjacent region layers; independent scalar frame schedule."""
    graph = spec.graph
    layers = [tuple(v for v, node in enumerate(graph.nodes) if node.region == r)
              for r in sorted(range(len(graph.regions)), key=lambda r: spec.ranks[r])]
    expected = sorted((a, b) for left, right in zip(layers, layers[1:]) for a in left for b in right)
    if (sorted((e.source, e.target) for e in graph.edges) != expected or
            sorted(graph.inputs) != list(layers[0]) or sorted(graph.outputs) != list(layers[-1])):
        raise ValueError("layered specialization requires complete adjacent layers and boundaries")
    if initial.cut % spec.stride or initial.pending or values.shape[0] != initial.batch_size:
        raise ValueError("SettleGraph needs a complete position cut and matching batch")
    stop = initial.cut + values.shape[1] * spec.stride
    result = _execute(graph, model, initial, spec.external(values, initial.cut // spec.stride),
                      stop, stop, layers, False, mode, zeta)
    groups = defaultdict(list)
    for b, time, port, value in result.outputs:
        output_time = time - spec.rank(graph.outputs[port]) + spec.output_rank
        groups[b, output_time].append((port, value))
    result.outputs = [(b, t, 0, sum(value for _, value in sorted(pairs)))
                      for (b, t), pairs in sorted(groups.items(), key=lambda item: item[0][::-1])]
    return result
