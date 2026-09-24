"""Independent fixed DAG dependency layers with local block kernels.

No generic executor or frontier planner is used. specialized.py retains the
independent scalar schedules; only formulas and topology validation are shared.
"""
from collections import defaultdict
import math
from .block_policy import validate
from .blocks import canonicalize, deliver, evaluate_block
from .records import Result
from .specialized import validate_topology
from .validation import validate_window


def _run(graph, model, initial, external, stop, seal, layers, mode, zeta, prefill, policy):
    validate(model, **policy)
    if mode not in {"hard", "hst", "softp"} or not math.isfinite(zeta):
        raise ValueError("invalid specialization options")
    atoms, ledger = validate_window(graph, model, initial, external, stop, seal)
    q = initial.fork(); q.ledger = ledger
    fibers = defaultdict(list)
    for a in list(q.pending)+atoms:
        fibers[a.batch, a.node, a.time].append(a)
    events, outputs, messages, stats = [], [], [], {}
    for layer in layers:
        members = set(layer)
        coordinates = defaultdict(set)
        for (b, v, t), xs in fibers.items():
            if v in members and xs and q.cut <= t < stop:
                coordinates[t, b].add(v)
        region = graph.nodes[layer[0]].region
        frames = [(b, region, t, nodes) for (t, b), nodes in sorted(coordinates.items())]
        block, counters = evaluate_block(graph, model, q, frames, fibers, mode=mode, zeta=zeta,
                                          prefill=prefill, **policy)
        deliver(graph, model, block, fibers, messages, outputs)
        events.extend(block)
        for name, value in counters.items():
            stats[name] = max(stats.get(name, 0), value) if name.startswith("max_") else stats.get(name, 0)+value
    q.pending = [a for xs in fibers.values() for a in xs if a.kind == 1 and a.time >= stop]
    q.cut = stop
    return canonicalize(graph, Result(q, events, outputs, messages, stats))


def run(graph, model, initial, external, stop, *, sealed_until, topology, mode="hard", zeta=1.0,
        prefill=True, packed=True, full_autograd="replay", aggregate_autograd="replay"):
    if topology not in {"chain", "diamond"}:
        raise ValueError("block specialization requires chain or diamond")
    layers = validate_topology(graph, topology)
    return _run(graph, model, initial, external, stop, sealed_until, layers, mode, zeta, prefill,
                dict(packed=packed, full_autograd=full_autograd, aggregate_autograd=aggregate_autograd))


def _settle(spec, model, initial, values, layers, mode, zeta, prefill, policy):
    if initial.cut % spec.stride or initial.pending or values.shape[0] != initial.batch_size:
        raise ValueError("SettleGraph needs a complete position cut and matching batch")
    stop = initial.cut + values.shape[1]*spec.stride
    result = _run(spec.graph, model, initial, spec.external(values, initial.cut//spec.stride),
                  stop, stop, layers, mode, zeta, prefill, policy)
    groups = defaultdict(list)
    for b, t, p, value in result.outputs:
        groups[b, t-spec.rank(spec.graph.outputs[p])+spec.output_rank].append((p, value))
    result.outputs = [(b, t, 0, sum(v for _, v in sorted(pairs)))
                      for (b, t), pairs in sorted(groups.items(), key=lambda item: item[0][::-1])]
    return result


def settle_chain(spec, model, initial, values, *, mode="hard", zeta=1.0, prefill=True,
                 packed=True, full_autograd="replay", aggregate_autograd="replay"):
    layers = validate_topology(spec.graph, "chain")
    return _settle(spec, model, initial, values, layers, mode, zeta, prefill,
                   dict(packed=packed, full_autograd=full_autograd, aggregate_autograd=aggregate_autograd))


def settle_layered(spec, model, initial, values, *, mode="hard", zeta=1.0, prefill=True,
                   packed=True, full_autograd="replay", aggregate_autograd="replay"):
    graph = spec.graph
    layers = [tuple(v for v, n in enumerate(graph.nodes) if n.region == r)
              for r in sorted(range(len(graph.regions)), key=lambda r: spec.ranks[r])]
    expected = sorted((a, b) for left, right in zip(layers, layers[1:]) for a in left for b in right)
    if (sorted((e.source, e.target) for e in graph.edges) != expected or
            sorted(graph.inputs) != list(layers[0]) or sorted(graph.outputs) != list(layers[-1])):
        raise ValueError("layered specialization requires complete adjacent layers and boundaries")
    return _settle(spec, model, initial, values, layers, mode, zeta, prefill,
                   dict(packed=packed, full_autograd=full_autograd, aggregate_autograd=aggregate_autograd))
