"""Runtime coordinate fixtures shared by API and checkpoint rejection tests."""
from dataclasses import replace
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.ops import Model
from tidegraph.records import Result
from tidegraph.reference import run
from tidegraph.frontier import run as frontier
from tidegraph.specialized import run as specialized
from tidegraph.native import Native

IMPLEMENTATIONS = ("reference", "python-frontier", "python-chain", "native-serial",
                   "native-packed", "native-frontier", "native-chain", "cursor")


def fixture(dtype, *, continued=False):
    graph = Graph((Node(0), Node(1)), (Edge(0, 1, 2),), (Region(1), Region(1)), (0,), (1,))
    model = Model(graph, dtype=dtype)
    q = Continuation(graph.identity, 2)
    xs = [External(b, 0, 0, 0, torch.full((3,), .2+b, dtype=dtype)) for b in range(2)]
    if continued:
        q = run(graph, model, q, xs, 1, sealed_until=1).continuation.detach()
    return graph, model, q, xs


def runner(implementation, graph, model, q):
    cursor = None
    if implementation == "reference":
        apply = lambda xs, stop, seal: run(graph, model, q, xs, stop, sealed_until=seal)
    elif implementation == "python-frontier":
        apply = lambda xs, stop, seal: frontier(graph, model, q, xs, stop, sealed_until=seal)
    elif implementation == "python-chain":
        apply = lambda xs, stop, seal: specialized(graph, model, q, xs, stop, sealed_until=seal, topology="chain")
    else:
        algorithm = {"native-frontier": "frontier", "native-chain": "chain"}.get(implementation, "streaming")
        engine = Native(graph, model, algorithm=algorithm, packed=implementation != "native-serial",
                        workers=1 if implementation == "native-serial" else 3)
        if implementation == "cursor":
            cursor = engine.cursor(q)
            def apply(xs, stop, seal):
                r = cursor.advance(xs, stop, sealed_until=seal)
                return Result(cursor.snapshot(), r.trace, r.outputs, r.messages, r.stats)
        else:
            apply = lambda xs, stop, seal: engine.run(q, xs, stop, sealed_until=seal)
    return apply, cursor


MALFORMED = ("cut", "batch", "state_batch", "state_node", "state_time", "observations",
             "history_batch", "history_region", "history_time", "history_scalar", "history_node", "history_count",
             "ledger_batch", "ledger_port", "ledger_position", "ledger_time",
             "pending_batch", "pending_node", "pending_time", "pending_kind", "pending_source", "pending_position")


def corrupt(q, name, convert):
    if name in {"cut", "batch"}:
        field = "cut" if name == "cut" else "batch_size"
        setattr(q, field, convert(getattr(q, field)))
    elif name in {"state_batch", "state_node", "history_batch", "history_region", "ledger_batch", "ledger_port"}:
        group = getattr(q, {"state": "states", "history": "history", "ledger": "ledger"}[name.split("_")[0]])
        key = next(iter(group)); row = group.pop(key)
        index = 0 if name.endswith("batch") else 1
        changed = list(key); changed[index] = convert(changed[index]); changed = tuple(changed)
        # bool(2) equals integer 1: remove a colliding key so dict cannot keep
        # that original int key and accidentally erase the intended corruption.
        group.pop(changed, None); group[changed] = row
    elif name in {"state_time", "observations"}:
        state = next(iter(q.states.values()))
        field = "last_time" if name == "state_time" else "observations"
        setattr(state, field, convert(getattr(state, field)))
    elif name.startswith("history_"):
        h = next(iter(q.history.values()))
        if name == "history_time": h.last_time = convert(h.last_time)
        elif name == "history_scalar": h.scalars["coordinate_probe"] = convert(1)
        else:
            node, count = next(iter(h.node_maps["selected"].items()))
            h.node_maps["selected"] = {convert(node) if name == "history_node" else node:
                                       convert(count) if name == "history_count" else count}
    elif name.startswith("ledger_"):
        key = next(iter(q.ledger)); row = list(q.ledger[key])
        index = 0 if name == "ledger_position" else 1
        row[index] = convert(row[index]); q.ledger[key] = tuple(row)
    else:
        field = name.removeprefix("pending_")
        q.pending[0] = replace(q.pending[0], **{field: convert(getattr(q.pending[0], field))})
