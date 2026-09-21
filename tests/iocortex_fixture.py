"""Independent Python scheduling against values exported by actual original LH C++."""
from collections import Counter
from dataclasses import replace
import torch
from tidegraph import Atom, Continuation, Edge, External, Graph, Node, Region
from tidegraph.compare import equivalent
from tidegraph.ops import Model
from tidegraph.ports import PortLayout
from tidegraph import reference, frontier
from tidegraph.lazy_add import decode as decode_add
from tidegraph.fiber_attention import decode_bias
from tidegraph.token_window import token_inputs


def tensor(record, dtype):
    return torch.tensor(record["values"], dtype=dtype).reshape(record["shape"])


def graph(record):
    return Graph(tuple(Node(**n) for n in record["nodes"]), tuple(Edge(**e) for e in record["edges"]),
                 tuple(Region(**r) for r in record["regions"]), tuple(record["inputs"]),
                 tuple(record["outputs"]), PortLayout(**record["layout"]))


def model(g, record, width, dtype):
    m = Model(g, width=width, dtype=dtype)
    assert len(m.nodes) == len(record["nodes"])
    for node, raw in zip(m.nodes, record["nodes"]):
        for name in ("decay", "weight", "bias", "read"):
            value = tensor(raw[name], dtype)
            assert getattr(node, name).shape == value.shape
            getattr(node, name).copy_(value)
        assert node.extra.keys() <= raw["extra"].keys()
        node.extra = torch.nn.ParameterDict({k: torch.nn.Parameter(tensor(v, dtype)) for k, v in raw["extra"].items()})
    for name in ("input_scale", "agg_scale", "edge_scale", "output_scale"):
        assert len(getattr(m, name)) == len(record[name])
        for p, raw in zip(getattr(m, name), record[name]): p.copy_(tensor(raw, dtype))
    return m


def hidden(m, q, expected, dtype):
    for b, v, raw in expected:
        w = m.nodes[v]; state = q.states.get((b, v), w.initial())
        actual = {"hidden": decode_add(w, state, q.cut)} if "hidden" in raw else {
            "key": state.slots["key"], "value": state.slots["value"], "log_bias": decode_bias(w, state, q.cut)}
        equivalent(actual, {k: tensor(value, dtype) for k, value in raw.items()}, f"physical[{b},{v}]")


def check_body(record, m, result, dtype):
    events = {(e["time"], e["batch"], e["node"]): e for e in result.trace}
    assert len(events) == len(record["events"]) == len(result.trace)
    seen = Counter(); last = {}
    for t, b, v, prop, active, full in record["events"]:
        e = events.pop((t, b, v)); proposal = tensor(prop, dtype)
        equivalent(e["proposal"], proposal); assert e["active"] == active
        if active: equivalent(e["full"], tensor(full, dtype))
        own_norm = e["proposal"].double().norm(); expected_norm = proposal.double().norm()
        equivalent(e["descriptor"], own_norm)
        perturbation = (e["proposal"].double()-proposal.double()).norm()
        assert abs(e["descriptor"]-expected_norm) <= perturbation+1e-10+1e-8*expected_norm
        seen[b, v] += 1; last[b, v] = t
    q = result.continuation
    assert not events and q.states.keys() == seen.keys()
    for owner, state in q.states.items():
        assert state.observations == seen[owner] and state.last_time == last[owner]
    expected = {(b, t, p): tensor(x, dtype) for b, t, p, x in record["outputs"]}
    actual = {(b, t, p): x for b, t, p, x in result.outputs}
    assert len(actual) == len(result.outputs); equivalent(actual, expected)
    hidden(m, q, record["hidden"], dtype)
    for b, region, v, selected, affected in record["counters"]:
        history = q.history.get((b, region))
        for name, expected in (("selected", selected), ("affected", affected)):
            assert (history.node_maps[name].get(v, 0) if history else 0) == expected
    expected = [Atom(*a[:-1], tensor(a[-1], dtype)) for a in record["pending"]]
    equivalent(sorted(q.pending, key=Atom.key), sorted(expected, key=Atom.key))


@torch.no_grad()
def check(record):
    assert record["schema"] == "lh-iocortex-fixture-v1" and record["dtype"] in {"float32", "float64"}
    dtype = getattr(torch, record["dtype"]); g = graph(record["body"])
    m = model(g, record["model"], record["width"], dtype)
    initial = Continuation(g.identity, record["batch_size"])
    inputs = [External(*row[:-1], tensor(row[-1], dtype)) for row in record["inputs"]]
    cut = record["cut"]; whole = reference.run(g, m, initial, inputs, cut, sealed_until=cut)
    check_body(record, m, whole, dtype)
    q = initial; outputs = []; trace = []; messages = []
    step = record["layers"] if record["scenario"] == "tokens" else 1
    for stop in range(step, cut+1, step):
        part = [x for x in inputs if q.cut <= x.time < stop]
        result = reference.run(g, m, q, part, stop, sealed_until=stop); q = result.continuation
        outputs.extend(result.outputs); trace.extend(result.trace); messages.extend(result.messages)
    equivalent(whole, replace(result, outputs=outputs, trace=trace, messages=messages))
    from single_graph_checks import check_original
    single_cuts = check_original(record, g, m, inputs, dtype)
    if record["scenario"] == "ragged": return len(trace), single_cuts
    assert record["scenario"] == "tokens"
    rg = graph(record["readout"]); rm = model(rg, record["read_model"], record["width"], dtype)
    rq = Continuation(rg.identity, record["batch_size"]); tokens = cut//step
    external = token_inputs(outputs, rq, step, tokens, body_cut=cut)
    expected = {(t, b): tensor(value, dtype)[b] for t, value in enumerate(record["logits"])
                for b in range(record["batch_size"])}
    final = None
    for run in (reference.run, frontier.run):
        result = run(rg, rm, rq, external, tokens, sealed_until=tokens)
        w = rm.nodes[0]
        def logits(outputs):
            return {(t, b): torch.nn.functional.linear(x, w.extra["token_head"], w.extra["token_head_bias"])
                    for b, t, _, x in outputs}
        equivalent(logits(result.outputs), expected); hidden(rm, result.continuation, record["read_hidden"], dtype)
        if final is not None: equivalent(final, result.continuation)
        final = result.continuation
        chunk = rq; chunk_outputs = []
        for token in range(tokens):
            rows = [o for o in outputs if token*step <= o[1] < (token+1)*step]
            xs = token_inputs(rows, chunk, step, token+1, body_cut=(token+1)*step)
            tail = run(rg, rm, chunk, xs, token+1, sealed_until=token+1)
            chunk = tail.continuation; chunk_outputs.extend(tail.outputs)
        equivalent(chunk, final); equivalent(logits(chunk_outputs), expected)
    return len(trace), single_cuts
