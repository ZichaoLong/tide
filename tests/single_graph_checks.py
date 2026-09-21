"""Complete body projection, readout state view and unfinished-window checks."""
from dataclasses import replace
from tidegraph import Continuation
from tidegraph import reference
from tidegraph.blocks import canonicalize
from tidegraph.compare import equivalent
from tidegraph.native import Native
from tidegraph.records import Result
from tidegraph.token_window import token_inputs
from single_graph_adapter import SingleGraph
from single_graph_compare import compare


def every_cut(body, model, readout, read_model, xs, layers, batch, stop, implementation="reference",
              require_read_windows=False):
    adapter = SingleGraph(body, model, readout, read_model, layers)
    graph, weights = adapter.graph, adapter.model
    initial = Continuation(graph.identity, batch); q = initial
    bq, rq = Continuation(body.identity, batch), Continuation(readout.identity, batch)
    engine = None if implementation == "reference" else Native(
        graph, weights, workers=1 if implementation == "native-serial" else 3,
        packed=implementation in {"native-packed", "cursor"})
    cursor = engine.cursor(q) if implementation == "cursor" else None

    def run(q, inputs, cut):
        return reference.run(graph, weights, q, inputs, cut, sealed_until=cut) if engine is None else (
            engine.run(q, inputs, cut, sealed_until=cut))

    global_inputs = adapter.inputs(xs)
    pending, traces, outputs, messages, body_outputs, read_outputs = [], [], [], [], [], []
    cut_records = []
    for cut in range(1, stop+1):
        body_cut = adapter.body_clock.cut(cut)
        br = reference.run(body, model, bq, [x for x in xs if bq.cut <= x.time < body_cut],
                           body_cut, sealed_until=body_cut)
        bq = br.continuation; pending.extend(br.outputs); body_outputs.extend(br.outputs)
        part = [x for x in global_inputs if q.cut <= x.time < cut]
        if cursor is not None:
            tail = cursor.advance(part, cut, sealed_until=cut)
            result = Result(cursor.snapshot(), tail.trace, tail.outputs, tail.messages, tail.stats)
        else:
            result = run(q, part, cut)
        q = result.continuation
        compare(body, adapter.body_view(result), canonicalize(body, br), f"cut[{cut}].body")
        read_cut = adapter.read_clock.cut(cut)
        rr = Result(rq, [], [], [], {})
        if rq.cut < read_cut:
            # Manual ragged body ticks have no original Pronounce call. An empty
            # window simply seals the graph; it is not an original-LH oracle case.
            assert pending or not require_read_windows
            external = token_inputs(pending, rq, layers, read_cut, body_cut=body_cut) if pending else []
            rr = reference.run(readout, read_model, rq, external, read_cut, sealed_until=read_cut)
            rq = rr.continuation; pending = []
        equivalent(adapter.read_view(q), replace(rq, ledger={}), f"cut[{cut}].read_state_history")
        equivalent(adapter.outputs(result), rr.outputs, f"cut[{cut}].read_outputs")
        expected_buffer = sorted(pending, key=lambda o: (o[1], o[0], o[2]))
        equivalent(adapter.buffer(q), expected_buffer, f"cut[{cut}].partial_window")
        cut_records.append((cut, adapter.body_clock.cut(cut), adapter.read_clock.cut(cut), len(pending)))
        traces.extend(result.trace); outputs.extend(result.outputs); messages.extend(result.messages)
        read_outputs.extend(rr.outputs)
    all_parts = canonicalize(graph, Result(q, traces, outputs, messages, {}))
    whole = run(initial, [x for x in global_inputs if x.time < stop], stop)
    compare(graph, all_parts, canonicalize(graph, whole), "single whole/cut")
    return adapter, whole, bq, rq, body_outputs, read_outputs, cut_records


def check_original(record, body, model, inputs, dtype):
    from iocortex_fixture import graph, model as load_model, check_body, hidden, tensor
    import torch
    rg = graph(record["readout"]); rm = load_model(rg, record["read_model"], record["width"], dtype)
    layers, ticks = record["layers"], record["cut"]
    stop = ticks//layers*(layers+1) if record["scenario"] == "tokens" else (
        (ticks-1)//layers*(layers+1)+(ticks-1)%layers+1)
    adapter, result, _, _, _, _, _ = every_cut(body, model, rg, rm, inputs, layers,
                                               record["batch_size"], stop,
                                               require_read_windows=record["scenario"] == "tokens")
    check_body(record, model, adapter.body_view(result), dtype)
    if record["scenario"] == "tokens":
        actual = {(t, b): torch.nn.functional.linear(x, rm.nodes[0].extra["token_head"],
                                                    rm.nodes[0].extra["token_head_bias"])
                  for b, t, _, x in adapter.outputs(result)}
        expected = {(t, b): tensor(value, dtype)[b] for t, value in enumerate(record["logits"])
                    for b in range(record["batch_size"])}
        equivalent(actual, expected, "single original logits")
        assert all(actual[k].argmax() == value.argmax() for k, value in expected.items())
        hidden(rm, adapter.read_view(result.continuation), record["read_hidden"], dtype)
    return stop
